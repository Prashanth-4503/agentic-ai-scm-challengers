"""
agents/forecasting_agent.py

Demand Forecasting Analyst
──────────────────────────
Role   : Predict near-term demand per SKU from sales history; flag stock-out / overstock risk.
Tools  : forecast_model (forecast_demand, get_sales_history, get_at_risk_skus)
         inventory_db  (get_stock_levels)
         calculator    (calculate_days_of_cover)
Memory : Long-term semantic memory via ChromaDB — stores past forecasts for trend comparison.

Two usage modes
───────────────
1. Standalone (analyse a specific SKU or all at-risk SKUs):
       from agents.forecasting_agent import run_forecast_analysis
       report = run_forecast_analysis("ELC-1009", horizon_days=7)

2. Integrated (LangGraph node — takes low_stock_items from Inventory Agent):
       from agents.forecasting_agent import run_forecasting_for_state
       forecasts = run_forecasting_for_state(low_stock_items)
       # Returns list ready for SCMState["forecasts"]
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI

from tools.forecast_model import FORECAST_TOOLS, compute_forecast, detect_all_at_risk
from tools.inventory_db import INVENTORY_TOOLS
from tools.calculator import CALCULATOR_TOOLS
from utils.logger import get_logger, log_agent_step, log_agent_output

load_dotenv()

logger = get_logger("ForecastingAgent")

# All tools available to the Forecasting Agent
_FORECASTING_TOOLS = FORECAST_TOOLS + [INVENTORY_TOOLS[0]] + [CALCULATOR_TOOLS[0]]
# (get_stock_levels + calculate_days_of_cover to complement forecast)


# ── Long-term memory (ChromaDB) ────────────────────────────────────────────────

_CHROMA_DIR = str(Path(__file__).parent.parent / "data" / "chroma_forecasts")


def _get_chroma_collection():
    """Return (or create) the ChromaDB forecasts collection."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=_CHROMA_DIR)
        return client.get_or_create_collection(
            name="past_forecasts",
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as e:
        logger.warning(f"ChromaDB unavailable ({e}). Long-term memory disabled.")
        return None


def _store_forecast_memory(forecast: dict) -> None:
    """Persist a forecast result to ChromaDB for long-term recall."""
    collection = _get_chroma_collection()
    if not collection:
        return
    try:
        doc_id = f"{forecast['sku']}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        text = (
            f"SKU: {forecast['sku']} | Risk: {forecast.get('risk_level')} | "
            f"DaysOfCover: {forecast.get('days_of_cover')} | "
            f"AvgDaily: {forecast.get('avg_daily_demand')} | "
            f"Trend: {forecast.get('trend')} | "
            f"Recommendation: {forecast.get('recommendation')}"
        )
        collection.add(
            documents=[text],
            metadatas=[{"sku": forecast["sku"], "timestamp": datetime.now().isoformat(),
                        "risk_level": forecast.get("risk_level", "UNKNOWN")}],
            ids=[doc_id],
        )
        logger.info(f"[MEMORY] Stored forecast for {forecast['sku']} in ChromaDB")
    except Exception as e:
        logger.warning(f"[MEMORY] Failed to store forecast: {e}")


def recall_past_forecasts(sku: str, n_results: int = 3) -> list[dict]:
    """Query ChromaDB for previous forecasts of a SKU."""
    collection = _get_chroma_collection()
    if not collection:
        return []
    try:
        results = collection.query(
            query_texts=[f"SKU: {sku}"],
            n_results=n_results,
            where={"sku": sku},
        )
        return results.get("metadatas", [[]])[0]
    except Exception as e:
        logger.warning(f"[MEMORY] Recall failed for {sku}: {e}")
        return []


# ── LLM setup ─────────────────────────────────────────────────────────────────

def _build_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        temperature=0,
    )


# ── Prompt template ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the Demand Forecasting Analyst for HexaShop E-Commerce.

Your mission is to predict near-term demand for SKUs using actual sales history data
and flag any SKUs at risk of stock-out or overstock.

Rules:
- ALWAYS use the forecast_demand tool to generate predictions. Never invent numbers.
- Use get_sales_history if you need to inspect raw sales patterns.
- Use get_at_risk_skus to find all problematic SKUs at once.
- Use calculate_days_of_cover to translate inventory into days of supply.
- Your output must include: risk level (CRITICAL/WARNING/OK/OVERSTOCK), days of cover,
  and a clear recommendation.
- If you see a CRITICAL risk, say so prominently.
"""


# ── Standalone runner ──────────────────────────────────────────────────────────

def run_forecast_analysis(question_or_sku: str, horizon_days: int = 7, verbose: bool = True) -> str:
    """
    Standalone mode: analyse demand for one SKU or ask a free-form question.

    Args:
        question_or_sku: A SKU (e.g. 'ELC-1009') or natural-language question.
        horizon_days:    Forecast window in days (default 7).
        verbose:         Print agent reasoning steps.

    Returns:
        Agent's analysis as a string.
    """
    # If a bare SKU was passed, turn it into a structured question
    if question_or_sku.upper() == question_or_sku and "-" in question_or_sku:
        question = (
            f"Analyse demand for SKU {question_or_sku} over the next {horizon_days} days. "
            f"Provide avg daily demand, trend, days of cover, risk level and recommendation."
        )
    else:
        question = question_or_sku

    log_agent_step(logger, "run_forecast_analysis", question)

    llm = _build_llm()
    agent = create_react_agent(model=llm, tools=_FORECASTING_TOOLS, prompt=SYSTEM_PROMPT)
    config = {"recursion_limit": 20}
    result = agent.invoke({"messages": [HumanMessage(content=question)]}, config=config)
    answer = result["messages"][-1].content
    log_agent_output(logger, "ForecastingAgent", answer)
    return answer


# ── Integration helper (LangGraph node) ───────────────────────────────────────

def run_forecasting_for_state(
    low_stock_items: Optional[list[dict]] = None,
    top_n: int = 20,
) -> list[dict]:
    """
    Called by the LangGraph Supervisor.
    Takes the low_stock_items list from the Inventory Agent (or scans all SKUs)
    and returns enriched forecast dicts for SCMState["forecasts"].

    Args:
        low_stock_items: Output from inventory_agent.get_low_stock_report().
                         If None, scans all SKUs automatically.
        top_n:           Max SKUs to process when scanning all.

    Returns:
        List of forecast dicts with risk_level, days_of_cover, recommendation.
    """
    if low_stock_items:
        log_agent_step(logger, "run_forecasting_for_state",
                       f"Processing {len(low_stock_items)} low-stock SKUs")
        # Forecast only the items flagged by Inventory Agent
        unique_skus = list({item["sku"] for item in low_stock_items})
        forecasts = []
        for sku in unique_skus:
            fc = compute_forecast(sku)
            if "error" not in fc:
                _store_forecast_memory(fc)
                forecasts.append(fc)
    else:
        log_agent_step(logger, "run_forecasting_for_state", "Scanning all SKUs for risk")
        forecasts = detect_all_at_risk(top_n=top_n)
        for fc in forecasts:
            _store_forecast_memory(fc)

    # Sort: CRITICAL first, then WARNING, then OVERSTOCK, then OK
    _rank = {"CRITICAL": 0, "WARNING": 1, "OVERSTOCK": 2, "OK": 3}
    forecasts.sort(key=lambda x: (_rank.get(x.get("risk_level", "OK"), 4),
                                   x.get("days_of_cover", 999)))

    log_agent_output(logger, "ForecastingAgent",
                     f"{len(forecasts)} forecasts generated. "
                     f"Critical: {sum(1 for f in forecasts if f.get('risk_level') == 'CRITICAL')}")
    return forecasts


# ── CrewAI Agent definition ────────────────────────────────────────────────────

def build_crewai_forecasting_agent():
    """
    Build and return the CrewAI Demand Forecasting Analyst agent.
    Imported by orchestration/graph.py to assemble the crew.
    """
    try:
        from crewai import Agent
        llm = _build_llm()
        return Agent(
            role="Demand Forecasting Analyst",
            goal=(
                "Predict near-term demand per SKU from historical sales data and "
                "identify every SKU at risk of stock-out or overstock before it becomes a problem."
            ),
            backstory=(
                "You are HexaShop's senior data scientist with deep expertise in e-commerce "
                "demand forecasting. You have built forecasting models that reduced stock-outs by 40%. "
                "You always base predictions on actual sales data from the database — "
                "never on gut feel or assumptions. "
                "You flag CRITICAL risks loudly and give clear, actionable recommendations."
            ),
            tools=_FORECASTING_TOOLS,
            llm=llm,
            verbose=True,
            memory=True,
            allow_delegation=False,
        )
    except ImportError:
        logger.warning("CrewAI not installed. build_crewai_forecasting_agent() unavailable.")
        return None


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("\n" + "=" * 60)
    print("  DEMAND FORECASTING AGENT — Standalone Mode")
    print("=" * 60)

    # Quick mode: python forecasting_agent.py ELC-1009
    if len(sys.argv) > 1:
        sku_arg = sys.argv[1].upper()
        print(f"\nRunning forecast for SKU: {sku_arg}\n")
        result = run_forecast_analysis(sku_arg, verbose=True)
        print(f"\n[Forecast Report]\n{result}\n")
        sys.exit(0)

    print("Type a SKU or a demand question below (or 'scan' to find all at-risk SKUs).\n")
    while True:
        try:
            q = input("Manager > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue
        if q.lower() == "scan":
            print("\n[Scanning all SKUs for demand risk...]\n")
            results = run_forecasting_for_state()
            for fc in results[:10]:
                print(f"  [{fc['risk_level']:8}] {fc['sku']} | {fc['days_of_cover']} days cover | {fc['recommendation']}")
            print()
            continue
        print("\n[Agent thinking...]\n")
        answer = run_forecast_analysis(q, verbose=True)
        print(f"\n[Analysis]\n{answer}\n")
        print("-" * 60)
