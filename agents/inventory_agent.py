"""
agents/inventory_agent.py

Inventory Monitoring Watcher
─────────────────────────────
Role   : Compare stock vs reorder thresholds; emit a prioritised replenishment list.
Tools  : inventory_db (get_stock_levels, get_items_below_reorder, get_inventory_summary)
Memory : Entity memory — persists SKU profiles across calls (JSON file).

Two usage modes
───────────────
1. Standalone (UC-1 single-agent):
       from agents.inventory_agent import run_inventory_check
       answer = run_inventory_check("Which SKUs are below reorder in the North warehouse?")

2. Integrated (LangGraph node):
       from agents.inventory_agent import get_low_stock_report
       low_stock_items = get_low_stock_report(region="North")
       # Returns a list ready to be written into SCMState["low_stock_items"]
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

from tools.inventory_db import INVENTORY_TOOLS, query_items_below_reorder, query_all_inventory_summary
from utils.logger import get_logger, log_tool_call, log_agent_step, log_agent_output

load_dotenv()

logger = get_logger("InventoryAgent")

# ── Entity memory store (JSON file for SKU profiles) ──────────────────────────
_ENTITY_STORE_PATH = Path(__file__).parent.parent / "data" / "entity_memory_inventory.json"


def _load_entity_memory() -> dict:
    if _ENTITY_STORE_PATH.exists():
        with open(_ENTITY_STORE_PATH) as f:
            return json.load(f)
    return {}


def _save_entity_memory(store: dict) -> None:
    _ENTITY_STORE_PATH.parent.mkdir(exist_ok=True)
    with open(_ENTITY_STORE_PATH, "w") as f:
        json.dump(store, f, indent=2)


def _update_entity_memory(items: list[dict]) -> None:
    """Persist SKU profile facts learned during this run."""
    store = _load_entity_memory()
    for item in items:
        sku = item.get("sku")
        if sku:
            store[sku] = {
                **store.get(sku, {}),
                "last_checked": datetime.now().isoformat(),
                "on_hand": item.get("on_hand"),
                "reorder_point": item.get("reorder_point"),
                "priority": item.get("priority"),
                "warehouse": item.get("warehouse"),
            }
    _save_entity_memory(store)


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

SYSTEM_PROMPT = """You are the Inventory Monitoring Watcher for HexaShop E-Commerce.

Your ONLY job is to inspect current inventory against reorder thresholds and report
the truth — no guessing, no hallucinating stock numbers.

Rules:
- ALWAYS use the inventory tools to fetch stock data. Never invent numbers.
- When asked about multiple SKUs or regions, call the appropriate tool.
- Return structured, factual answers. If a SKU is below reorder point, say so clearly.
- Prioritise items by urgency: CRITICAL > HIGH > MEDIUM > LOW.
- If the user asks a question you cannot answer with inventory tools, say so honestly.
"""


# ── Standalone runner (UC-1: single-agent Q&A) ────────────────────────────────

def run_inventory_check(question: str, verbose: bool = True) -> str:
    """
    UC-1: A manager asks a natural-language question about inventory.
    The agent uses inventory_db tools to answer — no hallucination.

    Args:
        question: Natural language question about inventory.
        verbose:  Print agent reasoning steps.

    Returns:
        Agent's grounded answer as a string.
    """
    log_agent_step(logger, "run_inventory_check", question)

    llm = _build_llm()
    system_msg = SYSTEM_PROMPT
    agent = create_react_agent(model=llm, tools=INVENTORY_TOOLS, prompt=system_msg)

    config = {"recursion_limit": 15}
    result = agent.invoke({"messages": [HumanMessage(content=question)]}, config=config)
    # Extract the last AI message
    answer = result["messages"][-1].content
    log_agent_output(logger, "InventoryAgent", answer)
    return answer


# ── Integration helper (LangGraph node) ───────────────────────────────────────

def get_low_stock_report(region: Optional[str] = None) -> list[dict]:
    """
    Called by the LangGraph Supervisor node.
    Returns a structured list of low-stock items ready for SCMState["low_stock_items"].

    Args:
        region: Optional region filter ('North', 'South', 'West'). None = all regions.

    Returns:
        List of dicts with rank, sku, warehouse, on_hand, reorder_point, deficit, priority.
    """
    log_agent_step(logger, "get_low_stock_report", f"region={region or 'all'}")
    items = query_items_below_reorder(region=region)
    _update_entity_memory(items)
    log_agent_output(logger, "InventoryAgent", f"{len(items)} low-stock items found")
    return items


def get_full_inventory_snapshot() -> list[dict]:
    """Return complete inventory snapshot (used by Forecasting Agent)."""
    return query_all_inventory_summary()


# ── CrewAI Agent definition ───────────────────────────────────────────────────

def build_crewai_inventory_agent():
    """
    Build and return the CrewAI Inventory Monitoring Watcher agent.
    Imported by orchestration/graph.py to assemble the crew.
    """
    try:
        from crewai import Agent
        llm = _build_llm()
        return Agent(
            role="Inventory Monitoring Watcher",
            goal=(
                "Compare current stock levels against reorder thresholds for all SKUs "
                "and warehouses. Produce a prioritised list of items that need replenishment, "
                "ordered from most to least urgent."
            ),
            backstory=(
                "You are HexaShop's senior inventory analyst with 8 years of experience "
                "monitoring warehouse stock across North, South, and West distribution centres. "
                "You are methodical, data-driven, and you NEVER guess stock numbers — "
                "every figure you report comes directly from the inventory database. "
                "Your replenishment alerts have prevented stock-outs on the busiest shopping days."
            ),
            tools=INVENTORY_TOOLS,
            llm=llm,
            verbose=True,
            memory=True,
            allow_delegation=False,
        )
    except ImportError:
        logger.warning("CrewAI not installed. build_crewai_inventory_agent() unavailable.")
        return None


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  INVENTORY MONITORING AGENT — Standalone Mode")
    print("=" * 60)
    print("Type your inventory question below (or 'quit' to exit).\n")

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
        print("\n[Agent thinking...]\n")
        answer = run_inventory_check(q, verbose=True)
        print(f"\n[Answer]\n{answer}\n")
        print("-" * 60)
