"""
CrewAI specialist agents — Inventory, Forecasting, Procurement, Logistics,
Customer Comms. Each is a LangGraph node function: (state) -> partial state.

Design notes
------------
- Every agent has real LangChain tools bound via `tools=[...]` on the CrewAI
  `Agent`, so tool calls are actually invoked by the agent/LLM loop, not just
  pre-fetched and pasted into the prompt (which was the case in the first
  draft of this file). We *also* pre-fetch the primary dataset before
  building the Task so the prompt is grounded even if the model chooses not
  to call a tool — belt and braces for a bootcamp demo.
- `_run_crew()` wraps every `crew.kickoff()` call with a short retry loop and
  a typed exception, so a flaky Azure call degrades gracefully (logged +
  fallback data) instead of crashing the whole graph — see the Resilience
  non-functional requirement.
- Procurement escalates to a human (UC-5) instead of silently inventing a
  "Mock Supplier" when no real supplier can meet demand.
"""
import json
import time

from crewai import Agent, Task, Crew, Process, LLM

from tools.inventory_db import (
    search_inventory,
    get_low_stock_items,
    get_sku_detail,
    get_inventory_summary,
)
from tools.forecast_model import forecast_demand
from tools.supplier_api import get_suppliers_for_sku
from tools.shipping_api import get_shipping_options
from tools.notify_tool import draft_delay_message, get_affected_orders
from tools.calculator import build_purchase_order
from tools.web_search import web_search

from utils.tool_adapter import to_crewai_tools
from utils.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
    PO_APPROVAL_THRESHOLD,
    VERBOSE_AGENTS,
)
from utils.memory_store import (
    save_forecast_record,
    save_supplier_decision,
    upsert_entity,
    get_supplier_history,
    get_forecast_history,
    semantic_recall,
)
from utils.logger import get_logger
from utils.prompt_loader import get_agent_prompt, get_task_prompt

log = get_logger(__name__)

llm = LLM(
    model=f"azure/{AZURE_OPENAI_DEPLOYMENT}",
    api_key=AZURE_OPENAI_API_KEY,
    base_url=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.0,
)

_MAX_RETRIES = 2


def _run_crew(crew: Crew, fallback: dict, context: str = ""):
    """Run a CrewAI crew with a short retry loop. Returns (parsed_dict, used_fallback: bool).

    Never raises — a persistent failure returns `fallback` so the calling
    agent node can still return a valid partial state and the graph keeps
    running (resilience requirement).
    """
    # Dynamically resolve agent name from context, e.g. "inventory_agent" or "forecasting_agent[SKU-1001]"
    agent_name = context.split("[")[0] if context else "app"
    logger = get_logger(agent_name)

    last_err = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            result = crew.kickoff()
            raw_text = getattr(result, "raw", str(result))
            parsed = _safe_json_parse(str(raw_text))
            if parsed:
                return parsed, False
            logger.warning("%s: crew returned non-JSON output, using fallback.", context)
            return dict(fallback), True
        except Exception as e:
            last_err = e
            logger.warning("%s: crew.kickoff() failed (attempt %d/%d): %s",
                           context, attempt, _MAX_RETRIES, e)
            time.sleep(0.5 * attempt)

    logger.error("%s: all retries exhausted (%s), falling back to safe default.",
                 context, last_err)
    return dict(fallback), True


def _safe_json_parse(text: str):
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        return json.loads(text)
    except Exception:
        import re
        try:
            match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
        except Exception:
            pass
        return None


# ── Inventory Agent (UC-1) ────────────────────────────────────────────

def inventory_agent(state):
    question = state.get("user_query", "")

    if not question.strip():
        inventory_data = get_inventory_summary.invoke({})
    elif "north" in question.lower() and ("low" in question.lower() or "reorder" in question.lower()):
        inventory_data = get_low_stock_items.invoke({"warehouse": "North"})
    elif "south" in question.lower() and ("low" in question.lower() or "reorder" in question.lower()):
        inventory_data = get_low_stock_items.invoke({"warehouse": "South"})
    elif "west" in question.lower() and ("low" in question.lower() or "reorder" in question.lower()):
        inventory_data = get_low_stock_items.invoke({"warehouse": "West"})
    else:
        inventory_data = search_inventory.invoke({"query": question})

    agent_prompt = get_agent_prompt("inventory_agent")
    inventory_specialist = Agent(
        role=agent_prompt.get("role", "Inventory Monitoring Agent"),
        goal=agent_prompt.get("goal", "Answer inventory questions using only tool-grounded stock data"),
        backstory=agent_prompt.get("backstory", "You are a careful inventory analyst. You never invent stock numbers."),
        tools=to_crewai_tools([get_inventory_summary, get_low_stock_items, get_sku_detail, search_inventory]),
        llm=llm,
        verbose=VERBOSE_AGENTS,
    )

    task_prompt = get_task_prompt("inventory_task")
    description_template = task_prompt.get("description", "")
    description = description_template.format(
        question=question,
        inventory_data=json.dumps(inventory_data, indent=2)
    )

    task = Task(
        description=description,
        agent=inventory_specialist,
        expected_output=task_prompt.get("expected_output", 'JSON with "answer" and "records"'),
    )

    crew = Crew(agents=[inventory_specialist], tasks=[task],
                process=Process.sequential, verbose=VERBOSE_AGENTS)

    fallback = {"answer": "Could not reach the reasoning engine — here is the raw grounded data.",
                "records": inventory_data}
    parsed, _ = _run_crew(crew, fallback, context="inventory_agent")

    return {
        "inventory_result": parsed,
        "final_response": parsed.get("answer", str(parsed)),
    }


# ── Forecasting Agent (UC-2, step 1) ──────────────────────────────────

def forecasting_agent(state):
    low_stock_items = state.get("low_stock_items")

    if not low_stock_items:
        low_stock_items = get_low_stock_items.invoke({})

    enriched = []

    agent_prompt = get_agent_prompt("forecasting_agent")
    forecasting_specialist = Agent(
        role=agent_prompt.get("role", "Demand Forecasting Agent"),
        goal=agent_prompt.get("goal", "Predict short-term demand risk for low-stock SKUs"),
        backstory=agent_prompt.get("backstory", "You analyze recent sales history and identify likely replenishment needs."),
        tools=to_crewai_tools([forecast_demand]),
        llm=llm,
        verbose=VERBOSE_AGENTS,
    )

    for item in low_stock_items:
        sku = item["sku"]
        forecast = forecast_demand.invoke({"sku": sku, "window_days": 7, "horizon_days": 7})
        history = get_forecast_history(sku)
        similar_past = semantic_recall("forecasts", f"SKU {sku} demand forecast", n_results=2)

        payload = {
            "sku": sku,
            "warehouse": item.get("warehouse"),
            "on_hand": item.get("on_hand"),
            "reorder_point": item.get("reorder_point"),
            "reorder_qty": item.get("reorder_qty"),
            "forecast": forecast,
            "past_forecasts_count": len(history),
            "similar_past_forecasts": similar_past,
        }

        task_prompt = get_task_prompt("forecasting_task")
        description_template = task_prompt.get("description", "")
        description = description_template.format(
            payload=json.dumps(payload, indent=2),
            sku=sku
        )

        task = Task(
            description=description,
            agent=forecasting_specialist,
            expected_output=task_prompt.get("expected_output", "JSON demand assessment"),
        )

        crew = Crew(agents=[forecasting_specialist], tasks=[task],
                    process=Process.sequential, verbose=VERBOSE_AGENTS)

        fallback = {
            "sku": sku,
            "forecast_units_7d": forecast.get("forecast_total_units", 0) if isinstance(forecast, dict) else 0,
            "recommended_reorder_qty": item.get("reorder_qty", 0),
            "risk": "MEDIUM",
            "reason": "Fallback forecast result (reasoning engine unavailable).",
        }
        parsed, _ = _run_crew(crew, fallback, context=f"forecasting_agent[{sku}]")

        save_forecast_record(parsed)
        upsert_entity("sku_profiles", sku, parsed)
        enriched.append(parsed)

    return {
        "forecast_result": {"items": enriched},
        "low_stock_items": enriched,
    }


# ── Procurement Agent (UC-2, step 2) ──────────────────────────────────

def procurement_agent(state):
    log = get_logger("procurement_agent")
    items = state.get("low_stock_items", [])

    if not items:
        return {
            "procurement_result": {"message": "No low-stock items found."},
            "po_approval_required": False,
            "escalation_required": False,
        }

    target = items[0]
    sku = target["sku"]
    needed_qty = int(target.get("recommended_reorder_qty", 0) or target.get("reorder_qty", 0) or 100)
    if needed_qty <= 0:
        needed_qty = 100

    quotes = get_suppliers_for_sku.invoke({"sku": sku})
    valid_quotes = [q for q in quotes if isinstance(q, dict) and "error" not in q
                    and q.get("available_qty", 0) >= needed_qty]

    # ── UC-5: no supplier can meet demand → escalate instead of guessing ──
    if not valid_quotes:
        any_quotes = [q for q in quotes if isinstance(q, dict) and "error" not in q]
        summary = (
            f"No supplier can fully meet the required quantity of {needed_qty} units "
            f"for SKU {sku}."
        )
        options = []
        if any_quotes:
            best_partial = max(any_quotes, key=lambda x: x.get("available_qty", 0))
            options.append(
                f"Accept a partial order of {best_partial.get('available_qty')} units "
                f"from {best_partial.get('supplier_name')} now, and re-order the rest later."
            )
            options.append("Split the order across multiple suppliers to cover full demand.")
        options.append("Search the web for an alternative supplier not in the current catalog.")
        options.append("Reduce the reorder quantity and accept a higher stock-out risk.")

        return {
            "procurement_result": {"message": summary, "quotes": quotes},
            "po_approval_required": False,
            "escalation_required": True,
            "escalation_payload": {
                "summary": summary,
                "sku": sku,
                "needed_qty": needed_qty,
                "available_quotes": any_quotes,
                "options": options,
            },
        }

    supplier_history = get_supplier_history(sku=sku)
    similar_past = semantic_recall("supplier_decisions", f"purchase order for SKU {sku}", n_results=2)

    agent_prompt = get_agent_prompt("procurement_agent")
    procurement_specialist = Agent(
        role=agent_prompt.get("role", "Procurement Specialist"),
        goal=agent_prompt.get("goal", "Choose the best supplier and draft a purchase order"),
        backstory=agent_prompt.get("backstory", "You optimize cost, lead time, and reliability without overspending."),
        tools=to_crewai_tools([get_suppliers_for_sku, build_purchase_order, web_search]),
        llm=llm,
        verbose=VERBOSE_AGENTS,
    )

    task_prompt = get_task_prompt("procurement_task")
    description_template = task_prompt.get("description", "")
    description = description_template.format(
        sku=sku,
        needed_qty=needed_qty,
        valid_quotes=json.dumps(valid_quotes, indent=2),
        supplier_history=json.dumps(supplier_history, indent=2),
        similar_past=json.dumps(similar_past, indent=2)
    )

    task = Task(
        description=description,
        agent=procurement_specialist,
        expected_output=task_prompt.get("expected_output", "PO draft JSON"),
    )

    crew = Crew(agents=[procurement_specialist], tasks=[task],
                process=Process.sequential, verbose=VERBOSE_AGENTS)

    best = min(valid_quotes, key=lambda x: (x.get("unit_cost", 999.0), x.get("lead_time_days", 99)))
    fallback = build_purchase_order.invoke({
        "sku": sku,
        "qty": needed_qty,
        "supplier_id": best["supplier_id"],
        "supplier_name": best["supplier_name"],
        "unit_cost": best["unit_cost"],
        "lead_time_days": best["lead_time_days"],
    })
    fallback["reason"] = "Fallback supplier choice based on lowest cost and lead time."

    parsed, used_fallback = _run_crew(crew, fallback, context=f"procurement_agent[{sku}]")

    # Whatever the agent (or the fallback) produced, recompute the total via
    # the calculator tool so the approval threshold check is always trustworthy,
    # even if the LLM did its own (possibly wrong) arithmetic.
    try:
        recomputed = build_purchase_order.invoke({
            "sku": parsed.get("sku", sku),
            "qty": int(parsed.get("qty", needed_qty)),
            "supplier_id": parsed.get("supplier_id", best["supplier_id"]),
            "supplier_name": parsed.get("supplier", best["supplier_name"]),
            "unit_cost": float(parsed.get("unit_cost", best["unit_cost"])),
            "lead_time_days": int(parsed.get("lead_time_days", best["lead_time_days"])),
        })
        parsed["total_cost"] = recomputed["total_cost"]
        parsed["approval_required"] = recomputed["approval_required"]
    except Exception as e:
        log.warning("procurement_agent[%s]: could not recompute PO cost via calculator: %s", sku, e)
        parsed["approval_required"] = parsed.get("total_cost", 0) > PO_APPROVAL_THRESHOLD

    save_supplier_decision(parsed)
    upsert_entity("supplier_profiles", parsed.get("supplier_id", "unknown"), parsed)

    return {
        "po_draft": parsed,
        "procurement_result": parsed,
        "po_approval_required": parsed["approval_required"],
        "escalation_required": False,
    }


# ── Logistics Agent (UC-3) ────────────────────────────────────────────

def logistics_agent(state):
    log = get_logger("logistics_agent")
    orders = state.get("orders_batch", [])

    if not orders:
        query = state.get("user_query", "").lower()
        region = "North"
        for r in ["North", "South", "West"]:
            if r.lower() in query:
                region = r
                break

        try:
            import pandas as pd
            from utils.config import DATA_DIR
            df_orders = pd.read_csv(DATA_DIR / "orders.csv")
            matched = df_orders[
                (df_orders["ship_to_region"].str.lower() == region.lower()) &
                (df_orders["status"].isin(["Pending", "Allocated"]))
            ].copy()
            orders = matched.to_dict(orient="records")[:5]
        except Exception as e:
            log.warning("Failed to load orders for logistics fallback: %s", e)
            orders = []

        if not orders:
            orders = [{
                "order_id": "MOCK-ORD-100",
                "customer_id": "CUS-MOCK",
                "sku": "ELC-1001",
                "qty": 1,
                "ship_to_region": region,
                "weight_kg": 2.0,
            }]

    agent_prompt = get_agent_prompt("logistics_agent")
    logistics_specialist = Agent(
        role=agent_prompt.get("role", "Logistics & Routing Agent"),
        goal=agent_prompt.get("goal", "Choose the best shipping option balancing cost and ETA"),
        backstory=agent_prompt.get("backstory", "You optimize shipping plans for pending orders across regions."),
        tools=to_crewai_tools([get_shipping_options]),
        llm=llm,
        verbose=VERBOSE_AGENTS,
    )

    enriched = []

    for order in orders:
        options = get_shipping_options.invoke({
            "region": order["ship_to_region"],
            "weight_kg": order.get("weight_kg", 1.0),
        })

        task_prompt = get_task_prompt("logistics_task")
        description_template = task_prompt.get("description", "")
        description = description_template.format(
            order=json.dumps(order, indent=2),
            options=json.dumps(options, indent=2),
            order_id=order['order_id']
        )

        task = Task(
            description=description,
            agent=logistics_specialist,
            expected_output=task_prompt.get("expected_output", "Shipping choice JSON"),
        )

        crew = Crew(agents=[logistics_specialist], tasks=[task],
                    process=Process.sequential, verbose=VERBOSE_AGENTS)

        if options:
            best = sorted(options, key=lambda x: (x["eta_days"], x["estimated_cost"]))[0]
            fallback = {
                "order_id": order["order_id"],
                "selected_carrier": best["carrier_name"],
                "cost": best["estimated_cost"],
                "eta_days": best["eta_days"],
                "reason": "Fallback best ETA/cost tradeoff (reasoning engine unavailable).",
            }
        else:
            fallback = {
                "order_id": order["order_id"],
                "selected_carrier": None,
                "cost": None,
                "eta_days": None,
                "reason": f"No carriers available for region {order.get('ship_to_region')}.",
            }

        parsed, _ = _run_crew(crew, fallback, context=f"logistics_agent[{order['order_id']}]")
        if parsed:
            enriched.append(parsed)

    plans_summary = []
    for plan in enriched:
        plans_summary.append(
            f"- Order {plan.get('order_id')}: Carrier {plan.get('selected_carrier')} "
            f"(${plan.get('cost')}, ETA {plan.get('eta_days')}d)"
        )
    final_response = "Logistics plans generated:\n" + "\n".join(plans_summary)

    return {
        "logistics_result": {"plans": enriched},
        "final_response": final_response,
    }


# ── Customer Comms Agent (UC-4) ───────────────────────────────────────

def customer_comms_agent(state):
    impacted_orders = state.get("impacted_orders", [])

    if not impacted_orders:
        low_items = state.get("low_stock_items", [])
        query = state.get("user_query", "")
        skus_to_check = []
        if low_items:
            skus_to_check = list({item["sku"] for item in low_items if "sku" in item})
        else:
            import re
            found = re.findall(r"[A-Z]{3}-\d{4}", query.upper())
            skus_to_check = found if found else []

        impacted_orders = []
        for sku in skus_to_check[:5]:
            affected = get_affected_orders.invoke({"sku": sku})
            if isinstance(affected, list):
                impacted_orders.extend(affected)

    if not impacted_orders:
        return {
            "comms_result": {"message": "No impacted orders supplied."},
            "final_response": "No delay notifications needed — no affected customer orders found.",
        }

    agent_prompt = get_agent_prompt("customer_comms_agent")
    comms_specialist = Agent(
        role=agent_prompt.get("role", "Customer Communications Agent"),
        goal=agent_prompt.get("goal", "Draft clear, empathetic customer delay messages"),
        backstory=agent_prompt.get("backstory", "You write concise, trustworthy updates for customers affected by supply delays."),
        tools=to_crewai_tools([draft_delay_message]),
        llm=llm,
        verbose=VERBOSE_AGENTS,
    )

    drafted = []

    for order in impacted_orders:
        base_message = draft_delay_message.invoke({"order": order})

        task_prompt = get_task_prompt("customer_comms_task")
        description_template = task_prompt.get("description", "")
        description = description_template.format(
            order=json.dumps(order, indent=2),
            base_message=base_message,
            order_id=order['order_id'],
            customer_id=order['customer_id']
        )

        task = Task(
            description=description,
            agent=comms_specialist,
            expected_output=task_prompt.get("expected_output", "Customer message JSON"),
        )

        crew = Crew(agents=[comms_specialist], tasks=[task],
                    process=Process.sequential, verbose=VERBOSE_AGENTS)

        fallback = {
            "order_id": order["order_id"],
            "customer_id": order["customer_id"],
            "message": base_message,
        }
        parsed, _ = _run_crew(crew, fallback, context=f"customer_comms_agent[{order['order_id']}]")

        upsert_entity("customer_profiles", order["customer_id"], parsed)
        drafted.append(parsed)

    msgs_summary = []
    for d in drafted:
        msgs_summary.append(
            f"💬 **Order {d.get('order_id')}** (Customer {d.get('customer_id')}):\n"
            f"   \"{d.get('message')}\""
        )
    final_response = "Delay notifications drafted:\n\n" + "\n\n".join(msgs_summary)

    return {
        "comms_result": {"messages": drafted},
        "final_response": final_response,
    }