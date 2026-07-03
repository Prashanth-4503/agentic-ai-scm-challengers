import json
# pyrefly: ignore [missing-import]
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
from tools.calculator import calculate_po_cost

from utils.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
    PO_APPROVAL_THRESHOLD,
)
from utils.memory_store import (
    save_forecast_record,
    save_supplier_decision,
    upsert_entity,
    get_supplier_history,
    get_forecast_history,
)


llm = LLM(
    model=f"azure/{AZURE_OPENAI_DEPLOYMENT}",
    api_key=AZURE_OPENAI_API_KEY,
    base_url=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.0
)


def _safe_json_parse(text: str):
    if not text:
        return None
    text = text.strip()
    # Strip markdown code blocks if present
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

    inventory_specialist = Agent(
        role="Inventory Monitoring Agent",
        goal="Answer inventory questions using only tool-grounded stock data",
        backstory="You are a careful inventory analyst. You never invent stock numbers.",
        llm=llm,
        verbose=False,
    )

    task = Task(
        description=f"""
Answer the manager question using ONLY this grounded inventory data.

Question:
{question}

Inventory data:
{json.dumps(inventory_data, indent=2)}

Return JSON with:
{{
  "answer": "...",
  "records": [...]
}}
""",
        agent=inventory_specialist,
        expected_output='JSON with "answer" and "records"',
    )

    crew = Crew(
        agents=[inventory_specialist],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()
    raw_text = getattr(result, "raw", str(result))
    parsed = _safe_json_parse(str(raw_text)) or {
        "answer": str(raw_text),
        "records": inventory_data,
    }

    return {
        "inventory_result": parsed,
        "final_response": parsed.get("answer", str(parsed)),
    }


def forecasting_agent(state):
    low_stock_items = state.get("low_stock_items")

    if not low_stock_items:
        low_stock_items = get_low_stock_items.invoke({})

    enriched = []

    forecasting_specialist = Agent(
        role="Demand Forecasting Agent",
        goal="Predict short-term demand risk for low-stock SKUs",
        backstory="You analyze recent sales history and identify likely replenishment needs.",
        llm=llm,
        verbose=False,
    )

    for item in low_stock_items:
        sku = item["sku"]
        forecast = forecast_demand.invoke({"sku": sku, "window_days": 7, "horizon_days": 7})
        history = get_forecast_history(sku)

        payload = {
            "sku": sku,
            "warehouse": item.get("warehouse"),
            "on_hand": item.get("on_hand"),
            "reorder_point": item.get("reorder_point"),
            "reorder_qty": item.get("reorder_qty"),
            "forecast": forecast,
            "past_forecasts_count": len(history),
        }

        task = Task(
            description=f"""
Analyze this SKU's near-term demand risk and summarize whether replenishment is needed.

Data:
{json.dumps(payload, indent=2)}

Return JSON:
{{
  "sku": "{sku}",
  "forecast_units_7d": number,
  "recommended_reorder_qty": number,
  "risk": "LOW|MEDIUM|HIGH",
  "reason": "..."
}}
""",
            agent=forecasting_specialist,
            expected_output="JSON demand assessment",
        )

        crew = Crew(
            agents=[forecasting_specialist],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        )

        result = crew.kickoff()
        raw_text = getattr(result, "raw", str(result))
        parsed = _safe_json_parse(str(raw_text)) or {
            "sku": sku,
            "forecast_units_7d": forecast.get("forecast_total_units", 0) if isinstance(forecast, dict) else 0,
            "recommended_reorder_qty": item.get("reorder_qty", 0),
            "risk": "MEDIUM",
            "reason": "Fallback forecast result.",
        }

        save_forecast_record(parsed)
        upsert_entity("sku_profiles", sku, parsed)
        enriched.append(parsed)

    return {
        "forecast_result": {"items": enriched},
        "low_stock_items": enriched,
    }


def procurement_agent(state):
    items = state.get("low_stock_items", [])

    if not items:
        return {
            "procurement_result": {"message": "No low-stock items found."},
            "po_approval_required": False,
        }

    target = items[0]
    sku = target["sku"]
    needed_qty = int(target.get("recommended_reorder_qty", 0) or target.get("reorder_qty", 0) or 100)
    if needed_qty <= 0:
        needed_qty = 100
    quotes = get_suppliers_for_sku.invoke({"sku": sku})
    supplier_history = get_supplier_history(sku=sku)

    procurement_specialist = Agent(
        role="Procurement Specialist",
        goal="Choose the best supplier and draft a purchase order",
        backstory="You optimize cost, lead time, and reliability without overspending.",
        llm=llm,
        verbose=False,
    )

    task = Task(
        description=f"""
Choose the best supplier for this SKU and draft a purchase order.

SKU: {sku}
Needed Qty: {needed_qty}

Available supplier quotes:
{json.dumps(quotes, indent=2)}

Past supplier history for this SKU:
{json.dumps(supplier_history, indent=2)}

Return strict JSON:
{{
  "sku": "{sku}",
  "qty": number,
  "supplier_id": "...",
  "supplier": "...",
  "unit_cost": number,
  "total_cost": number,
  "lead_time_days": number,
  "reason": "..."
}}
""",
        agent=procurement_specialist,
        expected_output="PO draft JSON",
    )

    crew = Crew(
        agents=[procurement_specialist],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()
    raw_text = getattr(result, "raw", str(result))
    parsed = _safe_json_parse(str(raw_text))

    if not parsed:
        valid_quotes = [q for q in quotes if isinstance(q, dict) and "error" not in q]
        if valid_quotes:
            best = min(valid_quotes, key=lambda x: (x.get("unit_cost", 999.0), x.get("lead_time_days", 99)))
            total_cost_value = round(best.get("unit_cost", 0.0) * needed_qty, 2)
            parsed = {
                "sku": sku,
                "qty": needed_qty,
                "supplier_id": best["supplier_id"],
                "supplier": best["supplier_name"],
                "unit_cost": best["unit_cost"],
                "total_cost": total_cost_value,
                "lead_time_days": best["lead_time_days"],
                "reason": "Fallback supplier choice based on lowest cost and lead time.",
            }
        else:
            parsed = {
                "sku": sku,
                "qty": needed_qty,
                "supplier_id": "SUPP-MOCK",
                "supplier": "Mock Supplier",
                "unit_cost": 10.0,
                "total_cost": round(10.0 * needed_qty, 2),
                "lead_time_days": 3,
                "reason": "No valid supplier quotes available. Mock fallback used.",
            }

    save_supplier_decision(parsed)
    upsert_entity("supplier_profiles", parsed["supplier_id"], parsed)

    return {
        "po_draft": parsed,
        "procurement_result": parsed,
        "po_approval_required": parsed["total_cost"] > PO_APPROVAL_THRESHOLD,
    }


def logistics_agent(state):
    orders = state.get("orders_batch", [])

    if not orders:
        # Try to infer region from query
        query = state.get("user_query", "").lower()
        region = "North"
        for r in ["North", "South", "West"]:
            if r.lower() in query:
                region = r
                break
        
        # Load pending orders for that region from orders.csv
        try:
            import pandas as pd
            from utils.config import DATA_DIR
            df_orders = pd.read_csv(DATA_DIR / "orders.csv")
            # Filter pending/allocated orders in the specified region
            matched = df_orders[
                (df_orders["ship_to_region"].str.lower() == region.lower()) &
                (df_orders["status"].isin(["Pending", "Allocated"]))
            ].copy()
            orders = matched.to_dict(orient="records")[:5] # limit to top 5
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
                "weight_kg": 2.0
            }]

    logistics_specialist = Agent(
        role="Logistics & Routing Agent",
        goal="Choose the best shipping option balancing cost and ETA",
        backstory="You optimize shipping plans for pending orders across regions.",
        llm=llm,
        verbose=False,
    )

    enriched = []

    for order in orders:
        options = get_shipping_options.invoke(
            {
                "region": order["ship_to_region"],
                "weight_kg": order.get("weight_kg", 1.0),
            }
        )

        task = Task(
            description=f"""
Choose the best carrier for this order.

Order:
{json.dumps(order, indent=2)}

Shipping options:
{json.dumps(options, indent=2)}

Return JSON:
{{
  "order_id": "{order['order_id']}",
  "selected_carrier": "...",
  "cost": number,
  "eta_days": number,
  "reason": "..."
}}
""",
            agent=logistics_specialist,
            expected_output="Shipping choice JSON",
        )

        crew = Crew(
            agents=[logistics_specialist],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        )

        result = crew.kickoff()
        raw_text = getattr(result, "raw", str(result))
        parsed = _safe_json_parse(str(raw_text))

        if not parsed and options:
            best = sorted(options, key=lambda x: (x["eta_days"], x["estimated_cost"]))[0]
            parsed = {
                "order_id": order["order_id"],
                "selected_carrier": best["carrier_name"],
                "cost": best["estimated_cost"],
                "eta_days": best["eta_days"],
                "reason": "Fallback best ETA/cost tradeoff.",
            }

        if parsed:
            enriched.append(parsed)

    # Format a nice final response summary
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
            "final_response": "No delay notifications needed — no affected customer orders found."
        }

    comms_specialist = Agent(
        role="Customer Communications Agent",
        goal="Draft clear, empathetic customer delay messages",
        backstory="You write concise, trustworthy updates for customers affected by supply delays.",
        llm=llm,
        verbose=False,
    )

    drafted = []

    for order in impacted_orders:
        base_message = draft_delay_message.invoke({"order": order})

        task = Task(
            description=f"""
Rewrite this delay message to be clear, empathetic, and professional.

Order data:
{json.dumps(order, indent=2)}

Base draft:
{base_message}

Return JSON:
{{
  "order_id": "{order['order_id']}",
  "customer_id": "{order['customer_id']}",
  "message": "..."
}}
""",
            agent=comms_specialist,
            expected_output="Customer message JSON",
        )

        crew = Crew(
            agents=[comms_specialist],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        )

        result = crew.kickoff()
        raw_text = getattr(result, "raw", str(result))
        parsed = _safe_json_parse(str(raw_text)) or {
            "order_id": order["order_id"],
            "customer_id": order["customer_id"],
            "message": base_message,
        }

        upsert_entity("customer_profiles", order["customer_id"], parsed)
        drafted.append(parsed)

    # Format a nice final response summary
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