"""
Logistics Agent (UC-3 stretch) — selects shipping carrier for a PO/region.
"""
from tools.shipping_api import get_shipping_options, select_best_carrier
from utils.logger import get_logger

log = get_logger(__name__)


def logistics_agent(state: dict) -> dict:
    """
    LangGraph node: given a PO or user query, find the best carrier.
    Uses the po_draft region or defaults to 'North'.
    """
    try:
        query = state.get("user_query", "").lower()

        # Try to infer region from query
        region = "North"
        for r in ["North", "South", "West"]:
            if r.lower() in query:
                region = r
                break

        options = get_shipping_options.invoke({"region": region, "weight_kg": 2.0})
        best = select_best_carrier.invoke({"region": region, "weight_kg": 2.0,
                                           "prefer": "cost"})

        log.info("Logistics: %d carriers for %s, best=%s",
                 len(options), region, best.get("carrier_name"))

        return {
            "logistics_result": {
                "region": region,
                "all_options": options,
                "recommended": best,
            },
            "final_response": (
                f"Logistics plan for {region} region:\n"
                f"Recommended carrier: {best.get('carrier_name')} "
                f"({best.get('service_level')}) — "
                f"ETA {best.get('eta_days')}d, "
                f"est. ${best.get('estimated_cost')}"
            ),
        }

    except Exception as e:
        log.error("Logistics agent error: %s", e)
        return {
            "logistics_result": {"error": str(e)},
            "final_response": f"Logistics error: {e}",
        }
