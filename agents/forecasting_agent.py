"""
Forecasting Agent — forecasts demand and detects low-stock SKUs needing replenishment.
Part of the UC-2 multi-agent procurement pipeline.
"""
from tools.forecast_model import forecast_demand
from tools.inventory_db import get_low_stock_items
from utils.logger import get_logger

log = get_logger(__name__)


def forecasting_agent(state: dict) -> dict:
    """
    LangGraph node:
    1. Detect all low-stock items across warehouses.
    2. For each low-stock SKU, forecast 7-day demand.
    3. Pass structured results downstream to the procurement agent.
    """
    try:
        # Step 1: find everything below reorder point
        low_items = get_low_stock_items.invoke({"warehouse": ""})
        if not low_items:
            log.info("No low-stock items found — nothing to replenish.")
            return {
                "forecast_result": {"message": "All SKUs above reorder level"},
                "low_stock_items": [],
            }

        # Step 2: enrich each with demand forecast
        enriched = []
        seen_skus = set()
        for item in low_items:
            sku = item["sku"]
            if sku in seen_skus:
                continue  # one forecast per SKU
            seen_skus.add(sku)

            forecast = forecast_demand.invoke({"sku": sku})
            item["forecast"] = forecast
            item["recommended_order_qty"] = max(
                item.get("reorder_qty", 0),
                forecast.get("forecast_total_units", 0),
            )
            enriched.append(item)

        log.info("Forecasting complete — %d SKUs need replenishment", len(enriched))
        return {
            "forecast_result": {"skus_analyzed": len(enriched)},
            "low_stock_items": enriched,
        }

    except Exception as e:
        log.error("Forecasting agent error: %s", e)
        return {
            "forecast_result": {"error": str(e)},
            "low_stock_items": [],
            "error": str(e),
        }
