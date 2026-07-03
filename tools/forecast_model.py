"""
Demand forecasting tool — simple moving average over sales_history.csv.
"""
import pandas as pd
from langchain_core.tools import tool
from utils.config import DATA_DIR
from utils.logger import get_logger

log = get_logger(__name__)

_sales = pd.read_csv(DATA_DIR / "sales_history.csv", parse_dates=["date"])


@tool
def forecast_demand(sku: str, window_days: int = 7, horizon_days: int = 7) -> dict:
    """
    Forecast demand for a SKU using a simple moving average.
    Args:
        sku: product SKU code (e.g. 'ELC-1001')
        window_days: number of trailing days for the average (default 7)
        horizon_days: how many days ahead to forecast total demand (default 7)
    Returns dict with sku, avg_daily_demand, forecast_horizon_days, forecast_total_units.
    """
    df = _sales[_sales["sku"].str.upper() == sku.upper()].sort_values("date")
    if df.empty:
        return {"error": f"No sales history for SKU {sku}"}

    recent = df.tail(window_days)
    avg_daily = round(recent["units_sold"].mean(), 2)
    total = round(avg_daily * horizon_days)

    log.info("Forecast %s: avg_daily=%.1f, horizon=%dd, total=%d",
             sku, avg_daily, horizon_days, total)
    return {
        "sku": sku.upper(),
        "avg_daily_demand": avg_daily,
        "window_days": window_days,
        "forecast_horizon_days": horizon_days,
        "forecast_total_units": total,
    }


@tool
def forecast_all_skus(window_days: int = 7, horizon_days: int = 7) -> list[dict]:
    """
    Forecast demand for ALL SKUs at once. Returns a list of forecast dicts.
    """
    results = []
    for sku in _sales["sku"].unique():
        df = _sales[_sales["sku"] == sku].sort_values("date")
        recent = df.tail(window_days)
        avg_daily = round(recent["units_sold"].mean(), 2)
        total = round(avg_daily * horizon_days)
        results.append({
            "sku": sku,
            "avg_daily_demand": avg_daily,
            "forecast_horizon_days": horizon_days,
            "forecast_total_units": total,
        })
    log.info("Batch-forecasted %d SKUs", len(results))
    return results
