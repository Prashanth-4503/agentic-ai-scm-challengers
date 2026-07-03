"""
Supplier lookup tool — reads suppliers.csv and supplier_catalog.csv.
Mocks an external supplier API.
"""
import pandas as pd
from langchain_core.tools import tool
from utils.config import DATA_DIR
from utils.logger import get_logger

log = get_logger(__name__)

_suppliers = pd.read_csv(DATA_DIR / "suppliers.csv")
_catalog = pd.read_csv(DATA_DIR / "supplier_catalog.csv")


@tool
def get_suppliers_for_sku(sku: str) -> list[dict]:
    """
    Return all suppliers that carry a given SKU, including unit_cost,
    moq, lead_time_days, and available_qty, enriched with supplier name & reliability.
    """
    cat = _catalog[_catalog["sku"].str.upper() == sku.upper()].copy()
    if cat.empty:
        return [{"error": f"No suppliers found for SKU {sku}"}]
    merged = cat.merge(_suppliers, on="supplier_id", how="left")
    cols = ["supplier_id", "supplier_name", "sku", "unit_cost", "moq",
            "lead_time_days", "available_qty", "reliability_score", "on_time_rate"]
    log.info("Found %d supplier options for %s", len(merged), sku)
    return merged[cols].to_dict(orient="records")


@tool
def select_best_supplier(sku: str, qty_needed: int) -> dict:
    """
    Pick the best supplier for a SKU based on:
      1. Can fulfil qty_needed (available_qty >= qty_needed and qty_needed >= moq)
      2. Highest reliability_score
      3. Lowest unit_cost as tiebreaker
    Returns the selected supplier row or an error.
    """
    cat = _catalog[_catalog["sku"].str.upper() == sku.upper()].copy()
    if cat.empty:
        return {"error": f"No suppliers for SKU {sku}"}

    merged = cat.merge(_suppliers, on="supplier_id", how="left")

    # Filter: supplier can fulfil the order
    eligible = merged[
        (merged["available_qty"] >= qty_needed) &
        (qty_needed >= merged["moq"])
    ]
    if eligible.empty:
        # Fall back: relax MOQ constraint, just pick by availability
        eligible = merged[merged["available_qty"] >= qty_needed]
    if eligible.empty:
        return {"error": f"No supplier can fulfil {qty_needed} units of {sku}"}

    # Rank: best reliability, then lowest cost
    best = eligible.sort_values(
        ["reliability_score", "unit_cost"], ascending=[False, True]
    ).iloc[0]

    result = {
        "supplier_id": best["supplier_id"],
        "supplier_name": best["supplier_name"],
        "sku": sku.upper(),
        "unit_cost": float(best["unit_cost"]),
        "moq": int(best["moq"]),
        "lead_time_days": int(best["lead_time_days"]),
        "available_qty": int(best["available_qty"]),
        "reliability_score": float(best["reliability_score"]),
    }
    log.info("Selected supplier %s for %s @ $%.2f",
             result["supplier_name"], sku, result["unit_cost"])
    return result
