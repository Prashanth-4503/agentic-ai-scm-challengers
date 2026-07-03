"""
Inventory database tool — reads data/inventory.csv and data/products.csv.
All functions return plain dicts so the LLM can ground its answers.
"""
import pandas as pd
from langchain_core.tools import tool
from utils.config import DATA_DIR
from utils.logger import get_logger

log = get_logger(__name__)

_inv = pd.read_csv(DATA_DIR / "inventory.csv")
_prod = pd.read_csv(DATA_DIR / "products.csv")


@tool
def get_inventory_summary() -> list[dict]:
    """Return every SKU row from inventory with product name, on_hand, reorder_point, and warehouse."""
    merged = _inv.merge(_prod[["sku", "product_name"]], on="sku", how="left")
    cols = ["sku", "product_name", "warehouse", "region", "on_hand",
            "safety_stock", "reorder_point", "reorder_qty"]
    return merged[cols].to_dict(orient="records")


@tool
def get_low_stock_items(warehouse: str = "") -> list[dict]:
    """Return SKUs where on_hand < reorder_point. Optionally filter by warehouse name (e.g. 'North DC')."""
    df = _inv.copy()
    if warehouse:
        df = df[df["warehouse"].str.contains(warehouse, case=False, na=False)]
    low = df[df["on_hand"] < df["reorder_point"]].copy()
    low = low.merge(_prod[["sku", "product_name"]], on="sku", how="left")
    low["deficit"] = low["reorder_point"] - low["on_hand"]
    cols = ["sku", "product_name", "warehouse", "region",
            "on_hand", "reorder_point", "deficit", "reorder_qty"]
    log.info("Found %d low-stock rows (warehouse=%s)", len(low), warehouse or "ALL")
    return low[cols].to_dict(orient="records")


@tool
def get_sku_detail(sku: str) -> list[dict]:
    """Get inventory detail for a specific SKU across all warehouses."""
    rows = _inv[_inv["sku"].str.upper() == sku.upper()]
    if rows.empty:
        return [{"error": f"SKU {sku} not found in inventory"}]
    merged = rows.merge(_prod[["sku", "product_name", "unit_price"]], on="sku", how="left")
    return merged.to_dict(orient="records")


@tool
def search_inventory(query: str) -> list[dict]:
    """Free-text search across SKU, product name, warehouse, or region."""
    q = query.lower().strip()
    merged = _inv.merge(_prod[["sku", "product_name"]], on="sku", how="left")
    
    # Try to find SKUs in the query text using regex
    import re
    skus_in_query = re.findall(r"[A-Za-z]{3}-\d{4}", query)
    if skus_in_query:
        skus_upper = [s.upper() for s in skus_in_query]
        mask = merged["sku"].str.upper().isin(skus_upper)
    else:
        # Substring match: either column contains query, or query contains column value
        mask = (
            merged["sku"].str.lower().str.contains(q, na=False)
            | merged["product_name"].str.lower().str.contains(q, na=False)
            | merged["warehouse"].str.lower().str.contains(q, na=False)
            | merged["region"].str.lower().str.contains(q, na=False)
            | merged["sku"].str.lower().apply(lambda x: x in q)
            | merged["product_name"].str.lower().apply(lambda x: x in q)
            | merged["warehouse"].str.lower().apply(lambda x: x in q)
            | merged["region"].str.lower().apply(lambda x: x in q)
        )
    result = merged[mask]
    return result.to_dict(orient="records")
