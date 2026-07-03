"""
Procurement Agent — selects suppliers and drafts purchase orders.
Part of the UC-2 multi-agent pipeline.
"""
from tools.supplier_api import select_best_supplier
from tools.calculator import build_purchase_order
from utils.config import PO_APPROVAL_THRESHOLD
from utils.logger import get_logger

log = get_logger(__name__)


def procurement_agent(state: dict) -> dict:
    """
    LangGraph node:
    1. For each low-stock item from forecasting, find best supplier.
    2. Draft a PO for the highest-priority SKU (the one with the biggest deficit).
    3. Flag whether human approval is required.
    """
    try:
        low_items = state.get("low_stock_items", [])
        if not low_items:
            return {
                "procurement_result": {"message": "No items to procure"},
                "po_draft": {},
                "po_approval_required": False,
            }

        # Sort by deficit descending — procure the most critical first
        sorted_items = sorted(low_items, key=lambda x: x.get("deficit", 0), reverse=True)
        top = sorted_items[0]
        sku = top["sku"]
        qty = top.get("recommended_order_qty", top.get("reorder_qty", 100))

        # Find best supplier
        supplier = select_best_supplier.invoke({"sku": sku, "qty_needed": qty})
        if "error" in supplier:
            log.warning("Supplier selection failed: %s", supplier["error"])
            return {
                "procurement_result": supplier,
                "po_draft": {},
                "po_approval_required": False,
                "error": supplier["error"],
            }

        # Build purchase order
        po = build_purchase_order.invoke({
            "sku": sku,
            "qty": qty,
            "supplier_id": supplier["supplier_id"],
            "supplier_name": supplier["supplier_name"],
            "unit_cost": supplier["unit_cost"],
            "lead_time_days": supplier["lead_time_days"],
        })

        needs_approval = po.get("approval_required", False)

        log.info("PO drafted: %s × %d from %s — total $%.2f (approval=%s)",
                 sku, qty, supplier["supplier_name"],
                 po["total_cost"], needs_approval)

        return {
            "procurement_result": {
                "status": "po_drafted",
                "all_low_stock_skus": [i["sku"] for i in sorted_items],
            },
            "po_draft": po,
            "po_approval_required": needs_approval,
        }

    except Exception as e:
        log.error("Procurement agent error: %s", e)
        return {
            "procurement_result": {"error": str(e)},
            "po_draft": {},
            "po_approval_required": False,
            "error": str(e),
        }
