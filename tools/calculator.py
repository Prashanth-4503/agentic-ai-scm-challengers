"""
PO calculator tool — builds purchase order JSON and checks approval threshold.
"""
from langchain_core.tools import tool
from utils.config import PO_APPROVAL_THRESHOLD
from utils.logger import get_logger

log = get_logger(__name__)


@tool
def build_purchase_order(sku: str, qty: int, supplier_id: str,
                         supplier_name: str, unit_cost: float,
                         lead_time_days: int) -> dict:
    """
    Build a purchase order dict and flag whether it needs human approval.
    """
    total_cost = round(unit_cost * qty, 2)
    needs_approval = total_cost > PO_APPROVAL_THRESHOLD

    po = {
        "sku": sku,
        "qty": qty,
        "supplier_id": supplier_id,
        "supplier": supplier_name,
        "unit_cost": unit_cost,
        "total_cost": total_cost,
        "lead_time_days": lead_time_days,
        "approval_required": needs_approval,
        "threshold": PO_APPROVAL_THRESHOLD,
    }
    log.info("PO built: %s × %d = $%.2f (approval=%s)",
             sku, qty, total_cost, needs_approval)
    return po


@tool
def calculate_po_cost(unit_cost: float, qty: int) -> dict:
    """
    Calculate total cost for a given unit cost and quantity.
    """
    total = round(unit_cost * qty, 2)
    return {"total_cost": total}
