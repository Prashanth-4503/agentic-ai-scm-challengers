import os
from langchain_openai import AzureChatOpenAI

from dotenv import load_dotenv
from tools.supplier_api import supplier_api, place_purchase_order
from tools.calculator import calculate_total_cost
from langgraph.types import interrupt

load_dotenv()

po_approval = os.getenv("PO_APPROVAL_THRESHOLD")

def procurement_agent(sku: str, quantity: int):
    """
    Procurement Agent for purchasing materials from suppliers.
    1.Fetch supplier information
    2.Calculate total cost
    3.Place purchase order
    4.Check approval threshold
    5.If above threshold, send to manager for approval
    6.If below threshold, place purchase order
    """

    supplier_result = supplier_api.invoke({
        "sku": sku,
        "quantity": quantity
    })

    if supplier_result["status"] != "SUCCESS":
        return supplier_result

    best_supplier = supplier_result["quotes"][0]

    total_cost = calculate_total_cost.invoke({
        "quantity": quantity,
        "unit_price": best_supplier["unit_cost"]
    })

    purchase_order = {
        "supplier_id": best_supplier["supplier_id"],
        "supplier_name": best_supplier["supplier_name"],
        "sku": sku,
        "quantity": quantity,
        "unit_price": best_supplier["unit_cost"],
        "total_cost": total_cost,
        "lead_time_days": best_supplier["lead_time_days"]
    }

    if total_cost > po_approval:

        approval = interrupt({
            "type": "PO_APPROVAL",
            "purchase_order": purchase_order,
            "message": "Purchase Order exceeds approval threshold."
        })

        if not approval.get("approved", False):
            return {
                "status": "REJECTED",
                "message": "Purchase Order rejected.",
                "purchase_order": purchase_order
            }

    return place_purchase_order.invoke({
        "po": purchase_order
    })


if __name__ == "__main__":

    result = procurement_agent("LAP001", 100)

    print(result)
