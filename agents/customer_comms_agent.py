"""
Customer Communications Agent (UC-4 stretch) — notifies customers about delays.
"""
from tools.notify_tool import get_affected_orders, send_delay_notification
from utils.logger import get_logger

log = get_logger(__name__)


def customer_comms_agent(state: dict) -> dict:
    """
    LangGraph node: find orders affected by low stock and send (mocked) notifications.
    """
    try:
        query = state.get("user_query", "")
        low_items = state.get("low_stock_items", [])

        # Gather SKUs to check — from low_stock_items or from query
        skus_to_check = []
        if low_items:
            skus_to_check = list({item["sku"] for item in low_items})
        else:
            # Try to extract SKU from query
            import re
            found = re.findall(r"[A-Z]{3}-\d{4}", query.upper())
            skus_to_check = found if found else []

        if not skus_to_check:
            return {
                "comms_result": {"message": "No SKUs identified for notification"},
                "final_response": "No delay notifications needed — no affected SKUs.",
            }

        all_notifications = []
        for sku in skus_to_check[:5]:  # limit to 5 SKUs
            affected = get_affected_orders.invoke({"sku": sku})
            for order in affected:
                msg = (
                    f"Dear {order['customer_name']}, your order {order['order_id']} "
                    f"for {sku} (promised {order['promised_date']}) may be delayed "
                    f"due to a stock shortage. We are expediting replenishment. "
                    f"We apologize for the inconvenience."
                )
                result = send_delay_notification.invoke({
                    "order_id": order["order_id"],
                    "customer_email": order["email"],
                    "message": msg,
                })
                all_notifications.append(result)

        log.info("Sent %d delay notifications", len(all_notifications))

        return {
            "comms_result": {
                "notifications_sent": len(all_notifications),
                "details": all_notifications,
            },
            "final_response": f"Sent {len(all_notifications)} delay notification(s) to affected customers.",
        }

    except Exception as e:
        log.error("Comms agent error: %s", e)
        return {
            "comms_result": {"error": str(e)},
            "final_response": f"Comms error: {e}",
        }
