"""
Customer notification tool — mocked email/SMS sender.
Reads customers.csv and orders.csv for context.
"""
import pandas as pd
from langchain_core.tools import tool
from utils.config import DATA_DIR
from utils.logger import get_logger

log = get_logger(__name__)

_customers = pd.read_csv(DATA_DIR / "customers.csv")
_orders = pd.read_csv(DATA_DIR / "orders.csv")


@tool
def get_affected_orders(sku: str) -> list[dict]:
    """
    Find pending/allocated orders that will be affected by a stock-out or delay for a SKU.
    """
    affected = _orders[
        (_orders["sku"].str.upper() == sku.upper()) &
        (_orders["status"].isin(["Pending", "Allocated"]))
    ].copy()
    if affected.empty:
        return []
    merged = affected.merge(_customers[["customer_id", "customer_name", "email"]],
                            on="customer_id", how="left")
    cols = ["order_id", "customer_id", "customer_name", "email",
            "sku", "qty", "promised_date", "status"]
    log.info("Found %d affected orders for %s", len(merged), sku)
    return merged[cols].to_dict(orient="records")


@tool
def send_delay_notification(order_id: str, customer_email: str,
                            message: str) -> dict:
    """
    [MOCKED] Send a delay notification to a customer.
    In production this would call an email/SMS API.
    """
    log.info("MOCK NOTIFICATION → %s (order %s): %s",
             customer_email, order_id, message[:80])
    return {
        "status": "sent",
        "channel": "email",
        "to": customer_email,
        "order_id": order_id,
        "message_preview": message[:120],
    }


@tool
def draft_delay_message(order: dict) -> str:
    """
    Draft a delay notification message for a customer order.
    """
    customer_name = order.get("customer_name", "Customer")
    order_id = order.get("order_id", "Unknown")
    sku = order.get("sku", "Unknown")
    promised_date = order.get("promised_date", "soon")
    return (
        f"Dear {customer_name}, your order {order_id} "
        f"for {sku} (promised {promised_date}) may be delayed "
        f"due to a stock shortage. We are expediting replenishment. "
        f"We apologize for the inconvenience."
    )
