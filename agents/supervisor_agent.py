"""
Supervisor Agent — routes user queries to the correct workflow.
Uses simple keyword matching (no LLM needed for routing).
"""
from utils.logger import get_logger

log = get_logger(__name__)

# Keyword → workflow mapping (checked in priority order)
_ROUTES = [
    (["replenish", "purchase order", "po ", "procurement", "auto-replen",
      "restock", "order from supplier", "buy", "replenishment"],  "procurement"),
    (["shipping", "carrier", "delivery", "logistics", "ship"],     "logistics"),
    (["delay", "notify", "customer message", "mail", "comms",
      "notification", "communicate"],                              "comms"),
    (["stock", "inventory", "reorder", "sku", "warehouse", "on_hand",
      "on hand", "how many", "quantity", "deficit"],              "inventory"),
]


def supervisor_agent(state: dict) -> dict:
    """LangGraph node: classify the user query and set workflow_type."""
    query = state.get("user_query", "").lower()

    for keywords, workflow in _ROUTES:
        if any(kw in query for kw in keywords):
            log.info("Supervisor routed to → %s", workflow)
            return {"workflow_type": workflow}

    # Default fallback
    log.info("Supervisor defaulted to → inventory")
    return {"workflow_type": "inventory"}