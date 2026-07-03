"""
Exception escalation node (UC-5).

When a specialist agent hits a situation it can't safely resolve on its own
(e.g. no supplier can meet demand for a SKU), it sets `escalation_required`
and `escalation_payload` in state instead of guessing. The graph routes here,
which pauses execution — same mechanism as the HITL approval gate — and
surfaces a summary plus recommended options to the manager.
"""
from langgraph.types import interrupt


def escalation_node(state):
    payload = dict(state.get("escalation_payload", {}))
    payload["status"] = "PENDING_HUMAN_DECISION"

    human_decision = interrupt(payload)

    return {
        "escalation_decision": str(human_decision),
    }
