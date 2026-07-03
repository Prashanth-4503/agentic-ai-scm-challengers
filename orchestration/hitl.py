from langgraph.types import interrupt


def human_approval_node(state):
    po_draft = state.get("po_draft", {})
    total_cost = po_draft.get("total_cost", 0)

    approval_payload = {
        "message": "Purchase Order requires approval",
        "sku": po_draft.get("sku"),
        "supplier": po_draft.get("supplier"),
        "qty": po_draft.get("qty"),
        "total_cost": total_cost,
        "lead_time_days": po_draft.get("lead_time_days"),
        "status": "PENDING_HUMAN_APPROVAL"
    }

    human_decision = interrupt(approval_payload)

    approved = str(human_decision).strip().lower() in ["approve", "approved", "yes", "y"]

    return {
        "po_approved": approved,
        "human_feedback": str(human_decision)
    }