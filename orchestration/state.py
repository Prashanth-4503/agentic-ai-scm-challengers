"""
Shared state schema for the LangGraph SCM workflow.
All agents read/write from this TypedDict.
"""
from typing import TypedDict, Optional, List, Dict, Any


class SCMState(TypedDict, total=False):
    # Input
    user_query: str
    workflow_type: str  # set by supervisor: inventory | procurement | logistics | comms

    # Inventory agent output
    inventory_result: Dict[str, Any]

    # Forecasting agent output
    forecast_result: Dict[str, Any]
    low_stock_items: List[Dict[str, Any]]

    # Procurement agent output
    procurement_result: Dict[str, Any]
    po_draft: Dict[str, Any]
    po_approval_required: bool

    # HITL output
    po_approved: Optional[bool]
    human_feedback: Optional[str]

    # Exception escalation (UC-5) — set by any specialist agent that hits an
    # ambiguous / high-risk situation it can't resolve on its own
    # (e.g. no supplier can meet demand).
    escalation_required: bool
    escalation_payload: Dict[str, Any]
    escalation_decision: Optional[str]

    # Logistics agent output
    logistics_result: Dict[str, Any]

    # Customer comms agent output
    comms_result: Dict[str, Any]

    # Final
    final_response: str
    error: Optional[str]