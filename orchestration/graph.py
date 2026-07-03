"""
LangGraph workflow — compiles the full SCM state graph.

Nodes: supervisor -> [inventory | forecasting -> procurement -> {escalation | hitl} | logistics | comms] -> finalizer

`procurement` can branch three ways:
  - escalation_required=True  -> escalation node (UC-5: no viable supplier, needs human judgement call)
  - po_approval_required=True -> hitl node (UC-2: PO value above threshold, needs human sign-off)
  - otherwise                 -> straight to finalizer (auto-approved, below threshold)
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from orchestration.state import SCMState
from orchestration.hitl import human_approval_node
from orchestration.escalation import escalation_node

from agents.supervisor_agent import supervisor_agent
from agents.crewai_agents import (
    inventory_agent,
    forecasting_agent,
    procurement_agent,
    logistics_agent,
    customer_comms_agent,
)


import json

memory = MemorySaver()


# ── Routing functions ──────────────────────────────────────────────

def route_from_supervisor(state: SCMState) -> str:
    wf = state.get("workflow_type", "")
    return {
        "inventory": "inventory",
        "procurement": "forecasting",
        "logistics": "logistics",
        "comms": "comms",
    }.get(wf, "finalizer")


def procurement_router(state: SCMState) -> str:
    if state.get("escalation_required"):
        return "escalation"
    if state.get("po_approval_required"):
        return "hitl"
    return "end_procurement"


# ── Finalizer node ─────────────────────────────────────────────────

def finalizer_node(state: SCMState) -> dict:
    """Build a human-readable final_response from whatever the workflow produced."""
    wf = state.get("workflow_type", "")

    if wf == "inventory":
        return {"final_response": state.get("inventory_result", {}).get(
            "answer", "No inventory data available.")}

    if wf == "procurement":
        if state.get("escalation_required"):
            payload = state.get("escalation_payload", {})
            decision = state.get("escalation_decision")
            return {"final_response":
                    f"🚨 ESCALATION RESOLVED\n"
                    f"Issue: {payload.get('summary', 'Unresolved procurement issue')}\n"
                    f"Manager decision: {decision}\n"
                    f"Options offered were:\n" +
                    "\n".join(f"  - {o}" for o in payload.get("options", []))}

        po = state.get("po_draft", {})
        if state.get("po_approved") is True:
            return {"final_response":
                    f"✅ PO APPROVED and ready for placement:\n"
                    f"{json.dumps(po, indent=2)}"}
        elif state.get("po_approved") is False:
            fb = state.get("human_feedback", "")
            return {"final_response":
                    f"❌ PO REJECTED by reviewer.\nFeedback: {fb}\n"
                    f"Draft was:\n{json.dumps(po, indent=2)}"}
        # No approval needed — auto-approved
        return {"final_response":
                f"✅ PO auto-approved (below threshold):\n"
                f"{json.dumps(po, indent=2)}"}

    if wf == "logistics":
        return {"final_response": state.get("final_response",
                str(state.get("logistics_result", {})))}

    if wf == "comms":
        return {"final_response": state.get("final_response",
                str(state.get("comms_result", {})))}

    return {"final_response": "Workflow completed."}


# ── Build the graph ────────────────────────────────────────────────

builder = StateGraph(SCMState)

builder.add_node("supervisor", supervisor_agent)
builder.add_node("inventory", inventory_agent)
builder.add_node("forecasting", forecasting_agent)
builder.add_node("procurement", procurement_agent)
builder.add_node("logistics", logistics_agent)
builder.add_node("comms", customer_comms_agent)
builder.add_node("hitl", human_approval_node)
builder.add_node("escalation", escalation_node)
builder.add_node("finalizer", finalizer_node)

# Edges
builder.add_edge(START, "supervisor")

builder.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "inventory": "inventory",
        "forecasting": "forecasting",
        "logistics": "logistics",
        "comms": "comms",
        "finalizer": "finalizer",
    },
)

builder.add_edge("inventory", "finalizer")
builder.add_edge("forecasting", "procurement")

builder.add_conditional_edges(
    "procurement",
    procurement_router,
    {
        "escalation": "escalation",
        "hitl": "hitl",
        "end_procurement": "finalizer",
    },
)

builder.add_edge("hitl", "finalizer")
builder.add_edge("escalation", "finalizer")
builder.add_edge("logistics", "finalizer")
builder.add_edge("comms", "finalizer")
builder.add_edge("finalizer", END)

# Compile with in-memory checkpointing (needed for interrupt/resume)
graph = builder.compile(checkpointer=memory)
