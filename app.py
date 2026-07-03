"""
HexaShop Agentic SCM — Streamlit Entry Point
Supports:
  UC-1: Inventory Q&A (single-agent)
  UC-2: Auto-Replenishment with HITL approval (multi-agent)
  UC-3: Logistics planning (stretch)
  UC-4: Customer delay notifications (stretch)
  UC-5: Exception escalation when no supplier can meet demand (stretch)

Run:  streamlit run app.py
"""
import sys
try:
    from streamlit.web import cli as stcli
    from streamlit.runtime import exists
    if not exists():
        sys.argv = ["streamlit", "run", sys.argv[0]] + sys.argv[1:]
        sys.exit(stcli.main())
except Exception:
    pass

import streamlit as st
import uuid
from langgraph.types import Command
from orchestration.graph import graph
from utils.config import PO_APPROVAL_THRESHOLD, validate_config
from utils.logger import get_logger

log = get_logger(__name__)

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(page_title="HexaShop SCM", page_icon="📦", layout="wide")

st.title("📦 HexaShop Agentic Supply Chain Manager")
st.caption(f"PO approval threshold: **${PO_APPROVAL_THRESHOLD:,}** · Powered by Azure OpenAI + LangGraph")

# ── Validate config on startup ─────────────────────────────────────
try:
    validate_config()
except ValueError as e:
    st.error(f"⚠️ Configuration error: {e}")
    st.stop()

# ── Session state init ─────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_approval" not in st.session_state:
    st.session_state.pending_approval = None
if "pending_query" not in st.session_state:
    # Set by sidebar example buttons; drained by the processing block below.
    st.session_state.pending_query = None


def run_query(query: str) -> None:
    """Send `query` through the graph and update session state with the result.

    This is the single place that actually calls graph.invoke() for a fresh
    user message, so both the chat box and the sidebar example buttons share
    identical behavior (including HITL pause detection).
    """
    st.session_state.messages.append({"role": "user", "content": query})

    thread_config = {"configurable": {"thread_id": st.session_state.thread_id}}
    try:
        result = graph.invoke({"user_query": query}, config=thread_config)

        # Check if graph was interrupted (HITL)
        snapshot = graph.get_state(thread_config)
        if snapshot.next:
            po_info = result.get("po_draft", {})
            st.session_state.pending_approval = po_info
            msg = (
                f"⏸️ **Workflow paused** — PO for **{po_info.get('sku')}** "
                f"totalling **${po_info.get('total_cost', 0):,.2f}** "
                f"exceeds the ${PO_APPROVAL_THRESHOLD:,} threshold.\n\n"
                f"Please approve or reject below."
            )
            st.session_state.messages.append({"role": "assistant", "content": msg})
        else:
            response = result.get("final_response", "Done.")
            st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        err_msg = f"❌ Error: {e}"
        log.error("App error: %s", e, exc_info=True)
        st.session_state.messages.append({"role": "assistant", "content": err_msg})


# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔧 Quick Actions")
    if st.button("🔄 New Session", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.pending_approval = None
        st.session_state.pending_query = None
        st.rerun()

    st.divider()
    st.subheader("📋 Example Queries")
    examples = [
        "Which SKUs are below reorder level in the North warehouse?",
        "Show me inventory for ELC-1009",
        "Run auto-replenishment workflow",
        "Find shipping options for the North region",
        "Notify customers about delayed SKU ELC-1003",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{hash(ex)}", use_container_width=True):
            # Don't invoke the graph mid-render (button click already
            # triggered a rerun cycle). Stash the query and process it on
            # the next run, in the same spot chat_input queries go through.
            st.session_state.pending_query = ex
            st.rerun()

    st.divider()
    st.caption(f"Session: `{st.session_state.thread_id[:8]}...`")

# ── Chat history ───────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Handle pending HITL approval ───────────────────────────────────
if st.session_state.pending_approval:
    payload = st.session_state.pending_approval
    st.warning("🔒 **Purchase Order Requires Human Approval**")

    col1, col2 = st.columns(2)
    with col1:
        st.json(payload)
    with col2:
        decision = st.radio("Decision:", ["Approve", "Reject"], index=0,
                            key="approval_radio")
        if st.button("Submit Decision", type="primary", key="submit_approval"):
            thread_config = {"configurable": {"thread_id": st.session_state.thread_id}}

            decision_text = decision.lower()
            try:
                result = graph.invoke(Command(resume=decision_text), config=thread_config)
                response = result.get("final_response", "Decision recorded.")
            except Exception as e:
                log.error("App error resuming HITL: %s", e, exc_info=True)
                response = f"❌ Error: {e}"

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.pending_approval = None
            st.rerun()

# ── Process a query queued by a sidebar example button ─────────────
if st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.pending_query = None
    with st.chat_message("assistant"):
        with st.spinner("🤖 Agents working..."):
            run_query(query)
    st.rerun()

# ── User input ─────────────────────────────────────────────────────
user_input = st.chat_input("Ask about inventory, run replenishment, logistics, or notifications...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Agents working..."):
            run_query(user_input)
    st.rerun()