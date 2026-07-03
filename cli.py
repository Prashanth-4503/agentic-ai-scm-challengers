"""
HexaShop Agentic SCM — CLI Entry Point
Quick way to test without Streamlit.

Usage:
  python cli.py "Which SKUs are below reorder level?"
  python cli.py "Run auto-replenishment workflow"
"""
import sys
import json
import uuid
from langgraph.types import Command
from orchestration.graph import graph
from utils.config import PO_APPROVAL_THRESHOLD, validate_config
from utils.logger import get_logger

log = get_logger(__name__)


def run_query(query: str):
    """Run a single query through the LangGraph workflow."""
    validate_config()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n{'='*60}")
    print(f"📦 HexaShop SCM — Query: {query}")
    print(f"{'='*60}\n")

    result = graph.invoke({"user_query": query}, config=config)

    # Check for HITL interrupt
    snapshot = graph.get_state(config)
    if snapshot.next:
        po = result.get("po_draft", {})
        print("\n🔒 PURCHASE ORDER REQUIRES APPROVAL")
        print(json.dumps(po, indent=2))
        print(f"\nTotal: ${po.get('total_cost', 0):,.2f} "
              f"(threshold: ${PO_APPROVAL_THRESHOLD:,})")

        decision = input("\n→ Approve or Reject? [approve/reject]: ").strip().lower()

        result = graph.invoke(Command(resume=decision), config=config)
        print(f"\n{'─'*40}")
        print(result.get("final_response", "Done."))
    else:
        print(result.get("final_response", "Done."))

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_query(" ".join(sys.argv[1:]))
    else:
        # Interactive mode
        print("📦 HexaShop Agentic SCM — Interactive Mode")
        print("Type 'quit' to exit.\n")
        while True:
            try:
                q = input("You: ").strip()
                if q.lower() in ("quit", "exit", "q"):
                    break
                if q:
                    run_query(q)
            except (KeyboardInterrupt, EOFError):
                break
        print("\nGoodbye! 👋")
