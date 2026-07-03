"""
Inventory Agent (UC-1) — answers grounded inventory questions using real stock data.
Uses LangChain tool-calling agent backed by Azure OpenAI.
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from utils.config import get_llm
from utils.logger import get_logger
from tools.inventory_db import (
    get_inventory_summary,
    get_low_stock_items,
    get_sku_detail,
    search_inventory,
)

log = get_logger(__name__)

SYSTEM_PROMPT = """You are the Inventory Analyst for HexaShop.
Your ONLY job is to answer inventory questions using the tools provided.
RULES:
- ALWAYS call a tool to get real data before answering. Never guess stock numbers.
- Present data clearly with SKU codes, quantities, and warehouse names.
- If a question is ambiguous, use the search_inventory tool.
- If asked about low stock or reorder, use get_low_stock_items.
- Be concise and factual. Format tables when useful."""

_tools = [get_inventory_summary, get_low_stock_items, get_sku_detail, search_inventory]

_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])


def _build_executor() -> AgentExecutor:
    llm = get_llm()
    agent = create_tool_calling_agent(llm, _tools, _prompt)
    return AgentExecutor(agent=agent, tools=_tools, verbose=True,
                         handle_parsing_errors=True, max_iterations=5)


def inventory_agent(state: dict) -> dict:
    """LangGraph node: run the inventory Q&A agent and return result."""
    query = state.get("user_query", "")
    log.info("Inventory agent handling: %s", query[:80])
    try:
        executor = _build_executor()
        result = executor.invoke({"input": query})
        answer = result.get("output", str(result))
    except Exception as e:
        log.error("Inventory agent error: %s", e)
        answer = f"Inventory agent error: {e}"
    return {"inventory_result": {"answer": answer}, "final_response": answer}
