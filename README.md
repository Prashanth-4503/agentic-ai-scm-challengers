# HexaShop — Agentic AI for Supply Chain Management

A **bootcamp capstone project** demonstrating how to build an agentic AI system for
supply chain operations using **LangChain**, **LangGraph**, and **Azure OpenAI**.

## Architecture

```
User Query → Supervisor Agent (router)
                ├── UC-1: Inventory Agent (single-agent Q&A)
                ├── UC-2: Forecasting → Procurement → [HITL approval | UC-5 escalation] (multi-agent)
                ├── UC-3: Logistics Agent (stretch)
                └── UC-4: Customer Comms Agent (stretch)
             → Finalizer → Response
```

| Component      | Technology               |
|----------------|--------------------------|
| LLM            | Azure OpenAI GPT-5.4 Mini |
| Prompts/Tools  | LangChain                |
| Orchestration  | LangGraph (StateGraph)   |
| HITL           | LangGraph interrupt()    |
| UI             | Streamlit                |
| Data           | CSV (mocked)             |

## Quick Start

```bash
# 1. Clone & enter repo
cd hexashop-agentic-scm

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your Azure OpenAI credentials

# 5. Run the Streamlit app
streamlit run app.py

# Or use CLI
python cli.py "Which SKUs are below reorder level?"
```

## Use Cases

### UC-1: Inventory Q&A (Single Agent)
Ask natural language questions about inventory. The agent uses real CSV data — never hallucinating numbers.

**Example queries:**
- `Which SKUs are below reorder level in the North warehouse?`
- `Show me inventory for ELC-1009`
- `What's the stock level of Wireless Earbuds Pro?`

### UC-2: Auto-Replenishment + HITL (Multi-Agent)
Automated pipeline: forecast demand → detect low stock → select supplier → draft PO → human approval.

**Example queries:**
- `Run auto-replenishment workflow`
- `Create purchase orders for low-stock items`

If a PO exceeds **$50,000** (configurable via `PO_APPROVAL_THRESHOLD`), the workflow pauses for human approval.

### UC-3: Logistics Planning (Stretch)
Find optimal shipping carriers by region, cost, or speed.

**Example:** `Find shipping options for the North region`

### UC-4: Customer Notifications (Stretch)
Notify customers about order delays caused by stock shortages.

**Example:** `Notify customers about delayed SKU ELC-1003`

### UC-5: Exception Escalation (Stretch)
If the Procurement Agent finds that **no supplier can fully cover** the needed
quantity for a SKU, it doesn't guess -- it pauses the graph (same `interrupt()`
mechanism as HITL approval) and surfaces a summary plus recommended options
(partial order, split across suppliers, web search for a new supplier, or
reduce quantity) to the manager for a decision.

**Try it:** run auto-replenishment and target `ELC-1003`, `ELC-1005`, or
`FSH-2003` -- none of these currently have a single supplier in
`data/supplier_catalog.csv` that covers the full `reorder_qty`, so the
workflow escalates instead of placing a PO.

### UC-5: Exception Escalation (Stretch)
If the Procurement Agent finds that **no supplier can fully cover** the needed
quantity for a SKU, it doesn't guess — it pauses the graph (same `interrupt()`
mechanism as HITL approval) and surfaces a summary plus recommended options
(partial order, split across suppliers, web search for a new supplier, or
reduce quantity) to the manager for a decision.

**Try it:** run auto-replenishment against `ELC-1003`, `ELC-1005`, or
`FSH-2003` — none of these have a single supplier that covers the current
`reorder_qty` in `data/supplier_catalog.csv`, so the workflow will escalate
instead of placing a PO.

## Project Structure

```
hexashop-agentic-scm/
├── app.py                    # Streamlit UI entry point
├── cli.py                    # CLI entry point
├── requirements.txt
├── .env.example
├── data/
│   ├── products.csv          # 36 SKUs across Electronics/Fashion/Home
│   ├── inventory.csv         # Stock levels per warehouse
│   ├── sales_history.csv     # 90 days of daily sales
│   ├── suppliers.csv         # 8 suppliers with reliability scores
│   ├── supplier_catalog.csv  # Supplier pricing & availability
│   ├── carriers.csv          # 5 shipping carriers
│   ├── customers.csv         # 25 customers
│   └── orders.csv            # 60 recent orders
├── agents/
│   ├── supervisor_agent.py   # Routes queries to workflows
│   ├── inventory_agent.py    # UC-1: LLM-powered inventory Q&A
│   ├── forecasting_agent.py  # UC-2: Demand forecasting
│   ├── procurement_agent.py  # UC-2: Supplier selection + PO
│   ├── logistics_agent.py    # UC-3: Carrier selection
│   └── customer_comms_agent.py # UC-4: Delay notifications
├── tools/
│   ├── inventory_db.py       # Inventory lookup tools
│   ├── forecast_model.py     # Moving-average forecasting
│   ├── supplier_api.py       # Supplier search & selection
│   ├── shipping_api.py       # Carrier lookup
│   ├── notify_tool.py        # Mocked notification sender
│   └── calculator.py         # PO cost calculation
├── orchestration/
│   ├── state.py              # SCMState TypedDict
│   ├── graph.py              # LangGraph StateGraph
│   ├── hitl.py               # Human-in-the-loop interrupt (UC-2)
│   └── escalation.py         # Exception escalation interrupt (UC-5)
└── utils/
    ├── config.py             # .env loading & LLM factory
    ├── logger.py             # Centralized logging
    └── memory_store.py       # JSON entity/history store + Chroma semantic memory
```

## Memory

| Tier | Implementation | Notes |
|---|---|---|
| Short-term | LangGraph `SCMState` | Lives for the duration of one graph run/thread |
| Entity | `memory/entity_memory.json` | SKU / supplier / customer profiles, keyed lookup |
| Long-term / Semantic | Chroma (`memory/chroma/`), JSON fallback | Past forecasts & PO decisions, queried by similarity in `forecasting_agent` / `procurement_agent` via `semantic_recall()`. If `chromadb` isn't installed or can't initialise, this silently falls back to JSON-only history -- the graph never crashes because of it. |

## Observability

Set `VERBOSE_AGENTS=true` in `.env` before a demo to print full CrewAI
agent/task reasoning traces to the console. Leave it `false` for normal runs
to keep the log clean.

## Resilience

Every `crew.kickoff()` call is wrapped in `_run_crew()` (see
`agents/crewai_agents.py`), which retries once on failure and falls back to a
deterministic, tool-computed default (e.g. lowest-cost supplier, best
ETA/cost carrier) if the LLM call keeps failing or returns unparseable
output -- so a flaky Azure OpenAI call degrades a single agent's answer
instead of crashing the whole graph.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL | — |
| `AZURE_OPENAI_API_KEY` | API key (or `AZURE_OPENAI_KEY`) | — |
| `AZURE_OPENAI_DEPLOYMENT` | Model deployment name | `gpt-5.4-mini` |
| `AZURE_OPENAI_API_VERSION` | API version | `2024-12-01-preview` |
| `PO_APPROVAL_THRESHOLD` | PO $ value requiring approval | `50000` |
| `TAVILY_API_KEY` | Optional -- enables the Procurement Agent's web_search tool | -- |
| `LOG_LEVEL` | Python log level | `INFO` |
| `VERBOSE_AGENTS` | Print full CrewAI reasoning traces | `false` |

## Key Design Decisions

- **Grounded answers**: The inventory agent MUST call tools before answering — no hallucinated stock numbers
- **Simple forecasting**: 7-day moving average (appropriate for a bootcamp demo)
- **Mocked externals**: All supplier/carrier/notification APIs read from CSVs
- **HITL via interrupt()**: LangGraph's built-in interrupt mechanism pauses the graph and resumes after human decision
- **Modular**: Each agent and tool is in its own file with a single responsibility
