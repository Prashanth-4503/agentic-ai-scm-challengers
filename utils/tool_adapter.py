"""
Adapts LangChain `@tool`-decorated callables (StructuredTool) into
crewai.tools.BaseTool instances so they can be passed to a CrewAI Agent's
`tools=[...]` list.

Why this exists
----------------
Every tool in tools/*.py is written once, using LangChain's `@tool` decorator,
and used two ways in this project:
  1. Directly, via `.invoke({...})`, when a node pre-fetches data in Python
     before building the Task prompt.
  2. Bound to a CrewAI Agent, so the LLM itself can call it mid-reasoning.

CrewAI's `Agent.tools` field validates every entry against
`crewai.tools.BaseTool` (a pydantic model) — it does NOT accept LangChain's
`StructuredTool`. Passing LangChain tools straight into `tools=[...]` throws:

    4 validation errors for Agent
    tools.0 Input should be a valid dictionary or instance of BaseTool
    [type=model_type, input_value=StructuredTool(name='get_...'), ...]

This module bridges the two so we keep a single tool implementation instead
of writing every tool twice.

Verified against crewai==1.15.1.
"""
import inspect

from crewai.tools.base_tool import Tool as CrewTool
from pydantic import create_model

from utils.logger import get_logger

log = get_logger(__name__)

_cache = {}


def _build_args_schema(func):
    """Infer a pydantic args schema from a plain function signature."""
    sig = inspect.signature(func)
    fields = {}
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        annotation = param.annotation if param.annotation != param.empty else str
        default = ... if param.default is param.empty else param.default
        fields[name] = (annotation, default)
    return create_model(f"{func.__name__}_input", **fields)


def to_crewai_tool(lc_tool):
    """Convert a single LangChain tool into a crewai.tools.Tool instance.

    Cached by object identity so the same LangChain tool always maps to the
    same CrewAI tool instance (avoids rebuilding schemas on every agent call).
    """
    key = id(lc_tool)
    if key in _cache:
        return _cache[key]

    try:
        # Preferred path: CrewAI ships a built-in LangChain adapter.
        crew_tool = CrewTool.from_langchain(lc_tool)
    except Exception as e:
        log.warning(
            "Tool.from_langchain failed for '%s' (%s) — building a manual adapter instead.",
            getattr(lc_tool, "name", lc_tool), e,
        )
        # Manual fallback so this keeps working even if a different crewai
        # version changes or removes from_langchain.
        func = getattr(lc_tool, "func", None) or lc_tool
        args_schema = getattr(lc_tool, "args_schema", None) or _build_args_schema(func)
        crew_tool = CrewTool(
            name=getattr(lc_tool, "name", getattr(func, "__name__", "tool")),
            description=getattr(lc_tool, "description", func.__doc__ or ""),
            func=func,
            args_schema=args_schema,
        )

    _cache[key] = crew_tool
    return crew_tool


def to_crewai_tools(lc_tools):
    """Convert a list of LangChain tools into a list of crewai Tool instances."""
    return [to_crewai_tool(t) for t in lc_tools]