"""
Web search tool — used by the Procurement Agent when the internal supplier
catalog doesn't have enough information (e.g. market price checks for a SKU
with no catalog entry).

Backed by Tavily's search API (TAVILY_API_KEY in .env). If the key is
missing or the request fails, returns a clearly-labelled empty result
instead of raising, so the graph never crashes because of an optional tool.
"""
import os
import requests
from langchain_core.tools import tool
from utils.logger import get_logger

log = get_logger(__name__)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_URL = "https://api.tavily.com/search"


@tool
def web_search(query: str, max_results: int = 3) -> list[dict]:
    """
    Search the public web for supplier / market information not available in
    the internal catalog (e.g. 'alternative suppliers for wireless earbuds').
    Returns a list of {title, url, snippet} dicts, or a single item with an
    'error' key if search is unavailable.
    """
    if not TAVILY_API_KEY:
        log.warning("web_search called but TAVILY_API_KEY is not configured.")
        return [{"error": "Web search is not configured (missing TAVILY_API_KEY)."}]

    try:
        resp = requests.post(
            TAVILY_URL,
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for r in data.get("results", [])[:max_results]:
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
            })
        return results or [{"error": f"No web results found for '{query}'."}]
    except Exception as e:
        log.warning("web_search failed for query '%s': %s", query, e)
        return [{"error": f"Web search failed: {e}"}]
