"""
Memory layer for HexaShop SCM agents.

Three tiers, matching the capstone spec (section 6.3):
  - Short-term  -> LangGraph state (see orchestration/state.py), not handled here.
  - Long-term / Semantic -> Chroma vector store (falls back to flat JSON if
    chromadb isn't installed or can't initialise, e.g. no network for the
    embedding model on first run). This is where past forecasts and supplier
    decisions live so agents can recall *similar* past situations, not just
    exact-match lookups.
  - Entity -> flat JSON key/value store keyed by SKU / supplier / customer id.

The JSON files are always written (cheap, offline, human-inspectable during
a demo). The vector store is written best-effort on top of that, so nothing
breaks if Chroma is unavailable.
"""
from pathlib import Path
import json
import time
from typing import Any, Dict, List

from utils.logger import get_logger

log = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MEMORY_DIR = BASE_DIR / "memory"
MEMORY_DIR.mkdir(exist_ok=True)

FORECAST_MEMORY_FILE = MEMORY_DIR / "forecast_history.json"
SUPPLIER_MEMORY_FILE = MEMORY_DIR / "supplier_history.json"
ENTITY_MEMORY_FILE = MEMORY_DIR / "entity_memory.json"
CHROMA_DIR = MEMORY_DIR / "chroma"


# ── Flat JSON store (always available) ───────────────────────────────

def _load_json(file_path: Path, default):
    if not file_path.exists():
        return default
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save_json(file_path: Path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_forecast_record(record: Dict[str, Any]):
    data = _load_json(FORECAST_MEMORY_FILE, [])
    data.append(record)
    _save_json(FORECAST_MEMORY_FILE, data)
    _semantic_upsert(
        collection="forecasts",
        doc_id=f"forecast-{record.get('sku')}-{int(time.time() * 1000)}",
        text=_forecast_to_text(record),
        metadata={"sku": record.get("sku", ""), "risk": record.get("risk", "")},
    )


def get_forecast_history(sku: str = None) -> List[Dict[str, Any]]:
    data = _load_json(FORECAST_MEMORY_FILE, [])
    if sku:
        return [x for x in data if x.get("sku") == sku]
    return data


def save_supplier_decision(record: Dict[str, Any]):
    data = _load_json(SUPPLIER_MEMORY_FILE, [])
    data.append(record)
    _save_json(SUPPLIER_MEMORY_FILE, data)
    _semantic_upsert(
        collection="supplier_decisions",
        doc_id=f"po-{record.get('sku')}-{int(time.time() * 1000)}",
        text=_supplier_decision_to_text(record),
        metadata={
            "sku": record.get("sku", ""),
            "supplier_id": record.get("supplier_id", ""),
        },
    )


def get_supplier_history(sku: str = None, supplier_id: str = None) -> List[Dict[str, Any]]:
    data = _load_json(SUPPLIER_MEMORY_FILE, [])
    if sku:
        data = [x for x in data if x.get("sku") == sku]
    if supplier_id:
        data = [x for x in data if x.get("supplier_id") == supplier_id]
    return data


def upsert_entity(entity_type: str, entity_id: str, payload: Dict[str, Any]):
    data = _load_json(ENTITY_MEMORY_FILE, {})
    if entity_type not in data:
        data[entity_type] = {}
    data[entity_type][entity_id] = payload
    _save_json(ENTITY_MEMORY_FILE, data)


def get_entity(entity_type: str, entity_id: str):
    data = _load_json(ENTITY_MEMORY_FILE, {})
    return data.get(entity_type, {}).get(entity_id)


def get_all_entities(entity_type: str):
    data = _load_json(ENTITY_MEMORY_FILE, {})
    return data.get(entity_type, {})


# ── Semantic / long-term memory (Chroma, best-effort) ────────────────

def _forecast_to_text(record: Dict[str, Any]) -> str:
    return (
        f"SKU {record.get('sku')} forecast: {record.get('forecast_units_7d')} units "
        f"over 7 days, risk={record.get('risk')}. Reason: {record.get('reason', '')}"
    )


def _supplier_decision_to_text(record: Dict[str, Any]) -> str:
    return (
        f"PO for SKU {record.get('sku')}: qty={record.get('qty')} from "
        f"{record.get('supplier')} at ${record.get('unit_cost')}/unit, "
        f"lead time {record.get('lead_time_days')} days. Reason: {record.get('reason', '')}"
    )


class _ChromaBackend:
    """Lazily-initialised Chroma client. Any failure disables semantic memory
    for the rest of the run rather than crashing the graph (resilience
    requirement)."""

    def __init__(self):
        self._client = None
        self._collections: Dict[str, Any] = {}
        self._disabled = False

    def _ensure_client(self):
        if self._client is not None or self._disabled:
            return
        try:
            import chromadb
            self._client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        except Exception as e:
            log.warning(
                "Semantic memory (Chroma) unavailable, falling back to JSON-only "
                "long-term memory: %s", e,
            )
            self._disabled = True

    def _get_collection(self, name: str):
        self._ensure_client()
        if self._disabled:
            return None
        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(name)
        return self._collections[name]

    def upsert(self, collection: str, doc_id: str, text: str, metadata: dict):
        try:
            col = self._get_collection(collection)
            if col is None:
                return
            col.upsert(ids=[doc_id], documents=[text], metadatas=[metadata])
        except Exception as e:
            log.warning("Semantic memory upsert failed (%s): %s", collection, e)

    def query(self, collection: str, text: str, n_results: int = 3) -> List[str]:
        try:
            col = self._get_collection(collection)
            if col is None:
                return []
            res = col.query(query_texts=[text], n_results=n_results)
            docs = res.get("documents", [[]])
            return docs[0] if docs else []
        except Exception as e:
            log.warning("Semantic memory query failed (%s): %s", collection, e)
            return []


_backend = _ChromaBackend()


def _semantic_upsert(collection: str, doc_id: str, text: str, metadata: dict):
    _backend.upsert(collection, doc_id, text, metadata)


def semantic_recall(collection: str, query_text: str, n_results: int = 3) -> List[str]:
    """Return the most similar past records (as plain text) for a query.

    Used by agents to ground reasoning in 'similar situations we saw before',
    on top of the exact-match JSON history. Returns [] if the vector store
    isn't available -- callers should treat this as optional context, never
    a hard dependency.
    """
    return _backend.query(collection, query_text, n_results)
