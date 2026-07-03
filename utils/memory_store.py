from pathlib import Path
import json
from typing import Any, Dict, List


BASE_DIR = Path(__file__).resolve().parent.parent
MEMORY_DIR = BASE_DIR / "memory"
MEMORY_DIR.mkdir(exist_ok=True)

FORECAST_MEMORY_FILE = MEMORY_DIR / "forecast_history.json"
SUPPLIER_MEMORY_FILE = MEMORY_DIR / "supplier_history.json"
ENTITY_MEMORY_FILE = MEMORY_DIR / "entity_memory.json"


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


def get_forecast_history(sku: str = None) -> List[Dict[str, Any]]:
    data = _load_json(FORECAST_MEMORY_FILE, [])
    if sku:
        return [x for x in data if x.get("sku") == sku]
    return data


def save_supplier_decision(record: Dict[str, Any]):
    data = _load_json(SUPPLIER_MEMORY_FILE, [])
    data.append(record)
    _save_json(SUPPLIER_MEMORY_FILE, data)


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