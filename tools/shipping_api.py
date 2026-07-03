"""
Shipping / carrier lookup tool — reads data/carriers.csv.
Mocks a logistics API.
"""
import pandas as pd
from langchain_core.tools import tool
from utils.config import DATA_DIR
from utils.logger import get_logger

log = get_logger(__name__)

_carriers = pd.read_csv(DATA_DIR / "carriers.csv")


@tool
def get_shipping_options(region: str, weight_kg: float = 1.0) -> list[dict]:
    """
    Return carriers that serve a given region, with estimated cost.
    Cost = base_cost + (cost_per_kg * weight_kg).
    """
    matches = []
    for _, row in _carriers.iterrows():
        covered = [r.strip() for r in str(row["regions_covered"]).split(",")]
        if region in covered:
            est_cost = round(row["base_cost"] + row["cost_per_kg"] * weight_kg, 2)
            matches.append({
                "carrier_id": row["carrier_id"],
                "carrier_name": row["carrier_name"],
                "service_level": row["service_level"],
                "eta_days": int(row["eta_days"]),
                "estimated_cost": est_cost,
                "reliability": float(row["reliability"]),
            })
    log.info("Found %d carriers for region=%s", len(matches), region)
    return sorted(matches, key=lambda x: x["estimated_cost"])


@tool
def select_best_carrier(region: str, weight_kg: float = 1.0,
                        prefer: str = "cost") -> dict:
    """
    Pick best carrier for a region. prefer='cost' for cheapest, 'speed' for fastest.
    """
    options = get_shipping_options.invoke({"region": region, "weight_kg": weight_kg})
    if not options:
        return {"error": f"No carriers serve region {region}"}
    if prefer == "speed":
        return min(options, key=lambda x: x["eta_days"])
    return options[0]  # already sorted by cost
