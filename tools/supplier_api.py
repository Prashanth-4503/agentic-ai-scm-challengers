import pandas as pd
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()

CATALOG_PATH = "data/supplier_catalog.csv"
SUPPLIERS_PATH = "data/suppliers.csv"


@tool
def supplier_api(sku: str, quantity: int):
    """
    Fetch supplier quotations for the given SKU and quantity.
    Returns eligible suppliers sorted by:
    1. Lowest unit cost
    2. Shortest lead time
    3. Highest reliability score
    """

    try:
        # Load supplier catalog and supplier master
        catalog_df = pd.read_csv(CATALOG_PATH)
        supplier_df = pd.read_csv(SUPPLIERS_PATH)

        # Filter by SKU
        sku_info = catalog_df[catalog_df["sku"] == sku]

        if sku_info.empty:
            return {
                "status": "FAILED",
                "message": f"SKU '{sku}' not found."
            }

        # Check supplier eligibility
        eligible = sku_info[
            (sku_info["available_qty"] >= quantity) &
            (quantity >= sku_info["moq"])
        ]

        if eligible.empty:
            return {
                "status": "FAILED",
                "message": f"No supplier can fulfill {quantity} units of '{sku}'."
            }

        # Merge supplier information
        supplier_data = eligible.merge(
            supplier_df,
            on="supplier_id",
            how="left"
        )

        # Sort suppliers
        supplier_data = supplier_data.sort_values(
            by=["unit_cost", "lead_time_days", "reliability_score"],
            ascending=[True, True, False]
        )

        return {
            "status": "SUCCESS",
            "sku": sku,
            "requested_quantity": quantity,
            "quotes": supplier_data.to_dict(orient="records")
        }

    except Exception as e:
        return {
            "status": "FAILED",
            "message": str(e)
        }


@tool
def place_purchase_order(po: dict):
    """
    Mock Purchase Order API.
    Called only after approval.
    """

    try:
        return {
            "status": "SUCCESS",
            "message": "Purchase Order placed successfully.",
            "purchase_order": {
                "supplier_id": po["supplier_id"],
                "supplier_name": po["supplier_name"],
                "sku": po["sku"],
                "quantity": po["quantity"],
                "unit_price": po["unit_price"],
                "total_cost": po["total_cost"],
                "lead_time_days": po["lead_time_days"]
            }
        }

    except Exception as e:
        return {
            "status": "FAILED",
            "message": str(e)
        }


if __name__ == "__main__":

    # Display available SKUs
    catalog = pd.read_csv(CATALOG_PATH)
    print("Available SKUs:")
    print(catalog["sku"].unique())

    # Replace with a valid SKU from the output above
    result = supplier_api.invoke(
        {
            "sku": "LAP001",
            "quantity": 100
        }
    )

    print("\nSupplier Quotes:\n")
    print(result)

    # Test PO placement
    if result["status"] == "SUCCESS":

        best_supplier = result["quotes"][0]

        po = {
            "supplier_id": best_supplier["supplier_id"],
            "supplier_name": best_supplier["supplier_name"],
            "sku": "LAP001",
            "quantity": 100,
            "unit_price": best_supplier["unit_cost"],
            "total_cost": best_supplier["unit_cost"] * 100,
            "lead_time_days": best_supplier["lead_time_days"]
        }

        print("\nPurchase Order Placement:\n")
        print(place_purchase_order.invoke({"po": po}))
