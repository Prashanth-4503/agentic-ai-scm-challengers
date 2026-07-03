from langchain.tools import tool

@tool
def calculate_total_cost(quantity: int, unit_price: float) -> float:
    """Calculate total cost based on quantity and unit price."""
    total_cost = quantity * unit_price
    return {
        "quantity": quantity,
        "unit_price": unit_price,
        "total_cost": total_cost
    }
    