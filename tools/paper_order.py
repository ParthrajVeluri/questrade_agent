"""Validation and submission for locally simulated paper orders."""

from decimal import Decimal

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class PaperOrderInput(BaseModel):
    """Structured input accepted by the paper-order tool."""

    side: str = Field(description="buy or sell")
    symbol: str = Field(description="Ticker symbol")
    quantity: str = Field(description="Number of shares")
    order_type: str = Field(description="market or limit")
    limit_price: float | None = Field(
        default=None,
        description="Limit price, or null for a market order",
    )


@tool("submit_paper_order", args_schema=PaperOrderInput)
def submit_paper_order(
    side: str,
    symbol: str,
    quantity: str,
    order_type: str,
    limit_price: float | None = None,
) -> dict:
    """Validate and submit an order to the local paper-trading implementation."""
    share_quantity = Decimal(quantity)
    if share_quantity <= 0:
        raise ValueError("Share quantity must be greater than zero.")
    if side not in {"buy", "sell"}:
        raise ValueError("Side must be buy or sell.")
    if order_type not in {"market", "limit"}:
        raise ValueError("Order type must be market or limit.")
    if order_type == "limit" and (limit_price is None or limit_price <= 0):
        raise ValueError("Limit orders require a positive limit price.")

    return {
        "status": "submitted",
        "side": side,
        "symbol": symbol.upper(),
        "quantity": str(share_quantity),
        "order_type": order_type,
        "limit_price": limit_price,
    }
