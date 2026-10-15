"""A self-contained MCP server used by the MCP examples and benchmarks.

This server is intentionally *reproducible*: it needs no credentials, makes no
network calls, and ships its own in-memory dataset. That makes it suitable for
CI and for apples-to-apples token/latency comparisons between a baseline
tool-calling agent and a strands-code-agent that calls the same tools as code.

The domain is a tiny e-commerce dataset (customers, products, orders). The
tools are deliberately shaped to expose the two scaling pains that the
"code execution with MCP" pattern addresses:

* ``list_orders`` returns a *large* list — in a tool-calling loop every row
  flows through the model's context; in a CodeAgent it can be filtered in the
  sandbox before anything reaches the model.
* answering realistic questions requires *chaining and joining* several tools,
  which a tool-calling agent must do across many turns.

Run it directly over stdio (what the examples do):

    python examples/mcp_tools/mcp_server.py

Requirements:
    pip install "mcp[cli]"
"""

from __future__ import annotations

import random
from datetime import date, timedelta

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("shop")


# --------------------------------------------------------------------------- #
# In-memory dataset (deterministic — seeded).
# --------------------------------------------------------------------------- #
_RNG = random.Random(42)

_CUSTOMERS = [
    {"id": 1, "name": "Ada Lovelace", "country": "UK", "tier": "gold"},
    {"id": 2, "name": "Linus Torvalds", "country": "US", "tier": "silver"},
    {"id": 3, "name": "Grace Hopper", "country": "US", "tier": "gold"},
    {"id": 4, "name": "Alan Turing", "country": "UK", "tier": "bronze"},
    {"id": 5, "name": "Margaret Hamilton", "country": "US", "tier": "gold"},
]

# Referral chain: each customer was referred by exactly one other (or None for the
# root). The chain is 5 -> 3 -> 2 -> 1 -> 4 -> (none). It can only be traversed one
# hop at a time, because the next id to look up is not known until the current hop
# returns — a genuine *sequential data dependency*. ``referred_by[id]`` is the id
# who referred ``id`` (the previous hop toward the root).
_REFERRED_BY = {5: 3, 3: 2, 2: 1, 1: 4, 4: None}

_PRODUCTS = [
    {"sku": "KBD-01", "name": "Mechanical Keyboard", "price": 120.0, "category": "peripherals"},
    {"sku": "MOU-01", "name": "Wireless Mouse", "price": 45.0, "category": "peripherals"},
    {"sku": "MON-27", "name": "27-inch Monitor", "price": 330.0, "category": "displays"},
    {"sku": "MON-32", "name": "32-inch Monitor", "price": 520.0, "category": "displays"},
    {"sku": "CBL-USB", "name": "USB-C Cable", "price": 15.0, "category": "accessories"},
    {"sku": "DOK-01", "name": "Docking Station", "price": 199.0, "category": "accessories"},
]


def _build_orders() -> list[dict]:
    orders = []
    start = date(2024, 1, 1)
    for i in range(1, 201):  # 200 orders — intentionally large
        customer = _RNG.choice(_CUSTOMERS)
        product = _RNG.choice(_PRODUCTS)
        qty = _RNG.randint(1, 4)
        order_day = start + timedelta(days=_RNG.randint(0, 364))
        orders.append(
            {
                "order_id": i,
                "customer_id": customer["id"],
                "sku": product["sku"],
                "quantity": qty,
                "total": round(product["price"] * qty, 2),
                "date": order_day.isoformat(),
                "status": _RNG.choice(["shipped", "shipped", "shipped", "pending", "cancelled"]),
            }
        )
    return orders


_ORDERS = _build_orders()


# --------------------------------------------------------------------------- #
# Tools.
# --------------------------------------------------------------------------- #
@mcp.tool()
def list_customers() -> list[dict]:
    """List all customers.

    Returns a list of customers, each a dict with keys:
    ``id`` (int), ``name`` (str), ``country`` (str, e.g. "US"/"UK"),
    and ``tier`` (str, one of "gold"/"silver"/"bronze").
    """
    return _CUSTOMERS


@mcp.tool()
def get_customer(customer_id: int) -> dict:
    """Fetch a single customer by id.

    Args:
        customer_id: The customer's numeric id.

    Returns a dict with keys ``id``, ``name``, ``country``, ``tier``.
    Raises an error if no customer has that id.
    """
    for c in _CUSTOMERS:
        if c["id"] == customer_id:
            return c
    raise ValueError(f"No customer with id {customer_id}")


@mcp.tool()
def list_products(category: str | None = None) -> list[dict]:
    """List products, optionally filtered by category.

    Args:
        category: If given, only products in this category are returned
            (one of "peripherals", "displays", "accessories"). If omitted,
            all products are returned.

    Returns a list of dicts with keys ``sku``, ``name``, ``price`` (float),
    ``category``.
    """
    if category is None:
        return _PRODUCTS
    return [p for p in _PRODUCTS if p["category"] == category]


@mcp.tool()
def get_product(sku: str) -> dict:
    """Fetch a single product by SKU.

    Args:
        sku: The product SKU, e.g. "MON-27".

    Returns a dict with keys ``sku``, ``name``, ``price``, ``category``.
    Raises an error if the SKU is unknown.
    """
    for p in _PRODUCTS:
        if p["sku"] == sku:
            return p
    raise ValueError(f"No product with sku {sku!r}")


@mcp.tool()
def list_orders(status: str | None = None) -> list[dict]:
    """List orders, optionally filtered by status.

    NOTE: This can return a *large* list (up to 200 orders). In a tool-calling
    agent every returned row is serialized into the model's context.

    Args:
        status: If given, only orders with this status are returned
            (one of "shipped", "pending", "cancelled"). If omitted, all
            orders are returned.

    Returns a list of dicts with keys ``order_id`` (int),
    ``customer_id`` (int), ``sku`` (str), ``quantity`` (int),
    ``total`` (float), ``date`` (ISO date str), ``status`` (str).
    """
    if status is None:
        return _ORDERS
    return [o for o in _ORDERS if o["status"] == status]


@mcp.tool()
def get_referrer(customer_id: int) -> dict:
    """Return who referred this customer — one hop up the referral chain.

    This is the *only* way to traverse referrals, and it reveals just one hop:
    the next id to look up is unknown until this call returns, so the chain can
    only be followed one step at a time (a sequential data dependency). A
    tool-calling agent must therefore spend one model turn per hop; a CodeAgent
    follows the whole chain in a single ``while`` loop.

    Args:
        customer_id: The customer whose referrer to look up.

    Returns a dict with keys ``customer_id`` (int), ``name`` (str), and
    ``referred_by`` (int id of the referrer, or ``None`` if this customer is the
    root of the chain / was not referred).
    """
    customer = next((c for c in _CUSTOMERS if c["id"] == customer_id), None)
    if customer is None:
        raise ValueError(f"No customer with id {customer_id}")
    return {
        "customer_id": customer_id,
        "name": customer["name"],
        "referred_by": _REFERRED_BY.get(customer_id),
    }


if __name__ == "__main__":
    mcp.run()
