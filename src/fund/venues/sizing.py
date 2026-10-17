# pyright: strict
"""Exchange boundary rounding. Sizes round down. Prices use five significant figures."""

from __future__ import annotations

import math

from fund.contracts import Side


def round_size(size: float, sz_decimals: int) -> float:
    """Round a size down to the market's size decimals."""
    factor = 10**sz_decimals
    return math.floor(size * factor + 1e-9) / factor


def round_price(px: float, sz_decimals: int) -> float:
    """Round a perp price to five significant figures and at most 6 - szDecimals decimals."""
    return round(float(f"{px:.5g}"), 6 - sz_decimals)


def ioc_limit_price(mid: float, side: Side, slippage: float, sz_decimals: int) -> float:
    """IOC limit: mid times one plus slippage for buys, one minus slippage for sells."""
    raw = mid * (1 + slippage) if side == "buy" else mid * (1 - slippage)
    return round_price(raw, sz_decimals)


def stop_limit_price(trigger: float, side: Side, sz_decimals: int) -> float:
    """Protective limit behind a trigger. Sell stops use 0.9, buy stops use 1.1."""
    raw = trigger * 0.9 if side == "sell" else trigger * 1.1
    return round_price(raw, sz_decimals)
