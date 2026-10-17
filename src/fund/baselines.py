"""Rule based benchmark portfolios. These bypass the risk gate."""

from __future__ import annotations

import random
from datetime import datetime

from fund.clock import utc_day
from fund.contracts import GatedTarget, GateResult, PortfolioState, Snapshot

BASELINES = ("sp500", "equal_weight", "trend", "random")
_BOOK = 11
_FLAT_GAP = 0.005


def baseline_targets(name: str, snapshot: Snapshot, now: datetime) -> dict[str, float]:
    """Target weights for one benchmark, in snapshot market order."""
    if name == "sp500":
        return _sp500(snapshot)
    if name == "equal_weight":
        return {market.symbol: 1.0 / _BOOK for market in snapshot.markets}
    if name == "trend":
        return {market.symbol: _trend_weight(market.sma20_gap) for market in snapshot.markets}
    if name == "random":
        rng = random.Random(utc_day(now))
        return {market.symbol: rng.uniform(-0.1, 0.1) for market in snapshot.markets}
    raise ValueError(f"Unknown baseline {name!r}. Choose one of {', '.join(BASELINES)}.")


def as_gate_result(targets: dict[str, float], portfolio: PortfolioState) -> GateResult:
    """Approved gate result for benchmark targets, with stops at zero."""
    return GateResult(
        targets=[
            GatedTarget(
                symbol=symbol,
                current_weight=portfolio.weight(symbol),
                requested_weight=weight,
                approved_weight=weight,
                stop_loss_pct=0.0,
                status="approved",
                reasons=[],
            )
            for symbol, weight in targets.items()
        ]
    )


def _sp500(snapshot: Snapshot) -> dict[str, float]:
    weights = {market.symbol: 0.0 for market in snapshot.markets}
    weights["xyz:SP500"] = 1.0
    return weights


def _trend_weight(gap: float) -> float:
    if abs(gap) < _FLAT_GAP:
        return 0.0
    sign = 1.0 if gap > 0.0 else -1.0
    return sign / _BOOK
