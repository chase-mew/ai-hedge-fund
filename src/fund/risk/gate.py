# pyright: strict
"""Deterministic risk gate. Claude proposes, this module disposes."""

from __future__ import annotations

from datetime import datetime, timedelta

from fund.config import RiskPolicy
from fund.contracts import (
    AssetClass,
    GatedTarget,
    GateResult,
    GateStatus,
    MarketDecision,
    MarketFeatures,
    MarketSpec,
    PortfolioDecision,
    PortfolioState,
    Snapshot,
)


def no_increase(target: float, current: float) -> float:
    """Block a new position, a flip, or any increase. Reductions and holds pass."""
    if current == 0 or target * current < 0:
        return 0.0
    return _sign(current) * min(abs(target), abs(current))


def apply_gate(
    decision: PortfolioDecision | None,
    portfolio: PortfolioState,
    snapshot: Snapshot,
    universe: list[MarketSpec],
    policy: RiskPolicy,
    now: datetime,
) -> GateResult:
    """Clip a decision to the risk policy. Any failure mode removes risk, never adds it."""
    if portfolio.equity <= portfolio.bookkeeping.high_water_mark * (1 - policy.max_drawdown):
        return _halt(portfolio, universe)

    specs = {spec.symbol: spec for spec in universe}
    quotes = {market.symbol: market for market in snapshot.markets}
    rows, rejected = _rows(decision, portfolio, universe, specs)
    for row in rows:
        capped = _clip(row.target, -policy.max_weight_per_market, policy.max_weight_per_market)
        _move(row, capped, "max_weight")
        _dislocation(row, quotes.get(row.symbol), policy)
        _session(row, snapshot, specs[row.symbol], policy)
        _flip_cooldown(row, portfolio, policy, now)
    if portfolio.equity <= portfolio.bookkeeping.day_start_equity * (1 - policy.daily_loss_limit):
        for row in rows:
            _move(row, no_increase(row.target, row.current), "daily_loss_limit")
    for row in rows:
        _min_trade(row, policy)
    # Later steps only shrink absolute targets and never flip their sign, so earlier caps stay true.
    _class_caps(rows, policy)
    _scale(rows, _over(sum(abs(row.target) for row in rows), policy.max_gross), "gross_cap")
    _new_risk(rows, policy)
    _scale(rows, _over(abs(sum(row.target for row in rows)), policy.max_net), "net_cap")
    for row in rows:
        _stop(row, policy)
    return GateResult(targets=[_finish(row) for row in rows] + rejected)


class _Row:
    def __init__(self, symbol: str, current: float, requested: float, asset_class: AssetClass) -> None:
        self.symbol = symbol
        self.current = current
        self.requested = requested
        self.target = requested
        self.asset_class = asset_class
        self.stop = 0.0
        self.reasons: list[str] = []
        self.requested_stop: float | None = None


def _halt(portfolio: PortfolioState, universe: list[MarketSpec]) -> GateResult:
    targets: list[GatedTarget] = []
    for spec in universe:
        current = portfolio.weight(spec.symbol)
        targets.append(
            GatedTarget(
                symbol=spec.symbol,
                current_weight=current,
                requested_weight=current,
                approved_weight=0.0,
                stop_loss_pct=0.0,
                status=_status(0.0, current, current),
                reasons=["max_drawdown"],
            )
        )
    return GateResult(
        targets=targets,
        halt=True,
        halt_reason="Equity is at or below the maximum drawdown limit, so every target is flat.",
    )


def _rows(
    decision: PortfolioDecision | None,
    portfolio: PortfolioState,
    universe: list[MarketSpec],
    specs: dict[str, MarketSpec],
) -> tuple[list[_Row], list[GatedTarget]]:
    chosen: dict[str, MarketDecision] = {}
    rejected: list[GatedTarget] = []
    if decision is not None:
        for item in decision.decisions:
            if item.symbol not in specs:
                rejected.append(_rejected(item, portfolio, "unknown_symbol"))
            elif item.symbol in chosen:
                rejected.append(_rejected(item, portfolio, "duplicate"))
            else:
                chosen[item.symbol] = item
    rows: list[_Row] = []
    for spec in universe:
        current = portfolio.weight(spec.symbol)
        item = chosen.get(spec.symbol)
        if decision is None:
            row = _Row(spec.symbol, current, current, spec.asset_class)
            row.reasons.append("no_decision")
        elif item is None:
            row = _Row(spec.symbol, current, current, spec.asset_class)
            row.reasons.append("missing")
        else:
            row = _Row(spec.symbol, current, item.target_weight, spec.asset_class)
            row.requested_stop = item.stop_loss_pct
        rows.append(row)
    return rows, rejected


def _rejected(item: MarketDecision, portfolio: PortfolioState, reason: str) -> GatedTarget:
    current = portfolio.weight(item.symbol)
    return GatedTarget(
        symbol=item.symbol,
        current_weight=current,
        requested_weight=item.target_weight,
        approved_weight=current,
        stop_loss_pct=item.stop_loss_pct,
        status="rejected",
        reasons=[reason],
    )


def _dislocation(row: _Row, quote: MarketFeatures | None, policy: RiskPolicy) -> None:
    moved = quote is None or quote.oracle <= 0
    if quote is not None and quote.oracle > 0:
        moved = abs(quote.mark / quote.oracle - 1) > policy.max_mark_oracle_deviation
    if moved:
        _move(row, no_increase(row.target, row.current), "price_dislocation")


def _session(row: _Row, snapshot: Snapshot, spec: MarketSpec, policy: RiskPolicy) -> None:
    if not spec.tradfi_hours:
        return
    if not snapshot.session.tradfi_open:
        _move(row, no_increase(row.target, row.current), "market_closed")
    if snapshot.session.pre_weekend:
        capped = _sign(row.target) * min(abs(row.target), policy.pre_weekend_max_weight)
        _move(row, capped, "pre_weekend_cap")


def _flip_cooldown(row: _Row, portfolio: PortfolioState, policy: RiskPolicy, now: datetime) -> None:
    last = portfolio.bookkeeping.last_flip_at.get(row.symbol)
    flipped = row.current != 0 and row.target != 0 and row.target * row.current < 0
    if flipped and last is not None and now - last <= timedelta(hours=policy.flip_cooldown_hours):
        _move(row, 0.0, "flip_cooldown")


def _min_trade(row: _Row, policy: RiskPolicy) -> None:
    growing = (row.current == 0 or row.target * row.current > 0) and abs(row.target) > abs(row.current)
    if growing and abs(row.target - row.current) < policy.min_trade_weight:
        _move(row, row.current, "below_min_trade")
        return
    flipping = row.current != 0 and row.target != 0 and row.target * row.current < 0
    if flipping and abs(row.target) < policy.min_trade_weight:
        _move(row, 0.0, "below_min_trade")


def _class_caps(rows: list[_Row], policy: RiskPolicy) -> None:
    for asset_class, cap in policy.class_caps.items():
        members = [row for row in rows if row.asset_class == asset_class]
        _scale(members, _over(sum(abs(row.target) for row in members), cap), "class_cap")


def _new_risk(rows: list[_Row], policy: RiskPolicy) -> None:
    fresh = [_risk_increase(row) for row in rows]
    factor = _over(sum(fresh), policy.max_new_risk_per_cycle)
    if factor is None:
        return
    for row, amount in zip(rows, fresh, strict=True):
        if amount <= 0:
            continue
        if row.current == 0 or row.target * row.current > 0:
            updated = _sign(row.target) * (abs(row.current) + amount * factor)
        else:
            updated = row.target * factor
        _move(row, updated, "new_risk_cap")


def _risk_increase(row: _Row) -> float:
    if row.current == 0 or row.target * row.current > 0:
        return max(0.0, abs(row.target) - abs(row.current))
    if row.target * row.current < 0:
        return abs(row.target)
    return 0.0


def _stop(row: _Row, policy: RiskPolicy) -> None:
    if row.target == 0:
        row.stop = 0.0
        return
    requested = policy.default_stop_pct if row.requested_stop is None else row.requested_stop
    clamped = _clip(requested, policy.stop_min_pct, policy.stop_max_pct)
    if abs(clamped - requested) > 1e-12:
        row.reasons.append("stop_clamped")
    row.stop = clamped


def _finish(row: _Row) -> GatedTarget:
    return GatedTarget(
        symbol=row.symbol,
        current_weight=row.current,
        requested_weight=row.requested,
        approved_weight=row.target,
        stop_loss_pct=row.stop,
        status=_status(row.target, row.requested, row.current),
        reasons=row.reasons,
    )


def _scale(rows: list[_Row], factor: float | None, reason: str) -> None:
    if factor is None:
        return
    for row in rows:
        _move(row, row.target * factor, reason)


def _over(total: float, cap: float) -> float | None:
    if total > cap:
        return cap / total
    return None


def _move(row: _Row, value: float, reason: str) -> None:
    if abs(value - row.target) <= 1e-12:
        return
    row.target = value
    row.reasons.append(reason)


def _status(approved: float, requested: float, current: float) -> GateStatus:
    if abs(approved - requested) <= 1e-9:
        return "approved"
    if abs(approved - current) <= 1e-9:
        return "held"
    return "clipped"


def _sign(value: float) -> float:
    if value > 0:
        return 1.0
    if value < 0:
        return -1.0
    return 0.0


def _clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)
