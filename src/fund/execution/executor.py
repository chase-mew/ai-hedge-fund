# pyright: strict
"""Turn gated targets into orders and send them to a venue."""

from __future__ import annotations

import logging
from typing import Literal

from fund.config import ExecutionSettings
from fund.contracts import (
    AccountSnapshot,
    GatedTarget,
    GateResult,
    MarketMeta,
    OrderOutcome,
    OrderPlan,
    Side,
    StopOrder,
)
from fund.venues.base import Venue
from fund.venues.sizing import round_price, round_size

logger = logging.getLogger(__name__)

_IncreaseReason = Literal["flip_open", "reduce", "increase"]
_CloseReason = Literal["close", "flip_close"]


def plan_orders(
    gate: GateResult,
    account: AccountSnapshot,
    marks: dict[str, float],
    meta: dict[str, MarketMeta],
    min_order_usd: float,
) -> list[OrderPlan]:
    """Build reduce only plans first, then opens and increases."""
    planned: list[OrderPlan] = []
    for target in gate.targets:
        if target.status == "rejected":
            continue
        planned.extend(_plans_for(target, account, marks, meta, min_order_usd))
    reduces = [plan for plan in planned if plan.reduce_only]
    increases = [plan for plan in planned if not plan.reduce_only]
    return reduces + increases


def desired_stops(
    gate: GateResult,
    account: AccountSnapshot,
    meta: dict[str, MarketMeta],
) -> dict[str, StopOrder | None]:
    """Build the stop for each gated symbol, or none when flat."""
    primary = [target for target in gate.targets if target.status != "rejected"]
    extras = [target for target in gate.targets if target.status == "rejected"]
    chosen: dict[str, GatedTarget] = {}
    for target in primary + extras:
        if target.symbol in chosen:
            continue
        chosen[target.symbol] = target
    return {symbol: _stop_for(target, account, meta) for symbol, target in chosen.items()}


def execute(
    gate: GateResult,
    venue: Venue,
    settings: ExecutionSettings,
    reference_marks: dict[str, float],
) -> list[OrderOutcome]:
    """Send plans, fail closed on error, then sync stops."""
    account = venue.account()
    marks = venue.marks()
    meta = venue.market_meta()
    plans = plan_orders(gate, account, marks, meta, settings.min_order_usd)
    outcomes: list[OrderOutcome] = []
    blocked = False
    for plan in plans:
        if not plan.reduce_only:
            skipped = _skip_increase(plan, marks, reference_marks, settings.max_price_gap, blocked)
            if skipped is not None:
                outcomes.append(skipped)
                logger.info("skip %s %s", plan.symbol, skipped.message)
                continue
            market = meta[plan.symbol]
            venue.ensure_leverage(
                plan.symbol,
                min(settings.exchange_leverage, market.max_leverage),
                not market.only_isolated,
            )
        outcome = venue.close_position(plan.symbol) if plan.close_all else _send(venue, plan)
        outcomes.append(outcome)
        logger.info("order %s %s %s %s", plan.symbol, plan.side, plan.size, outcome.status)
        if outcome.status == "error":
            blocked = True
    fresh = venue.account()
    for symbol, stop in desired_stops(gate, fresh, meta).items():
        venue.sync_stop(symbol, stop)
    return outcomes


def _plans_for(
    target: GatedTarget,
    account: AccountSnapshot,
    marks: dict[str, float],
    meta: dict[str, MarketMeta],
    min_order_usd: float,
) -> list[OrderPlan]:
    current = _size(account, target.symbol)
    if target.approved_weight == 0:
        if current == 0:
            return []
        return [_close(target.symbol, current, "close")]
    mark = marks.get(target.symbol, 0.0)
    market = meta.get(target.symbol)
    if mark <= 0 or market is None:
        return []
    target_size = target.approved_weight * account.equity / mark
    if current * target_size < 0:
        return _flip(target.symbol, current, target_size, mark, market, min_order_usd)
    resized = _resize(target.symbol, target_size - current, target_size, current, mark, market, min_order_usd)
    if resized is None:
        return []
    return [resized]


def _flip(
    symbol: str,
    current: float,
    target_size: float,
    mark: float,
    market: MarketMeta,
    min_order_usd: float,
) -> list[OrderPlan]:
    plans: list[OrderPlan] = [_close(symbol, current, "flip_close")]
    opened = _sized(symbol, target_size, mark, market, min_order_usd, "flip_open")
    if opened is not None:
        plans.append(opened)
    return plans


def _resize(
    symbol: str,
    delta: float,
    target_size: float,
    current: float,
    mark: float,
    market: MarketMeta,
    min_order_usd: float,
) -> OrderPlan | None:
    reason: Literal["reduce", "increase"] = "reduce" if abs(target_size) < abs(current) else "increase"
    return _sized(symbol, delta, mark, market, min_order_usd, reason)


def _sized(
    symbol: str,
    signed_size: float,
    mark: float,
    market: MarketMeta,
    min_order_usd: float,
    reason: _IncreaseReason,
) -> OrderPlan | None:
    size = round_size(abs(signed_size), market.sz_decimals)
    if size * mark < min_order_usd:
        return None
    side: Side = "buy" if signed_size > 0 else "sell"
    return OrderPlan(
        symbol=symbol,
        side=side,
        size=size,
        reduce_only=reason == "reduce",
        close_all=False,
        reason=reason,
    )


def _close(symbol: str, current: float, reason: _CloseReason) -> OrderPlan:
    side: Side = "sell" if current > 0 else "buy"
    return OrderPlan(
        symbol=symbol,
        side=side,
        size=abs(current),
        reduce_only=True,
        close_all=True,
        reason=reason,
    )


def _stop_for(target: GatedTarget, account: AccountSnapshot, meta: dict[str, MarketMeta]) -> StopOrder | None:
    position = account.positions.get(target.symbol)
    market = meta.get(target.symbol)
    if position is None or target.stop_loss_pct == 0 or market is None:
        return None
    if position.size > 0:
        side: Side = "sell"
        raw = position.entry_px * (1 - target.stop_loss_pct / 100)
    else:
        side = "buy"
        raw = position.entry_px * (1 + target.stop_loss_pct / 100)
    return StopOrder(
        symbol=target.symbol,
        side=side,
        size=abs(position.size),
        trigger_px=round_price(raw, market.sz_decimals),
    )


def _skip_increase(
    plan: OrderPlan,
    marks: dict[str, float],
    reference: dict[str, float],
    limit: float,
    blocked: bool,
) -> OrderOutcome | None:
    if _price_gap(plan.symbol, marks, reference, limit):
        return _skipped(plan, "price_gap")
    if blocked:
        return _skipped(plan, "prior_error")
    return None


def _price_gap(symbol: str, marks: dict[str, float], reference: dict[str, float], limit: float) -> bool:
    if symbol not in reference:
        return False
    ref = reference[symbol]
    if ref == 0:
        return True
    return abs(marks[symbol] / ref - 1) > limit


def _send(venue: Venue, plan: OrderPlan) -> OrderOutcome:
    return venue.market_order(plan.symbol, plan.side, plan.size, plan.reduce_only)


def _skipped(plan: OrderPlan, message: str) -> OrderOutcome:
    return OrderOutcome(
        symbol=plan.symbol,
        side=plan.side,
        requested_size=plan.size,
        filled_size=0.0,
        avg_px=None,
        fee_usd=0.0,
        status="skipped",
        message=message,
    )


def _size(account: AccountSnapshot, symbol: str) -> float:
    position = account.positions.get(symbol)
    if position is None:
        return 0.0
    return position.size
