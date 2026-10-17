# pyright: strict
"""Paper venue accounting."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from fund.config import PaperSettings
from fund.contracts import (
    AccountSnapshot,
    Bookkeeping,
    Candle,
    MarketMeta,
    MarketQuote,
    OrderOutcome,
    PaperAccount,
    PaperPosition,
    Position,
    Side,
    StopOrder,
)


def new_paper_account(portfolio_id: str, equity: float, now: datetime) -> PaperAccount:
    """Create a flat paper account holding the starting equity as cash."""
    return PaperAccount(
        portfolio_id=portfolio_id,
        cash=equity,
        positions={},
        last_marks={},
        updated_at=now,
        bookkeeping=Bookkeeping(
            day=now.strftime("%Y-%m-%d"),
            day_start_equity=equity,
            high_water_mark=equity,
        ),
    )


def paper_equity(account: PaperAccount, marks: dict[str, float]) -> float:
    """Return cash plus open pnl at the supplied marks."""
    pnl = 0.0
    for symbol, position in account.positions.items():
        pnl += position.size * (_mark(account, marks, symbol, position.entry_px) - position.entry_px)
    return account.cash + pnl


def accrue_funding(account: PaperAccount, quotes: dict[str, MarketQuote], now: datetime) -> PaperAccount:
    """Charge funding since updated_at and advance the clock."""
    hours = (now - account.updated_at).total_seconds() / 3600
    if hours <= 0:
        return account
    cash = account.cash
    funding = account.funding_cum
    last_marks = dict(account.last_marks)
    for symbol, position in account.positions.items():
        quote = quotes.get(symbol)
        if quote is None:
            continue
        payment = position.size * quote.mark * quote.funding_hourly * hours
        cash -= payment
        funding += payment
        last_marks[symbol] = quote.mark
    return _account(account, cash=cash, funding_cum=funding, last_marks=last_marks, updated_at=now)


def process_stops(
    account: PaperAccount,
    hourly: dict[str, list[Candle]],
    settings: PaperSettings,
    now: datetime,
) -> PaperAccount:
    """Fill stops that hourly candles touched and remove those positions."""
    updated = account
    for symbol, position in account.positions.items():
        stop = position.stop
        if stop is None:
            continue
        if not _stop_hit(position.size, stop.trigger_px, hourly.get(symbol, []), account.updated_at, now):
            continue
        side: Side = "sell" if position.size > 0 else "buy"
        slip = settings.slippage_bps / 10_000
        price = stop.trigger_px * (1 - slip) if position.size > 0 else stop.trigger_px * (1 + slip)
        updated = _fill(updated, symbol, side, abs(position.size), price, settings, None, None)
    return updated


class PaperVenue:
    """Simulated venue that fills against marks with fees and slippage."""

    name: str = "paper"

    def __init__(
        self,
        account: PaperAccount,
        marks: dict[str, float],
        meta: dict[str, MarketMeta],
        settings: PaperSettings,
        now: datetime,
    ) -> None:
        self._account = account
        self._marks = dict(marks)
        self._meta = dict(meta)
        self._settings = settings
        self._now = now
        self._next_id = 1

    def state(self) -> PaperAccount:
        """Return the account to persist."""
        return self._account

    def market_meta(self) -> dict[str, MarketMeta]:
        """Return the market rules supplied at construction."""
        return self._meta

    def marks(self) -> dict[str, float]:
        """Return the latest marks."""
        return self._marks

    def account(self) -> AccountSnapshot:
        """Return equity and positions marked to market."""
        positions = {
            symbol: Position(
                symbol=symbol,
                size=position.size,
                entry_px=position.entry_px,
                mark=_mark(self._account, self._marks, symbol, position.entry_px),
            )
            for symbol, position in self._account.positions.items()
        }
        return AccountSnapshot(equity=paper_equity(self._account, self._marks), positions=positions)

    def market_order(self, symbol: str, side: Side, size: float, reduce_only: bool) -> OrderOutcome:
        """Fill a market order at the slipped mark."""
        mark = self._marks.get(symbol)
        if size <= 0:
            return _outcome(symbol, side, size, 0.0, None, 0.0, "unfilled")
        if mark is None or mark <= 0:
            return _outcome(symbol, side, size, 0.0, None, 0.0, "unfilled")
        position = self._account.positions.get(symbol)
        fill_size = _closable(position, side, size) if reduce_only else size
        if fill_size <= 0:
            return _outcome(symbol, side, size, 0.0, None, 0.0, "unfilled")
        price = _trade_price(mark, side, self._settings)
        self._account = _fill(self._account, symbol, side, fill_size, price, self._settings, self._now, mark)
        order_id = f"paper-{self._next_id}"
        self._next_id += 1
        status: Literal["filled", "partial"] = "filled" if fill_size == size else "partial"
        fee = _fee(symbol, fill_size, price, self._settings)
        return _outcome(symbol, side, size, fill_size, price, fee, status, order_id)

    def close_position(self, symbol: str) -> OrderOutcome:
        """Close the whole position, or skip when already flat."""
        position = self._account.positions.get(symbol)
        if position is None:
            return _outcome(symbol, "sell", 0.0, 0.0, None, 0.0, "skipped", message="flat")
        side: Side = "sell" if position.size > 0 else "buy"
        return self.market_order(symbol, side, abs(position.size), True)

    def ensure_leverage(self, symbol: str, leverage: int, cross: bool) -> None:
        """Ignore leverage on the paper venue."""

    def sync_stop(self, symbol: str, stop: StopOrder | None) -> None:
        """Store or clear the stop on an open position."""
        position = self._account.positions.get(symbol)
        if position is None:
            return
        positions = dict(self._account.positions)
        positions[symbol] = position.model_copy(update={"stop": stop})
        self._account = _account(self._account, positions=positions)


def _mark(account: PaperAccount, marks: dict[str, float], symbol: str, entry_px: float) -> float:
    if symbol in marks:
        return marks[symbol]
    if symbol in account.last_marks:
        return account.last_marks[symbol]
    return entry_px


def _trade_price(mark: float, side: Side, settings: PaperSettings) -> float:
    slip = settings.slippage_bps / 10_000
    if side == "buy":
        return mark * (1 + slip)
    return mark * (1 - slip)


def _fee(symbol: str, size: float, price: float, settings: PaperSettings) -> float:
    rate = settings.xyz_taker_fee if symbol.startswith("xyz:") else settings.core_taker_fee
    return abs(size) * price * rate


def _closable(position: PaperPosition | None, side: Side, size: float) -> float:
    if position is None:
        return 0.0
    opposing = (position.size > 0 and side == "sell") or (position.size < 0 and side == "buy")
    if not opposing:
        return 0.0
    return min(size, abs(position.size))


def _fill(
    account: PaperAccount,
    symbol: str,
    side: Side,
    size: float,
    price: float,
    settings: PaperSettings,
    now: datetime | None,
    mark: float | None,
) -> PaperAccount:
    position = account.positions.get(symbol)
    updated, realised = _position_after(position, symbol, side, size, price)
    fee = _fee(symbol, size, price, settings)
    positions = dict(account.positions)
    if updated is None:
        positions.pop(symbol, None)
    else:
        positions[symbol] = updated
    last_marks = dict(account.last_marks)
    if mark is not None:
        last_marks[symbol] = mark
    return _account(
        account,
        cash=account.cash + realised - fee,
        positions=positions,
        last_marks=last_marks,
        updated_at=account.updated_at if now is None else now,
        fees_cum=account.fees_cum + fee,
    )


def _position_after(
    position: PaperPosition | None,
    symbol: str,
    side: Side,
    size: float,
    price: float,
) -> tuple[PaperPosition | None, float]:
    delta = size if side == "buy" else -size
    if position is None:
        return PaperPosition(symbol=symbol, size=delta, entry_px=price), 0.0
    old = position.size
    if old * delta > 0:
        new_size = old + delta
        entry = (old * position.entry_px + delta * price) / new_size
        return position.model_copy(update={"size": new_size, "entry_px": entry}), 0.0
    closed = min(abs(delta), abs(old))
    sign = 1.0 if old > 0 else -1.0
    realised = (price - position.entry_px) * closed * sign
    leftover = abs(delta) - closed
    if leftover == 0:
        reduced = old + delta
        if reduced == 0:
            return None, realised
        return position.model_copy(update={"size": reduced}), realised
    opened = PaperPosition(symbol=symbol, size=leftover if delta > 0 else -leftover, entry_px=price)
    return opened, realised


def _stop_hit(size: float, trigger: float, candles: list[Candle], start: datetime, end: datetime) -> bool:
    for candle in sorted(candles, key=lambda item: item.open_time):
        if candle.open_time <= start or candle.open_time > end:
            continue
        if size > 0 and candle.low <= trigger:
            return True
        if size < 0 and candle.high >= trigger:
            return True
    return False


def _account(
    account: PaperAccount,
    *,
    cash: float | None = None,
    positions: dict[str, PaperPosition] | None = None,
    last_marks: dict[str, float] | None = None,
    updated_at: datetime | None = None,
    fees_cum: float | None = None,
    funding_cum: float | None = None,
) -> PaperAccount:
    return PaperAccount(
        portfolio_id=account.portfolio_id,
        cash=account.cash if cash is None else cash,
        positions=account.positions if positions is None else positions,
        last_marks=account.last_marks if last_marks is None else last_marks,
        updated_at=account.updated_at if updated_at is None else updated_at,
        bookkeeping=account.bookkeeping,
        fees_cum=account.fees_cum if fees_cum is None else fees_cum,
        funding_cum=account.funding_cum if funding_cum is None else funding_cum,
        llm_cost_cum=account.llm_cost_cum,
    )


def _outcome(
    symbol: str,
    side: Side,
    requested: float,
    filled: float,
    price: float | None,
    fee: float,
    status: Literal["filled", "partial", "unfilled", "error", "skipped"],
    order_id: str | None = None,
    message: str = "",
) -> OrderOutcome:
    return OrderOutcome(
        symbol=symbol,
        side=side,
        requested_size=requested,
        filled_size=filled,
        avg_px=price,
        fee_usd=fee,
        status=status,
        order_id=order_id,
        message=message,
    )
