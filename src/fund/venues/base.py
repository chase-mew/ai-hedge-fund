"""Execution venue protocol."""

from __future__ import annotations

from typing import Protocol

from fund.contracts import AccountSnapshot, MarketMeta, OrderOutcome, Side, StopOrder


class Venue(Protocol):
    name: str

    def market_meta(self) -> dict[str, MarketMeta]: ...

    def marks(self) -> dict[str, float]: ...

    def account(self) -> AccountSnapshot: ...

    def market_order(self, symbol: str, side: Side, size: float, reduce_only: bool) -> OrderOutcome: ...

    def close_position(self, symbol: str) -> OrderOutcome: ...

    def ensure_leverage(self, symbol: str, leverage: int, cross: bool) -> None: ...

    def sync_stop(self, symbol: str, stop: StopOrder | None) -> None: ...
