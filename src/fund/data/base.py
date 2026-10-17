"""Market data provider protocol."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from fund.contracts import Candle, MarketMeta, MarketQuote


class MarketData(Protocol):
    def quotes(self, symbols: list[str]) -> dict[str, MarketQuote]: ...

    def meta(self, symbols: list[str]) -> dict[str, MarketMeta]: ...

    def candles(self, symbol: str, interval: str, start: datetime, end: datetime) -> list[Candle]: ...
