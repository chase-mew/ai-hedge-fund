"""Shared pytest options."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from fund.contracts import Candle, MarketMeta, MarketQuote
from fund.data.hyperliquid_data import recorded_client


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Rewrite golden prompt files.",
    )


@pytest.fixture
def update_golden(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--update-golden"))


class FixtureData:
    """Market data loaded from recorded Hyperliquid responses."""

    def __init__(self, directory: Path | None = None) -> None:
        root = directory or Path(__file__).resolve().parent / "fixtures"
        self._client = recorded_client(root)

    def quotes(self, symbols: list[str]) -> dict[str, MarketQuote]:
        """Return quotes parsed from recorded meta and asset contexts."""
        return self._client.quotes(symbols)

    def meta(self, symbols: list[str]) -> dict[str, MarketMeta]:
        """Return exchange metadata parsed from recorded meta and asset contexts."""
        return self._client.meta(symbols)

    def candles(self, symbol: str, interval: str, start: datetime, end: datetime) -> list[Candle]:
        """Return recorded candles for one symbol sorted by open time."""
        return self._client.candles(symbol, interval, start, end)


@pytest.fixture
def fixture_data() -> FixtureData:
    """Recorded Hyperliquid responses for offline tests."""
    return FixtureData()
