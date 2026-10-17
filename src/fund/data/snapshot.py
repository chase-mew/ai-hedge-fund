# pyright: strict
"""Build a market snapshot for one cycle."""

from __future__ import annotations

from datetime import datetime, timedelta

from fund.clock import cycle_id, cycle_slot, session_state
from fund.contracts import MarketSpec, NewsDigest, Snapshot
from fund.data.base import MarketData
from fund.data.features import compute_features

_LOOKBACK_DAYS = 120


def build_snapshot(
    data: MarketData,
    universe: list[MarketSpec],
    now: datetime,
    cycle_hours: int,
    news: NewsDigest | None = None,
) -> Snapshot:
    """Fetch quotes and daily candles and build a snapshot for every market."""
    symbols = [spec.symbol for spec in universe]
    quotes = data.quotes(symbols)
    start = now - timedelta(days=_LOOKBACK_DAYS)
    session = session_state(now)
    markets = [
        compute_features(spec, quotes[spec.symbol], data.candles(spec.symbol, "1d", start, now), session)
        for spec in universe
    ]
    return Snapshot(
        cycle_id=cycle_id(cycle_slot(now, cycle_hours)),
        as_of=now,
        session=session,
        markets=markets,
        news=news,
    )
