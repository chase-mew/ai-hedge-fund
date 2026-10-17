# pyright: strict
"""Pure feature calculations from quotes and daily candles."""

from __future__ import annotations

import math
import statistics

from fund.contracts import Candle, MarketFeatures, MarketQuote, MarketSpec, SessionState

_MIN_CANDLES = 51


def compute_features(
    spec: MarketSpec,
    quote: MarketQuote,
    daily: list[Candle],
    session: SessionState,
) -> MarketFeatures:
    """Compute returns, volatility, and moving average gaps for one market."""
    if len(daily) < _MIN_CANDLES:
        raise ValueError(f"Need at least {_MIN_CANDLES} daily candles for {spec.symbol}; got {len(daily)}.")
    closes = [candle.close for candle in daily]
    mark = quote.mark
    window = closes[-31:]
    log_returns = [math.log(window[index] / window[index - 1]) for index in range(1, len(window))]
    highs = [candle.high for candle in daily[-20:]]
    lows = [candle.low for candle in daily[-20:]]
    return MarketFeatures(
        symbol=spec.symbol,
        display_name=spec.display_name,
        asset_class=spec.asset_class,
        tradfi_hours=spec.tradfi_hours,
        market_open=session.tradfi_open or not spec.tradfi_hours,
        mark=mark,
        oracle=quote.oracle,
        ret_24h=mark / quote.prev_day_px - 1,
        ret_7d=mark / closes[-8] - 1,
        ret_30d=mark / closes[-31] - 1,
        vol_30d_ann=statistics.pstdev(log_returns) * math.sqrt(365),
        pct_from_20d_high=mark / max(highs) - 1,
        pct_from_20d_low=mark / min(lows) - 1,
        sma20_gap=mark / _mean(closes[-20:]) - 1,
        sma50_gap=mark / _mean(closes[-50:]) - 1,
        funding_ann=quote.funding_hourly * 24 * 365,
        open_interest_usd=quote.open_interest * mark,
        volume_24h_usd=quote.day_volume_usd,
        daily_closes=closes[-10:],
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)
