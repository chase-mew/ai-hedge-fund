"""Feature formulas on synthetic daily series."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from fund.contracts import AssetClass, Candle, MarketQuote, MarketSpec, SessionState
from fund.data.features import compute_features

START = datetime(2026, 1, 1, tzinfo=UTC)


def _candles(
    closes: list[float],
    highs: list[float] | None = None,
    lows: list[float] | None = None,
) -> list[Candle]:
    rows: list[Candle] = []
    for index, close in enumerate(closes):
        high = highs[index] if highs is not None else close
        low = lows[index] if lows is not None else close
        rows.append(
            Candle(
                open_time=START + timedelta(days=index),
                open=close,
                high=high,
                low=low,
                close=close,
                volume=1.0,
            )
        )
    return rows


def _quote(symbol: str, mark: float, prev: float, funding: float = 0.0) -> MarketQuote:
    return MarketQuote(
        symbol=symbol,
        mark=mark,
        oracle=mark,
        prev_day_px=prev,
        funding_hourly=funding,
        open_interest=2.0,
        day_volume_usd=50.0,
    )


def _spec(symbol: str, tradfi_hours: bool) -> MarketSpec:
    return MarketSpec(
        symbol=symbol,
        display_name=symbol,
        asset_class=AssetClass.STOCK if tradfi_hours else AssetClass.CRYPTO,
        tradfi_hours=tradfi_hours,
    )


def test_constant_prices_have_zero_return_and_volatility() -> None:
    closes = [100.0] * 60
    features = compute_features(
        _spec("BTC", False),
        _quote("BTC", 100.0, 100.0, funding=0.0001),
        _candles(closes),
        SessionState(tradfi_open=False, pre_weekend=False),
    )
    assert features.ret_24h == 0.0
    assert features.ret_7d == 0.0
    assert features.ret_30d == 0.0
    assert features.vol_30d_ann == 0.0
    assert features.pct_from_20d_high == 0.0
    assert features.pct_from_20d_low == 0.0
    assert features.sma20_gap == 0.0
    assert features.sma50_gap == 0.0
    assert features.funding_ann == pytest.approx(0.0001 * 24 * 365)
    assert features.open_interest_usd == pytest.approx(200.0)
    assert features.volume_24h_usd == 50.0
    assert features.daily_closes == [100.0] * 10
    assert features.market_open is True


def test_geometric_series_has_known_returns() -> None:
    ratio = 1.01
    closes = [100.0 * ratio**index for index in range(60)]
    mark = closes[-1]
    features = compute_features(
        _spec("ETH", False),
        _quote("ETH", mark, closes[-2]),
        _candles(closes),
        SessionState(tradfi_open=True, pre_weekend=False),
    )
    assert features.ret_24h == pytest.approx(ratio - 1)
    assert features.ret_7d == pytest.approx(ratio**7 - 1)
    assert features.ret_30d == pytest.approx(ratio**30 - 1)
    assert features.vol_30d_ann == pytest.approx(0.0)
    assert features.sma20_gap == pytest.approx(mark / (sum(closes[-20:]) / 20) - 1)
    assert features.sma50_gap == pytest.approx(mark / (sum(closes[-50:]) / 50) - 1)
    assert features.pct_from_20d_high == pytest.approx(0.0)
    assert features.pct_from_20d_low == pytest.approx(mark / closes[-20] - 1)
    assert features.daily_closes == pytest.approx(closes[-10:])


def test_volatility_annualises_population_std_of_log_returns() -> None:
    returns = [0.01 if index % 2 == 0 else -0.01 for index in range(30)]
    window = [100.0]
    for value in returns:
        window.append(window[-1] * math.exp(value))
    closes = [window[0]] * 20 + window
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / len(returns)
    expected = math.sqrt(variance) * math.sqrt(365)
    features = compute_features(
        _spec("BTC", False),
        _quote("BTC", closes[-1], closes[-2]),
        _candles(closes),
        SessionState(tradfi_open=True, pre_weekend=False),
    )
    assert features.vol_30d_ann == pytest.approx(expected)


def test_distance_from_twenty_day_high_and_low() -> None:
    closes = [100.0] * 60
    highs = [100.0] * 60
    lows = [100.0] * 60
    highs[-5] = 130.0
    lows[-3] = 80.0
    features = compute_features(
        _spec("xyz:GOLD", True),
        _quote("xyz:GOLD", 110.0, 100.0),
        _candles(closes, highs, lows),
        SessionState(tradfi_open=True, pre_weekend=False),
    )
    assert features.pct_from_20d_high == pytest.approx(110.0 / 130.0 - 1)
    assert features.pct_from_20d_low == pytest.approx(110.0 / 80.0 - 1)


def test_too_few_candles_raise() -> None:
    with pytest.raises(ValueError, match="51"):
        compute_features(
            _spec("BTC", False),
            _quote("BTC", 100.0, 100.0),
            _candles([100.0] * 50),
            SessionState(tradfi_open=True, pre_weekend=False),
        )


def test_fifty_one_candles_are_enough() -> None:
    features = compute_features(
        _spec("BTC", False),
        _quote("BTC", 100.0, 100.0),
        _candles([100.0] * 51),
        SessionState(tradfi_open=True, pre_weekend=False),
    )
    assert features.sma50_gap == 0.0
    assert features.ret_30d == 0.0


def test_crypto_stays_open_when_tradfi_is_closed() -> None:
    features = compute_features(
        _spec("BTC", False),
        _quote("BTC", 100.0, 100.0),
        _candles([100.0] * 60),
        SessionState(tradfi_open=False, pre_weekend=False),
    )
    assert features.market_open is True
    assert features.tradfi_hours is False


def test_stock_follows_the_session() -> None:
    closed = compute_features(
        _spec("xyz:AAPL", True),
        _quote("xyz:AAPL", 100.0, 100.0),
        _candles([100.0] * 60),
        SessionState(tradfi_open=False, pre_weekend=True),
    )
    opened = compute_features(
        _spec("xyz:AAPL", True),
        _quote("xyz:AAPL", 100.0, 100.0),
        _candles([100.0] * 60),
        SessionState(tradfi_open=True, pre_weekend=False),
    )
    assert closed.market_open is False
    assert opened.market_open is True
    assert closed.tradfi_hours is True
