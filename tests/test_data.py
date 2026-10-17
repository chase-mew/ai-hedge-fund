"""Hyperliquid parsing, retries, and snapshot assembly."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
import requests
from hyperliquid.info import Info
from hyperliquid.utils.error import ClientError

from fund.config import load_config
from fund.contracts import Candle, MarketQuote, NewsDigest, Snapshot
from fund.data.base import MarketData
from fund.data.hyperliquid_data import HyperliquidData
from fund.data.snapshot import build_snapshot

FIXTURES = Path(__file__).resolve().parent / "fixtures"
NOW = datetime(2026, 10, 14, 16, 0, tzinfo=UTC)
START = datetime(2026, 10, 1, tzinfo=UTC)
END = datetime(2026, 10, 14, tzinfo=UTC)


def _asset(
    name: str,
    decimals: object = "5",
    leverage: object = "40",
    isolated: bool | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {"name": name, "szDecimals": decimals, "maxLeverage": leverage}
    if isolated is not None:
        row["onlyIsolated"] = isolated
    return row


def _ctx(mark: str = "10") -> dict[str, str]:
    return {
        "markPx": mark,
        "oraclePx": "11",
        "prevDayPx": "9",
        "funding": "0.0001",
        "openInterest": "4",
        "dayNtlVlm": "8.5",
    }


class FlakyInfo:
    def __init__(
        self,
        dexes: dict[str, tuple[list[dict[str, object]], list[dict[str, str]]]] | None = None,
        candles: list[dict[str, object]] | None = None,
        errors: Sequence[BaseException] | None = None,
    ) -> None:
        self.dexes = dexes or {}
        self._candles = candles or []
        self.errors = list(errors or [])
        self.posts: list[dict[str, Any]] = []
        self.calls = 0

    def post(self, url_path: str, payload: dict[str, Any] | None = None) -> Any:
        del url_path
        self.calls += 1
        self._raise_scheduled()
        body = payload or {}
        self.posts.append(body)
        dex = str(body.get("dex", ""))
        assets, contexts = self.dexes[dex]
        return [{"universe": assets}, contexts]

    def candles_snapshot(
        self, name: str, interval: str, startTime: int, endTime: int
    ) -> list[dict[str, object]]:
        del name, interval, startTime, endTime
        self.calls += 1
        self._raise_scheduled()
        return list(self._candles)

    def _raise_scheduled(self) -> None:
        if self.errors:
            raise self.errors.pop(0)


def _client(info: FlakyInfo) -> HyperliquidData:
    return HyperliquidData(info=cast(Info, info))


def _sample_dexes() -> dict[str, tuple[list[dict[str, object]], list[dict[str, str]]]]:
    return {
        "": ([_asset("BTC")], [_ctx("10")]),
        "xyz": (
            [
                _asset("xyz:JPY", decimals="2", leverage="50", isolated=True),
                _asset("xyz:SP500", isolated=False),
            ],
            [_ctx("157.8"), _ctx("7600")],
        ),
    }


def _flat_candles(count: int = 60) -> list[Candle]:
    return [
        Candle(
            open_time=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=index),
            open=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            volume=1.0,
        )
        for index in range(count)
    ]


class StubData:
    def __init__(self, fail: str | None = None) -> None:
        self.fail = fail
        self.quote_calls = 0
        self.candle_calls: list[tuple[str, str, datetime, datetime]] = []

    def quotes(self, symbols: list[str]) -> dict[str, MarketQuote]:
        self.quote_calls += 1
        return {
            symbol: MarketQuote(
                symbol=symbol,
                mark=100.0,
                oracle=100.0,
                prev_day_px=100.0,
                funding_hourly=0.0,
                open_interest=1.0,
                day_volume_usd=1.0,
            )
            for symbol in symbols
        }

    def meta(self, symbols: list[str]) -> dict[str, Any]:
        del symbols
        raise AssertionError("snapshot must not fetch meta")

    def candles(self, symbol: str, interval: str, start: datetime, end: datetime) -> list[Candle]:
        self.candle_calls.append((symbol, interval, start, end))
        if symbol == self.fail:
            raise RuntimeError(f"candles failed for {symbol}")
        return _flat_candles()


def test_quotes_parse_string_numbers_and_post_both_dexes() -> None:
    info = FlakyInfo(dexes=_sample_dexes())
    quotes = _client(info).quotes(["BTC", "xyz:JPY"])
    assert [post["dex"] for post in info.posts] == ["", "xyz"]
    assert [post["type"] for post in info.posts] == ["metaAndAssetCtxs", "metaAndAssetCtxs"]
    assert quotes["BTC"].mark == 10.0
    assert quotes["BTC"].oracle == 11.0
    assert quotes["BTC"].prev_day_px == 9.0
    assert quotes["BTC"].funding_hourly == pytest.approx(0.0001)
    assert quotes["BTC"].open_interest == 4.0
    assert quotes["BTC"].day_volume_usd == 8.5
    assert quotes["xyz:JPY"].mark == 157.8


def test_meta_reads_string_ints_and_defaults_only_isolated() -> None:
    metas = _client(FlakyInfo(dexes=_sample_dexes())).meta(["BTC", "xyz:JPY", "xyz:SP500"])
    assert metas["BTC"].sz_decimals == 5
    assert metas["BTC"].max_leverage == 40
    assert metas["BTC"].only_isolated is False
    assert metas["xyz:JPY"].sz_decimals == 2
    assert metas["xyz:JPY"].max_leverage == 50
    assert metas["xyz:JPY"].only_isolated is True
    assert metas["xyz:SP500"].only_isolated is False


def test_missing_symbol_raises_key_error() -> None:
    data = _client(FlakyInfo(dexes=_sample_dexes()))
    with pytest.raises(KeyError, match="xyz:GONE"):
        data.quotes(["xyz:GONE"])
    with pytest.raises(KeyError, match="xyz:GONE"):
        data.meta(["xyz:GONE"])


def test_candles_sort_string_fields_by_open_time() -> None:
    info = FlakyInfo(
        candles=[
            {"t": 2_000, "o": "3", "h": "4", "l": "2", "c": "3.5", "v": "9"},
            {"t": 1_000, "o": "1", "h": "2", "l": "0.5", "c": "1.5", "v": "8"},
        ]
    )
    candles = _client(info).candles("BTC", "1d", START, END)
    assert [candle.open_time for candle in candles] == [
        datetime.fromtimestamp(1, tz=UTC),
        datetime.fromtimestamp(2, tz=UTC),
    ]
    assert candles[0].close == 1.5
    assert candles[0].low == 0.5
    assert candles[0].volume == 8.0
    assert candles[1].open == 3.0


def test_candles_reject_naive_datetimes() -> None:
    data = _client(FlakyInfo())
    with pytest.raises(ValueError, match="timezone aware"):
        data.candles("BTC", "1d", END.replace(tzinfo=None), END)


def test_unknown_candle_symbol_is_not_retried() -> None:
    info = FlakyInfo(errors=[KeyError("xyz:GONE")])
    with pytest.raises(KeyError, match="xyz:GONE"):
        _client(info).candles("xyz:GONE", "1d", START, END)
    assert info.calls == 1


def test_retries_network_errors_with_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("fund.data.hyperliquid_data.time.sleep", lambda seconds: sleeps.append(seconds))
    info = FlakyInfo(
        candles=[{"t": 1_000, "o": "1", "h": "1", "l": "1", "c": "1", "v": "1"}],
        errors=[requests.ConnectionError("down"), requests.Timeout("slow"), requests.ConnectionError("down")],
    )
    candles = _client(info).candles("BTC", "1h", START, END)
    assert len(candles) == 1
    assert info.calls == 4
    assert sleeps == [1, 2, 4]


def test_retries_http_429(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("fund.data.hyperliquid_data.time.sleep", lambda seconds: sleeps.append(seconds))
    info = FlakyInfo(
        candles=[{"t": 1_000, "o": "1", "h": "1", "l": "1", "c": "1", "v": "1"}],
        errors=[ClientError(429, None, "rate", None)],
    )
    candles = _client(info).candles("BTC", "1d", START, END)
    assert len(candles) == 1
    assert info.calls == 2
    assert sleeps == [1]


def test_does_not_retry_other_http_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("fund.data.hyperliquid_data.time.sleep", lambda seconds: sleeps.append(seconds))
    info = FlakyInfo(errors=[ClientError(400, None, "bad", None)])
    with pytest.raises(ClientError):
        _client(info).candles("BTC", "1d", START, END)
    assert info.calls == 1
    assert sleeps == []


def test_gives_up_after_three_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("fund.data.hyperliquid_data.time.sleep", lambda seconds: sleeps.append(seconds))
    info = FlakyInfo(errors=[requests.ConnectionError("down")] * 4)
    with pytest.raises(requests.ConnectionError):
        _client(info).candles("BTC", "1d", START, END)
    assert info.calls == 4
    assert sleeps == [1, 2, 4]


def test_write_raw_saves_meta_and_daily_candles(tmp_path: Path) -> None:
    info = FlakyInfo(
        dexes=_sample_dexes(),
        candles=[{"t": 1_000, "o": "1", "h": "2", "l": "0.5", "c": "1.5", "v": "8"}],
    )
    _client(info).write_raw(tmp_path, ["BTC", "xyz:JPY"], datetime(2026, 10, 14, 16, tzinfo=UTC))
    core = json.loads((tmp_path / "hl_meta_ctxs_core.json").read_text())
    xyz = json.loads((tmp_path / "hl_meta_ctxs_xyz.json").read_text())
    assert core[0]["universe"][0]["name"] == "BTC"
    assert xyz[0]["universe"][0]["name"] == "xyz:JPY"
    btc = json.loads((tmp_path / "hl_candles_BTC.json").read_text())
    jpy = json.loads((tmp_path / "hl_candles_xyz__JPY.json").read_text())
    assert btc[0]["v"] == "8"
    assert jpy[0]["c"] == "1.5"


def test_constructor_requests_both_perp_dexes(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_info(base_url: str, skip_ws: bool = False, perp_dexs: list[str] | None = None) -> FlakyInfo:
        seen["base_url"] = base_url
        seen["skip_ws"] = skip_ws
        seen["perp_dexs"] = list(perp_dexs or [])
        return FlakyInfo()

    monkeypatch.setattr("fund.data.hyperliquid_data.Info", fake_info)
    HyperliquidData(base_url="https://example.test")
    assert seen == {"base_url": "https://example.test", "skip_ws": True, "perp_dexs": ["", "xyz"]}


def test_fixture_quotes_parse_recorded_strings(fixture_data: MarketData) -> None:
    raw = json.loads((FIXTURES / "hl_meta_ctxs_core.json").read_text())
    meta, contexts = raw
    btc = next(ctx for asset, ctx in zip(meta["universe"], contexts, strict=True) if asset["name"] == "BTC")
    quote = fixture_data.quotes(["BTC", "xyz:JPY"])["BTC"]
    assert quote.mark == float(btc["markPx"])
    assert quote.oracle == float(btc["oraclePx"])
    assert quote.prev_day_px == float(btc["prevDayPx"])
    assert quote.funding_hourly == float(btc["funding"])
    assert quote.open_interest == float(btc["openInterest"])
    assert quote.day_volume_usd == float(btc["dayNtlVlm"])
    metas = fixture_data.meta(["BTC", "xyz:JPY", "xyz:SP500"])
    assert metas["xyz:JPY"].only_isolated is True
    assert metas["xyz:SP500"].only_isolated is False
    assert metas["BTC"].only_isolated is False
    assert isinstance(metas["BTC"].sz_decimals, int)
    assert metas["BTC"].sz_decimals > 0


def test_fixture_candles_parse_strings_in_open_time_order(fixture_data: MarketData) -> None:
    raw = json.loads((FIXTURES / "hl_candles_BTC.json").read_text())
    ordered = sorted(raw, key=lambda row: int(row["t"]))
    candles = fixture_data.candles(
        "BTC",
        "1d",
        datetime(2020, 1, 1, tzinfo=UTC),
        datetime(2030, 1, 1, tzinfo=UTC),
    )
    assert len(candles) == len(raw)
    assert [candle.open_time for candle in candles] == sorted(candle.open_time for candle in candles)
    assert candles[0].open == float(ordered[0]["o"])
    assert candles[0].volume == float(ordered[0]["v"])
    assert candles[-1].close == float(ordered[-1]["c"])


def test_recorded_candles_reject_other_intervals(fixture_data: MarketData) -> None:
    with pytest.raises(ValueError, match="1d"):
        fixture_data.candles("BTC", "1h", START, END)


def test_build_snapshot_fetches_quotes_once_and_daily_candles() -> None:
    config = load_config()
    universe = [spec for spec in config.universe if spec.symbol in {"BTC", "xyz:AAPL"}]
    stub = StubData()
    news = NewsDigest(text="Quiet.", sources=["desk"], compiled_at=NOW)
    snapshot = build_snapshot(stub, universe, NOW, config.fund.cycle_hours, news)
    assert stub.quote_calls == 1
    assert [call[0] for call in stub.candle_calls] == ["xyz:AAPL", "BTC"]
    assert {call[1] for call in stub.candle_calls} == {"1d"}
    assert stub.candle_calls[0][2] == NOW - timedelta(days=120)
    assert stub.candle_calls[0][3] == NOW
    assert snapshot.cycle_id == "2026-10-14T16:00Z"
    assert snapshot.as_of == NOW
    assert snapshot.news == news
    assert [market.symbol for market in snapshot.markets] == ["xyz:AAPL", "BTC"]


def test_build_snapshot_raises_when_one_symbol_fails() -> None:
    universe = [spec for spec in load_config().universe if spec.symbol in {"BTC", "ETH"}]
    stub = StubData(fail="ETH")
    with pytest.raises(RuntimeError, match="ETH"):
        build_snapshot(stub, universe, NOW, 4)
    assert stub.quote_calls == 1
    assert [call[0] for call in stub.candle_calls] == ["BTC", "ETH"]


def test_build_snapshot_covers_recorded_universe(fixture_data: MarketData) -> None:
    recorded = Snapshot.model_validate_json((FIXTURES / "snapshot_real.json").read_text())
    config = load_config()
    snapshot = build_snapshot(fixture_data, list(config.universe), recorded.as_of, config.fund.cycle_hours)
    assert len(snapshot.markets) == 11
    assert [market.symbol for market in snapshot.markets] == [spec.symbol for spec in config.universe]
    assert snapshot.cycle_id == recorded.cycle_id
    assert snapshot.session == recorded.session
    for got, saved in zip(snapshot.markets, recorded.markets, strict=True):
        assert got.symbol == saved.symbol
        assert got.mark == pytest.approx(saved.mark)
        assert got.ret_24h == pytest.approx(saved.ret_24h)
        assert got.ret_7d == pytest.approx(saved.ret_7d)
        assert got.ret_30d == pytest.approx(saved.ret_30d)
        assert got.vol_30d_ann == pytest.approx(saved.vol_30d_ann)
        assert got.sma20_gap == pytest.approx(saved.sma20_gap)
        assert got.sma50_gap == pytest.approx(saved.sma50_gap)
        assert got.funding_ann == pytest.approx(saved.funding_ann)
        assert got.market_open is saved.market_open
        assert got.daily_closes == pytest.approx(saved.daily_closes)


@pytest.mark.live
def test_live_mainnet_covers_the_universe() -> None:
    data = HyperliquidData()
    symbols = [spec.symbol for spec in load_config().universe]
    quotes = data.quotes(symbols)
    metas = data.meta(symbols)
    assert set(quotes) == set(symbols)
    assert set(metas) == set(symbols)
    for symbol in symbols:
        assert quotes[symbol].mark > 0
        assert metas[symbol].max_leverage > 0
    now = datetime.now(UTC)
    candles = data.candles("BTC", "1d", now - timedelta(days=120), now)
    assert len(candles) >= 51
    assert [candle.open_time for candle in candles] == sorted(candle.open_time for candle in candles)
