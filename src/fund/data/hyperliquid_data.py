# pyright: strict
"""Hyperliquid market data provider."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, cast

import requests
from hyperliquid.info import Info  # type: ignore[reportMissingTypeStubs]
from hyperliquid.utils import constants  # type: ignore[reportMissingTypeStubs]
from hyperliquid.utils.error import ClientError  # type: ignore[reportMissingTypeStubs]

from fund.contracts import Candle, MarketMeta, MarketQuote

_LOG = logging.getLogger(__name__)
_RETRY_SLEEPS_S = (1, 2, 4)


class _InfoApi(Protocol):
    def post(self, url_path: str, payload: Any = None) -> Any: ...

    def candles_snapshot(self, name: str, interval: str, startTime: int, endTime: int) -> Any: ...


def _call[T](operation: Callable[[], T]) -> T:
    """Run one request, then retry up to 3 times after sleeps of 1, 2, and 4 seconds."""
    attempt = 0
    while True:
        try:
            return operation()
        except requests.RequestException as exc:
            error: BaseException = exc
        except ClientError as exc:
            if cast(int, exc.status_code) != 429:
                raise
            error = exc
        if attempt == len(_RETRY_SLEEPS_S):
            raise error
        delay = _RETRY_SLEEPS_S[attempt]
        attempt += 1
        _LOG.warning(
            "Hyperliquid request failed (%s); retrying in %s seconds",
            type(error).__name__,
            delay,
        )
        time.sleep(delay)


def _as_float(value: Any) -> float:
    return float(cast(str | int | float, value))


def _as_int(value: Any) -> int:
    return int(cast(str | int, value))


def _require_aware(moment: datetime, label: str) -> None:
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError(f"{label} must be timezone aware")


def meta_fixture_path(directory: Path, dex: str) -> Path:
    """Return the recorded metaAndAssetCtxs path for one dex."""
    name = "hl_meta_ctxs_xyz.json" if dex == "xyz" else "hl_meta_ctxs_core.json"
    return directory / name


def candle_fixture_path(directory: Path, symbol: str) -> Path:
    """Return the recorded daily candle path for one symbol."""
    slug = symbol.replace("/", "__").replace(":", "__")
    return directory / f"hl_candles_{slug}.json"


class _RecordedInfo:
    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def post(self, url_path: str, payload: Any = None) -> Any:
        del url_path
        body = cast(dict[str, Any], payload)
        dex = cast(str, body.get("dex", ""))
        return json.loads(meta_fixture_path(self._directory, dex).read_text())

    def candles_snapshot(self, name: str, interval: str, startTime: int, endTime: int) -> Any:
        del startTime, endTime
        if interval != "1d":
            raise ValueError(f"Recorded fixtures only include 1d candles for {name}.")
        return json.loads(candle_fixture_path(self._directory, name).read_text())


def recorded_client(directory: Path) -> HyperliquidData:
    """Read recorded meta and candle JSON through the Hyperliquid parser."""
    return HyperliquidData(info=cast(Info, _RecordedInfo(directory)))


class HyperliquidData:
    """Market data from the Hyperliquid info API."""

    def __init__(self, base_url: str = constants.MAINNET_API_URL, info: Info | None = None) -> None:
        """Open a mainnet info client, or use an injected one."""
        if info is None:
            info = _call(lambda: Info(base_url, skip_ws=True, perp_dexs=["", "xyz"]))
        self._info = cast(_InfoApi, info)

    def quotes(self, symbols: list[str]) -> dict[str, MarketQuote]:
        """Return mark, oracle, funding, open interest, and day volume for each symbol."""
        rows = self._rows()
        quotes: dict[str, MarketQuote] = {}
        for symbol in symbols:
            ctx = _context(rows, symbol)
            quotes[symbol] = MarketQuote(
                symbol=symbol,
                mark=_as_float(ctx["markPx"]),
                oracle=_as_float(ctx["oraclePx"]),
                prev_day_px=_as_float(ctx["prevDayPx"]),
                funding_hourly=_as_float(ctx["funding"]),
                open_interest=_as_float(ctx["openInterest"]),
                day_volume_usd=_as_float(ctx["dayNtlVlm"]),
            )
        return quotes

    def meta(self, symbols: list[str]) -> dict[str, MarketMeta]:
        """Return size decimals, max leverage, and the isolated only flag for each symbol."""
        rows = self._rows()
        metas: dict[str, MarketMeta] = {}
        for symbol in symbols:
            asset = _asset(rows, symbol)
            metas[symbol] = MarketMeta(
                symbol=symbol,
                sz_decimals=_as_int(asset["szDecimals"]),
                max_leverage=_as_int(asset["maxLeverage"]),
                only_isolated=asset.get("onlyIsolated") is True,
            )
        return metas

    def candles(self, symbol: str, interval: str, start: datetime, end: datetime) -> list[Candle]:
        """Return candles for one symbol sorted by open time."""
        _require_aware(start, "start")
        _require_aware(end, "end")
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        raw = _call(lambda: self._info.candles_snapshot(symbol, interval, start_ms, end_ms))
        parsed = [_candle(cast(dict[str, Any], row)) for row in cast(list[Any], raw)]
        parsed.sort(key=lambda candle: candle.open_time)
        return parsed

    def write_raw(self, directory: Path, symbols: list[str], now: datetime) -> None:
        """Write meta and daily candle JSON for the given symbols."""
        directory.mkdir(parents=True, exist_ok=True)
        for dex in ("", "xyz"):
            payload = _call(
                lambda dex=dex: self._info.post("/info", {"type": "metaAndAssetCtxs", "dex": dex})
            )
            meta_fixture_path(directory, dex).write_text(json.dumps(payload, indent=2) + "\n")
        start_ms = int((now - timedelta(days=120)).timestamp() * 1000)
        end_ms = int(now.timestamp() * 1000)
        for symbol in symbols:
            rows = _call(lambda symbol=symbol: self._info.candles_snapshot(symbol, "1d", start_ms, end_ms))
            candle_fixture_path(directory, symbol).write_text(json.dumps(rows, indent=2) + "\n")

    def _rows(self) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
        found: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        for dex in ("", "xyz"):
            payload = _call(
                lambda dex=dex: self._info.post("/info", {"type": "metaAndAssetCtxs", "dex": dex})
            )
            meta = cast(dict[str, Any], payload[0])
            contexts = cast(list[dict[str, Any]], payload[1])
            universe = cast(list[dict[str, Any]], meta["universe"])
            for asset, ctx in zip(universe, contexts, strict=True):
                found[cast(str, asset["name"])] = (asset, ctx)
        return found


def _asset(rows: dict[str, tuple[dict[str, Any], dict[str, Any]]], symbol: str) -> dict[str, Any]:
    try:
        return rows[symbol][0]
    except KeyError:
        raise KeyError(symbol) from None


def _context(rows: dict[str, tuple[dict[str, Any], dict[str, Any]]], symbol: str) -> dict[str, Any]:
    try:
        return rows[symbol][1]
    except KeyError:
        raise KeyError(symbol) from None


def _candle(row: dict[str, Any]) -> Candle:
    return Candle(
        open_time=datetime.fromtimestamp(int(cast(int | str, row["t"])) / 1000, tz=UTC),
        open=_as_float(row["o"]),
        high=_as_float(row["h"]),
        low=_as_float(row["l"]),
        close=_as_float(row["c"]),
        volume=_as_float(row["v"]),
    )
