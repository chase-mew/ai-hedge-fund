"""Hyperliquid execution venue for the core dex and xyz."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Literal, cast

from eth_account import Account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

from fund.contracts import AccountSnapshot, MarketMeta, OrderOutcome, Position, Side, StopOrder
from fund.venues.sizing import ioc_limit_price, round_price, round_size, stop_limit_price

log = logging.getLogger(__name__)

_DEXS = ("", "xyz")
_NO_LIQUIDITY = "could not immediately match against any resting orders"
Status = Literal["filled", "partial", "unfilled", "error", "skipped"]


class HyperliquidVenue:
    """Live Hyperliquid venue for the core dex and xyz."""

    name: str = "hyperliquid"

    def __init__(
        self,
        network: Literal["testnet", "mainnet"],
        account_address: str,
        agent_private_key: str,
        symbols: list[str],
        slippage: float,
        allow_mainnet: bool,
        info: Info | None = None,
        exchange: Exchange | None = None,
    ) -> None:
        if network == "mainnet" and not allow_mainnet:
            raise PermissionError("Mainnet trading is disabled. Set FUND_ALLOW_MAINNET=yes to enable it.")
        self.network = network
        self.account_address = account_address
        self.symbols = list(symbols)
        self.slippage = slippage
        self._leverage_set: set[str] = set()
        url = constants.MAINNET_API_URL if network == "mainnet" else constants.TESTNET_API_URL
        if info is None:
            info = Info(url, skip_ws=True, perp_dexs=["", "xyz"])
        if exchange is None:
            wallet = Account.from_key(agent_private_key)
            exchange = Exchange(wallet, url, account_address=account_address, perp_dexs=["", "xyz"])
        self.info = info
        self.exchange = exchange
        log.info("hyperliquid venue %s for %s symbols", network, len(self.symbols))

    def market_meta(self) -> dict[str, MarketMeta]:
        """Load perp metadata for the core dex and xyz."""
        found: dict[str, MarketMeta] = {}
        for meta, _ctxs in self._meta_and_ctxs():
            for asset in meta["universe"]:
                symbol = str(asset["name"])
                found[symbol] = MarketMeta(
                    symbol=symbol,
                    sz_decimals=int(asset["szDecimals"]),
                    max_leverage=int(asset["maxLeverage"]),
                    only_isolated=bool(asset.get("onlyIsolated", False)),
                )
        return found

    def marks(self) -> dict[str, float]:
        """Load mark prices for the core dex and xyz."""
        prices: dict[str, float] = {}
        for meta, ctxs in self._meta_and_ctxs():
            for asset, ctx in zip(meta["universe"], ctxs, strict=True):
                prices[str(asset["name"])] = float(ctx["markPx"])
        return prices

    def account(self) -> AccountSnapshot:
        """Equity is the larger marginSummary.accountValue, and the two dex values are not added."""
        # Phase 5 replaces this after scripts/inspect_account.py.
        states = [cast(dict[str, Any], self.info.user_state(self.account_address, dex)) for dex in _DEXS]
        equity = _equity(states)
        prices = self.marks()
        positions: dict[str, Position] = {}
        for state in states:
            for wrapper in state["assetPositions"]:
                raw = wrapper["position"]
                size = float(raw["szi"])
                if size == 0.0:
                    continue
                symbol = str(raw["coin"])
                entry_raw = raw.get("entryPx")
                entry = float(entry_raw) if entry_raw not in (None, "") else prices[symbol]
                positions[symbol] = Position(symbol=symbol, size=size, entry_px=entry, mark=prices[symbol])
        return AccountSnapshot(equity=equity, positions=positions)

    def market_order(self, symbol: str, side: Side, size: float, reduce_only: bool) -> OrderOutcome:
        """Place an IOC limit and parse the status. The fee stays zero until fees_since."""
        meta = self.market_meta()[symbol]
        mid = self._mids()[symbol]
        submitted = round_size(size, meta.sz_decimals)
        limit_px = ioc_limit_price(mid, side, self.slippage, meta.sz_decimals)
        result = self.exchange.order(
            symbol,
            side == "buy",
            submitted,
            limit_px,
            {"limit": {"tif": "Ioc"}},
            reduce_only=reduce_only,
        )
        outcome = _outcome(symbol, side, submitted, result)
        log.info(
            "order %s %s size %s limit %s status %s id %s",
            symbol,
            side,
            submitted,
            limit_px,
            outcome.status,
            outcome.order_id,
        )
        return outcome

    def close_position(self, symbol: str) -> OrderOutcome:
        """Close the full position. A None response means there was no position."""
        position = self.account().positions.get(symbol)
        side: Side = "sell" if position is None or position.size > 0 else "buy"
        requested = 0.0 if position is None else abs(position.size)
        result = self.exchange.market_close(symbol, slippage=self.slippage)
        if result is None:
            return _result(symbol, side, requested, status="skipped", message="no position")
        return _outcome(symbol, side, requested, result)

    def ensure_leverage(self, symbol: str, leverage: int, cross: bool) -> None:
        """Set leverage once per symbol for this process."""
        if symbol in self._leverage_set:
            return
        result = self.exchange.update_leverage(leverage, symbol, is_cross=cross)
        rejection = _rejection(result)
        if rejection is not None:
            raise RuntimeError(f"Could not set leverage for {symbol}: {rejection}")
        self._leverage_set.add(symbol)
        log.info("leverage %s x%s cross %s", symbol, leverage, cross)

    def sync_stop(self, symbol: str, stop: StopOrder | None) -> None:
        """Keep one matching trigger, otherwise cancel the stops and place the new one."""
        dex = _dex(symbol)
        listed = cast(list[dict[str, Any]], self.info.frontend_open_orders(self.account_address, dex))
        stops = [order for order in listed if _is_stop(order, symbol)]
        if stop is not None and len(stops) == 1 and _same_stop(stops[0], stop):
            log.info("keeping stop %s oid %s", symbol, stops[0].get("oid"))
            return
        for order in stops:
            self.exchange.cancel(symbol, int(order["oid"]))
        if stop is None:
            return
        self._place_stop(stop)

    def fees_since(self, start: datetime) -> dict[str, float]:
        """Sum fill fees in USD by order id since start."""
        if start.tzinfo is None or start.utcoffset() is None:
            raise ValueError("start must be timezone aware")
        start_ms = int(start.timestamp() * 1000)
        fills = cast(list[dict[str, Any]], self.info.user_fills_by_time(self.account_address, start_ms))
        fees: dict[str, float] = {}
        for fill in fills:
            oid = str(fill["oid"])
            fees[oid] = fees.get(oid, 0.0) + float(fill["fee"])
        return fees

    def _meta_and_ctxs(self) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
        rows: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        for dex in _DEXS:
            payload = cast(list[Any], self.info.post("/info", {"type": "metaAndAssetCtxs", "dex": dex}))
            meta = cast(dict[str, Any], payload[0])
            ctxs = cast(list[dict[str, Any]], payload[1])
            rows.append((meta, ctxs))
        return rows

    def _mids(self) -> dict[str, float]:
        prices: dict[str, float] = {}
        for dex in _DEXS:
            raw = cast(dict[str, Any], self.info.all_mids(dex))
            for symbol, px in raw.items():
                prices[str(symbol)] = float(px)
        return prices

    def _place_stop(self, stop: StopOrder) -> None:
        meta = self.market_meta()[stop.symbol]
        size = round_size(stop.size, meta.sz_decimals)
        trigger = round_price(stop.trigger_px, meta.sz_decimals)
        limit_px = stop_limit_price(trigger, stop.side, meta.sz_decimals)
        result = self.exchange.order(
            stop.symbol,
            stop.side == "buy",
            size,
            limit_px,
            {"trigger": {"triggerPx": trigger, "isMarket": True, "tpsl": "sl"}},
            reduce_only=True,
        )
        rejection = _rejection(result)
        if rejection is not None:
            raise RuntimeError(f"Stop for {stop.symbol} was rejected: {rejection}")
        log.info("placed stop %s %s size %s trigger %s", stop.symbol, stop.side, size, trigger)


def _dex(symbol: str) -> str:
    return symbol.split(":", 1)[0] if ":" in symbol else ""


def _equity(states: list[dict[str, Any]]) -> float:
    values: list[float] = []
    for state in states:
        summary = state.get("marginSummary")
        if not isinstance(summary, dict):
            continue
        raw = summary.get("accountValue")
        if raw in (None, ""):
            continue
        values.append(float(raw))
    if not values:
        raise RuntimeError(
            "user_state did not include marginSummary.accountValue. Run scripts/inspect_account.py."
        )
    return max(values)


def _is_stop(order: dict[str, Any], symbol: str) -> bool:
    return bool(order.get("coin") == symbol and order.get("isTrigger") and order.get("reduceOnly"))


def _same_stop(order: dict[str, Any], stop: StopOrder) -> bool:
    if _book_side(order.get("side")) != stop.side:
        return False
    if not _same_size(float(order["sz"]), stop.size):
        return False
    return _trigger_close(float(order["triggerPx"]), stop.trigger_px)


def _book_side(raw: object) -> Side | None:
    if raw == "B":
        return "buy"
    if raw == "A":
        return "sell"
    return None


def _same_size(left: float, right: float) -> bool:
    return abs(left - right) <= max(1e-9, abs(right) * 1e-6)


def _trigger_close(actual: float, desired: float) -> bool:
    if desired == 0:
        return actual == 0
    return abs(actual - desired) / abs(desired) <= 0.001


def _no_liquidity(message: str) -> bool:
    return _NO_LIQUIDITY in message.lower()


def _text(value: object) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


def _rejection(result: object) -> str | None:
    if not isinstance(result, dict):
        return "Unexpected exchange response."
    if result.get("status") != "ok":
        return _text(result.get("response")) or "Exchange rejected the action."
    response = result.get("response")
    if not isinstance(response, dict):
        return None
    data = response.get("data")
    if not isinstance(data, dict):
        return None
    statuses = data.get("statuses")
    if not isinstance(statuses, list) or not statuses:
        return None
    first = statuses[0]
    if isinstance(first, dict) and "error" in first:
        return _text(first.get("error")) or "Exchange rejected the action."
    return None


def _outcome(symbol: str, side: Side, requested: float, result: object) -> OrderOutcome:
    if not isinstance(result, dict) or result.get("status") != "ok":
        message = _text(result.get("response")) if isinstance(result, dict) else "Unexpected order response."
        status: Status = "unfilled" if _no_liquidity(message) else "error"
        return _result(symbol, side, requested, status=status, message=message)
    item = _first_status(result)
    if item is None:
        return _result(symbol, side, requested, status="error", message="Order response had no statuses.")
    if "error" in item:
        message = _text(item.get("error"))
        status = "unfilled" if _no_liquidity(message) else "error"
        return _result(symbol, side, requested, status=status, message=message)
    filled = item.get("filled")
    if isinstance(filled, dict):
        total = float(filled["totalSz"])
        kind: Status = "partial" if total + 1e-9 < requested else "filled"
        return _result(
            symbol,
            side,
            requested,
            status=kind,
            filled_size=total,
            avg_px=float(filled["avgPx"]),
            order_id=str(filled["oid"]),
        )
    resting = item.get("resting")
    order_id = None
    if isinstance(resting, dict) and "oid" in resting:
        order_id = str(resting["oid"])
    return _result(
        symbol,
        side,
        requested,
        status="error",
        message="IOC order was not filled.",
        order_id=order_id,
    )


def _first_status(result: dict[str, Any]) -> dict[str, Any] | None:
    response = result.get("response")
    if not isinstance(response, dict):
        return None
    data = response.get("data")
    if not isinstance(data, dict):
        return None
    statuses = data.get("statuses")
    if not isinstance(statuses, list) or not statuses:
        return None
    first = statuses[0]
    if not isinstance(first, dict):
        return None
    return first


def _result(
    symbol: str,
    side: Side,
    requested: float,
    *,
    status: Status,
    filled_size: float = 0.0,
    avg_px: float | None = None,
    order_id: str | None = None,
    message: str = "",
) -> OrderOutcome:
    return OrderOutcome(
        symbol=symbol,
        side=side,
        requested_size=requested,
        filled_size=filled_size,
        avg_px=avg_px,
        fee_usd=0.0,
        status=status,
        order_id=order_id,
        message=message,
    )
