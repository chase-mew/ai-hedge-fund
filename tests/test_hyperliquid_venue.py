"""Offline Hyperliquid venue tests with synthetic recorded responses."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

from fund.contracts import StopOrder
from fund.venues.hyperliquid import HyperliquidVenue
from fund.venues.sizing import ioc_limit_price, stop_limit_price

FIXTURES = Path(__file__).parent / "fixtures"
ADDRESS = "0xMAIN"
SYMBOLS = ["BTC", "ETH", "xyz:SP500", "xyz:CL"]
MAINNET_BLOCK = "Mainnet trading is disabled. Set FUND_ALLOW_MAINNET=yes to enable it."


def _load(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text())


class FakeInfo:
    def __init__(self) -> None:
        self.posts: list[tuple[str, dict[str, Any]]] = []
        self.user_calls: list[tuple[str, str]] = []
        self.metas = {"": _load("hlx_meta_core.json"), "xyz": _load("hlx_meta_xyz.json")}
        self.user_states = {"": _load("hlx_user_core.json"), "xyz": _load("hlx_user_xyz.json")}
        self.mids = _load("hlx_mids.json")
        self.orders = {"": _load("hlx_open_orders.json"), "xyz": []}
        self.fills = _load("hlx_fills.json")
        self.fill_start: int | None = None

    def post(self, path: str, payload: dict[str, Any]) -> Any:
        self.posts.append((path, payload))
        return self.metas[payload["dex"]]

    def user_state(self, address: str, dex: str = "") -> Any:
        self.user_calls.append((address, dex))
        return self.user_states[dex]

    def all_mids(self, dex: str = "") -> Any:
        return self.mids[dex]

    def frontend_open_orders(self, address: str, dex: str = "") -> Any:
        return self.orders[dex]

    def user_fills_by_time(self, address: str, start_time: int, *_args: object, **_kwargs: object) -> Any:
        self.fill_start = start_time
        return [fill for fill in self.fills if fill["time"] >= start_time]


class FakeExchange:
    def __init__(self) -> None:
        self.orders: list[dict[str, Any]] = []
        self.closes: list[dict[str, Any]] = []
        self.cancels: list[tuple[str, int]] = []
        self.leverages: list[tuple[int, str, bool]] = []
        self.order_result: Any = _load("hlx_order_filled.json")
        self.close_result: Any = None
        self.leverage_result: Any = {"status": "ok", "response": {"type": "default"}}

    def order(
        self,
        name: str,
        is_buy: bool,
        sz: float,
        limit_px: float,
        order_type: dict[str, Any],
        reduce_only: bool = False,
    ) -> Any:
        self.orders.append(
            {
                "name": name,
                "is_buy": is_buy,
                "sz": sz,
                "limit_px": limit_px,
                "order_type": order_type,
                "reduce_only": reduce_only,
            }
        )
        return self.order_result

    def market_close(
        self,
        coin: str,
        sz: float | None = None,
        px: float | None = None,
        slippage: float = 0.05,
    ) -> Any:
        self.closes.append({"coin": coin, "slippage": slippage})
        return self.close_result

    def cancel(self, name: str, oid: int) -> Any:
        self.cancels.append((name, oid))
        return {"status": "ok", "response": {"type": "cancel"}}

    def update_leverage(self, leverage: int, name: str, is_cross: bool = True) -> Any:
        self.leverages.append((leverage, name, is_cross))
        return self.leverage_result


def _venue(
    info: FakeInfo | None = None,
    exchange: FakeExchange | None = None,
    *,
    network: str = "testnet",
    allow_mainnet: bool = False,
    slippage: float = 0.05,
) -> tuple[HyperliquidVenue, FakeInfo, FakeExchange]:
    info = info or FakeInfo()
    exchange = exchange or FakeExchange()
    venue = HyperliquidVenue(
        cast(Any, network),
        ADDRESS,
        "unused",
        SYMBOLS,
        slippage,
        allow_mainnet,
        info=cast(Info, info),
        exchange=cast(Exchange, exchange),
    )
    return venue, info, exchange


def _explode(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("real client constructed")


def test_mainnet_without_permission_raises_before_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fund.venues.hyperliquid.Info", _explode)
    monkeypatch.setattr("fund.venues.hyperliquid.Exchange", _explode)
    monkeypatch.setattr("fund.venues.hyperliquid.Account.from_key", _explode)
    with pytest.raises(PermissionError) as caught:
        HyperliquidVenue("mainnet", ADDRESS, "unused", SYMBOLS, 0.01, False)
    assert str(caught.value) == MAINNET_BLOCK


def test_injected_clients_are_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fund.venues.hyperliquid.Info", _explode)
    monkeypatch.setattr("fund.venues.hyperliquid.Exchange", _explode)
    monkeypatch.setattr("fund.venues.hyperliquid.Account.from_key", _explode)
    info, exchange = FakeInfo(), FakeExchange()
    venue, _, _ = _venue(info, exchange)
    assert venue.info is info
    assert venue.exchange is exchange


def test_real_clients_receive_both_dexs(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class InfoStub:
        def __init__(self, base_url: str, skip_ws: bool = False, perp_dexs: list[str] | None = None) -> None:
            seen["info"] = (base_url, skip_ws, perp_dexs)

    class ExchangeStub:
        def __init__(
            self,
            wallet: object,
            base_url: str,
            account_address: str | None = None,
            perp_dexs: list[str] | None = None,
        ) -> None:
            seen["exchange"] = (base_url, account_address, perp_dexs, wallet)

    monkeypatch.setattr("fund.venues.hyperliquid.Info", InfoStub)
    monkeypatch.setattr("fund.venues.hyperliquid.Exchange", ExchangeStub)
    monkeypatch.setattr("fund.venues.hyperliquid.Account.from_key", lambda key: {"agent": key})
    HyperliquidVenue("testnet", ADDRESS, "agentkey", SYMBOLS, 0.05, False)
    assert seen["info"] == (constants.TESTNET_API_URL, True, ["", "xyz"])
    assert seen["exchange"][:3] == (constants.TESTNET_API_URL, ADDRESS, ["", "xyz"])
    HyperliquidVenue("mainnet", ADDRESS, "agentkey", SYMBOLS, 0.01, True)
    assert seen["info"][0] == constants.MAINNET_API_URL
    assert seen["exchange"][0] == constants.MAINNET_API_URL
    assert seen["info"][2] == ["", "xyz"]
    assert seen["exchange"][2] == ["", "xyz"]


def test_market_meta_and_marks_post_both_dexs() -> None:
    venue, info, _ = _venue()
    meta = venue.market_meta()
    marks = venue.marks()
    assert meta["BTC"].sz_decimals == 5
    assert meta["BTC"].max_leverage == 40
    assert meta["BTC"].only_isolated is False
    assert meta["xyz:SP500"].only_isolated is True
    assert marks == {"BTC": 80000.0, "ETH": 3000.0, "xyz:SP500": 5610.0, "xyz:CL": 70.5}
    assert info.posts == [
        ("/info", {"type": "metaAndAssetCtxs", "dex": ""}),
        ("/info", {"type": "metaAndAssetCtxs", "dex": "xyz"}),
        ("/info", {"type": "metaAndAssetCtxs", "dex": ""}),
        ("/info", {"type": "metaAndAssetCtxs", "dex": "xyz"}),
    ]


def test_account_equity_does_not_double_count() -> None:
    venue, info, _ = _venue()
    assert venue.account().equity == pytest.approx(1000.5)
    assert info.user_calls == [(ADDRESS, ""), (ADDRESS, "xyz")]
    info.user_states["xyz"]["marginSummary"]["accountValue"] = "250"
    assert venue.account().equity == pytest.approx(1000.5)
    del info.user_states[""]["marginSummary"]["accountValue"]
    assert venue.account().equity == pytest.approx(250)


def test_account_equity_requires_an_account_value() -> None:
    venue, info, _ = _venue()
    for state in info.user_states.values():
        state["marginSummary"] = {}
    with pytest.raises(RuntimeError, match=r"marginSummary\.accountValue"):
        venue.account()


def test_account_reads_positions_from_both_dexs() -> None:
    venue, _, _ = _venue()
    positions = venue.account().positions
    assert set(positions) == {"BTC", "xyz:SP500"}
    assert positions["BTC"].size == pytest.approx(0.01)
    assert positions["BTC"].entry_px == pytest.approx(79000)
    assert positions["BTC"].mark == pytest.approx(80000)
    assert positions["xyz:SP500"].size == pytest.approx(-0.2)
    assert positions["xyz:SP500"].entry_px == pytest.approx(5610)
    assert positions["xyz:SP500"].mark == pytest.approx(5610)


def test_market_order_filled_is_an_ioc_limit() -> None:
    venue, _, exchange = _venue()
    outcome = venue.market_order("BTC", "buy", 0.01, True)
    assert outcome.status == "filled"
    assert outcome.filled_size == pytest.approx(0.01)
    assert outcome.avg_px == pytest.approx(84000)
    assert outcome.order_id == "4242"
    assert outcome.fee_usd == 0.0
    sent = exchange.orders[0]
    assert sent["is_buy"] is True
    assert sent["reduce_only"] is True
    assert sent["sz"] == pytest.approx(0.01)
    assert sent["limit_px"] == pytest.approx(ioc_limit_price(80000, "buy", 0.05, 5))
    assert sent["order_type"] == {"limit": {"tif": "Ioc"}}


def test_market_order_partial_when_fill_is_below_requested() -> None:
    venue, _, exchange = _venue()
    exchange.order_result = _load("hlx_order_partial.json")
    outcome = venue.market_order("BTC", "sell", 0.01, False)
    assert outcome.status == "partial"
    assert outcome.filled_size == pytest.approx(0.004)
    assert outcome.requested_size == pytest.approx(0.01)
    assert outcome.order_id == "4243"
    assert exchange.orders[0]["is_buy"] is False
    assert exchange.orders[0]["limit_px"] == pytest.approx(ioc_limit_price(80000, "sell", 0.05, 5))


def test_market_order_no_liquidity_is_unfilled() -> None:
    venue, _, exchange = _venue()
    exchange.order_result = _load("hlx_order_no_liquidity.json")
    outcome = venue.market_order("BTC", "buy", 0.01, False)
    assert outcome.status == "unfilled"
    assert outcome.filled_size == 0.0
    assert "could not immediately match" in outcome.message
    exchange.order_result = {
        "status": "err",
        "response": "Order could not immediately match against any resting orders.",
    }
    top = venue.market_order("ETH", "buy", 0.1, False)
    assert top.status == "unfilled"


def test_market_order_other_error() -> None:
    venue, _, exchange = _venue()
    exchange.order_result = _load("hlx_order_error.json")
    outcome = venue.market_order("BTC", "buy", 0.01, False)
    assert outcome.status == "error"
    assert outcome.order_id is None
    assert "minimum value" in outcome.message


def test_resting_ioc_is_an_error() -> None:
    venue, _, exchange = _venue()
    exchange.order_result = {"status": "ok", "response": {"data": {"statuses": [{"resting": {"oid": 77}}]}}}
    outcome = venue.market_order("BTC", "buy", 0.01, False)
    assert outcome.status == "error"
    assert outcome.order_id == "77"
    assert outcome.message == "IOC order was not filled."


def test_close_position_none_is_skipped() -> None:
    venue, _, exchange = _venue()
    outcome = venue.close_position("ETH")
    assert outcome.status == "skipped"
    assert outcome.message == "no position"
    assert outcome.side == "sell"
    assert exchange.closes == [{"coin": "ETH", "slippage": 0.05}]


def test_close_position_parses_a_fill() -> None:
    venue, _, exchange = _venue()
    exchange.close_result = _load("hlx_order_filled.json")
    outcome = venue.close_position("BTC")
    assert outcome.status == "filled"
    assert outcome.side == "sell"
    assert outcome.requested_size == pytest.approx(0.01)
    assert outcome.order_id == "4242"


def test_close_position_buys_to_cover_a_short() -> None:
    venue, _, exchange = _venue()
    filled = _load("hlx_order_filled.json")
    filled["response"]["data"]["statuses"][0]["filled"]["totalSz"] = "0.2"
    exchange.close_result = filled
    outcome = venue.close_position("xyz:SP500")
    assert outcome.status == "filled"
    assert outcome.side == "buy"
    assert outcome.requested_size == pytest.approx(0.2)
    assert exchange.closes[0]["slippage"] == pytest.approx(0.05)


def test_ensure_leverage_caches_per_symbol() -> None:
    venue, _, exchange = _venue()
    venue.ensure_leverage("BTC", 5, True)
    venue.ensure_leverage("BTC", 5, True)
    venue.ensure_leverage("xyz:SP500", 3, False)
    assert exchange.leverages == [(5, "BTC", True), (3, "xyz:SP500", False)]


def test_ensure_leverage_does_not_cache_a_rejection() -> None:
    venue, _, exchange = _venue()
    exchange.leverage_result = {"status": "err", "response": "too high"}
    with pytest.raises(RuntimeError, match="too high"):
        venue.ensure_leverage("BTC", 50, True)
    with pytest.raises(RuntimeError, match="too high"):
        venue.ensure_leverage("BTC", 50, True)
    assert exchange.leverages == [(50, "BTC", True), (50, "BTC", True)]


def test_sync_stop_keeps_a_matching_stop() -> None:
    venue, _, exchange = _venue()
    venue.sync_stop("BTC", StopOrder(symbol="BTC", side="sell", size=0.01, trigger_px=76000))
    assert exchange.cancels == []
    assert exchange.orders == []


def test_sync_stop_replaces_a_mismatch() -> None:
    venue, info, exchange = _venue()
    info.orders[""][0]["triggerPx"] = "70000"
    exchange.order_result = {"status": "ok", "response": {"data": {"statuses": [{"resting": {"oid": 900}}]}}}
    venue.sync_stop("BTC", StopOrder(symbol="BTC", side="sell", size=0.01, trigger_px=76000))
    assert exchange.cancels == [("BTC", 555)]
    sent = exchange.orders[0]
    assert sent["reduce_only"] is True
    assert sent["is_buy"] is False
    assert sent["sz"] == pytest.approx(0.01)
    assert sent["order_type"]["trigger"]["triggerPx"] == pytest.approx(76000)
    assert sent["order_type"]["trigger"]["isMarket"] is True
    assert sent["order_type"]["trigger"]["tpsl"] == "sl"
    assert sent["limit_px"] == pytest.approx(stop_limit_price(76000, "sell", 5))


def test_sync_stop_none_cancels_without_placing() -> None:
    venue, info, exchange = _venue()
    info.orders[""].append(
        {
            "coin": "BTC",
            "oid": 557,
            "side": "A",
            "sz": "0.01",
            "isTrigger": True,
            "reduceOnly": True,
            "triggerPx": "75000",
        }
    )
    venue.sync_stop("BTC", None)
    assert exchange.cancels == [("BTC", 555), ("BTC", 557)]
    assert exchange.orders == []


def test_sync_stop_raises_when_placement_is_rejected() -> None:
    venue, info, exchange = _venue()
    info.orders[""] = []
    exchange.order_result = _load("hlx_order_error.json")
    with pytest.raises(RuntimeError, match="minimum value"):
        venue.sync_stop("BTC", StopOrder(symbol="BTC", side="sell", size=0.01, trigger_px=76000))


def test_fees_since_sums_by_order_id() -> None:
    venue, info, _ = _venue()
    start = datetime(2026, 1, 1, tzinfo=UTC)
    fees = venue.fees_since(start)
    assert info.fill_start == int(start.timestamp() * 1000)
    assert fees["101"] == pytest.approx(0.035)
    assert fees["303"] == pytest.approx(0.5)
    assert "202" not in fees


def test_fees_since_rejects_a_naive_datetime() -> None:
    venue, _, _ = _venue()
    naive = datetime(2026, 1, 1)  # noqa: DTZ001  naive input is the behaviour under test
    with pytest.raises(ValueError, match="timezone aware"):
        venue.fees_since(naive)


def test_module_does_not_call_schedule_cancel() -> None:
    import fund.venues.hyperliquid as venue_module

    source = Path(venue_module.__file__ or "").read_text()
    assert "schedule_cancel" not in source


@pytest.mark.live
def test_live_testnet_account_read() -> None:
    """Read the testnet account without placing an order."""
    from fund.config import load_secrets

    secrets = load_secrets()
    address = secrets.hl_account_address
    if not address:
        raise RuntimeError("HL_ACCOUNT_ADDRESS is not set. Add the main wallet address to .env.")
    info = Info(constants.TESTNET_API_URL, skip_ws=True, perp_dexs=["", "xyz"])
    venue = HyperliquidVenue(
        "testnet",
        address,
        "unused",
        SYMBOLS,
        0.05,
        False,
        info=info,
        exchange=cast(Exchange, FakeExchange()),
    )
    snapshot = venue.account()
    assert snapshot.equity >= 0
