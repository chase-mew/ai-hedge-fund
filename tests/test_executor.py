"""Order planning, fail closed execution, and stop placement."""

from datetime import UTC, datetime

import pytest

from fund.config import ExecutionSettings, load_config
from fund.contracts import (
    AccountSnapshot,
    GatedTarget,
    GateResult,
    GateStatus,
    MarketMeta,
    OrderOutcome,
    PaperPosition,
    Position,
    Side,
    StopOrder,
)
from fund.execution.executor import desired_stops, execute, plan_orders
from fund.venues.paper import PaperVenue, new_paper_account

NOW = datetime(2026, 10, 14, 16, 0, tzinfo=UTC)
PAPER = load_config().paper


def test_plans_put_reduces_before_increases() -> None:
    account = _account(
        100_000,
        [
            _position("BBB", 10, 100),
            _position("CCC", 10, 100),
            _position("DDD", 50, 100),
        ],
    )
    gate = _gate(
        [
            _target("AAA", 0.1, status="clipped"),
            _target("BBB", 0),
            _target("CCC", -0.1),
            _target("DDD", 0.02),
            _target("EEE", 0.2, status="rejected"),
        ]
    )
    marks = {symbol: 100.0 for symbol in ("AAA", "BBB", "CCC", "DDD", "EEE")}
    meta = {symbol: _meta(symbol, 2) for symbol in marks}
    plans = plan_orders(gate, account, marks, meta, 11)
    assert [plan.reason for plan in plans] == ["close", "flip_close", "reduce", "increase", "flip_open"]
    assert [plan.symbol for plan in plans] == ["BBB", "CCC", "DDD", "AAA", "CCC"]
    assert plans[0].close_all is True
    assert plans[0].reduce_only is True
    assert plans[0].side == "sell"
    assert plans[0].size == pytest.approx(10)
    assert plans[1].close_all is True
    assert plans[2].reduce_only is True
    assert plans[2].side == "sell"
    assert plans[2].size == pytest.approx(30)
    assert plans[3].reduce_only is False
    assert plans[3].side == "buy"
    assert plans[3].size == pytest.approx(100)
    assert plans[4].side == "sell"
    assert plans[4].size == pytest.approx(100)
    assert "EEE" not in {plan.symbol for plan in plans}


def test_plan_rounds_size_down() -> None:
    account = _account(10_000, [])
    gate = _gate([_target("BTC", 0.01239)])
    plans = plan_orders(gate, account, {"BTC": 100}, {"BTC": _meta("BTC", 2)}, 11)
    assert len(plans) == 1
    assert plans[0].reason == "increase"
    assert plans[0].size == pytest.approx(1.23)
    assert plans[0].side == "buy"


def test_plan_skips_notional_below_the_minimum() -> None:
    account = _account(10_000, [])
    tiny = plan_orders(_gate([_target("BTC", 0.001)]), account, {"BTC": 100}, {"BTC": _meta("BTC", 2)}, 11)
    assert tiny == []
    exact = plan_orders(_gate([_target("ETH", 0.0011)]), account, {"ETH": 100}, {"ETH": _meta("ETH", 2)}, 11)
    assert len(exact) == 1
    assert exact[0].size == pytest.approx(0.11)


def test_flip_open_is_skipped_when_the_new_order_is_below_the_minimum() -> None:
    account = _account(10_000, [_position("BTC", 1, 100)])
    plans = plan_orders(_gate([_target("BTC", -0.0005)]), account, {"BTC": 100}, {"BTC": _meta("BTC", 2)}, 11)
    assert [plan.reason for plan in plans] == ["flip_close"]
    assert plans[0].close_all is True
    assert plans[0].side == "sell"


def test_flat_target_closes_longs_and_shorts() -> None:
    account = _account(10_000, [_position("BTC", 1.5, 100), _position("ETH", -2, 100)])
    gate = _gate([_target("BTC", 0, status="held"), _target("ETH", 0), _target("SOL", 0)])
    plans = plan_orders(
        gate, account, {"BTC": 100, "ETH": 100, "SOL": 100}, _metas(("BTC", "ETH", "SOL")), 11
    )
    assert [(plan.symbol, plan.side, plan.size, plan.reason) for plan in plans] == [
        ("BTC", "sell", 1.5, "close"),
        ("ETH", "buy", 2.0, "close"),
    ]


def test_plan_opens_and_reduces_shorts() -> None:
    account = _account(100_000, [_position("ETH", -50, 100)])
    gate = _gate([_target("BTC", -0.1), _target("ETH", -0.02)])
    marks = {"BTC": 100.0, "ETH": 100.0}
    plans = plan_orders(gate, account, marks, _metas(("BTC", "ETH")), 11)
    assert [(plan.symbol, plan.side, plan.reason, plan.reduce_only, plan.size) for plan in plans] == [
        ("ETH", "buy", "reduce", True, 30),
        ("BTC", "sell", "increase", False, 100),
    ]


def test_matching_position_plans_nothing() -> None:
    account = _account(10_000, [_position("BTC", 10, 100)])
    plans = plan_orders(
        _gate([_target("BTC", 0.1, status="held")]), account, {"BTC": 100}, {"BTC": _meta("BTC", 2)}, 11
    )
    assert plans == []


def test_plan_skips_when_mark_or_meta_is_missing() -> None:
    account = _account(10_000, [_position("BTC", 1, 100)])
    no_mark = plan_orders(_gate([_target("BTC", 0.1)]), account, {}, {"BTC": _meta("BTC", 2)}, 11)
    assert no_mark == []
    no_meta = plan_orders(_gate([_target("ETH", 0.1)]), account, {"ETH": 100}, {}, 11)
    assert no_meta == []
    still_closes = plan_orders(_gate([_target("BTC", 0)]), account, {}, {}, 11)
    assert [plan.reason for plan in still_closes] == ["close"]


def test_price_gap_skips_increases_above_the_limit() -> None:
    account = _account(10_000, [_position("GOLD", 1, 80)])
    gate = _gate([_target("BTC", 0.1), _target("ETH", 0.1), _target("GOLD", 0)])
    marks = {"BTC": 106.0, "ETH": 104.0, "GOLD": 80.0}
    meta = _metas(("BTC", "ETH", "GOLD"), decimals=4)
    venue = RecordingVenue(account, marks, meta)
    outcomes = execute(gate, venue, _settings(), {"BTC": 100, "ETH": 100, "GOLD": 100})
    assert [(item.symbol, item.status, item.message) for item in outcomes] == [
        ("GOLD", "filled", ""),
        ("BTC", "skipped", "price_gap"),
        ("ETH", "filled", ""),
    ]
    assert ("close_position", "GOLD") in venue.calls
    assert ("ensure_leverage", "ETH", 5, True) in venue.calls
    assert not any(call[0] == "ensure_leverage" and call[1] == "BTC" for call in venue.calls)
    assert not any(call[0] == "market_order" and call[1] == "BTC" for call in venue.calls)


def test_missing_reference_mark_does_not_skip() -> None:
    account = _account(10_000, [])
    venue = RecordingVenue(account, {"BTC": 100}, {"BTC": _meta("BTC", 2)})
    outcomes = execute(_gate([_target("BTC", 0.1)]), venue, _settings(), {})
    assert outcomes[0].status == "filled"
    assert any(call[0] == "market_order" for call in venue.calls)


def test_zero_reference_mark_skips_the_increase() -> None:
    account = _account(10_000, [])
    venue = RecordingVenue(account, {"BTC": 100}, {"BTC": _meta("BTC", 2)})
    outcomes = execute(_gate([_target("BTC", 0.1)]), venue, _settings(), {"BTC": 0})
    assert outcomes[0].status == "skipped"
    assert outcomes[0].message == "price_gap"
    assert outcomes[0].filled_size == 0


def test_error_skips_later_increases_and_still_reduces() -> None:
    account = _account(100_000, [_position("BTC", 2, 100), _position("ETH", 50, 100)])
    gate = _gate(
        [
            _target("SOL", 0.1),
            _target("BTC", 0),
            _target("XRP", 0.1),
            _target("ETH", 0.02),
        ]
    )
    marks = {symbol: 100.0 for symbol in ("SOL", "BTC", "XRP", "ETH")}
    error = OrderOutcome(
        symbol="BTC",
        side="sell",
        requested_size=2,
        filled_size=0,
        avg_px=None,
        fee_usd=0,
        status="error",
        message="rejected",
    )
    filled = OrderOutcome(
        symbol="ETH",
        side="sell",
        requested_size=30,
        filled_size=30,
        avg_px=100,
        fee_usd=0,
        status="filled",
    )
    venue = RecordingVenue(account, marks, _metas(("SOL", "BTC", "XRP", "ETH")), [error, filled])
    outcomes = execute(gate, venue, _settings(), marks)
    assert [(item.symbol, item.status, item.message) for item in outcomes] == [
        ("BTC", "error", "rejected"),
        ("ETH", "filled", ""),
        ("SOL", "skipped", "prior_error"),
        ("XRP", "skipped", "prior_error"),
    ]
    submitted = [call for call in venue.calls if call[0] in ("close_position", "market_order")]
    assert submitted == [("close_position", "BTC"), ("market_order", "ETH", "sell", 30, True)]
    assert not any(call[0] == "ensure_leverage" for call in venue.calls)
    synced = {call[1] for call in venue.calls if call[0] == "sync_stop"}
    assert synced == {"SOL", "BTC", "XRP", "ETH"}


def test_stops_use_entry_for_longs_and_shorts() -> None:
    account = _account(
        10_000,
        [
            _position("BTC", 0.2, 81_491.37),
            _position("ETH", -3, 157.775),
            _position("GOLD", 1, 100),
        ],
    )
    gate = _gate(
        [
            _target("BTC", 0.1, stop=10),
            _target("ETH", -0.1, stop=10),
            _target("SOL", 0, stop=5),
            _target("GOLD", 0.1, stop=0),
        ]
    )
    meta = {
        "BTC": _meta("BTC", 5),
        "ETH": _meta("ETH", 2),
        "SOL": _meta("SOL", 2),
        "GOLD": _meta("GOLD", 2),
    }
    stops = desired_stops(gate, account, meta)
    assert stops["BTC"] == StopOrder(symbol="BTC", side="sell", size=0.2, trigger_px=73_342)
    assert stops["ETH"] == StopOrder(symbol="ETH", side="buy", size=3, trigger_px=173.55)
    assert stops["SOL"] is None
    assert stops["GOLD"] is None


def test_rejected_duplicate_does_not_replace_the_stop() -> None:
    account = _account(10_000, [_position("BTC", 1, 100)])
    gate = _gate([_target("BTC", 0.1, stop=10), _target("BTC", 0.2, stop=1, status="rejected")])
    stops = desired_stops(gate, account, {"BTC": _meta("BTC", 2)})
    stop = stops["BTC"]
    assert stop is not None
    assert stop.trigger_px == pytest.approx(90)


def test_stop_is_absent_without_market_meta() -> None:
    account = _account(10_000, [_position("BTC", 1, 100)])
    stops = desired_stops(_gate([_target("BTC", 0.1, stop=5)]), account, {})
    assert stops["BTC"] is None


def test_ensure_leverage_respects_the_market_cap_and_isolation() -> None:
    account = _account(10_000, [])
    gate = _gate([_target("BTC", 0.1), _target("xyz:JPY", 0.1)])
    meta = {
        "BTC": _meta("BTC", 5, leverage=3, isolated=False),
        "xyz:JPY": _meta("xyz:JPY", 2, leverage=50, isolated=True),
    }
    marks = {"BTC": 100.0, "xyz:JPY": 100.0}
    venue = RecordingVenue(account, marks, meta)
    execute(gate, venue, _settings(), marks)
    assert ("ensure_leverage", "BTC", 3, True) in venue.calls
    assert ("ensure_leverage", "xyz:JPY", 5, False) in venue.calls
    btc_leverage = venue.calls.index(("ensure_leverage", "BTC", 3, True))
    btc_order = next(
        index for index, call in enumerate(venue.calls) if call[0] == "market_order" and call[1] == "BTC"
    )
    assert btc_leverage < btc_order


def test_execute_on_paper_opens_and_stores_the_stop() -> None:
    meta = {"BTC": _meta("BTC", 5, leverage=40)}
    venue = PaperVenue(
        new_paper_account("config:opus-5.5/single", 10_000, NOW), {"BTC": 80_000}, meta, PAPER, NOW
    )
    outcomes = execute(_gate([_target("BTC", 0.1, stop=5)]), venue, _settings(), {"BTC": 80_000})
    assert outcomes[0].status == "filled"
    assert outcomes[0].side == "buy"
    assert outcomes[0].filled_size == pytest.approx(0.0125)
    assert outcomes[0].avg_px == pytest.approx(80_040)
    position = venue.state().positions["BTC"]
    assert position.entry_px == pytest.approx(80_040)
    assert position.stop is not None
    assert position.stop.side == "sell"
    assert position.stop.size == pytest.approx(0.0125)
    assert position.stop.trigger_px == 76_038


def test_execute_on_paper_closes_a_flat_target() -> None:
    meta = {"BTC": _meta("BTC", 5, leverage=40)}
    held = PaperPosition(symbol="BTC", size=0.0125, entry_px=80_000)
    account = new_paper_account("p", 10_000, NOW).model_copy(update={"positions": {"BTC": held}})
    venue = PaperVenue(account, {"BTC": 80_000}, meta, PAPER, NOW)
    outcomes = execute(_gate([_target("BTC", 0)]), venue, _settings(), {"BTC": 80_000})
    assert outcomes[0].status == "filled"
    assert outcomes[0].side == "sell"
    assert "BTC" not in venue.state().positions


class RecordingVenue:
    """Venue double that records every call and replays scripted outcomes."""

    name = "recording"

    def __init__(
        self,
        account: AccountSnapshot,
        marks: dict[str, float],
        meta: dict[str, MarketMeta],
        script: list[OrderOutcome] | None = None,
    ) -> None:
        self._account = account
        self._marks = marks
        self._meta = meta
        self._script = list(script or [])
        self.calls: list[tuple[object, ...]] = []

    def market_meta(self) -> dict[str, MarketMeta]:
        return self._meta

    def marks(self) -> dict[str, float]:
        return self._marks

    def account(self) -> AccountSnapshot:
        return self._account

    def market_order(self, symbol: str, side: Side, size: float, reduce_only: bool) -> OrderOutcome:
        self.calls.append(("market_order", symbol, side, size, reduce_only))
        return self._pop(symbol, side, size)

    def close_position(self, symbol: str) -> OrderOutcome:
        self.calls.append(("close_position", symbol))
        return self._pop(symbol, "sell", 0)

    def ensure_leverage(self, symbol: str, leverage: int, cross: bool) -> None:
        self.calls.append(("ensure_leverage", symbol, leverage, cross))

    def sync_stop(self, symbol: str, stop: StopOrder | None) -> None:
        self.calls.append(("sync_stop", symbol, stop))

    def _pop(self, symbol: str, side: Side, size: float) -> OrderOutcome:
        if self._script:
            return self._script.pop(0)
        return OrderOutcome(
            symbol=symbol,
            side=side,
            requested_size=size,
            filled_size=size,
            avg_px=self._marks.get(symbol),
            fee_usd=0,
            status="filled",
        )


def _meta(symbol: str, decimals: int, leverage: int = 10, isolated: bool = False) -> MarketMeta:
    return MarketMeta(symbol=symbol, sz_decimals=decimals, max_leverage=leverage, only_isolated=isolated)


def _metas(symbols: tuple[str, ...], decimals: int = 2) -> dict[str, MarketMeta]:
    return {symbol: _meta(symbol, decimals) for symbol in symbols}


def _position(symbol: str, size: float, mark: float) -> Position:
    return Position(symbol=symbol, size=size, entry_px=mark, mark=mark)


def _account(equity: float, positions: list[Position]) -> AccountSnapshot:
    return AccountSnapshot(equity=equity, positions={item.symbol: item for item in positions})


def _target(
    symbol: str,
    approved: float,
    stop: float = 5,
    status: GateStatus = "approved",
    current: float = 0,
) -> GatedTarget:
    return GatedTarget(
        symbol=symbol,
        current_weight=current,
        requested_weight=approved,
        approved_weight=approved,
        stop_loss_pct=stop,
        status=status,
        reasons=[],
    )


def _gate(targets: list[GatedTarget]) -> GateResult:
    return GateResult(targets=targets)


def _settings() -> ExecutionSettings:
    return load_config().execution
