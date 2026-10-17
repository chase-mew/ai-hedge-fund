"""Benchmark target weights."""

import random
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from fund.baselines import BASELINES, as_gate_result, baseline_targets
from tests.factories import make_features, make_portfolio, make_snapshot

NOW = datetime(2026, 10, 17, 12, 0, tzinfo=UTC)


def test_sp500_is_fully_long_the_index() -> None:
    snapshot = make_snapshot(now=NOW)
    targets = baseline_targets("sp500", snapshot, NOW)
    assert list(targets) == [market.symbol for market in snapshot.markets]
    assert targets["xyz:SP500"] == 1.0
    assert sum(targets.values()) == 1.0
    short = make_snapshot(now=NOW, markets=[make_features("BTC")])
    extra = baseline_targets("sp500", short, NOW)
    assert extra == {"BTC": 0.0, "xyz:SP500": 1.0}


def test_equal_weight_is_long_one_eleventh_of_each_market() -> None:
    snapshot = make_snapshot(now=NOW)
    targets = baseline_targets("equal_weight", snapshot, NOW)
    assert list(targets) == [market.symbol for market in snapshot.markets]
    assert len(targets) == 11
    assert list(targets.values()) == pytest.approx([1 / 11] * 11)
    short = make_snapshot(now=NOW, markets=[make_features("BTC"), make_features("ETH")])
    short_targets = baseline_targets("equal_weight", short, NOW)
    assert list(short_targets) == ["BTC", "ETH"]
    assert short_targets["BTC"] == pytest.approx(1 / 11)
    assert short_targets["ETH"] == pytest.approx(1 / 11)


def test_trend_follows_the_sign_of_sma20_gap() -> None:
    markets = [
        make_features("BTC", sma20_gap=0.01),
        make_features("ETH", sma20_gap=-0.02),
        make_features("xyz:SP500", sma20_gap=0.005),
        make_features("xyz:GOLD", sma20_gap=-0.005),
        make_features("xyz:CL", sma20_gap=0.0049),
    ]
    snapshot = make_snapshot(now=NOW, markets=markets)
    targets = baseline_targets("trend", snapshot, NOW)
    assert targets["BTC"] == pytest.approx(1 / 11)
    assert targets["ETH"] == pytest.approx(-1 / 11)
    assert targets["xyz:SP500"] == pytest.approx(1 / 11)
    assert targets["xyz:GOLD"] == pytest.approx(-1 / 11)
    assert targets["xyz:CL"] == 0.0


def test_random_baseline_is_reproducible_for_a_utc_day() -> None:
    snapshot = make_snapshot(now=NOW)
    morning = datetime(2026, 10, 17, 1, 0, tzinfo=UTC)
    evening = datetime(2026, 10, 17, 23, 0, tzinfo=UTC)
    first = baseline_targets("random", snapshot, morning)
    assert first == baseline_targets("random", snapshot, evening)
    rng = random.Random("2026-10-17")
    expected = {market.symbol: rng.uniform(-0.1, 0.1) for market in snapshot.markets}
    assert first == expected
    assert all(-0.1 <= weight <= 0.1 for weight in first.values())
    later = baseline_targets("random", snapshot, datetime(2026, 10, 18, 0, 0, tzinfo=UTC))
    assert later != first
    new_york = datetime(2026, 10, 16, 22, 0, tzinfo=ZoneInfo("America/New_York"))
    assert baseline_targets("random", snapshot, new_york) == first


def test_as_gate_result_approves_targets_without_stops() -> None:
    portfolio = make_portfolio(weights={"BTC": 0.1}, now=NOW)
    result = as_gate_result({"BTC": 1.0, "ETH": -0.2}, portfolio)
    assert result.halt is False
    assert result.halt_reason is None
    assert [row.symbol for row in result.targets] == ["BTC", "ETH"]
    btc, eth = result.targets
    assert btc.current_weight == pytest.approx(0.1)
    assert btc.requested_weight == 1.0
    assert btc.approved_weight == 1.0
    assert btc.stop_loss_pct == 0.0
    assert btc.status == "approved"
    assert btc.reasons == []
    assert eth.current_weight == 0.0
    assert eth.approved_weight == -0.2
    assert eth.stop_loss_pct == 0.0
    assert eth.status == "approved"


def test_unknown_baseline_raises() -> None:
    assert BASELINES == ("sp500", "equal_weight", "trend", "random")
    with pytest.raises(ValueError, match="Unknown baseline"):
        baseline_targets("momentum", make_snapshot(now=NOW), NOW)
