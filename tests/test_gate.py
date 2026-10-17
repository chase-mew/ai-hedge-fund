"""One test per risk gate reason, with hand computed weights."""

from datetime import UTC, datetime, timedelta

from fund.config import RiskPolicy, load_config
from fund.contracts import GatedTarget, GateResult, PortfolioDecision, PortfolioState, SessionState, Snapshot
from fund.risk.gate import apply_gate, no_increase
from tests.factories import make_decision, make_policy, make_portfolio, make_snapshot

NOW = datetime(2026, 10, 14, 16, 0, tzinfo=UTC)
UNIVERSE = [spec.symbol for spec in load_config().universe]


def _gate(
    decision: PortfolioDecision | None,
    portfolio: PortfolioState,
    snapshot: Snapshot,
    policy: RiskPolicy | None = None,
) -> GateResult:
    return apply_gate(
        decision,
        portfolio,
        snapshot,
        load_config().universe,
        policy or make_policy(),
        NOW,
    )


def _at(result: GateResult, symbol: str) -> GatedTarget:
    matches = [target for target in result.targets if target.symbol == symbol]
    assert len(matches) == 1
    return matches[0]


def _session(open_market: bool = True, pre_weekend: bool = False) -> Snapshot:
    snapshot = make_snapshot(now=NOW)
    session = SessionState(tradfi_open=open_market, pre_weekend=pre_weekend)
    return snapshot.model_copy(update={"session": session})


def _quote(snapshot: Snapshot, symbol: str, mark: float, oracle: float) -> Snapshot:
    markets = []
    for market in snapshot.markets:
        if market.symbol == symbol:
            markets.append(market.model_copy(update={"mark": mark, "oracle": oracle}))
        else:
            markets.append(market)
    return snapshot.model_copy(update={"markets": markets})


def _drop(snapshot: Snapshot, symbol: str) -> Snapshot:
    kept = [market for market in snapshot.markets if market.symbol != symbol]
    return snapshot.model_copy(update={"markets": kept})


def _book(portfolio: PortfolioState, **updates: object) -> PortfolioState:
    return portfolio.model_copy(update={"bookkeeping": portfolio.bookkeeping.model_copy(update=updates)})


def _loose(**overrides: object) -> RiskPolicy:
    values: dict[str, object] = {
        "max_gross": 5.0,
        "max_net": 5.0,
        "max_new_risk_per_cycle": 5.0,
    }
    values.update(overrides)
    return make_policy(**values)


def test_no_increase_blocks_entries_flips_and_adds() -> None:
    assert no_increase(0.2, 0.0) == 0.0
    assert no_increase(-0.1, 0.2) == 0.0
    assert no_increase(0.5, 0.2) == 0.2
    assert no_increase(0.1, 0.2) == 0.1
    assert no_increase(-0.05, -0.2) == -0.05
    assert no_increase(-0.4, -0.2) == -0.2


def test_drawdown_halt_flattens_everything() -> None:
    portfolio = _book(make_portfolio(equity=8000.0, weights={"BTC": 0.1}, now=NOW), high_water_mark=10000.0)
    result = _gate(make_decision(weights={"BTC": 0.1}), portfolio, _session())
    assert result.halt is True
    assert result.halt_reason is not None
    assert [target.approved_weight for target in result.targets] == [0.0] * 11
    assert _at(result, "BTC").reasons == ["max_drawdown"]
    assert _at(result, "BTC").status == "clipped"
    assert _at(result, "ETH").status == "approved"


def test_drawdown_boundary_is_inclusive() -> None:
    on_line = _book(make_portfolio(equity=8000.0, now=NOW), high_water_mark=10000.0)
    above = _book(make_portfolio(equity=8000.01, now=NOW), high_water_mark=10000.0)
    assert _gate(make_decision(), on_line, _session()).halt is True
    assert _gate(make_decision(), above, _session()).halt is False


def test_no_decision_holds_current_weights() -> None:
    portfolio = make_portfolio(equity=10000.0, weights={"BTC": 0.05}, now=NOW)
    result = _gate(None, portfolio, _session())
    assert result.halt is False
    assert _at(result, "BTC").approved_weight == 0.05
    assert _at(result, "BTC").reasons == ["no_decision"]
    assert _at(result, "BTC").status == "approved"
    assert _at(result, "BTC").stop_loss_pct == 5.0
    assert _at(result, "ETH").approved_weight == 0.0
    assert "no_decision" in _at(result, "ETH").reasons


def test_missing_symbol_holds_and_unknown_and_duplicate_are_rejected() -> None:
    decision = make_decision(weights={"BTC": 0.05}, symbols=["BTC", "BTC", "DOGE"])
    portfolio = make_portfolio(equity=10000.0, weights={"BTC": 0.05, "ETH": 0.04}, now=NOW)
    result = _gate(decision, portfolio, _session())
    assert [target.symbol for target in result.targets] == [*UNIVERSE, "BTC", "DOGE"]
    assert _at(result, "ETH").reasons == ["missing"]
    assert _at(result, "ETH").approved_weight == 0.04
    assert _at(result, "ETH").status == "approved"
    rejected = [target for target in result.targets if target.status == "rejected"]
    assert [target.reasons for target in rejected] == [["duplicate"], ["unknown_symbol"]]
    assert rejected[0].approved_weight == 0.05
    assert rejected[1].approved_weight == 0.0


def test_max_weight_clips_to_policy() -> None:
    result = _gate(make_decision(weights={"BTC": 0.25}), make_portfolio(now=NOW), _session())
    target = _at(result, "BTC")
    assert target.approved_weight == 0.20
    assert target.reasons == ["max_weight"]
    assert target.status == "clipped"


def test_price_dislocation_blocks_the_increase() -> None:
    snapshot = _quote(_session(), "xyz:GOLD", mark=102.0, oracle=100.0)
    portfolio = make_portfolio(equity=10000.0, weights={"xyz:GOLD": 0.10}, now=NOW)
    result = _gate(make_decision(weights={"xyz:GOLD": 0.20}), portfolio, snapshot)
    target = _at(result, "xyz:GOLD")
    assert target.approved_weight == 0.10
    assert target.reasons == ["price_dislocation"]
    assert target.status == "held"


def test_zero_oracle_and_missing_quote_block_new_risk() -> None:
    zero = _quote(_session(), "BTC", mark=100.0, oracle=0.0)
    missing = _drop(_session(), "ETH")
    flat = make_portfolio(now=NOW)
    assert _at(_gate(make_decision(weights={"BTC": 0.10}), flat, zero), "BTC").approved_weight == 0.0
    eth = _gate(make_decision(weights={"ETH": 0.10}), flat, missing)
    assert "price_dislocation" in _at(eth, "ETH").reasons


def test_closed_market_blocks_new_exposure_but_crypto_can_trade() -> None:
    portfolio = make_portfolio(equity=10000.0, weights={"xyz:SP500": 0.10}, now=NOW)
    result = _gate(
        make_decision(weights={"xyz:SP500": 0.15, "BTC": 0.10}),
        portfolio,
        _session(open_market=False),
    )
    closed = _at(result, "xyz:SP500")
    assert closed.approved_weight == 0.10
    assert closed.reasons == ["market_closed"]
    assert closed.status == "held"
    assert _at(result, "BTC").approved_weight == 0.10
    assert _at(result, "BTC").status == "approved"


def test_pre_weekend_cap_shrinks_longs_and_shorts() -> None:
    result = _gate(
        make_decision(weights={"xyz:SP500": 0.15, "xyz:JPY": -0.15, "BTC": 0.15}),
        make_portfolio(now=NOW),
        _session(pre_weekend=True),
    )
    assert _at(result, "xyz:SP500").approved_weight == 0.10
    assert _at(result, "xyz:SP500").reasons == ["pre_weekend_cap"]
    assert _at(result, "xyz:JPY").approved_weight == -0.10
    assert _at(result, "BTC").approved_weight == 0.15


def test_flip_cooldown_flattens_a_recent_flip() -> None:
    portfolio = _book(
        make_portfolio(equity=10000.0, weights={"BTC": 0.10}, now=NOW),
        last_flip_at={"BTC": NOW - timedelta(hours=1)},
    )
    result = _gate(make_decision(weights={"BTC": -0.10}), portfolio, _session())
    target = _at(result, "BTC")
    assert target.approved_weight == 0.0
    assert target.reasons == ["flip_cooldown"]
    assert target.status == "clipped"


def test_flip_is_allowed_once_the_cooldown_has_passed() -> None:
    portfolio = _book(
        make_portfolio(equity=10000.0, weights={"BTC": 0.10}, now=NOW),
        last_flip_at={"BTC": NOW - timedelta(hours=12, seconds=1)},
    )
    result = _gate(make_decision(weights={"BTC": -0.10}), portfolio, _session())
    assert _at(result, "BTC").approved_weight == -0.10
    assert _at(result, "BTC").status == "approved"


def test_daily_loss_limit_blocks_increases_including_the_boundary() -> None:
    portfolio = _book(
        make_portfolio(equity=9600.0, weights={"xyz:NVDA": 0.10}, now=NOW),
        day_start_equity=10000.0,
    )
    result = _gate(make_decision(weights={"xyz:NVDA": 0.20}), portfolio, _session())
    target = _at(result, "xyz:NVDA")
    assert target.approved_weight == 0.10
    assert target.reasons == ["daily_loss_limit"]
    assert target.status == "held"


def test_below_min_trade_drops_tiny_entries_and_tiny_flips_but_not_reductions() -> None:
    entry = _gate(make_decision(weights={"BTC": 0.01}), make_portfolio(now=NOW), _session())
    assert _at(entry, "BTC").approved_weight == 0.0
    assert _at(entry, "BTC").reasons == ["below_min_trade"]
    assert _at(entry, "BTC").status == "held"

    flip = _gate(
        make_decision(weights={"ETH": -0.01}),
        make_portfolio(equity=10000.0, weights={"ETH": 0.10}, now=NOW),
        _session(),
    )
    assert _at(flip, "ETH").approved_weight == 0.0
    assert _at(flip, "ETH").reasons == ["below_min_trade"]

    reduction = _gate(
        make_decision(weights={"xyz:AAPL": 0.09}),
        make_portfolio(equity=10000.0, weights={"xyz:AAPL": 0.10}, now=NOW),
        _session(),
    )
    reduced = _at(reduction, "xyz:AAPL")
    assert reduced.approved_weight == 0.09
    assert reduced.reasons == []
    assert reduced.status == "approved"


def test_class_cap_scales_only_that_class() -> None:
    weights = {"xyz:NVDA": 0.20, "xyz:AAPL": 0.20, "xyz:MSFT": 0.20, "xyz:TSLA": 0.20, "BTC": 0.20}
    result = _gate(make_decision(weights=weights), make_portfolio(now=NOW), _session(), _loose())
    for symbol in ("xyz:NVDA", "xyz:AAPL", "xyz:MSFT", "xyz:TSLA"):
        target = _at(result, symbol)
        assert abs(target.approved_weight - 0.15) < 1e-12
        assert target.reasons == ["class_cap"]
    assert _at(result, "BTC").approved_weight == 0.20
    assert _at(result, "BTC").reasons == []


def test_gross_cap_scales_every_target() -> None:
    policy = _loose(max_gross=0.30)
    result = _gate(
        make_decision(weights={"BTC": 0.20, "ETH": 0.20}),
        make_portfolio(now=NOW),
        _session(),
        policy,
    )
    assert abs(_at(result, "BTC").approved_weight - 0.15) < 1e-12
    assert abs(_at(result, "ETH").approved_weight - 0.15) < 1e-12
    assert _at(result, "BTC").reasons == ["gross_cap"]


def test_new_risk_cap_shrinks_entries_and_flips_but_not_exits() -> None:
    policy = _loose(max_new_risk_per_cycle=0.30, max_weight_per_market=0.25)
    opened = _gate(
        make_decision(weights={"BTC": 0.20, "ETH": 0.20}),
        make_portfolio(now=NOW),
        _session(),
        policy,
    )
    assert abs(_at(opened, "BTC").approved_weight - 0.15) < 1e-12
    assert _at(opened, "BTC").reasons == ["new_risk_cap"]

    flipping = _gate(
        make_decision(weights={"BTC": -0.20, "ETH": 0.20}),
        make_portfolio(equity=10000.0, weights={"BTC": 0.20}, now=NOW),
        _session(),
        _loose(max_new_risk_per_cycle=0.20, max_weight_per_market=0.25),
    )
    assert abs(_at(flipping, "BTC").approved_weight - -0.10) < 1e-12
    assert "new_risk_cap" in _at(flipping, "BTC").reasons
    assert abs(_at(flipping, "ETH").approved_weight - 0.10) < 1e-12


def test_net_cap_scales_the_book() -> None:
    policy = _loose(max_net=0.30)
    result = _gate(
        make_decision(weights={"BTC": 0.20, "ETH": 0.20}),
        make_portfolio(now=NOW),
        _session(),
        policy,
    )
    assert abs(_at(result, "BTC").approved_weight - 0.15) < 1e-12
    assert _at(result, "BTC").reasons == ["net_cap"]


def test_stop_is_clamped_and_flat_targets_have_no_stop() -> None:
    wide = _gate(
        make_decision(weights={"BTC": 0.05}, stops={"BTC": 20.0}),
        make_portfolio(now=NOW),
        _session(),
    )
    tight = _gate(
        make_decision(weights={"ETH": 0.05}, stops={"ETH": 0.5}),
        make_portfolio(now=NOW),
        _session(),
    )
    flat = _gate(make_decision(stops={"BTC": 5.0}), make_portfolio(now=NOW), _session())
    assert _at(wide, "BTC").stop_loss_pct == 12.0
    assert _at(wide, "BTC").reasons == ["stop_clamped"]
    assert _at(wide, "BTC").status == "approved"
    assert _at(tight, "ETH").stop_loss_pct == 1.0
    assert "stop_clamped" in _at(tight, "ETH").reasons
    assert _at(flat, "BTC").stop_loss_pct == 0.0
    assert _at(flat, "BTC").reasons == []


def test_compliant_decision_is_approved_unchanged() -> None:
    weights = {symbol: 0.05 for symbol in UNIVERSE}
    decision = make_decision(weights=weights, stops={symbol: 5.0 for symbol in UNIVERSE})
    result = _gate(decision, make_portfolio(now=NOW), _session())
    assert result.halt is False
    assert [target.symbol for target in result.targets] == UNIVERSE
    for target in result.targets:
        assert target.approved_weight == 0.05
        assert target.status == "approved"
        assert target.reasons == []
        assert target.stop_loss_pct == 5.0
