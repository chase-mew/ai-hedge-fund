"""Property tests for the risk gate invariants."""

from datetime import UTC, datetime, timedelta

from hypothesis import given, settings
from hypothesis import strategies as st

from fund.config import RiskPolicy, load_config
from fund.contracts import AssetClass, MarketFeatures, SessionState
from fund.risk.gate import apply_gate
from tests.factories import make_decision, make_policy, make_portfolio, make_snapshot

NOW = datetime(2026, 10, 14, 16, 0, tzinfo=UTC)
SYMBOLS = [spec.symbol for spec in load_config().universe]
CLASSES = {spec.symbol: spec.asset_class for spec in load_config().universe}
TRADFI = {spec.symbol for spec in load_config().universe if spec.tradfi_hours}
WEIGHT = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)
REQUEST = st.floats(min_value=-0.25, max_value=0.25, allow_nan=False, allow_infinity=False)
STOP = st.floats(min_value=0.5, max_value=20.0, allow_nan=False, allow_infinity=False)
MONEY = st.floats(min_value=1_000.0, max_value=20_000.0, allow_nan=False, allow_infinity=False)
SHOCK = st.floats(min_value=-0.05, max_value=0.05, allow_nan=False, allow_infinity=False)


def _new_risk(approved: float, current: float) -> float:
    if current == 0 or approved * current > 0:
        return max(0.0, abs(approved) - abs(current))
    if approved * current < 0:
        return abs(approved)
    return 0.0


@given(
    currents=st.fixed_dictionaries({symbol: WEIGHT for symbol in SYMBOLS}),
    requests=st.fixed_dictionaries({symbol: REQUEST for symbol in SYMBOLS}),
    stops=st.fixed_dictionaries({symbol: STOP for symbol in SYMBOLS}),
    shocks=st.fixed_dictionaries({symbol: SHOCK for symbol in SYMBOLS}),
    equity=MONEY,
    day_start=MONEY,
    high_water=MONEY,
    tradfi_open=st.booleans(),
    pre_weekend=st.booleans(),
    flip_hours=st.integers(min_value=0, max_value=48),
    tighten=st.sampled_from(
        [
            "max_weight_per_market",
            "max_gross",
            "max_net",
            "max_new_risk_per_cycle",
            "pre_weekend_max_weight",
            "class",
        ]
    ),
)
@settings(max_examples=500, deadline=None)
def test_gate_invariants(
    currents: dict[str, float],
    requests: dict[str, float],
    stops: dict[str, float],
    shocks: dict[str, float],
    equity: float,
    day_start: float,
    high_water: float,
    tradfi_open: bool,
    pre_weekend: bool,
    flip_hours: int,
    tighten: str,
) -> None:
    policy = make_policy()
    portfolio = make_portfolio(equity=equity, weights=currents, now=NOW)
    portfolio = portfolio.model_copy(
        update={
            "bookkeeping": portfolio.bookkeeping.model_copy(
                update={
                    "day_start_equity": day_start,
                    "high_water_mark": high_water,
                    "last_flip_at": {symbol: NOW - timedelta(hours=flip_hours) for symbol in SYMBOLS},
                }
            )
        }
    )
    snapshot = make_snapshot(now=NOW).model_copy(
        update={"session": SessionState(tradfi_open=tradfi_open, pre_weekend=pre_weekend)}
    )
    markets: list[MarketFeatures] = []
    for market in snapshot.markets:
        markets.append(market.model_copy(update={"oracle": market.mark * (1 + shocks[market.symbol])}))
    snapshot = snapshot.model_copy(update={"markets": markets})
    decision = make_decision(weights=requests, stops=stops)
    universe = load_config().universe
    result = apply_gate(decision, portfolio, snapshot, universe, policy, NOW)
    again = apply_gate(decision, portfolio, snapshot, universe, policy, NOW)
    assert result == again

    approved = {target.symbol: target.approved_weight for target in result.targets}
    current = {symbol: portfolio.weight(symbol) for symbol in SYMBOLS}
    assert set(approved) == set(SYMBOLS)

    if result.halt:
        assert all(weight == 0.0 for weight in approved.values())
    for symbol, weight in approved.items():
        assert abs(weight) <= policy.max_weight_per_market + 1e-9
        if weight != 0:
            stop = next(target.stop_loss_pct for target in result.targets if target.symbol == symbol)
            assert policy.stop_min_pct - 1e-9 <= stop <= policy.stop_max_pct + 1e-9
    assert sum(abs(weight) for weight in approved.values()) <= policy.max_gross + 1e-9
    assert abs(sum(approved.values())) <= policy.max_net + 1e-9
    for asset_class, cap in policy.class_caps.items():
        gross = sum(abs(weight) for symbol, weight in approved.items() if CLASSES[symbol] == asset_class)
        assert gross <= cap + 1e-9
    added = sum(_new_risk(approved[symbol], current[symbol]) for symbol in SYMBOLS)
    assert added <= policy.max_new_risk_per_cycle + 1e-9

    if not tradfi_open:
        for symbol in TRADFI:
            assert abs(approved[symbol]) <= abs(current[symbol]) + 1e-9
            assert approved[symbol] * current[symbol] >= -1e-12
    if equity <= day_start * (1 - policy.daily_loss_limit):
        for symbol in SYMBOLS:
            assert abs(approved[symbol]) <= abs(current[symbol]) + 1e-9
            assert approved[symbol] * current[symbol] >= -1e-12
    if equity <= high_water * (1 - policy.max_drawdown):
        assert result.halt is True

    tighter = _tighten(policy, tighten)
    other = apply_gate(decision, portfolio, snapshot, universe, tighter, NOW)
    other_gross = sum(abs(target.approved_weight) for target in other.targets)
    this_gross = sum(abs(weight) for weight in approved.values())
    assert other_gross <= this_gross + 1e-9


def _tighten(policy: RiskPolicy, name: str) -> RiskPolicy:
    if name == "class":
        caps = dict(policy.class_caps)
        caps[AssetClass.STOCK] = policy.class_caps[AssetClass.STOCK] * 0.5
        return policy.model_copy(update={"class_caps": caps})
    return policy.model_copy(update={name: getattr(policy, name) * 0.5})
