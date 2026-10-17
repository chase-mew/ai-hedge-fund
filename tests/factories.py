"""Builders for contract objects. Lanes use these instead of hand built fixtures."""

from __future__ import annotations

from datetime import UTC, datetime

from fund.config import RiskPolicy, load_config
from fund.contracts import (
    AccountSnapshot,
    AssetClass,
    Bookkeeping,
    MarketDecision,
    MarketFeatures,
    MarketSpec,
    PortfolioDecision,
    PortfolioState,
    Position,
    SessionState,
    Snapshot,
)

DEFAULT_MARKS: dict[str, float] = {
    "xyz:SP500": 7639.4,
    "xyz:XYZ100": 29141.0,
    "xyz:NVDA": 214.35,
    "xyz:AAPL": 342.23,
    "xyz:MSFT": 526.27,
    "xyz:TSLA": 351.68,
    "xyz:GOLD": 4160.1,
    "xyz:CL": 99.02,
    "xyz:JPY": 157.78,
    "BTC": 81530.0,
    "ETH": 2416.5,
}


def make_features(symbol: str = "BTC", mark: float = 80000.0, **overrides: object) -> MarketFeatures:
    """Build one market feature row."""
    spec = _spec(symbol)
    values: dict[str, object] = {
        "symbol": symbol,
        "display_name": spec.display_name if spec else symbol,
        "asset_class": spec.asset_class if spec else AssetClass.CRYPTO,
        "tradfi_hours": spec.tradfi_hours if spec else False,
        "market_open": True,
        "mark": mark,
        "oracle": mark,
        "ret_24h": 0.0,
        "ret_7d": 0.0,
        "ret_30d": 0.0,
        "vol_30d_ann": 0.2,
        "pct_from_20d_high": -0.01,
        "pct_from_20d_low": 0.04,
        "sma20_gap": 0.0,
        "sma50_gap": 0.0,
        "funding_ann": 0.0,
        "open_interest_usd": 1_000_000.0,
        "volume_24h_usd": 1_000_000.0,
        "daily_closes": [mark] * 10,
    }
    values.update(overrides)
    return MarketFeatures.model_validate(values)


def make_snapshot(now: datetime | None = None, **overrides: object) -> Snapshot:
    """Build a snapshot covering every market in fund.toml."""
    moment = now or datetime(2026, 10, 14, 16, 0, tzinfo=UTC)
    config = load_config()
    markets = [
        make_features(
            symbol=spec.symbol,
            mark=DEFAULT_MARKS[spec.symbol],
            display_name=spec.display_name,
            asset_class=spec.asset_class,
            tradfi_hours=spec.tradfi_hours,
        )
        for spec in config.universe
    ]
    values: dict[str, object] = {
        "cycle_id": "2026-10-14T16:00Z",
        "as_of": moment,
        "session": SessionState(tradfi_open=True, pre_weekend=False),
        "markets": markets,
        "news": None,
    }
    values.update(overrides)
    return Snapshot.model_validate(values)


def make_portfolio(
    equity: float = 10000.0,
    weights: dict[str, float] | None = None,
    marks: dict[str, float] | None = None,
    now: datetime | None = None,
    portfolio_id: str = "config:opus-5.5/single",
) -> PortfolioState:
    """Build a portfolio whose position sizes match the given weights at the given marks."""
    moment = now or datetime(2026, 10, 14, 16, 0, tzinfo=UTC)
    price = {**DEFAULT_MARKS, **(marks or {})}
    positions: dict[str, Position] = {}
    for symbol, weight in (weights or {}).items():
        if weight == 0:
            continue
        mark = price[symbol]
        positions[symbol] = Position(symbol=symbol, size=weight * equity / mark, entry_px=mark, mark=mark)
    return PortfolioState(
        portfolio_id=portfolio_id,
        as_of=moment,
        account=AccountSnapshot(equity=equity, positions=positions),
        bookkeeping=Bookkeeping(
            day=moment.strftime("%Y-%m-%d"),
            day_start_equity=equity,
            high_water_mark=equity,
        ),
    )


def make_decision(
    weights: dict[str, float] | None = None,
    probs: dict[str, float] | None = None,
    stops: dict[str, float] | None = None,
    symbols: list[str] | None = None,
) -> PortfolioDecision:
    """Build a portfolio decision with one row per universe symbol."""
    chosen = symbols if symbols is not None else [spec.symbol for spec in load_config().universe]
    weight_map = weights or {}
    prob_map = probs or {}
    stop_map = stops or {}
    return PortfolioDecision(
        market_view="No strong regime view in this fixture.",
        decisions=[
            MarketDecision(
                symbol=symbol,
                thesis="Fixture thesis grounded in the supplied snapshot.",
                invalidation="A close through the 20 day average.",
                prob_up_24h=prob_map.get(symbol, 0.5),
                target_weight=weight_map.get(symbol, 0.0),
                stop_loss_pct=stop_map.get(symbol, 5.0),
            )
            for symbol in chosen
        ],
    )


def make_policy(**overrides: object) -> RiskPolicy:
    """Load the risk policy from fund.toml and apply keyword overrides."""
    data = load_config().risk.model_dump()
    data.update(overrides)
    return RiskPolicy.model_validate(data)


def _spec(symbol: str) -> MarketSpec | None:
    for spec in load_config().universe:
        if spec.symbol == symbol:
            return spec
    return None
