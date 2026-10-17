"""Every contract model survives a JSON round trip."""

from datetime import UTC, datetime

from pydantic import BaseModel

from fund.contracts import (
    AccountSnapshot,
    AnalystReport,
    AnalystView,
    AssetClass,
    Bookkeeping,
    CalibrationBin,
    Candle,
    ConfigRun,
    CycleRecord,
    EquityPoint,
    Forecast,
    ForecastScore,
    GatedTarget,
    GateResult,
    LlmUsage,
    MarketDecision,
    MarketFeatures,
    MarketMeta,
    MarketQuote,
    MarketSpec,
    NewsDigest,
    OrderOutcome,
    OrderPlan,
    PaperAccount,
    PaperPosition,
    PortfolioDecision,
    PortfolioScore,
    PortfolioState,
    Position,
    RecentDecision,
    Resolution,
    ScoreReport,
    SessionState,
    Snapshot,
    StopOrder,
    TrackRecord,
)

NOW = datetime(2026, 10, 14, 16, 0, tzinfo=UTC)


def _features() -> MarketFeatures:
    return MarketFeatures(
        symbol="BTC",
        display_name="Bitcoin",
        asset_class=AssetClass.CRYPTO,
        tradfi_hours=False,
        market_open=True,
        mark=80000.0,
        oracle=80010.0,
        ret_24h=0.01,
        ret_7d=0.02,
        ret_30d=0.03,
        vol_30d_ann=0.4,
        pct_from_20d_high=-0.01,
        pct_from_20d_low=0.05,
        sma20_gap=0.01,
        sma50_gap=0.02,
        funding_ann=0.1,
        open_interest_usd=1.0,
        volume_24h_usd=2.0,
        daily_closes=[1.0, 2.0],
    )


def _decision() -> PortfolioDecision:
    return PortfolioDecision(
        market_view="Quiet tape.",
        decisions=[
            MarketDecision(
                symbol="BTC",
                thesis="Holding the 20 day average.",
                invalidation="A daily close under 78000.",
                prob_up_24h=0.55,
                target_weight=0.05,
                stop_loss_pct=5.0,
            )
        ],
    )


def _snapshot() -> Snapshot:
    return Snapshot(
        cycle_id="2026-10-14T16:00Z",
        as_of=NOW,
        session=SessionState(tradfi_open=True, pre_weekend=False),
        markets=[_features()],
        news=NewsDigest(text="Nothing notable.", sources=["example.com"], compiled_at=NOW),
    )


def _portfolio() -> PortfolioState:
    position = Position(symbol="BTC", size=0.01, entry_px=80000.0, mark=81000.0)
    return PortfolioState(
        portfolio_id="config:opus-5.5/single",
        as_of=NOW,
        account=AccountSnapshot(equity=10000.0, positions={"BTC": position}),
        bookkeeping=Bookkeeping(day="2026-10-14", day_start_equity=10000.0, high_water_mark=10100.0),
    )


def _samples() -> list[BaseModel]:
    features = _features()
    decision = _decision()
    snapshot = _snapshot()
    news = snapshot.news
    assert news is not None
    portfolio = _portfolio()
    stop = StopOrder(symbol="BTC", side="sell", size=0.01, trigger_px=76000.0)
    outcome = OrderOutcome(
        symbol="BTC",
        side="buy",
        requested_size=0.01,
        filled_size=0.01,
        avg_px=80000.0,
        fee_usd=0.36,
        status="filled",
        order_id="1",
    )
    usage = LlmUsage(
        purpose="decide",
        model_id="claude-opus-5-5",
        input_tokens=10,
        output_tokens=20,
        cache_write_tokens=0,
        cache_read_tokens=0,
        web_searches=0,
        cost_usd=0.01,
    )
    gate = GateResult(
        targets=[
            GatedTarget(
                symbol="BTC",
                current_weight=0.0,
                requested_weight=0.05,
                approved_weight=0.05,
                stop_loss_pct=5.0,
                status="approved",
                reasons=[],
            )
        ]
    )
    return [
        MarketSpec(symbol="BTC", display_name="Bitcoin", asset_class=AssetClass.CRYPTO, tradfi_hours=False),
        MarketMeta(symbol="BTC", sz_decimals=5, max_leverage=40, only_isolated=False),
        Candle(open_time=NOW, open=1, high=2, low=0.5, close=1.5, volume=3),
        MarketQuote(
            symbol="BTC",
            mark=1,
            oracle=1,
            prev_day_px=1,
            funding_hourly=0,
            open_interest=1,
            day_volume_usd=1,
        ),
        SessionState(tradfi_open=True, pre_weekend=False),
        features,
        news,
        snapshot,
        portfolio.account.positions["BTC"],
        portfolio.account,
        portfolio.bookkeeping,
        portfolio,
        RecentDecision(
            cycle_id="2026-10-14T12:00Z",
            symbol="BTC",
            target_weight=0.05,
            prob_up_24h=0.55,
            thesis="Holding.",
            price_then=80000,
            price_now=81000,
        ),
        TrackRecord(recent=[], resolved_forecasts=0, brier=None),
        decision.decisions[0],
        decision,
        AnalystReport(
            summary="Quiet.",
            views=[
                AnalystView(
                    symbol="BTC",
                    key_points=["Above the average."],
                    risks=["A gap down."],
                    stance="neutral",
                    prob_up_24h=0.5,
                )
            ],
        ),
        gate.targets[0],
        gate,
        OrderPlan(symbol="BTC", side="buy", size=0.01, reduce_only=False, close_all=False, reason="increase"),
        outcome,
        stop,
        PaperAccount(
            portfolio_id="config:test",
            cash=10000,
            positions={"BTC": PaperPosition(symbol="BTC", size=0.01, entry_px=80000, stop=stop)},
            last_marks={"BTC": 81000},
            updated_at=NOW,
            bookkeeping=portfolio.bookkeeping,
        ),
        usage,
        ConfigRun(
            config_id="opus-5.5/single",
            portfolio_id="config:opus-5.5/single",
            decision=decision,
            error=None,
            gate=gate,
            outcomes=[outcome],
            usage=[usage],
            transcript_path=None,
            equity_before=10000,
            equity_after=10010,
        ),
        CycleRecord(
            cycle_id="2026-10-14T16:00Z",
            mode="paper",
            started_at=NOW,
            finished_at=NOW,
            snapshot=snapshot,
            news_usage=usage,
            runs=[],
            errors=[],
        ),
        Forecast(
            forecast_id="f1",
            cycle_id="2026-10-14T16:00Z",
            config_id="opus-5.5/single",
            symbol="BTC",
            made_at=NOW,
            resolves_at=NOW,
            prob_up=0.55,
            ref_price=80000,
        ),
        Resolution(forecast_id="f1", resolved_at=NOW, price=81000, outcome=True),
        EquityPoint(
            at=NOW,
            cycle_id="2026-10-14T16:00Z",
            portfolio_id="config:opus-5.5/single",
            equity=10000,
            gross=0.1,
            net=0.1,
            fees_cum=0,
            funding_cum=0,
            llm_cost_cum=0,
        ),
        CalibrationBin(lower=0.5, upper=0.6, mean_prob=0.55, hit_rate=0.5, count=2),
        PortfolioScore(
            portfolio_id="config:opus-5.5/single",
            kind="config",
            cycles=1,
            start_equity=10000,
            equity=10010,
            total_return=0.001,
            return_net_of_llm=0.001,
            ann_vol=None,
            sharpe=None,
            sortino=None,
            max_drawdown=0,
            fees=0,
            funding=0,
            llm_cost=0,
        ),
        ForecastScore(config_id="opus-5.5/single", resolved=1, brier=0.2, brier_skill=0.2, calibration=[]),
        ScoreReport(generated_at=NOW, portfolios=[], forecasts=[]),
    ]


def test_models_round_trip_json() -> None:
    for model in _samples():
        assert model is not None
        restored = type(model).model_validate_json(model.model_dump_json())
        assert restored == model


def test_weight_uses_notional() -> None:
    portfolio = _portfolio()
    assert portfolio.equity == 10000.0
    assert abs(portfolio.weight("BTC") - 810.0 / 10000.0) < 1e-12
