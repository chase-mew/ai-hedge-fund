"""Golden and rule tests for decision prompts."""

from datetime import UTC, datetime
from pathlib import Path

from fund.brain.prompts import SYSTEM_ANALYST, SYSTEM_NEWS, SYSTEM_PM, SYSTEM_TEAM_PM, render_limits
from fund.brain.render import render_analyst_reports, render_snapshot
from fund.contracts import (
    AccountSnapshot,
    AnalystReport,
    AnalystView,
    AssetClass,
    Bookkeeping,
    NewsDigest,
    PortfolioState,
    Position,
    RecentDecision,
    SessionState,
    TrackRecord,
)
from tests.factories import make_features, make_policy, make_portfolio, make_snapshot

GOLDEN = Path(__file__).resolve().parent / "golden"
MOMENT = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)


def _plan_fence(marker: str) -> str:
    text = Path("BUILD_PLAN.md").read_text()
    start = text.index(marker)
    fence = text.index("```\n", start)
    body_start = fence + len("```\n")
    body_end = text.index("\n```", body_start)
    return text[body_start:body_end] + "\n"


def _assert_golden(name: str, text: str, update: bool) -> None:
    path = GOLDEN / name
    if update:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    assert path.read_text() == text


def _book() -> PortfolioState:
    equity = 10321.40
    gold_mark = 4159.6
    btc_mark = 81491.0
    return PortfolioState(
        portfolio_id="config:opus-5.5/single",
        as_of=MOMENT,
        account=AccountSnapshot(
            equity=equity,
            positions={
                "xyz:GOLD": Position(
                    symbol="xyz:GOLD",
                    size=0.12 * equity / gold_mark,
                    entry_px=4120.0,
                    mark=gold_mark,
                ),
                "BTC": Position(
                    symbol="BTC",
                    size=-0.05 * equity / btc_mark,
                    entry_px=81900.0,
                    mark=btc_mark,
                ),
            },
        ),
        bookkeeping=Bookkeeping(day="2026-10-19", day_start_equity=10250.0, high_water_mark=10402.10),
    )


def _markets() -> list:
    return [
        make_features(
            "xyz:SP500",
            mark=7639.40,
            display_name="S&P 500",
            asset_class=AssetClass.EQUITY_INDEX,
            tradfi_hours=True,
            market_open=True,
            ret_24h=0.0042,
            ret_7d=0.011,
            ret_30d=0.0295,
            vol_30d_ann=0.142,
            pct_from_20d_high=-0.008,
            pct_from_20d_low=0.041,
            sma20_gap=0.012,
            sma50_gap=0.028,
            funding_ann=0.055,
            open_interest_usd=120_300_000,
            volume_24h_usd=103_000_000,
            daily_closes=[7512.1, 7530.4, 7600.0, 7610.0, 7620.0, 7630.0, 7639.4],
        ),
        make_features(
            "xyz:GOLD",
            mark=4159.6,
            display_name="Gold",
            asset_class=AssetClass.COMMODITY,
            tradfi_hours=True,
            market_open=False,
            ret_24h=0.0096,
            vol_30d_ann=0.16,
            sma20_gap=0.01,
            open_interest_usd=50_000_000,
            volume_24h_usd=20_000_000,
            daily_closes=[4100.0, 4159.6],
        ),
        make_features(
            "BTC",
            mark=81491.0,
            display_name="Bitcoin",
            asset_class=AssetClass.CRYPTO,
            tradfi_hours=False,
            market_open=True,
            ret_24h=-0.005,
            vol_30d_ann=0.48,
            sma20_gap=-0.02,
            open_interest_usd=2_500_000_000,
            volume_24h_usd=800_000_000,
            daily_closes=[81900.0, 81491.0],
        ),
    ]


def _track() -> TrackRecord:
    return TrackRecord(
        recent=[
            RecentDecision(
                cycle_id="2026-10-17T12:00Z",
                symbol="xyz:GOLD",
                target_weight=0.10,
                prob_up_24h=0.58,
                thesis="Older cycle that should drop off.",
                price_then=4000.0,
                price_now=4159.6,
            ),
            RecentDecision(
                cycle_id="2026-10-18T00:00Z",
                symbol="BTC",
                target_weight=0.04,
                prob_up_24h=0.55,
                thesis="Bitcoin still above its longer average.",
                price_then=82000.0,
                price_now=81491.0,
            ),
            RecentDecision(
                cycle_id="2026-10-18T12:00Z",
                symbol="xyz:GOLD",
                target_weight=0.12,
                prob_up_24h=0.62,
                thesis="Gold holding above its 20 day average while real yields fall.",
                price_then=4120.0,
                price_now=4159.6,
            ),
            RecentDecision(
                cycle_id="2026-10-18T12:00Z",
                symbol="xyz:SP500",
                target_weight=0.0,
                prob_up_24h=0.5,
                thesis="Flat targets are omitted.",
                price_then=7600.0,
                price_now=7639.4,
            ),
            RecentDecision(
                cycle_id="2026-10-19T08:00Z",
                symbol="BTC",
                target_weight=-0.05,
                prob_up_24h=0.42,
                thesis="Bitcoin slipped under its 20 day average.",
                price_then=81900.0,
                price_now=81491.0,
            ),
        ],
        resolved_forecasts=132,
        brier=0.244,
    )


def _snapshot():
    return make_snapshot(
        now=MOMENT,
        cycle_id="2026-10-19T12:00Z",
        session=SessionState(tradfi_open=True, pre_weekend=False),
        markets=_markets(),
        news=NewsDigest(
            text="Macro\nThe Federal Reserve left rates unchanged.\n",
            sources=["https://example.com"],
            compiled_at=datetime(2026, 10, 19, 11, 58, tzinfo=UTC),
        ),
    )


def _reports() -> dict[str, AnalystReport]:
    return {
        "crypto": AnalystReport(
            summary="Crypto is mixed.",
            views=[
                AnalystView(
                    symbol="BTC",
                    key_points=["Price is under the 20 day average."],
                    risks=["A short squeeze."],
                    stance="bearish",
                    prob_up_24h=0.42,
                )
            ],
        ),
        "equities": AnalystReport(
            summary="Equities are quiet.",
            views=[
                AnalystView(
                    symbol="xyz:SP500",
                    key_points=["The index is above its 20 day average."],
                    risks=["A weekend gap."],
                    stance="bullish",
                    prob_up_24h=0.56,
                )
            ],
        ),
    }


def test_prompts_match_the_plan() -> None:
    assert _plan_fence("`SYSTEM_PM`:") == SYSTEM_PM
    assert _plan_fence("`SYSTEM_ANALYST`:") == SYSTEM_ANALYST
    assert _plan_fence("`SYSTEM_NEWS`:") == SYSTEM_NEWS
    team = SYSTEM_PM + "\n" + _plan_fence("`SYSTEM_TEAM_PM` is `SYSTEM_PM` plus:")
    assert team == SYSTEM_TEAM_PM


def test_system_prompt_inserts_limits() -> None:
    limits = render_limits(make_policy())
    text = SYSTEM_PM.format(limits=limits)
    assert "{limits}" not in text
    assert limits in text
    assert "Analyst reports" in SYSTEM_TEAM_PM.format(limits=limits)


def test_render_limits_states_the_policy() -> None:
    assert render_limits(make_policy()) == "\n".join(
        [
            "Per market weight: -0.20 to +0.20.",
            "Gross exposure at most 1.50.",
            "Net exposure: -1.00 to +1.00.",
            "New risk per cycle at most 0.60.",
            "Minimum trade weight 0.02.",
            "Daily loss limit 4 percent.",
            "Maximum drawdown 20 percent.",
            "Flip cooldown 12 hours.",
            "Pre weekend weight on US hours markets: -0.10 to +0.10.",
            "New exposure is blocked when mark and oracle differ by more than 1 percent.",
            "Stop distance 1.0 to 12.0 percent, default 5.0 percent.",
            "Class gross caps: equity_index 0.60, stock 0.60, commodity 0.40, fx 0.30, crypto 0.40.",
            "Closed markets and the daily loss limit block added exposure.",
        ]
    )


def test_render_limits_keep_fractional_values() -> None:
    text = render_limits(make_policy(daily_loss_limit=0.045, flip_cooldown_hours=1.5))
    assert "Daily loss limit 4.5 percent." in text
    assert "Flip cooldown 1.5 hours." in text


def test_snapshot_matches_golden(update_golden: bool) -> None:
    text = render_snapshot(_snapshot(), _book(), _track())
    _assert_golden("snapshot_single.txt", text, update_golden)


def test_team_user_message_matches_golden(update_golden: bool) -> None:
    snapshot = render_snapshot(_snapshot(), _book(), _track()).rstrip()
    text = snapshot + "\n\n" + render_analyst_reports(_reports())
    _assert_golden("snapshot_team.txt", text, update_golden)


def test_recent_decisions_keep_the_last_three_cycles() -> None:
    text = render_snapshot(_snapshot(), _book(), _track())
    assert "2026-10-17T12:00Z" not in text
    assert "Flat targets are omitted." not in text
    assert "2026-10-18T00:00Z BTC target +0.04" in text
    assert 'target +0.12 p_up 0.62 "Gold holding above its 20 day average while real yields fall."' in text
    assert "2026-10-19T08:00Z BTC target -0.05" in text
    assert text.index("2026-10-18T00:00Z") < text.index("2026-10-18T12:00Z") < text.index("2026-10-19T08:00Z")
    assert "Forecast record: 132 resolved, Brier 0.244 (0.250 is no skill)." in text


def test_multiline_thesis_stays_on_one_line() -> None:
    track = TrackRecord(
        recent=[
            RecentDecision(
                cycle_id="2026-10-19T08:00Z",
                symbol="BTC",
                target_weight=0.05,
                prob_up_24h=0.55,
                thesis="First line.\nSecond line.",
                price_then=100.0,
                price_now=101.0,
            )
        ],
        resolved_forecasts=1,
        brier=0.2,
    )
    text = render_snapshot(make_snapshot(), make_portfolio(), track)
    assert '"First line. Second line."' in text


def test_flat_book_says_there_are_no_positions() -> None:
    portfolio = make_portfolio()
    account = portfolio.account.model_copy(
        update={"positions": {"BTC": Position(symbol="BTC", size=0.0, entry_px=1.0, mark=1.0)}}
    )
    text = render_snapshot(make_snapshot(), portfolio.model_copy(update={"account": account}), _empty_track())
    assert "No open positions." in text
    assert "Gross 0.00 | net +0.00" in text


def test_closed_session_and_missing_brier() -> None:
    snapshot = make_snapshot(session=SessionState(tradfi_open=False, pre_weekend=True), news=None)
    text = render_snapshot(snapshot, make_portfolio(), _empty_track())
    assert "US market session: CLOSED. Pre weekend window: yes." in text
    assert "Brier n/a" in text
    assert "None." in text
    assert "NEWS DIGEST" not in text


def test_missing_prices_render_as_not_available() -> None:
    portfolio = make_portfolio(equity=1000.0, weights={"BTC": 0.1})
    position = portfolio.account.positions["BTC"]
    account = portfolio.account.model_copy(
        update={"positions": {"BTC": position.model_copy(update={"entry_px": 0.0})}}
    )
    track = TrackRecord(
        recent=[
            RecentDecision(
                cycle_id="2026-10-19T08:00Z",
                symbol="BTC",
                target_weight=0.1,
                prob_up_24h=0.6,
                thesis="No reference price.",
                price_then=0.0,
                price_now=10.0,
            )
        ],
        resolved_forecasts=0,
        brier=None,
    )
    text = render_snapshot(make_snapshot(), portfolio.model_copy(update={"account": account}), track)
    assert "n/a" in text
    assert "(n/a)." in text


def test_market_filter_hides_other_symbols() -> None:
    text = render_snapshot(_snapshot(), _book(), _track(), markets=["xyz:GOLD"])
    assert "xyz:GOLD" in text
    assert "xyz:SP500" not in text
    assert "BTC" not in text
    assert "CLOSED" in text


def test_daily_closes_keep_the_last_ten() -> None:
    closes = [float(index) for index in range(12)]
    snapshot = make_snapshot(markets=[make_features("BTC", daily_closes=closes)], news=None)
    text = render_snapshot(snapshot, make_portfolio(), _empty_track())
    assert "BTC: 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00, 11.00" in text
    assert "0.00, 1.00" not in text


def test_analyst_reports_follow_group_order() -> None:
    extra = AnalystReport(summary="Something else.", views=[])
    text = render_analyst_reports({**_reports(), "other": extra})
    assert text.index("equities") < text.index("crypto") < text.index("other")
    assert "1. Price is under the 20 day average." in text
    assert "bearish" in text


def test_empty_analyst_reports() -> None:
    assert render_analyst_reports({}) == "ANALYST REPORTS\nNo analyst reports.\n"


def _empty_track() -> TrackRecord:
    return TrackRecord(recent=[], resolved_forecasts=0, brier=None)
