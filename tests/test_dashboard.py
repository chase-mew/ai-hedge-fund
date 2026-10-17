"""Dashboard static site rendering."""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import pytest

from fund.config import load_config
from fund.contracts import (
    Bookkeeping,
    CalibrationBin,
    ConfigRun,
    CycleRecord,
    EquityPoint,
    ForecastScore,
    GatedTarget,
    GateResult,
    GateStatus,
    LlmUsage,
    Mode,
    NewsDigest,
    OrderOutcome,
    PaperAccount,
    PaperPosition,
    PortfolioDecision,
    PortfolioScore,
    ScoreReport,
    SessionState,
    Snapshot,
    StopOrder,
)
from fund.dashboard.build import build_site
from tests.factories import make_decision, make_snapshot

NOW = datetime(2026, 10, 17, 20, 0, tzinfo=UTC)
NEWER = datetime(2026, 10, 17, 16, 0, tzinfo=UTC)
OLDER = datetime(2026, 10, 17, 12, 0, tzinfo=UTC)
CHAMPION = "opus-5.5/single"
PORTFOLIO = "config:opus-5.5/single"
SCRIPT = "Watch <script>alert(1)</script> before adding risk."
LONG_VIEW = "The tape is quiet and the fund keeps risk small. " * 4
HEADINGS = [
    "<h2>Headline numbers</h2>",
    "<h2>Equity</h2>",
    "<h2>Leaderboard</h2>",
    "<h2>Calibration</h2>",
    "<h2>Current positions</h2>",
    "<h2>Latest decision</h2>",
    "<h2>Recent cycles</h2>",
    "<h2>Method and caveats</h2>",
]


class FakeJournal:
    def __init__(
        self,
        cycles: list[CycleRecord] | None = None,
        equity: list[EquityPoint] | None = None,
        scores: ScoreReport | None = None,
        halt: dict[str, str] | None = None,
        paper: dict[str, PaperAccount] | None = None,
        live_bookkeeping: Bookkeeping | None = None,
    ) -> None:
        self.root = Path(".")
        self._cycles = list(cycles or [])
        self._equity = list(equity or [])
        self._scores = scores
        self._halt = halt
        self._paper = dict(paper or {})
        self._live = live_bookkeeping

    def load_cycles(self, limit: int | None = None) -> list[CycleRecord]:
        if limit is None:
            return list(self._cycles)
        return list(self._cycles[-limit:])

    def load_equity(self) -> list[EquityPoint]:
        return list(self._equity)

    def load_scores(self) -> ScoreReport | None:
        return self._scores

    def halt_state(self) -> dict[str, str] | None:
        return self._halt

    def load_paper(self, portfolio_id: str) -> PaperAccount | None:
        return self._paper.get(portfolio_id)

    def load_live_bookkeeping(self) -> Bookkeeping | None:
        return self._live


@pytest.fixture
def built(tmp_path: Path) -> Path:
    out = tmp_path / "site"
    build_site(_rich_journal(tmp_path), load_config(), out, NOW)
    return out


def test_header_shows_identity_mode_halt_and_last_cycle(built: Path) -> None:
    html = (built / "index.html").read_text()
    assert "<h1>AI Hedge Fund</h1>" in html
    assert "Claude proposes, code disposes. Every decision is journaled, scored and published." in html
    assert '<span class="badge">PAPER</span>' in html
    assert "Last cycle 17 Oct 2026, 16:05 UTC" in html
    assert "Halted. Drawdown halt" in html


def test_section_headings_appear_in_order(built: Path) -> None:
    html = (built / "index.html").read_text()
    positions = [html.index(item) for item in HEADINGS]
    assert positions == sorted(positions)


def test_model_script_text_is_escaped(built: Path) -> None:
    escaped = "&lt;script&gt;alert(1)&lt;/script&gt;"
    pages = [
        built / "index.html",
        built / "cycles" / "2026-10-17T16__00Z.html",
    ]
    for path in pages:
        html = path.read_text()
        assert "<script>alert(1)</script>" not in html
        assert escaped in html


def test_headline_uses_published_scores(built: Path) -> None:
    html = (built / "index.html").read_text()
    head = html.split("<h2>Headline numbers</h2>", 1)[1].split("<h2>Equity</h2>", 1)[0]
    assert "$10,120.00" in head
    assert "+1.20%" in head
    assert "S&amp;P 500 return" in head
    assert "+0.80%" in head
    assert "+0.120" in head
    assert "$4.75" in head


def test_leaderboard_sorts_by_net_return(built: Path) -> None:
    html = (built / "index.html").read_text()
    board = html.split("<h2>Leaderboard</h2>", 1)[1].split("<h2>Calibration</h2>", 1)[0]
    order = [
        "config:opus-5.5/single",
        "baseline:sp500",
        "config:haiku-5.5/single",
        "baseline:equal_weight",
    ]
    positions = [board.index(item) for item in order]
    assert positions == sorted(positions)


def test_sharpe_is_withheld_until_thirty_cycles(built: Path) -> None:
    html = (built / "index.html").read_text()
    board = html.split("<h2>Leaderboard</h2>", 1)[1].split("<h2>Calibration</h2>", 1)[0]
    assert "n/a, needs 30 cycles" in board
    assert "1.25" in board


def test_calibration_lists_config_scores(built: Path) -> None:
    html = (built / "index.html").read_text()
    block = html.split("<h2>Calibration</h2>", 1)[1].split("<h2>Current positions</h2>", 1)[0]
    assert 'id="calibration-chart"' in block
    assert "opus-5.5/single" in block
    assert "haiku-5.5/single" in block
    assert "-0.050" in block


def test_current_positions_come_from_the_champion_paper_account(built: Path) -> None:
    html = (built / "index.html").read_text()
    block = html.split("<h2>Current positions</h2>", 1)[1].split("<h2>Latest decision</h2>", 1)[0]
    assert "Champion paper account" in block
    assert "7,600.00" in block
    assert "Bitcoin" in block
    assert "none" in block


def test_recent_cycles_link_to_pages(built: Path) -> None:
    html = (built / "index.html").read_text()
    block = html.split("<h2>Recent cycles</h2>", 1)[1].split("<h2>Method and caveats</h2>", 1)[0]
    assert 'href="cycles/2026-10-17T16__00Z.html"' in block
    assert 'href="cycles/2026-10-17T12__00Z.html"' in block
    assert "The tape is quiet and the fund keeps risk small." in block
    assert "..." in block
    assert "TESTNET" in block
    assert "PAPER" in block


def test_method_section_states_the_experiment_rules(built: Path) -> None:
    html = (built / "index.html").read_text()
    block = html.split("<h2>Method and caveats</h2>", 1)[1]
    assert "forward only" in block
    assert "backtest" in block
    assert "Hyperliquid" in block
    assert "This is an experiment, not investment advice." in block
    assert "Approve" in block


def test_site_writes_pages_style_and_data_file(built: Path) -> None:
    assert (built / "index.html").is_file()
    assert (built / "data.json").is_file()
    assert (built / "cycles" / "2026-10-17T16__00Z.html").is_file()
    assert (built / "cycles" / "2026-10-17T12__00Z.html").is_file()
    source = Path("src/fund/dashboard/static/style.css").read_text()
    assert (built / "style.css").read_text() == source
    assert "1100px" in source
    assert "prefers-color-scheme" in source
    index = (built / "index.html").read_text()
    cycle = (built / "cycles" / "2026-10-17T16__00Z.html").read_text()
    assert 'href="style.css"' in index
    assert 'href="../style.css"' in cycle
    assert "cdn.jsdelivr.net/npm/chart.js@4/" in index


def test_data_json_contains_scores_and_equity(built: Path) -> None:
    payload = json.loads((built / "data.json").read_text())
    assert payload["generated_at"]
    assert payload["scores"]["portfolios"]
    assert payload["equity"][0]["portfolio_id"]


def test_cycle_page_shows_snapshot_news_decisions_and_costs(built: Path) -> None:
    page = (built / "cycles" / "2026-10-17T16__00Z.html").read_text()
    assert "US market session: open." in page
    assert "Pre weekend window: no." in page
    assert "Central banks held rates and oil inventories fell." in page
    assert 'href="https://example.com/fed"' in page
    assert 'href="http://example.com/oil"' in page
    assert 'href="javascript:' not in page
    assert "The dollar softened and equities held the 20 day average." in page
    assert "The second pass cites the same snapshot." in page
    assert "The team call failed." in page
    assert "$2.70" in page
    older = (built / "cycles" / "2026-10-17T12__00Z.html").read_text()
    assert "US market session: closed." in older
    assert "Pre weekend window: yes." in older
    assert "No news digest for this cycle." in older


def test_empty_journal_renders_no_cycles_page(tmp_path: Path) -> None:
    out = tmp_path / "site"
    build_site(FakeJournal(), load_config(), out, NOW)
    html = (out / "index.html").read_text()
    assert "No cycles yet" in html
    assert "This is an experiment, not investment advice." in html
    assert "Leaderboard" not in html
    assert "chart.js" not in html
    payload = json.loads((out / "data.json").read_text())
    assert payload["scores"] is None
    assert payload["equity"] == []
    assert (out / "style.css").is_file()


def test_only_the_latest_thirty_cycles_are_published(tmp_path: Path) -> None:
    snapshot = make_snapshot(now=NOW)
    cycles = []
    for index in range(31):
        moment = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=index)
        cycles.append(
            _cycle(
                moment,
                snapshot,
                cycle_id=moment.strftime("%Y-%m-%dT%H:%MZ"),
            )
        )
    scores = ScoreReport(
        generated_at=NOW,
        portfolios=[_score("baseline:sp500", "baseline", equity=10000, total_return=0, net_return=0)],
        forecasts=[],
    )
    equity = [_point(NOW, "only", PORTFOLIO, 10500)]
    out = tmp_path / "site"
    build_site(FakeJournal(cycles=cycles, equity=equity, scores=scores), load_config(), out, NOW)
    pages = list((out / "cycles").glob("*.html"))
    names = {item.name for item in pages}
    assert len(pages) == 30
    assert "2026-01-01T00__00Z.html" not in names
    assert "2026-01-31T00__00Z.html" in names
    index = (out / "index.html").read_text()
    assert "2026-01-01T00__00Z.html" not in index
    assert "No forecast scores for the champion yet." in index


def test_live_bookkeeping_selects_the_live_account(tmp_path: Path) -> None:
    moment = datetime(2026, 10, 17, 16, tzinfo=UTC)
    snapshot = make_snapshot(now=moment, cycle_id="2026-10-17T16:00Z")
    run = _run(CHAMPION, decision=None, error="No decision")
    cycle = _cycle(moment, snapshot, mode="live", runs=[run])
    live = _account(
        "live",
        cash=5000,
        positions={
            "xyz:GOLD": PaperPosition(
                symbol="xyz:GOLD",
                size=0.1,
                entry_px=9999,
                stop=StopOrder(symbol="xyz:GOLD", side="sell", size=0.1, trigger_px=9000),
            )
        },
        marks={"xyz:GOLD": 10000},
        moment=moment,
    )
    journal = FakeJournal(
        cycles=[cycle],
        paper={"live": live},
        live_bookkeeping=_book(moment, 5000),
        halt={"note": "Manual halt"},
    )
    out = tmp_path / "site"
    build_site(journal, load_config(), out, NOW)
    html = (out / "index.html").read_text()
    assert '<span class="badge">LIVE</span>' in html
    assert "Live account" in html
    assert "9,999.00" in html
    assert "Halted. Manual halt" in html
    assert "No champion decision was recorded for the latest cycle." in html


def test_headline_uses_equity_series_without_scores(tmp_path: Path) -> None:
    moment = datetime(2026, 10, 17, 16, tzinfo=UTC)
    earlier = datetime(2026, 10, 17, 12, tzinfo=UTC)
    snapshot = make_snapshot(now=moment, cycle_id="2026-10-17T16:00Z")
    cycle = _cycle(moment, snapshot, news=_usage(1.75, "news"))
    equity = [
        _point(earlier, "c1", PORTFOLIO, 10000),
        _point(moment, "c2", PORTFOLIO, 11000),
        _point(earlier, "c0", "baseline:random", 0),
    ]
    out = tmp_path / "site"
    build_site(FakeJournal(cycles=[cycle], equity=equity), load_config(), out, NOW)
    html = (out / "index.html").read_text()
    head = html.split("<h2>Headline numbers</h2>", 1)[1].split("<h2>Equity</h2>", 1)[0]
    assert "$11,000.00" in head
    assert "+10.00%" in head
    assert "$1.75" in head
    assert "n/a" in head
    assert "No account snapshot is available." in html
    assert "No scores yet." in html


def test_headline_uses_paper_equity_without_series(tmp_path: Path) -> None:
    moment = datetime(2026, 10, 17, 16, tzinfo=UTC)
    snapshot = make_snapshot(now=moment, cycle_id="2026-10-17T16:00Z")
    cycle = _cycle(moment, snapshot)
    account = _account(PORTFOLIO, cash=12345, positions={}, marks={}, moment=moment)
    scores = ScoreReport(
        generated_at=NOW,
        portfolios=[],
        forecasts=[
            ForecastScore(
                config_id=CHAMPION,
                resolved=0,
                brier=None,
                brier_skill=None,
                calibration=[],
            )
        ],
    )
    journal = FakeJournal(
        cycles=[cycle],
        scores=scores,
        paper={PORTFOLIO: account},
        live_bookkeeping=_book(moment, 1),
        halt={},
    )
    out = tmp_path / "site"
    build_site(journal, load_config(), out, NOW)
    html = (out / "index.html").read_text()
    assert "$12,345.00" in html
    assert "No open positions." in html
    assert "Champion paper account" in html
    assert "A halt is active" in html
    assert "No calibration bins for the champion yet." in html


def test_naive_now_is_rejected(tmp_path: Path) -> None:
    naive = datetime(2026, 10, 17, 20, 0, tzinfo=UTC).replace(tzinfo=None)
    with pytest.raises(ValueError, match="timezone aware"):
        build_site(FakeJournal(), load_config(), tmp_path, naive)


def _rich_journal(root: Path) -> FakeJournal:
    paths = _transcripts(root)
    decision = _scripted(
        make_decision(weights={"xyz:SP500": 0.10, "BTC": -0.05}, probs={"xyz:SP500": 0.62, "BTC": 0.42})
    )
    newer_snapshot = make_snapshot(
        now=NEWER,
        cycle_id="2026-10-17T16:00Z",
        session=SessionState(tradfi_open=True, pre_weekend=False),
        news=NewsDigest(
            text="Central banks held rates and oil inventories fell.",
            sources=[
                "https://example.com/fed",
                "http://example.com/oil",
                "reuters.com",
                "javascript:alert(1)",
            ],
            compiled_at=NEWER,
        ),
    )
    older_snapshot = make_snapshot(
        now=OLDER,
        cycle_id="2026-10-17T12:00Z",
        session=SessionState(tradfi_open=False, pre_weekend=True),
    )
    newer = _cycle(
        NEWER,
        newer_snapshot,
        finished=datetime(2026, 10, 17, 16, 5, tzinfo=UTC),
        news=_usage(0.40, "news"),
        errors=["Marked a data gap"],
        runs=[
            _run(
                "haiku-5.5/single",
                decision=None,
                gate=GateResult(
                    targets=[
                        GatedTarget(
                            symbol="BTC",
                            current_weight=0,
                            requested_weight=-0.05,
                            approved_weight=-0.05,
                            stop_loss_pct=5,
                            status="approved",
                            reasons=[],
                        )
                    ]
                ),
                transcript_path=paths["missing"],
            ),
            _run(
                CHAMPION,
                decision=decision,
                gate=_gate(decision),
                outcomes=[
                    OrderOutcome(
                        symbol="xyz:SP500",
                        side="buy",
                        requested_size=0.13,
                        filled_size=0.13,
                        avg_px=7639.4,
                        fee_usd=0.15,
                        status="filled",
                    ),
                    OrderOutcome(
                        symbol="BTC",
                        side="sell",
                        requested_size=0.02,
                        filled_size=0,
                        avg_px=None,
                        fee_usd=0,
                        status="skipped",
                    ),
                ],
                usage=[_usage(2.25)],
                transcript_path=paths["champion"],
                equity_before=10000,
                equity_after=10120,
            ),
            _run(
                "sonnet-5.5/single",
                decision=make_decision().model_copy(update={"market_view": "A second configuration agrees."}),
                usage=[_usage(0.05)],
                transcript_path=paths["sonnet"],
            ),
            _run(
                "opus-5.5/team", decision=None, error="The team call failed.", transcript_path=paths["team"]
            ),
        ],
    )
    older_decision = make_decision().model_copy(update={"market_view": LONG_VIEW})
    older = _cycle(
        OLDER,
        older_snapshot,
        mode="testnet",
        finished=datetime(2026, 10, 17, 12, 2, tzinfo=UTC),
        runs=[_run(CHAMPION, decision=older_decision)],
    )
    return FakeJournal(
        cycles=[older, newer],
        equity=[
            _point(OLDER, older.cycle_id, PORTFOLIO, 10000),
            _point(NEWER, newer.cycle_id, PORTFOLIO, 10120),
            _point(OLDER, older.cycle_id, "baseline:sp500", 10000),
            _point(NEWER, newer.cycle_id, "baseline:sp500", 10080),
            _point(NEWER, newer.cycle_id, "baseline:equal_weight", 9950),
            _point(OLDER, older.cycle_id, "config:haiku-5.5/single", 10000),
            _point(NEWER, newer.cycle_id, "config:haiku-5.5/single", 10000),
        ],
        scores=_rich_scores(),
        halt={"reason": "Drawdown halt"},
        paper={PORTFOLIO: _champion_paper()},
    )


def _rich_scores() -> ScoreReport:
    return ScoreReport(
        generated_at=NOW,
        portfolios=[
            _score(
                PORTFOLIO,
                "config",
                equity=10120,
                total_return=0.012,
                net_return=0.010,
                drawdown=0.02,
                fees=1.2,
                funding=0.3,
                llm_cost=4.50,
            ),
            _score(
                "baseline:sp500",
                "baseline",
                equity=10080,
                total_return=0.008,
                net_return=0.008,
                cycles=40,
                sharpe=1.25,
                drawdown=0.01,
            ),
            _score(
                "config:haiku-5.5/single",
                "config",
                equity=9990,
                total_return=0,
                net_return=-0.001,
                fees=0.1,
                funding=0.2,
                llm_cost=0.25,
            ),
            _score(
                "baseline:equal_weight",
                "baseline",
                equity=9950,
                total_return=-0.004,
                net_return=-0.004,
                drawdown=0.05,
            ),
        ],
        forecasts=[
            ForecastScore(
                config_id="haiku-5.5/single",
                resolved=4,
                brier=0.31,
                brier_skill=-0.05,
                calibration=[],
            ),
            ForecastScore(
                config_id=CHAMPION,
                resolved=12,
                brier=0.22,
                brier_skill=0.12,
                calibration=[
                    CalibrationBin(lower=0.4, upper=0.5, mean_prob=0.45, hit_rate=0.40, count=5),
                    CalibrationBin(lower=0.5, upper=0.6, mean_prob=0.58, hit_rate=0.62, count=7),
                ],
            ),
            ForecastScore(
                config_id="sonnet-5.5/single",
                resolved=0,
                brier=None,
                brier_skill=None,
                calibration=[],
            ),
        ],
    )


def _champion_paper() -> PaperAccount:
    return _account(
        PORTFOLIO,
        cash=8000,
        positions={
            "xyz:SP500": PaperPosition(
                symbol="xyz:SP500",
                size=0.13,
                entry_px=7600,
                stop=StopOrder(symbol="xyz:SP500", side="sell", size=0.13, trigger_px=7200),
            ),
            "BTC": PaperPosition(symbol="BTC", size=-0.02, entry_px=82000, stop=None),
            "ETH": PaperPosition(symbol="ETH", size=0, entry_px=2400, stop=None),
            "ZZZ": PaperPosition(symbol="ZZZ", size=1, entry_px=0, stop=None),
        },
        marks={"xyz:SP500": 7639.4, "BTC": 80000, "ETH": 2416.5},
        moment=NEWER,
    )


def _transcripts(root: Path) -> dict[str, str]:
    champion = root / "champion.json.gz"
    payload = [
        {
            "response": {
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "The dollar softened and equities held the 20 day average.",
                    },
                    {"type": "thinking", "thinking": "   "},
                    {"type": "thinking", "thinking": 5},
                    {"type": "text", "text": "ok"},
                ]
            }
        }
    ]
    with gzip.open(champion, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    sonnet = root / "sonnet.json"
    sonnet.write_text(
        json.dumps(
            [
                {
                    "response": {
                        "content": [
                            {"type": "thinking", "thinking": "The second pass cites the same snapshot."}
                        ]
                    }
                }
            ]
        ),
        encoding="utf-8",
    )
    (root / "team.json.gz").write_bytes(b"not gzip")
    return {
        "champion": str(champion),
        "sonnet": str(sonnet),
        "team": str(root / "team.json.gz"),
        "missing": str(root / "missing.json.gz"),
    }


def _scripted(decision: PortfolioDecision) -> PortfolioDecision:
    rows = []
    for item in decision.decisions:
        if item.symbol == "xyz:SP500":
            rows.append(item.model_copy(update={"thesis": SCRIPT}))
        else:
            rows.append(item)
    return decision.model_copy(
        update={"market_view": "Equities are firm and the dollar is soft.", "decisions": rows}
    )


def _gate(decision: PortfolioDecision) -> GateResult:
    targets: list[GatedTarget] = []
    for item in decision.decisions:
        if item.symbol == "ETH":
            continue
        status: GateStatus = "approved"
        approved = item.target_weight
        reasons: list[str] = []
        if item.symbol == "xyz:SP500":
            status = "clipped"
            approved = 0.08
            reasons = ["max_weight"]
        targets.append(
            GatedTarget(
                symbol=item.symbol,
                current_weight=0,
                requested_weight=item.target_weight,
                approved_weight=approved,
                stop_loss_pct=item.stop_loss_pct,
                status=status,
                reasons=reasons,
            )
        )
    return GateResult(targets=targets)


def _cycle(
    moment: datetime,
    snapshot: Snapshot,
    *,
    cycle_id: str | None = None,
    mode: Mode = "paper",
    runs: list[ConfigRun] | None = None,
    news: LlmUsage | None = None,
    errors: list[str] | None = None,
    finished: datetime | None = None,
) -> CycleRecord:
    return CycleRecord(
        cycle_id=cycle_id or snapshot.cycle_id,
        mode=mode,
        started_at=moment,
        finished_at=finished or moment,
        snapshot=snapshot,
        news_usage=news,
        runs=runs or [],
        errors=errors or [],
    )


def _run(
    config_id: str,
    *,
    decision: PortfolioDecision | None = None,
    error: str | None = None,
    gate: GateResult | None = None,
    outcomes: list[OrderOutcome] | None = None,
    usage: list[LlmUsage] | None = None,
    transcript_path: str | None = None,
    equity_before: float = 10000,
    equity_after: float = 10000,
) -> ConfigRun:
    portfolio_id = "live" if config_id == "live" else f"config:{config_id}"
    return ConfigRun(
        config_id=config_id,
        portfolio_id=portfolio_id,
        decision=decision,
        error=error,
        gate=gate,
        outcomes=outcomes or [],
        usage=usage or [],
        transcript_path=transcript_path,
        equity_before=equity_before,
        equity_after=equity_after,
    )


def _usage(cost: float, purpose: str = "decide") -> LlmUsage:
    return LlmUsage(
        purpose=purpose,
        model_id="claude-opus-5-5",
        input_tokens=1000,
        output_tokens=200,
        cache_write_tokens=0,
        cache_read_tokens=0,
        web_searches=0,
        cost_usd=cost,
    )


def _point(at: datetime, cycle_id: str, portfolio_id: str, equity: float) -> EquityPoint:
    return EquityPoint(
        at=at,
        cycle_id=cycle_id,
        portfolio_id=portfolio_id,
        equity=equity,
        gross=0,
        net=0,
        fees_cum=0,
        funding_cum=0,
        llm_cost_cum=0,
    )


def _score(
    portfolio_id: str,
    kind: Literal["live", "config", "baseline"],
    *,
    equity: float,
    total_return: float,
    net_return: float,
    cycles: int = 4,
    sharpe: float | None = None,
    drawdown: float = 0,
    fees: float = 0,
    funding: float = 0,
    llm_cost: float = 0,
) -> PortfolioScore:
    return PortfolioScore(
        portfolio_id=portfolio_id,
        kind=kind,
        cycles=cycles,
        start_equity=10000,
        equity=equity,
        total_return=total_return,
        return_net_of_llm=net_return,
        ann_vol=None,
        sharpe=sharpe,
        sortino=None,
        max_drawdown=drawdown,
        fees=fees,
        funding=funding,
        llm_cost=llm_cost,
    )


def _account(
    portfolio_id: str,
    *,
    cash: float,
    positions: dict[str, PaperPosition],
    marks: dict[str, float],
    moment: datetime,
) -> PaperAccount:
    return PaperAccount(
        portfolio_id=portfolio_id,
        cash=cash,
        positions=positions,
        last_marks=marks,
        updated_at=moment,
        bookkeeping=_book(moment, cash),
    )


def _book(moment: datetime, equity: float) -> Bookkeeping:
    return Bookkeeping(day=moment.strftime("%Y-%m-%d"), day_start_equity=equity, high_water_mark=equity)
