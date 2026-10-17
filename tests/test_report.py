"""Score report assembly from a journal."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from fund.contracts import EquityPoint, Forecast, Resolution
from fund.journal.store import Journal
from fund.scoring.report import build_report

NOW = datetime(2026, 10, 17, 16, 0, tzinfo=UTC)


def test_empty_journal_has_no_scores(tmp_path: Path) -> None:
    report = build_report(Journal(tmp_path), NOW)
    assert report.generated_at == NOW
    assert report.portfolios == []
    assert report.forecasts == []


def _point(
    at: datetime,
    portfolio_id: str,
    equity: float,
    fees: float,
    funding: float,
    llm: float,
) -> EquityPoint:
    return EquityPoint(
        at=at,
        cycle_id=at.strftime("%Y-%m-%dT%H:%MZ"),
        portfolio_id=portfolio_id,
        equity=equity,
        gross=0.0,
        net=0.0,
        fees_cum=fees,
        funding_cum=funding,
        llm_cost_cum=llm,
    )


def _forecast(forecast_id: str, config_id: str, prob: float) -> Forecast:
    made = datetime(2026, 10, 16, 12, 0, tzinfo=UTC)
    return Forecast(
        forecast_id=forecast_id,
        cycle_id="2026-10-16T12:00Z",
        config_id=config_id,
        symbol="BTC",
        made_at=made,
        resolves_at=made + timedelta(hours=24),
        prob_up=prob,
        ref_price=100.0,
    )


def test_build_report_assembles_scores_from_the_journal(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    t0 = datetime(2026, 10, 17, 8, 0, tzinfo=UTC)
    t1 = datetime(2026, 10, 17, 12, 0, tzinfo=UTC)
    t2 = datetime(2026, 10, 17, 16, 0, tzinfo=UTC)
    config = "config:opus-5.5/single"
    journal.append_equity(
        [
            _point(t2, config, 88.0, 0.4, 0.1, 4.0),
            _point(t0, "live", 200.0, 0.0, 0.0, 0.0),
            _point(t0, "baseline:sp500", 0.0, 0.0, 0.0, 5.0),
            _point(t0, config, 100.0, 0.1, 0.0, 1.0),
            _point(t1, "live", 250.0, 1.0, 0.5, 0.0),
            _point(t1, config, 110.0, 0.2, 0.05, 2.0),
        ]
    )
    journal.append_forecasts(
        [
            _forecast("f1", "opus-5.5/single", 1.0),
            _forecast("f4", "shadow", 0.5),
            _forecast("f3", "opus-5.5/single", 0.25),
            _forecast("f2", "opus-5.5/single", 0.0),
            _forecast("f5", "quiet", 0.4),
        ]
    )
    journal.append_resolutions(
        [
            Resolution(forecast_id="f1", resolved_at=NOW, price=110.0, outcome=True),
            Resolution(forecast_id="f2", resolved_at=NOW, price=90.0, outcome=False),
            Resolution(forecast_id="f4", resolved_at=NOW, price=90.0, outcome=False),
            Resolution(forecast_id="orphan", resolved_at=NOW, price=1.0, outcome=True),
        ]
    )

    report = build_report(journal, NOW)
    assert [row.portfolio_id for row in report.portfolios] == [config, "live", "baseline:sp500"]
    scored = report.portfolios[0]
    assert scored.kind == "config"
    assert scored.cycles == 3
    assert scored.start_equity == 100.0
    assert scored.equity == 88.0
    assert scored.total_return == pytest.approx(-0.12)
    assert scored.return_net_of_llm == pytest.approx(-0.16)
    assert scored.ann_vol is None
    assert scored.sharpe is None
    assert scored.sortino is None
    assert scored.max_drawdown == pytest.approx(0.2)
    assert scored.fees == 0.4
    assert scored.funding == 0.1
    assert scored.llm_cost == 4.0

    live = report.portfolios[1]
    assert live.kind == "live"
    assert live.cycles == 2
    assert live.total_return == pytest.approx(0.25)
    assert live.return_net_of_llm == pytest.approx(0.25)
    assert live.max_drawdown == 0.0
    assert live.fees == 1.0
    assert live.funding == 0.5
    assert live.llm_cost == 0.0

    flat = report.portfolios[2]
    assert flat.kind == "baseline"
    assert flat.cycles == 1
    assert flat.start_equity == 0.0
    assert flat.total_return == 0.0
    assert flat.return_net_of_llm == 0.0
    assert flat.llm_cost == 5.0

    assert [row.config_id for row in report.forecasts] == ["opus-5.5/single", "shadow", "quiet"]
    opus, shadow, quiet = report.forecasts
    assert opus.resolved == 2
    assert opus.brier == 0.0
    assert opus.brier_skill == 1.0
    pairs = [(item.lower, item.hit_rate, item.count) for item in opus.calibration]
    assert pairs == [(0.0, 0.0, 1), (0.9, 1.0, 1)]
    assert shadow.resolved == 1
    assert shadow.brier == pytest.approx(0.25)
    assert shadow.brier_skill == pytest.approx(0.0)
    assert shadow.calibration[0].lower == 0.5
    assert shadow.calibration[0].upper == pytest.approx(0.6)
    assert shadow.calibration[0].mean_prob == 0.5
    assert shadow.calibration[0].hit_rate == 0.0
    assert shadow.calibration[0].count == 1
    assert quiet.resolved == 0
    assert quiet.brier is None
    assert quiet.brier_skill is None
    assert quiet.calibration == []
