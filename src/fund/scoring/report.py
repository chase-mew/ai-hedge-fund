"""Build score reports from a journal."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from fund.contracts import (
    EquityPoint,
    Forecast,
    ForecastScore,
    PortfolioScore,
    Resolution,
    ScoreReport,
)
from fund.journal.store import Journal, _load_jsonl
from fund.scoring.metrics import (
    annualised_vol,
    brier,
    brier_skill,
    calibration,
    max_drawdown,
    period_returns,
    sharpe,
    sortino,
)


def build_report(journal: Journal, now: datetime) -> ScoreReport:
    """Score every portfolio and forecast config stored in the journal."""
    grouped: dict[str, list[EquityPoint]] = {}
    for point in journal.load_equity():
        grouped.setdefault(point.portfolio_id, []).append(point)
    portfolios = [_portfolio_score(portfolio_id, points) for portfolio_id, points in grouped.items()]
    return ScoreReport(generated_at=now, portfolios=portfolios, forecasts=_forecast_scores(journal))


def _portfolio_score(portfolio_id: str, points: list[EquityPoint]) -> PortfolioScore:
    ordered = sorted(points, key=lambda point: (point.at, point.cycle_id))
    equities = [point.equity for point in ordered]
    returns = period_returns(equities)
    start = equities[0]
    end = equities[-1]
    last = ordered[-1]
    return PortfolioScore(
        portfolio_id=portfolio_id,
        kind=_kind(portfolio_id),
        cycles=len(ordered),
        start_equity=start,
        equity=end,
        total_return=_ratio(end, start),
        return_net_of_llm=_ratio(end - last.llm_cost_cum, start),
        ann_vol=annualised_vol(returns),
        sharpe=sharpe(returns),
        sortino=sortino(returns),
        max_drawdown=max_drawdown(equities),
        fees=last.fees_cum,
        funding=last.funding_cum,
        llm_cost=last.llm_cost_cum,
    )


def _forecast_scores(journal: Journal) -> list[ForecastScore]:
    loaded = _load_jsonl(journal.root / "resolutions.jsonl", Resolution)
    resolutions = {row.forecast_id: row for row in loaded}
    grouped: dict[str, list[tuple[float, bool]]] = {}
    order: list[str] = []
    for forecast in _load_jsonl(journal.root / "forecasts.jsonl", Forecast):
        bucket = grouped.get(forecast.config_id)
        if bucket is None:
            bucket = []
            grouped[forecast.config_id] = bucket
            order.append(forecast.config_id)
        resolution = resolutions.get(forecast.forecast_id)
        if resolution is None:
            continue
        bucket.append((forecast.prob_up, resolution.outcome))
    return [_forecast_score(config_id, grouped[config_id]) for config_id in order]


def _forecast_score(config_id: str, rows: list[tuple[float, bool]]) -> ForecastScore:
    probs = [prob for prob, _ in rows]
    outcomes = [outcome for _, outcome in rows]
    score = brier(probs, outcomes)
    return ForecastScore(
        config_id=config_id,
        resolved=len(rows),
        brier=score,
        brier_skill=None if score is None else brier_skill(score),
        calibration=calibration(probs, outcomes),
    )


def _kind(portfolio_id: str) -> Literal["live", "config", "baseline"]:
    if portfolio_id == "live":
        return "live"
    if portfolio_id.startswith("baseline:"):
        return "baseline"
    return "config"


def _ratio(end: float, start: float) -> float:
    if start == 0.0:
        return 0.0
    return end / start - 1.0
