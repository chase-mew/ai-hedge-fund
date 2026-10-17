# pyright: strict
"""Pure return, risk and forecast scores. Rates are fractions."""

from __future__ import annotations

import math
import statistics

from fund.contracts import CalibrationBin

PERIODS_PER_YEAR = 365 * 24 / 4
_MIN_RETURNS = 30


def period_returns(equity: list[float]) -> list[float]:
    """Simple returns between successive equity values."""
    return [equity[index] / equity[index - 1] - 1.0 for index in range(1, len(equity))]


def max_drawdown(equity: list[float]) -> float:
    """Largest peak to trough decline as a positive fraction."""
    peak = 0.0
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0.0:
            worst = max(worst, (peak - value) / peak)
    return worst


def annualised_vol(returns: list[float]) -> float | None:
    """Annualised sample standard deviation, or None before 30 returns."""
    if len(returns) < _MIN_RETURNS:
        return None
    return statistics.stdev(returns) * math.sqrt(PERIODS_PER_YEAR)


def sharpe(returns: list[float]) -> float | None:
    """Annualised Sharpe ratio with a risk free rate of zero."""
    if len(returns) < _MIN_RETURNS:
        return None
    deviation = statistics.stdev(returns)
    if deviation == 0.0:
        return None
    return statistics.fmean(returns) / deviation * math.sqrt(PERIODS_PER_YEAR)


def sortino(returns: list[float]) -> float | None:
    """Annualised Sortino ratio against zero, or None before 30 returns."""
    if len(returns) < _MIN_RETURNS:
        return None
    squared = sum(min(value, 0.0) ** 2 for value in returns)
    downside = math.sqrt(squared / len(returns))
    if downside == 0.0:
        return None
    return statistics.fmean(returns) / downside * math.sqrt(PERIODS_PER_YEAR)


def brier(probs: list[float], outcomes: list[bool]) -> float | None:
    """Mean squared error of probabilities against binary outcomes."""
    if not probs or len(probs) != len(outcomes):
        return None
    total = 0.0
    for prob, outcome in zip(probs, outcomes, strict=True):
        actual = 1.0 if outcome else 0.0
        total += (prob - actual) ** 2
    return total / len(probs)


def brier_skill(brier_score: float) -> float:
    """Skill of a Brier score against a constant one half forecast."""
    return 1.0 - brier_score / 0.25


def calibration(probs: list[float], outcomes: list[bool], bins: int = 10) -> list[CalibrationBin]:
    """Bin probabilities against hit rates. Empty bins are omitted."""
    if not probs or len(probs) != len(outcomes):
        return []
    grouped: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for prob, outcome in zip(probs, outcomes, strict=True):
        grouped[_bin_index(prob, bins)].append((prob, outcome))
    rows: list[CalibrationBin] = []
    width = 1.0 / bins
    for index, bucket in enumerate(grouped):
        if not bucket:
            continue
        count = len(bucket)
        hits = sum(1 for _, outcome in bucket if outcome)
        rows.append(
            CalibrationBin(
                lower=index * width,
                upper=(index + 1) * width,
                mean_prob=sum(prob for prob, _ in bucket) / count,
                hit_rate=hits / count,
                count=count,
            )
        )
    return rows


def _bin_index(prob: float, bins: int) -> int:
    return min(bins - 1, max(0, math.floor(prob * bins)))
