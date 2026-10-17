"""Hand computed return, risk and Brier metrics."""

import math

import pytest

from fund.contracts import CalibrationBin
from fund.scoring.metrics import (
    PERIODS_PER_YEAR,
    annualised_vol,
    brier,
    brier_skill,
    calibration,
    max_drawdown,
    period_returns,
    sharpe,
    sortino,
)

_SCALE = math.sqrt(PERIODS_PER_YEAR)
_MIXED = [0.02] * 15 + [-0.01] * 15


def test_periods_per_year_is_four_hour_slots() -> None:
    assert PERIODS_PER_YEAR == 365 * 24 / 4 == 2190


def test_period_returns_are_simple_successive_changes() -> None:
    assert period_returns([]) == []
    assert period_returns([100.0]) == []
    assert period_returns([100.0, 110.0, 99.0]) == pytest.approx([0.1, -0.1])
    assert period_returns([50.0, 50.0]) == pytest.approx([0.0])


def test_max_drawdown_of_a_known_twenty_percent_drop() -> None:
    assert max_drawdown([100.0, 150.0, 120.0]) == pytest.approx(0.2)
    assert max_drawdown([100.0, 80.0, 200.0, 180.0]) == pytest.approx(0.2)


def test_max_drawdown_of_a_constant_series() -> None:
    assert max_drawdown([100.0, 100.0, 100.0, 100.0]) == 0.0
    assert max_drawdown([]) == 0.0
    assert max_drawdown([0.0, 0.0]) == 0.0


def test_vol_and_sharpe_match_a_hand_computed_series() -> None:
    assert sum(_MIXED) / len(_MIXED) == pytest.approx(0.005)
    variance = (30 * (0.015**2)) / 29
    deviation = math.sqrt(variance)
    assert annualised_vol(_MIXED) == pytest.approx(deviation * _SCALE)
    assert sharpe(_MIXED) == pytest.approx(0.005 / deviation * _SCALE)


def test_vol_and_sharpe_are_none_before_thirty_returns() -> None:
    short = [0.01] * 29
    assert annualised_vol(short) is None
    assert sharpe(short) is None
    assert annualised_vol([]) is None


def test_sharpe_is_none_when_volatility_is_zero() -> None:
    flat = [0.01] * 30
    assert annualised_vol(flat) == 0.0
    assert sharpe(flat) is None


def test_sortino_matches_a_hand_computed_series() -> None:
    downside = math.sqrt((15 * (0.01**2)) / 30)
    assert sortino(_MIXED) == pytest.approx(0.005 / downside * _SCALE)
    assert sortino([-0.01] * 30) == pytest.approx(-_SCALE)


def test_sortino_is_none_without_enough_returns_or_downside() -> None:
    assert sortino([0.01] * 29) is None
    assert sortino([0.01] * 30) is None


def test_brier_for_perfect_worst_and_even_forecasts() -> None:
    outcomes = [True, False, True]
    assert brier([1.0, 0.0, 1.0], outcomes) == 0.0
    assert brier([0.0, 1.0, 0.0], outcomes) == 1.0
    assert brier([0.5, 0.5, 0.5], outcomes) == 0.25


def test_brier_skill_is_one_minus_score_over_a_quarter() -> None:
    assert brier_skill(0.0) == 1.0
    assert brier_skill(0.25) == 0.0
    assert brier_skill(1.0) == -3.0


def test_brier_is_none_without_a_complete_pair() -> None:
    assert brier([], []) is None
    assert brier([0.5], []) is None
    assert brier([0.5], [True, False]) is None


def test_calibration_omits_empty_bins() -> None:
    bins = calibration([0.0, 0.12, 0.18, 1.0], [False, True, False, True])
    assert bins == [
        CalibrationBin(lower=0.0, upper=0.1, mean_prob=0.0, hit_rate=0.0, count=1),
        CalibrationBin(lower=0.1, upper=0.2, mean_prob=0.15, hit_rate=0.5, count=2),
        CalibrationBin(lower=0.9, upper=1.0, mean_prob=1.0, hit_rate=1.0, count=1),
    ]
    edges = calibration([0.0, 0.1, 1.0], [False, True, True])
    assert [(item.lower, item.upper, item.count) for item in edges] == [
        (0.0, 0.1, 1),
        (0.1, 0.2, 1),
        (0.9, 1.0, 1),
    ]
    assert calibration([], []) == []
    assert calibration([0.2], []) == []
    assert calibration([0.5], [True, False]) == []
