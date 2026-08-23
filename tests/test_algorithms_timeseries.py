"""The from-scratch smoothing recursions, checked against statsmodels' implementations."""

from __future__ import annotations

import numpy as np
from statsmodels.tsa.holtwinters import Holt, SimpleExpSmoothing

from classical_ml.algorithms import holt_linear_trend, simple_exponential_smoothing


def _series():
    rng = np.random.RandomState(3)
    return 50 + np.cumsum(rng.normal(0, 1, size=40))


def test_simple_exponential_smoothing_matches_statsmodels():
    series = _series()
    alpha = 0.35

    mine = simple_exponential_smoothing(series, alpha)
    theirs = SimpleExpSmoothing(
        series, initialization_method="known", initial_level=series[0]
    ).fit(smoothing_level=alpha, optimized=False).fittedvalues

    np.testing.assert_allclose(mine, theirs, atol=1e-9)


def test_holt_linear_trend_matches_statsmodels():
    series = _series()
    alpha, beta = 0.4, 0.15

    level, trend = holt_linear_trend(series, alpha, beta)
    fitted = Holt(
        series, initialization_method="known", initial_level=series[0], initial_trend=series[1] - series[0]
    ).fit(smoothing_level=alpha, smoothing_trend=beta, optimized=False)

    np.testing.assert_allclose(level, fitted.level, atol=1e-9)
    np.testing.assert_allclose(trend, fitted.trend, atol=1e-9)


def test_holt_forecast_extrapolates_the_last_level_and_trend():
    series = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    level, trend = holt_linear_trend(series, alpha=0.8, beta=0.8)

    forecast = level[-1] + np.arange(1, 4) * trend[-1]
    assert forecast[0] < forecast[1] < forecast[2]  # a positive trend keeps extrapolating upward


def test_simple_exponential_smoothing_first_value_is_the_series_own_first_observation():
    series = _series()
    assert simple_exponential_smoothing(series, alpha=0.5)[0] == series[0]
