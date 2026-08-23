"""Exponential smoothing, written by hand from the recursions Holt-Winters is built on.

Simple exponential smoothing carries a single level forward, weighted between the
newest observation and its own last forecast. Holt's method adds a second state, a
trend, updated the same way: a weighted blend of what just changed and what the trend
already was. Both are checked in ``tests/`` against ``statsmodels.tsa.holtwinters``,
which is the same reason ``algorithms/linear.py`` checks its gradient descent against
scikit-learn's closed form.
"""

from __future__ import annotations

import numpy as np


def simple_exponential_smoothing(series: np.ndarray, alpha: float) -> np.ndarray:
    """One-step-ahead fitted values under a single smoothed level.

    ``fitted[0]`` is the series' own first observation, standing in for a level no data
    has updated yet; every later point is the smoothed forecast made just before it was
    observed, ``fitted[t] = alpha * series[t - 1] + (1 - alpha) * fitted[t - 1]``.
    """
    series = np.asarray(series, dtype=float)
    fitted = np.empty(len(series))
    fitted[0] = series[0]
    for t in range(1, len(series)):
        fitted[t] = alpha * series[t - 1] + (1 - alpha) * fitted[t - 1]
    return fitted


def holt_linear_trend(series: np.ndarray, alpha: float, beta: float) -> tuple[np.ndarray, np.ndarray]:
    """Level and trend state under Holt's linear method, one entry per observation.

    The level and trend are seeded from the first two observations (level at ``series[0]``,
    trend at the first difference), then both update on every later point: the level
    blends the new observation with where the trend said it would be, and the trend
    blends how much the level just moved with where it already thought it was heading.
    Forecasting ``h`` steps past the end is ``level[-1] + h * trend[-1]``, since nothing
    after the last state is left to smooth against.
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    level = np.empty(n)
    trend = np.empty(n)

    level_prev = series[0]
    trend_prev = series[1] - series[0]
    for t in range(n):
        level_t = alpha * series[t] + (1 - alpha) * (level_prev + trend_prev)
        trend_t = beta * (level_t - level_prev) + (1 - beta) * trend_prev
        level[t] = level_t
        trend[t] = trend_t
        level_prev, trend_prev = level_t, trend_t

    return level, trend
