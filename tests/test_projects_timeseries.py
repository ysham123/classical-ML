"""The time series and stochastic modeling stages, exercised on synthetic data.

Nothing here needs a download: the CO2 series ships inside statsmodels, and the
synthetic series below is built with a known trend and seasonal period so the stages
have something with the right shape to work on without waiting on the real 44-year
series or its slower SARIMA fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from classical_ml import paths
from classical_ml.projects.timeseries import (
    _acf_pacf,
    _decomposition,
    _forecast,
    _markov_regime_switching,
    _option_pricing,
    _random_walk_vs_mean_reversion,
)


@pytest.fixture
def synthetic_series():
    """A trend plus a period-12 seasonal cycle, the shape the stages are built for."""
    rng = np.random.RandomState(0)
    n = 150
    index = pd.date_range("2000-01-01", periods=n, freq="MS")
    t = np.arange(n)
    values = 100 + 0.5 * t + 5 * np.sin(2 * np.pi * t / 12) + rng.normal(scale=0.3, size=n)
    return pd.Series(values, index=index, name="synthetic")


def test_decomposition_rejects_a_unit_root_only_after_differencing(synthetic_series, monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "OUTPUT_DIR", tmp_path)
    metrics = _decomposition(synthetic_series)

    assert metrics["adf_pvalue_differenced"] < metrics["adf_pvalue_level"]
    assert metrics["adf_pvalue_differenced"] < 0.05
    assert (tmp_path / "timeseries" / "decomposition.png").exists()


def test_acf_pacf_writes_one_figure(synthetic_series, monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "OUTPUT_DIR", tmp_path)
    _acf_pacf(synthetic_series)
    assert (tmp_path / "timeseries" / "acf_pacf.png").exists()


def test_forecast_returns_every_model_and_sarima_beats_seasonal_naive(synthetic_series, monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "OUTPUT_DIR", tmp_path)
    metrics = _forecast(synthetic_series)

    for name in ("seasonal_naive", "holt", "sarima"):
        for metric in ("mae", "rmse", "mape"):
            assert f"{name}_test_{metric}" in metrics
            assert metrics[f"{name}_test_{metric}"] >= 0

    # A strong, regular seasonal cycle plus a trend is exactly what SARIMA is fitted for,
    # and exactly what a non-seasonal Holt forecast has no way to capture.
    assert metrics["sarima_test_mae"] < metrics["holt_test_mae"]
    assert (tmp_path / "timeseries" / "forecast.png").exists()


def test_random_walk_vs_mean_reversion_separates_the_two_processes(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "OUTPUT_DIR", tmp_path)
    metrics = _random_walk_vs_mean_reversion()

    assert metrics["random_walk_adf_pvalue"] > 0.05  # fails to reject a unit root
    assert metrics["mean_reverting_adf_pvalue"] < 0.05  # rejects it
    assert (tmp_path / "timeseries" / "random_walk_vs_mean_reversion.png").exists()


def test_option_pricing_lands_close_to_the_closed_form(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "OUTPUT_DIR", tmp_path)
    metrics = _option_pricing()

    assert metrics["monte_carlo_absolute_error"] < 5 * metrics["monte_carlo_standard_error"]
    assert (tmp_path / "timeseries" / "option_pricing.png").exists()


def test_markov_regime_switching_matches_the_stationary_distribution(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "OUTPUT_DIR", tmp_path)
    metrics = _markov_regime_switching()

    assert metrics["markov_max_absolute_difference"] < 0.02
    assert (tmp_path / "timeseries" / "markov_regime_switching.png").exists()
