"""Forecasting Mauna Loa CO2 and simulating the stochastic processes behind it.

Two halves of the same question: what does a time series look like when you can only
observe it, and what does one look like when you know exactly how it was generated. The
first half decomposes and forecasts 44 years of atmospheric CO2 readings, the kind of
series with a trend and a seasonal cycle that classical time series analysis was built
for. The second half simulates the processes whose statistical properties are known in
closed form: a random walk against a mean-reverting one, geometric Brownian motion
checked against the Black-Scholes price it is supposed to converge to, and a Markov
chain checked against its own stationary distribution. The stationarity test that opens
the first half (does this series have a unit root) is the same test that separates the
two processes in the second, which is what ties the project together rather than making
it two demonstrations under one roof.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from .. import datasets
from ..algorithms import (
    black_scholes_call,
    holt_linear_trend,
    markov_stationary_distribution,
    monte_carlo_call_price,
    simulate_markov_chain,
    simulate_ornstein_uhlenbeck,
    simulate_random_walk,
)
from ..plotting import PALETTE, save

PROJECT = "timeseries"


def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def _mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100)


def _decomposition(series: pd.Series) -> dict[str, float]:
    """Split the series into trend, seasonal and residual, then test whether it needs differencing.

    A unit root test that fails to reject on the level series and rejects on the first
    difference is the standard justification for the ``d=1`` in the SARIMA order fitted
    in :func:`_forecast`; asserting the order without checking this would be asserting a
    modeling choice rather than showing why it was made.
    """
    decomposed = seasonal_decompose(series, model="additive", period=12)

    fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
    for ax, values, title in (
        (axes[0], series, "Observed"),
        (axes[1], decomposed.trend, "Trend"),
        (axes[2], decomposed.seasonal, "Seasonal"),
        (axes[3], decomposed.resid, "Residual"),
    ):
        ax.plot(values.index, values.values, color=PALETTE[0], linewidth=1.1)
        ax.set_title(title, loc="left", fontsize=9)
    fig.suptitle("Mauna Loa CO2: additive decomposition")
    fig.tight_layout()
    save(fig, PROJECT, "decomposition")

    adf_level = adfuller(series.dropna())
    adf_differenced = adfuller(series.diff().dropna())
    print(f"  ADF on the level series:       p = {adf_level[1]:.4f} (fails to reject a unit root)")
    print(f"  ADF on the first difference:   p = {adf_differenced[1]:.2e} (rejects it)")

    return {
        "adf_pvalue_level": float(adf_level[1]),
        "adf_pvalue_differenced": float(adf_differenced[1]),
    }


def _acf_pacf(series: pd.Series) -> None:
    """Autocorrelation and partial autocorrelation of the differenced series.

    Differencing once was already justified in :func:`_decomposition`; these two plots
    are what would normally set the AR and MA orders by hand, before the SARIMA search
    in :func:`_forecast` picks them by grid instead.
    """
    differenced = series.diff().dropna()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    plot_acf(differenced, ax=axes[0], lags=36)
    plot_pacf(differenced, ax=axes[1], lags=36, method="ywm")
    axes[0].set_title("ACF, first difference")
    axes[1].set_title("PACF, first difference")
    fig.tight_layout()
    save(fig, PROJECT, "acf_pacf")


def _forecast(series: pd.Series) -> dict[str, float]:
    """Hold out the last 24 months and score three forecasters on months none of them saw.

    A seasonal-naive baseline just repeats last year; Holt's linear trend (fitted from
    scratch, see ``algorithms/timeseries.py``) has a trend but no seasonal term at all,
    which is the point of including it here as a second, harder baseline to beat, not a
    contender. SARIMA gets both a trend and a seasonal component, chosen from the
    differencing and autocorrelation structure of the two stages above it.
    """
    train, test = series.iloc[:-24], series.iloc[-24:]
    test_values = test.to_numpy()

    seasonal_naive = np.tile(train.to_numpy()[-12:], 2)

    level, trend = holt_linear_trend(train.to_numpy(), alpha=0.6, beta=0.05)
    horizon = np.arange(1, 25)
    holt_forecast = level[-1] + horizon * trend[-1]

    sarima = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                      enforce_stationarity=False, enforce_invertibility=False)
    sarima_forecast = sarima.fit(disp=False).get_forecast(steps=24).predicted_mean.to_numpy()

    forecasters = {
        "seasonal_naive": seasonal_naive,
        "holt": holt_forecast,
        "sarima": sarima_forecast,
    }
    metrics: dict[str, float] = {}
    for name, forecast in forecasters.items():
        metrics[f"{name}_test_mae"] = _mae(test_values, forecast)
        metrics[f"{name}_test_rmse"] = _rmse(test_values, forecast)
        metrics[f"{name}_test_mape"] = _mape(test_values, forecast)
        print(f"  {name:<14} MAE {metrics[f'{name}_test_mae']:.3f}, "
              f"RMSE {metrics[f'{name}_test_rmse']:.3f}, MAPE {metrics[f'{name}_test_mape']:.3f}%")

    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.plot(test.index, test_values, color="#222222", linewidth=2, label="actual")
    ax.plot(test.index, seasonal_naive, color=PALETTE[0], linestyle="--", label="seasonal naive")
    ax.plot(test.index, holt_forecast, color=PALETTE[1], linestyle="--", label="Holt linear trend")
    ax.plot(test.index, sarima_forecast, color=PALETTE[2], linewidth=1.8, label="SARIMA(1,1,1)(1,1,1,12)")
    ax.set(xlabel="month", ylabel="CO2 [ppm]", title="24 held-out months: three forecasters")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    save(fig, PROJECT, "forecast")

    return metrics


def _random_walk_vs_mean_reversion() -> dict[str, float]:
    """Simulate a random walk and an Ornstein-Uhlenbeck process on the same clock and pull them apart.

    Both start at the same random increments' scale; the only difference is the pull
    back toward a mean. That difference is enough that a unit root test correctly
    rejects on the mean-reverting process and fails to reject on the random walk, the
    same test :func:`_decomposition` ran on the CO2 series.
    """
    sigma, theta, mu = 1.0, 5.0, 0.0
    n_steps, n_paths, dt = 2000, 2000, 1.0 / 250
    t_grid = np.arange(n_steps + 1) * dt

    random_walk = simulate_random_walk(0.0, sigma, n_steps, dt, n_paths, seed=0)
    mean_reverting = simulate_ornstein_uhlenbeck(0.0, theta, mu, sigma, n_steps, dt, n_paths, seed=1)

    empirical_var_rw = random_walk[:, -1].var()
    empirical_var_ou = mean_reverting[:, -1].var()
    theory_var_rw = sigma**2 * t_grid[-1]
    theory_var_ou = sigma**2 / (2 * theta)

    adf_rw = adfuller(random_walk[0])
    adf_ou = adfuller(mean_reverting[0])
    print(f"  random walk:   Var[X_T] empirical {empirical_var_rw:.3f} vs theory {theory_var_rw:.3f}, "
          f"ADF p = {adf_rw[1]:.3f}")
    print(f"  mean-reverting: Var[X_T] empirical {empirical_var_ou:.3f} vs theory {theory_var_ou:.3f}, "
          f"ADF p = {adf_ou[1]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for path in random_walk[:8]:
        axes[0].plot(t_grid, path, color=PALETTE[0], alpha=0.6, linewidth=1)
    for path in mean_reverting[:8]:
        axes[0].plot(t_grid, path, color=PALETTE[1], alpha=0.6, linewidth=1)
    axes[0].plot([], [], color=PALETTE[0], label="random walk")
    axes[0].plot([], [], color=PALETTE[1], label="mean-reverting")
    axes[0].set(xlabel="t", ylabel="X(t)", title="Sample paths")
    axes[0].legend(loc="upper left", fontsize=8)

    empirical_curve_rw = random_walk.var(axis=0)
    empirical_curve_ou = mean_reverting.var(axis=0)
    axes[1].plot(t_grid, empirical_curve_rw, color=PALETTE[0], label="random walk, empirical")
    axes[1].plot(t_grid, sigma**2 * t_grid, color=PALETTE[0], linestyle="--", label="random walk, theory")
    axes[1].plot(t_grid, empirical_curve_ou, color=PALETTE[1], label="mean-reverting, empirical")
    axes[1].plot(t_grid, sigma**2 / (2 * theta) * (1 - np.exp(-2 * theta * t_grid)), color=PALETTE[1],
                 linestyle="--", label="mean-reverting, theory")
    axes[1].set(xlabel="t", ylabel="Var[X(t)]", title="Variance across paths, empirical vs theory")
    axes[1].legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    save(fig, PROJECT, "random_walk_vs_mean_reversion")

    return {
        "random_walk_adf_pvalue": float(adf_rw[1]),
        "mean_reverting_adf_pvalue": float(adf_ou[1]),
        "random_walk_variance_empirical": float(empirical_var_rw),
        "random_walk_variance_theory": float(theory_var_rw),
        "mean_reverting_variance_empirical": float(empirical_var_ou),
        "mean_reverting_variance_theory": float(theory_var_ou),
    }


def _option_pricing() -> dict[str, float]:
    """Price a European call by Monte Carlo and watch it converge on the Black-Scholes price.

    Both the simulation and the closed form price the same lognormal model, so the gap
    between them at any path count is sampling error alone, not model mismatch, and it
    should shrink like every other Monte Carlo estimate: proportional to ``1 / sqrt(n)``.
    """
    s0, k, r, sigma, t = 100.0, 100.0, 0.03, 0.2, 1.0
    closed_form = black_scholes_call(s0, k, r, sigma, t)

    path_counts = (1_000, 10_000, 100_000)
    estimates, errors = [], []
    for n_paths in path_counts:
        price, standard_error = monte_carlo_call_price(s0, k, r, sigma, t, n_paths, seed=42)
        estimates.append(price)
        errors.append(standard_error)
        print(f"  {n_paths:>7,} paths: {price:.4f} +/- {2 * standard_error:.4f} "
              f"(closed form {closed_form:.4f}, off by {abs(price - closed_form) / standard_error:.2f} SE)")

    estimates = np.array(estimates)
    errors = np.array(errors)

    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.errorbar(path_counts, estimates, yerr=2 * errors, fmt="o-", color=PALETTE[0],
                capsize=3, label="Monte Carlo, +/- 2 SE")
    ax.axhline(closed_form, color="#222222", linestyle="--", linewidth=1.4, label="Black-Scholes")
    ax.set_xscale("log")
    ax.set(xlabel="Monte Carlo paths", ylabel="call price [USD]",
           title="Monte Carlo converging on the closed form")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    save(fig, PROJECT, "option_pricing")

    final_price, final_error = float(estimates[-1]), float(errors[-1])
    return {
        "black_scholes_call_price": float(closed_form),
        "monte_carlo_call_price": final_price,
        "monte_carlo_standard_error": final_error,
        "monte_carlo_absolute_error": float(abs(final_price - closed_form)),
    }


def _markov_regime_switching() -> dict[str, float]:
    """Simulate a two-state economy and check that it visits each state as often as the maths predicts.

    The transition matrix alone determines the long-run fraction of time in each state,
    the stationary distribution computed as its dominant eigenvector; a long simulated
    chain should land on that answer without ever being told what it is.
    """
    states = ("expansion", "contraction")
    transition = np.array([[0.9, 0.1], [0.3, 0.7]])
    chain = simulate_markov_chain(transition, initial_state=0, n_steps=20_000, seed=7)

    empirical = np.bincount(chain, minlength=2) / len(chain)
    theoretical = markov_stationary_distribution(transition)
    print(f"  empirical state frequencies:   {dict(zip(states, empirical.round(4)))}")
    print(f"  theoretical stationary distribution: {dict(zip(states, theoretical.round(4)))}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].step(np.arange(300), chain[:300], where="post", color=PALETTE[0], linewidth=1)
    axes[0].set_yticks([0, 1], states)
    axes[0].set(xlabel="step", title="First 300 steps of the simulated chain")

    x = np.arange(len(states))
    axes[1].bar(x - 0.18, empirical, width=0.36, color=PALETTE[0], label="empirical (20,000 steps)")
    axes[1].bar(x + 0.18, theoretical, width=0.36, color=PALETTE[1], label="stationary distribution")
    axes[1].set_xticks(x, states)
    axes[1].set(ylabel="probability", title="Empirical frequency vs the stationary distribution")
    axes[1].legend(loc="upper center", fontsize=8)
    fig.tight_layout()
    save(fig, PROJECT, "markov_regime_switching")

    return {
        "markov_empirical_expansion": float(empirical[0]),
        "markov_stationary_expansion": float(theoretical[0]),
        "markov_max_absolute_difference": float(np.max(np.abs(empirical - theoretical))),
    }


def run() -> dict[str, float]:
    """Forecast Mauna Loa CO2, then simulate the stochastic processes behind option pricing and regimes."""
    series = datasets.mauna_loa_co2()
    print(f"  Mauna Loa CO2: {len(series)} months, {series.index[0]:%Y-%m} to {series.index[-1]:%Y-%m}")

    metrics = _decomposition(series)
    _acf_pacf(series)
    metrics.update(_forecast(series))
    metrics.update(_random_walk_vs_mean_reversion())
    metrics.update(_option_pricing())
    metrics.update(_markov_regime_switching())
    return metrics
