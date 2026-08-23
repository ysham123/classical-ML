"""Stochastic processes simulated from scratch, checked against what theory predicts.

A random walk and an Ornstein-Uhlenbeck process share the same driving noise but behave
completely differently over time: one process's variance grows without bound, the
other's converges to a fixed value. Geometric Brownian motion is simulated by its exact
solution, not by discretizing the differential equation, which is what lets a Monte
Carlo option price be compared directly against the closed-form Black-Scholes formula
rather than against a solution that is itself only approximate. A Markov chain closes
the set: simulate transitions for long enough and the state frequencies should settle
on the chain's stationary distribution, computed here as the transition matrix's
dominant left eigenvector.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def simulate_random_walk(x0: float, sigma: float, n_steps: int, dt: float, n_paths: int, seed: int) -> np.ndarray:
    """Paths of shape ``(n_paths, n_steps + 1)`` with independent Normal(0, sigma^2 * dt) increments.

    Scaling each increment by ``sqrt(dt)`` puts this on the same continuous-time footing
    as :func:`simulate_ornstein_uhlenbeck`, so the two are comparable at the same ``t``
    rather than merely after the same number of steps.
    """
    rng = np.random.RandomState(seed)
    increments = rng.normal(0.0, sigma * np.sqrt(dt), size=(n_paths, n_steps))
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = x0
    paths[:, 1:] = x0 + np.cumsum(increments, axis=1)
    return paths


def simulate_ornstein_uhlenbeck(
    x0: float, theta: float, mu: float, sigma: float, n_steps: int, dt: float, n_paths: int, seed: int
) -> np.ndarray:
    """Mean-reverting paths via Euler-Maruyama: ``x[t+1] = x[t] + theta * (mu - x[t]) * dt + sigma * sqrt(dt) * eps``.

    Unlike the random walk, this process is pulled back toward ``mu`` at a rate set by
    ``theta``, so its variance stops growing and settles at ``sigma^2 / (2 * theta)``.
    """
    rng = np.random.RandomState(seed)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = x0
    for t in range(n_steps):
        eps = rng.normal(0.0, 1.0, size=n_paths)
        paths[:, t + 1] = paths[:, t] + theta * (mu - paths[:, t]) * dt + sigma * np.sqrt(dt) * eps
    return paths


def simulate_gbm(s0: float, mu: float, sigma: float, n_steps: int, dt: float, n_paths: int, seed: int) -> np.ndarray:
    """Geometric Brownian motion paths from its exact solution, not an Euler discretization.

    ``S_t = S0 * exp((mu - sigma^2 / 2) * t + sigma * W_t)`` is exact for any step size,
    so the only error left in a Monte Carlo estimate built on these paths is sampling
    error, which is what makes the standard-error comparison against Black-Scholes in
    :func:`monte_carlo_call_price` a fair one.
    """
    rng = np.random.RandomState(seed)
    increments = rng.normal(0.0, np.sqrt(dt), size=(n_paths, n_steps))
    brownian = np.cumsum(increments, axis=1)
    t_grid = np.arange(1, n_steps + 1) * dt
    log_paths = np.log(s0) + (mu - 0.5 * sigma**2) * t_grid + sigma * brownian

    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = s0
    paths[:, 1:] = np.exp(log_paths)
    return paths


def black_scholes_call(s0: float, k: float, r: float, sigma: float, t: float) -> float:
    """Closed-form European call price under the same lognormal model :func:`simulate_gbm` draws from."""
    d1 = (np.log(s0 / k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return float(s0 * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2))


def monte_carlo_call_price(
    s0: float, k: float, r: float, sigma: float, t: float, n_paths: int, seed: int
) -> tuple[float, float]:
    """Discounted mean payoff over simulated risk-neutral terminal prices, with its standard error.

    Drift is set to the risk-free rate ``r`` (the risk-neutral measure the closed form
    also prices under), and the whole horizon is one step, since only the terminal price
    of a European option matters.
    """
    terminal = simulate_gbm(s0, r, sigma, 1, t, n_paths, seed)[:, -1]
    discounted_payoff = np.exp(-r * t) * np.maximum(terminal - k, 0.0)
    return float(discounted_payoff.mean()), float(discounted_payoff.std(ddof=1) / np.sqrt(n_paths))


def simulate_markov_chain(transition: np.ndarray, initial_state: int, n_steps: int, seed: int) -> np.ndarray:
    """A length ``n_steps + 1`` sequence of state indices visited under ``transition``."""
    rng = np.random.RandomState(seed)
    n_states = transition.shape[0]
    states = np.empty(n_steps + 1, dtype=int)
    states[0] = initial_state
    for t in range(n_steps):
        states[t + 1] = rng.choice(n_states, p=transition[states[t]])
    return states


def markov_stationary_distribution(transition: np.ndarray) -> np.ndarray:
    """The distribution a long chain settles into: the left eigenvector of ``transition`` for eigenvalue 1."""
    eigenvalues, eigenvectors = np.linalg.eig(transition.T)
    stationary = np.real(eigenvectors[:, np.argmin(np.abs(eigenvalues - 1.0))])
    return stationary / stationary.sum()
