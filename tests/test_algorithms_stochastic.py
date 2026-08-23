"""The from-scratch stochastic simulations, checked against what theory says they should do."""

from __future__ import annotations

import numpy as np

from classical_ml.algorithms import (
    black_scholes_call,
    markov_stationary_distribution,
    monte_carlo_call_price,
    simulate_gbm,
    simulate_markov_chain,
    simulate_ornstein_uhlenbeck,
    simulate_random_walk,
)


def test_random_walk_variance_grows_linearly_with_time():
    sigma, n_steps, n_paths, dt = 1.0, 2000, 4000, 1.0 / 250
    paths = simulate_random_walk(0.0, sigma, n_steps, dt, n_paths, seed=0)

    t_final = n_steps * dt
    empirical = paths[:, -1].var()
    theory = sigma**2 * t_final
    assert abs(empirical - theory) / theory < 0.1


def test_ornstein_uhlenbeck_variance_converges_to_the_stationary_value():
    sigma, theta, mu, n_steps, n_paths, dt = 1.0, 5.0, 0.0, 2000, 4000, 1.0 / 250
    paths = simulate_ornstein_uhlenbeck(0.0, theta, mu, sigma, n_steps, dt, n_paths, seed=1)

    empirical = paths[:, -1].var()
    theory = sigma**2 / (2 * theta)
    assert abs(empirical - theory) / theory < 0.1


def test_ornstein_uhlenbeck_reverts_to_its_mean_regardless_of_starting_point():
    paths = simulate_ornstein_uhlenbeck(x0=50.0, theta=5.0, mu=0.0, sigma=0.5, n_steps=1000, dt=1.0 / 250,
                                        n_paths=200, seed=2)
    assert abs(paths[:, -1].mean()) < 1.0  # started at 50, should have reverted near 0


def test_simulate_gbm_stays_lognormal_and_positive():
    paths = simulate_gbm(s0=100.0, mu=0.05, sigma=0.2, n_steps=252, dt=1.0 / 252, n_paths=5000, seed=3)
    assert np.all(paths > 0.0)

    terminal = paths[:, -1]
    theoretical_mean = 100.0 * np.exp(0.05 * 1.0)
    assert abs(terminal.mean() - theoretical_mean) / theoretical_mean < 0.05


def test_monte_carlo_call_price_matches_black_scholes():
    s0, k, r, sigma, t = 100.0, 100.0, 0.03, 0.2, 1.0
    closed_form = black_scholes_call(s0, k, r, sigma, t)
    price, standard_error = monte_carlo_call_price(s0, k, r, sigma, t, n_paths=200_000, seed=42)

    assert abs(price - closed_form) < 4 * standard_error


def test_black_scholes_call_is_worth_more_with_higher_volatility():
    low = black_scholes_call(s0=100.0, k=100.0, r=0.03, sigma=0.1, t=1.0)
    high = black_scholes_call(s0=100.0, k=100.0, r=0.03, sigma=0.4, t=1.0)
    assert high > low


def test_markov_chain_settles_on_its_stationary_distribution():
    transition = np.array([[0.9, 0.1], [0.3, 0.7]])
    chain = simulate_markov_chain(transition, initial_state=0, n_steps=50_000, seed=7)

    empirical = np.bincount(chain, minlength=2) / len(chain)
    theoretical = markov_stationary_distribution(transition)

    np.testing.assert_allclose(empirical, theoretical, atol=0.02)


def test_markov_stationary_distribution_is_a_valid_probability_vector():
    transition = np.array([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1], [0.4, 0.4, 0.2]])
    stationary = markov_stationary_distribution(transition)

    assert np.isclose(stationary.sum(), 1.0)
    assert np.all(stationary >= 0.0)
    # the stationary distribution is a fixed point of the transition
    np.testing.assert_allclose(stationary @ transition, stationary, atol=1e-8)
