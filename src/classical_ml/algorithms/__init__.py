"""Machine learning algorithms implemented from scratch with NumPy.

These are the implementations the projects lean on where a library call would hide the
mechanics. Each one is verified against its scikit-learn equivalent in the test suite,
so the claim that they are correct is checked rather than asserted.
"""

from .decomposition import lda_from_scratch, pca_from_scratch
from .ensemble import MajorityVoteClassifier, ensemble_error
from .feature_selection import SBS
from .linear import AdalineGD, AdalineSGD, LinearRegressionGD, LogisticRegressionGD, Perceptron
from .neural_net import NeuralNetMLP, compute_mse_and_acc, int_to_onehot, minibatch_generator, mse_loss, train
from .stochastic import (
    black_scholes_call,
    markov_stationary_distribution,
    monte_carlo_call_price,
    simulate_gbm,
    simulate_markov_chain,
    simulate_ornstein_uhlenbeck,
    simulate_random_walk,
)
from .timeseries import holt_linear_trend, simple_exponential_smoothing

__all__ = [
    "SBS",
    "AdalineGD",
    "AdalineSGD",
    "LinearRegressionGD",
    "LogisticRegressionGD",
    "MajorityVoteClassifier",
    "NeuralNetMLP",
    "Perceptron",
    "black_scholes_call",
    "compute_mse_and_acc",
    "ensemble_error",
    "holt_linear_trend",
    "int_to_onehot",
    "lda_from_scratch",
    "markov_stationary_distribution",
    "minibatch_generator",
    "monte_carlo_call_price",
    "mse_loss",
    "pca_from_scratch",
    "simple_exponential_smoothing",
    "simulate_gbm",
    "simulate_markov_chain",
    "simulate_ornstein_uhlenbeck",
    "simulate_random_walk",
    "train",
]
