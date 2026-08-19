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

__all__ = [
    "SBS",
    "AdalineGD",
    "AdalineSGD",
    "LinearRegressionGD",
    "LogisticRegressionGD",
    "MajorityVoteClassifier",
    "NeuralNetMLP",
    "Perceptron",
    "compute_mse_and_acc",
    "ensemble_error",
    "int_to_onehot",
    "lda_from_scratch",
    "minibatch_generator",
    "mse_loss",
    "pca_from_scratch",
    "train",
]
