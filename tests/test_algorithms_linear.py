"""The from-scratch linear models, checked against scikit-learn and against theory."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

from classical_ml.algorithms import (
    AdalineGD,
    AdalineSGD,
    LinearRegressionGD,
    LogisticRegressionGD,
    Perceptron,
)


def test_perceptron_converges_on_linearly_separable_data(iris_binary):
    model = Perceptron(eta=0.1, n_iter=10).fit(iris_binary.X, iris_binary.y)
    assert model.errors_[-1] == 0
    assert (model.predict(iris_binary.X) == iris_binary.y).all()


def test_adaline_loss_decreases_monotonically(iris_binary):
    X_std = StandardScaler().fit_transform(iris_binary.X)
    model = AdalineGD(eta=0.5, n_iter=20).fit(X_std, iris_binary.y)
    assert (np.diff(model.losses_) < 0).all()


def test_adaline_diverges_when_the_learning_rate_is_too_large(iris_binary):
    """Unscaled features plus eta = 0.1 is the failure that motivates standardization."""
    model = AdalineGD(eta=0.1, n_iter=15).fit(iris_binary.X, iris_binary.y)
    assert model.losses_[-1] > model.losses_[0]


def test_adaline_sgd_partial_fit_keeps_learning(iris_binary):
    X_std = StandardScaler().fit_transform(iris_binary.X)
    model = AdalineSGD(eta=0.01, n_iter=5, random_state=1).fit(X_std, iris_binary.y)
    before = (model.predict(X_std) == iris_binary.y).mean()
    model.partial_fit(X_std[:10], iris_binary.y[:10])
    assert before > 0.95
    assert (model.predict(X_std) == iris_binary.y).mean() >= before - 0.05


def test_logistic_regression_gd_matches_sklearn(iris_binary):
    X = StandardScaler().fit_transform(iris_binary.X)
    y = iris_binary.y

    # The classes are perfectly separable, so the unpenalized likelihood has no finite
    # maximum and both optimizers keep travelling; 5,000 epochs is where the two weight
    # vectors have settled onto the same direction.
    mine = LogisticRegressionGD(eta=0.3, n_iter=5000, random_state=1).fit(X, y)
    theirs = LogisticRegression(C=1000.0, solver="lbfgs", max_iter=1000).fit(X, y)

    assert (mine.predict(X) == theirs.predict(X)).all()
    cosine = np.dot(mine.w_, theirs.coef_[0]) / (np.linalg.norm(mine.w_) * np.linalg.norm(theirs.coef_[0]))
    assert cosine > 0.99


def test_logistic_loss_decreases_and_stays_positive(iris_binary):
    X = StandardScaler().fit_transform(iris_binary.X)
    model = LogisticRegressionGD(eta=0.3, n_iter=200).fit(X, iris_binary.y)
    assert model.losses_[-1] < model.losses_[0]
    assert model.losses_[-1] > 0


def test_sigmoid_is_bounded_at_extreme_inputs():
    extreme = LogisticRegressionGD().activation(np.array([-1e6, 0.0, 1e6]))
    assert np.all((extreme >= 0.0) & (extreme <= 1.0))
    assert np.isclose(extreme[1], 0.5)


@pytest.fixture(scope="module")
def noisy_line():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(200, 1))
    y = 3.0 * X.ravel() + 1.5 + rng.normal(scale=0.1, size=200)
    return X, y


def test_gradient_descent_matches_the_closed_form(noisy_line):
    X, y = noisy_line
    gd = LinearRegressionGD(eta=0.1, n_iter=500).fit(X, y)
    ols = LinearRegression().fit(X, y)

    assert np.isclose(gd.w_[0], ols.coef_[0], atol=1e-3)
    assert np.isclose(gd.b_[0], ols.intercept_, atol=1e-3)


def test_regression_loss_decreases_every_epoch(noisy_line):
    X, y = noisy_line
    model = LinearRegressionGD(eta=0.05, n_iter=100).fit(X, y)
    assert (np.diff(model.losses_) < 0).all()
