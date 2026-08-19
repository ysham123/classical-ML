"""The gradient boosting stage, exercised on synthetic data so no download is needed."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from classical_ml import paths
from classical_ml.projects.housing import _gradient_boosting


@pytest.fixture
def curved_problem():
    """A target a straight line cannot fit: interactions and a squared term."""
    rng = np.random.RandomState(0)
    X = rng.uniform(0, 10, size=(800, 5))
    y = (X[:, 0] ** 2 * 3 + X[:, 1] * X[:, 2] * 2 + X[:, 3] * 5
         + rng.normal(scale=3.0, size=800))
    return train_test_split(X, y, test_size=0.3, random_state=1)


def test_gradient_boosting_stops_early_and_beats_a_linear_fit(curved_problem, monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "OUTPUT_DIR", tmp_path)  # keep the figure out of outputs/
    X_train, X_test, y_train, y_test = curved_problem

    metrics = _gradient_boosting(X_train, y_train, X_test, y_test, [f"f{i}" for i in range(5)])
    linear_r2 = r2_score(y_test, LinearRegression().fit(X_train, y_train).predict(X_test))

    assert metrics["xgboost_test_r2"] > linear_r2
    assert metrics["xgboost_test_r2"] > 0.9
    # Early stopping should fire well before the 2,000 round cap.
    assert 0 < metrics["xgboost_best_iteration"] < 2000
    assert metrics["xgboost_rounds_available"] <= 2000
    assert (tmp_path / "housing" / "gradient_boosting.png").exists()


def test_gradient_boosting_never_sees_the_test_set_while_choosing_its_stopping_round(
    curved_problem, monkeypatch, tmp_path
):
    """The stopping round must come from a fold of the training data, not from y_test.

    Corrupting the test labels after the split cannot change where the booster stopped;
    if it did, the held-out score would not be held out.
    """
    monkeypatch.setattr(paths, "OUTPUT_DIR", tmp_path)
    X_train, X_test, y_train, y_test = curved_problem

    honest = _gradient_boosting(X_train, y_train, X_test, y_test, [f"f{i}" for i in range(5)])
    corrupted = _gradient_boosting(X_train, y_train, X_test, np.random.RandomState(7).permutation(y_test),
                                   [f"f{i}" for i in range(5)])

    assert honest["xgboost_best_iteration"] == corrupted["xgboost_best_iteration"]
    assert honest["xgboost_train_r2"] == corrupted["xgboost_train_r2"]
