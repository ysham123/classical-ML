"""A majority vote classifier that behaves like a first-class scikit-learn estimator."""

from __future__ import annotations

from math import comb

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.pipeline import _name_estimators
from sklearn.preprocessing import LabelEncoder


class MajorityVoteClassifier(ClassifierMixin, BaseEstimator):
    """A majority vote ensemble that plugs into pipelines, cloning and grid search.

    Args:
        classifiers: The estimators to combine.
        vote: ``classlabel`` takes a plurality vote over predictions, ``probability``
            averages the predicted probabilities, which is recommended for calibrated
            classifiers.
        weights: Optional per-classifier weights.
    """

    def __init__(self, classifiers, vote: str = "classlabel", weights=None) -> None:
        self.classifiers = classifiers
        self.named_classifiers = dict(_name_estimators(classifiers))
        self.vote = vote
        self.weights = weights

    def fit(self, X, y) -> MajorityVoteClassifier:
        if self.vote not in ("probability", "classlabel"):
            raise ValueError(f"vote must be 'probability' or 'classlabel', got '{self.vote}'")
        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError(f"got {len(self.weights)} weights for {len(self.classifiers)} classifiers")

        # LabelEncoder makes the argmax over votes line up with the original class labels.
        self.lablenc_ = LabelEncoder().fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = [clone(clf).fit(X, self.lablenc_.transform(y)) for clf in self.classifiers]
        return self

    def predict(self, X) -> np.ndarray:
        if self.vote == "probability":
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:
            predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T
            maj_vote = np.apply_along_axis(
                lambda row: np.argmax(np.bincount(row, weights=self.weights)), axis=1, arr=predictions
            )
        return self.lablenc_.inverse_transform(maj_vote)

    def predict_proba(self, X) -> np.ndarray:
        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        return np.average(probas, axis=0, weights=self.weights)

    def get_params(self, deep: bool = True) -> dict:
        """Expose the nested estimators so GridSearchCV can address them by name."""
        if not deep:
            return super().get_params(deep=False)
        out = self.named_classifiers.copy()
        for name, step in self.named_classifiers.items():
            for key, value in step.get_params(deep=True).items():
                out[f"{name}__{key}"] = value
        return out


def ensemble_error(n_classifier: int, error: float) -> float:
    """Probability that a majority vote of independent base learners is wrong."""
    k_start = int(np.ceil(n_classifier / 2.0))
    probs = [comb(n_classifier, k) * error**k * (1 - error) ** (n_classifier - k)
             for k in range(k_start, n_classifier + 1)]
    return float(sum(probs))
