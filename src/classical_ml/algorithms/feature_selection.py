"""Sequential backward selection, a wrapper method that works with any estimator."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class SBS:
    """Sequential backward selection.

    A greedy wrapper method: starting from all d features, repeatedly drop the single
    feature whose removal costs the least validation performance, until ``k_features``
    remain. Unlike an L1 penalty this works with any estimator, including ones with no
    notion of a coefficient.

    Args:
        estimator: Any classifier; it is cloned so the caller's instance is untouched.
        k_features: Size of the subset to stop at.
        scoring: Callable taking (y_true, y_pred).
        test_size: Fraction of the training data held out to score subsets on.

    Attributes:
        subsets_: Feature index tuples, largest subset first.
        scores_: Validation score of the best subset at each size.
        k_score_: Score of the final subset.
    """

    def __init__(self, estimator, k_features: int, scoring=accuracy_score,
                 test_size: float = 0.25, random_state: int = 1) -> None:
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> SBS:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        self.scores_ = [self._calc_score(X_train, y_train, X_test, y_test, self.indices_)]

        while dim > self.k_features:
            scores, subsets = [], []
            for subset in combinations(self.indices_, r=dim - 1):
                scores.append(self._calc_score(X_train, y_train, X_test, y_test, subset))
                subsets.append(subset)
            best = int(np.argmax(scores))
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            self.scores_.append(scores[best])
            dim -= 1

        self.k_score_ = self.scores_[-1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices) -> float:
        self.estimator.fit(X_train[:, indices], y_train)
        return self.scoring(y_test, self.estimator.predict(X_test[:, indices]))
