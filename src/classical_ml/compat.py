"""Shims for scikit-learn API changes since the book was published.

The book targets scikit-learn 1.0. Two calls it makes have changed since: the ``penalty``
argument of ``LogisticRegression`` was deprecated in 1.8 in favour of ``l1_ratio``, and
``liblinear`` no longer wraps itself in a one-versus-rest scheme for multiclass problems.
Both are kept in one place so the chapter code stays readable.
"""

from __future__ import annotations

import sklearn
from sklearn.linear_model import LogisticRegression

SKLEARN_VERSION: tuple[int, ...] = tuple(int(part) for part in sklearn.__version__.split(".")[:2])


def l1_logistic_regression(C: float = 1.0, **kwargs) -> LogisticRegression:
    """A binary logistic regression with an L1 penalty, on any supported version."""
    if SKLEARN_VERSION >= (1, 8):
        return LogisticRegression(l1_ratio=1.0, C=C, solver="liblinear", **kwargs)
    return LogisticRegression(penalty="l1", C=C, solver="liblinear", **kwargs)
