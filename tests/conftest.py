"""Shared fixtures. The package is imported from ``src`` without needing an install."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from classical_ml import datasets  # noqa: E402


@pytest.fixture(scope="session")
def iris_binary():
    """Two linearly separable Iris species, the setting the perceptron needs."""
    return datasets.iris_two_class()


@pytest.fixture(scope="session")
def wine():
    """Three classes and thirteen features, for the decomposition tests."""
    return datasets.wine_frame()


@pytest.fixture(scope="session")
def wdbc():
    """Breast Cancer Wisconsin, the dataset the diagnosis project runs on."""
    return datasets.breast_cancer()
