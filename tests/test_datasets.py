"""The loaders must hand every project exactly the matrices it expects."""

from __future__ import annotations

import numpy as np
import pytest

from classical_ml import datasets
from classical_ml.paths import DATA_DIR


def test_iris_two_class_is_the_separable_pair(iris_binary):
    assert iris_binary.X.shape == (100, 2)
    assert set(np.unique(iris_binary.y)) == {0, 1}
    assert iris_binary.target_names == ["setosa", "versicolor"]


def test_iris_versicolor_virginica_is_the_overlapping_pair():
    data = datasets.iris_versicolor_virginica()
    assert data.X.shape == (100, 2)
    assert np.bincount(data.y).tolist() == [50, 50]


def test_wine_frame_uses_the_canonical_column_names(wine):
    assert wine.shape == (178, 14)
    assert wine.columns[0] == "Class label"
    assert sorted(wine["Class label"].unique()) == [1, 2, 3]


def test_breast_cancer_marks_malignant_as_the_positive_class(wdbc):
    """scikit-learn labels benign as 1; every metric in the diagnosis project assumes
    the opposite, so the loader flips it."""
    assert wdbc.X.shape == (569, 30)
    assert int(wdbc.y.sum()) == 212  # malignant cases
    assert wdbc.target_names[1] == "malignant"


@pytest.mark.needs_data
@pytest.mark.skipif(not (DATA_DIR / "AmesHousing.txt").exists(), reason="run `make data` first")
def test_ames_housing_is_clean_after_loading():
    frame = datasets.ames_housing()
    assert list(frame.columns) == datasets.AMES_COLUMNS
    assert frame.isnull().sum().sum() == 0
    assert set(frame["Central Air"].unique()) <= {0, 1}


@pytest.mark.needs_data
@pytest.mark.skipif(not (DATA_DIR / "movie_data.csv").exists(), reason="run `make data` first")
def test_imdb_reviews_are_balanced_and_shuffled():
    frame = datasets.imdb_reviews()
    assert frame.shape == (50000, 2)
    assert frame["sentiment"].value_counts().to_dict() == {0: 25000, 1: 25000}
    # A shuffled corpus does not open with 25,000 consecutive labels of one class.
    assert frame["sentiment"].head(1000).nunique() == 2
