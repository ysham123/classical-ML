"""PCA and LDA built by hand have to agree with scikit-learn's versions."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from classical_ml.algorithms import lda_from_scratch, pca_from_scratch


def test_pca_explained_variance_matches_sklearn(wine):
    X = StandardScaler().fit_transform(wine.iloc[:, 1:].values)
    _, _, explained = pca_from_scratch(X, n_components=2)
    assert np.allclose(explained, PCA().fit(X).explained_variance_ratio_, atol=1e-10)


def test_pca_projection_matches_sklearn_up_to_sign(wine):
    X = StandardScaler().fit_transform(wine.iloc[:, 1:].values)
    w, _, _ = pca_from_scratch(X, n_components=2)
    mine, theirs = X.dot(w), PCA(n_components=2).fit_transform(X)
    for component in range(2):
        assert abs(np.corrcoef(mine[:, component], theirs[:, component])[0, 1]) > 0.9999


def test_pca_components_are_orthonormal(wine):
    X = StandardScaler().fit_transform(wine.iloc[:, 1:].values)
    w, _, _ = pca_from_scratch(X, n_components=3)
    assert np.allclose(w.T.dot(w), np.eye(3), atol=1e-10)


def test_lda_finds_exactly_c_minus_one_useful_discriminants(wine):
    """The between-class scatter matrix has rank at most c - 1, so with three classes
    everything past the second discriminant is numerical noise."""
    X = StandardScaler().fit_transform(wine.iloc[:, 1:].values)
    y = wine["Class label"].values
    _, eigen_vals, discriminability, S_W, S_B = lda_from_scratch(X, y)

    assert discriminability[:2].sum() > 0.999
    assert np.allclose(np.abs(eigen_vals[2:]), 0.0, atol=1e-6)
    assert S_W.shape == S_B.shape == (13, 13)


def test_lda_projection_matches_sklearn(wine):
    X = StandardScaler().fit_transform(wine.iloc[:, 1:].values)
    y = wine["Class label"].values
    w, _, _, _, _ = lda_from_scratch(X, y)
    mine = X.dot(w)
    theirs = LinearDiscriminantAnalysis(n_components=2).fit_transform(X, y)
    assert abs(np.corrcoef(mine[:, 0], theirs[:, 0])[0, 1]) > 0.99


def test_lda_separates_classes_better_than_pca_on_the_first_axis(wine):
    """The point of using the labels: LDA's first axis is built to separate, PCA's is not."""
    X = StandardScaler().fit_transform(wine.iloc[:, 1:].values)
    y = wine["Class label"].values

    def separation(projection):
        means = [projection[y == label].mean() for label in np.unique(y)]
        return (max(means) - min(means)) / projection.std()

    pca_axis = X.dot(pca_from_scratch(X, n_components=1)[0])[:, 0]
    lda_axis = X.dot(lda_from_scratch(X, y, n_components=1)[0])[:, 0]
    assert separation(lda_axis) > separation(pca_axis)
