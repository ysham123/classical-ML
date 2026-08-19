"""Principal component analysis and linear discriminant analysis, from the linear algebra up.

PCA is unsupervised and keeps the directions of greatest variance; LDA uses the labels
and keeps the directions that separate the classes best. Both are implemented here as
an eigendecomposition of an explicitly constructed matrix, and both are verified against
scikit-learn's versions in ``tests/``.
"""

from __future__ import annotations

import numpy as np


def pca_from_scratch(X_std: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run PCA the long way: covariance matrix, eigendecomposition, projection matrix.

    Returns:
        A tuple of (projection matrix of shape (d, n_components), sorted eigenvalues,
        explained variance ratios).
    """
    cov_mat = np.cov(X_std.T)
    eigen_vals, eigen_vecs = np.linalg.eigh(cov_mat)

    order = np.argsort(eigen_vals)[::-1]
    eigen_vals, eigen_vecs = eigen_vals[order], eigen_vecs[:, order]
    explained = eigen_vals / eigen_vals.sum()

    # Each column of w is one principal component, sign-fixed so runs are reproducible.
    w = eigen_vecs[:, :n_components]
    w = w * np.sign(w[np.argmax(np.abs(w), axis=0), range(n_components)])
    return w, eigen_vals, explained


def lda_from_scratch(X_std: np.ndarray, y: np.ndarray, n_components: int = 2):
    """Compute the LDA projection from the within-class and between-class scatter matrices."""
    n_features = X_std.shape[1]
    labels = np.unique(y)
    mean_overall = np.mean(X_std, axis=0).reshape(n_features, 1)

    S_W = np.zeros((n_features, n_features))
    S_B = np.zeros((n_features, n_features))
    for label in labels:
        X_class = X_std[y == label]
        # The scaled within-class scatter is the class covariance matrix.
        S_W += np.cov(X_class.T) * (X_class.shape[0] - 1)
        mean_vec = np.mean(X_class, axis=0).reshape(n_features, 1)
        deviation = mean_vec - mean_overall
        S_B += X_class.shape[0] * deviation.dot(deviation.T)

    eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    eigen_vals, eigen_vecs = eigen_vals.real, eigen_vecs.real
    order = np.argsort(np.abs(eigen_vals))[::-1]
    eigen_vals, eigen_vecs = eigen_vals[order], eigen_vecs[:, order]

    discriminability = np.abs(eigen_vals) / np.abs(eigen_vals).sum()
    w = eigen_vecs[:, :n_components]
    w = w * np.sign(w[np.argmax(np.abs(w), axis=0), range(n_components)])
    return w, eigen_vals, discriminability, S_W, S_B
