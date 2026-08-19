"""Sequential backward selection must shrink the subset without touching the caller's estimator."""

from __future__ import annotations

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from classical_ml.algorithms import SBS


def _split(wine):
    X = wine.iloc[:, 1:].values
    y = wine["Class label"].values
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    return StandardScaler().fit_transform(X_train), y_train


def test_sbs_removes_one_feature_at_a_time(wine):
    X_train, y_train = _split(wine)
    sbs = SBS(KNeighborsClassifier(n_neighbors=5), k_features=1).fit(X_train, y_train)

    assert [len(subset) for subset in sbs.subsets_] == list(range(13, 0, -1))
    assert len(sbs.scores_) == len(sbs.subsets_)
    assert sbs.transform(X_train).shape[1] == 1
    assert sbs.k_score_ == sbs.scores_[-1]


def test_sbs_stops_at_the_requested_size(wine):
    X_train, y_train = _split(wine)
    sbs = SBS(KNeighborsClassifier(n_neighbors=5), k_features=5).fit(X_train, y_train)
    assert len(sbs.indices_) == 5
    assert sbs.transform(X_train).shape[1] == 5


def test_sbs_does_not_fit_the_estimator_it_is_given(wine):
    """SBS clones its estimator, so the caller's instance stays unfitted."""
    X_train, y_train = _split(wine)
    knn = KNeighborsClassifier(n_neighbors=5)
    SBS(knn, k_features=11).fit(X_train, y_train)
    assert not hasattr(knn, "n_samples_fit_")
