"""The majority vote classifier must behave like a real scikit-learn estimator."""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from classical_ml import datasets
from classical_ml.algorithms import MajorityVoteClassifier, ensemble_error


def _members():
    return [
        Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(C=0.001, random_state=1))]),
        DecisionTreeClassifier(max_depth=1, criterion="entropy", random_state=0),
        Pipeline([("sc", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=1))]),
    ]


def _split():
    data = datasets.iris_versicolor_virginica()
    return train_test_split(data.X, data.y, test_size=0.5, random_state=1, stratify=data.y)


def test_voting_helps_only_when_the_base_learner_beats_chance():
    assert ensemble_error(n_classifier=11, error=0.25) < 0.25
    assert ensemble_error(n_classifier=11, error=0.75) > 0.75
    assert 0.49 < ensemble_error(n_classifier=11, error=0.5) < 0.51


def test_majority_vote_is_clonable_and_cross_validates():
    X_train, _, y_train, _ = _split()
    ensemble = MajorityVoteClassifier(classifiers=_members())
    clone(ensemble)  # raises if get_params and __init__ disagree

    scores = cross_val_score(ensemble, X_train, y_train, cv=10, scoring="roc_auc")
    assert np.isfinite(scores).all()
    assert scores.mean() > 0.9


def test_majority_vote_beats_its_weakest_member():
    X_train, X_test, y_train, y_test = _split()
    member_scores = [m.fit(X_train, y_train).score(X_test, y_test) for m in _members()]
    ensemble = MajorityVoteClassifier(classifiers=_members()).fit(X_train, y_train)
    assert ensemble.score(X_test, y_test) >= min(member_scores)


def test_probability_and_classlabel_voting_both_work():
    X_train, X_test, y_train, _ = _split()
    by_label = MajorityVoteClassifier(classifiers=_members(), vote="classlabel").fit(X_train, y_train)
    by_proba = MajorityVoteClassifier(classifiers=_members(), vote="probability").fit(X_train, y_train)
    assert by_label.predict(X_test).shape == by_proba.predict(X_test).shape
    assert np.allclose(by_proba.predict_proba(X_test).sum(axis=1), 1.0)


def test_invalid_vote_and_weights_are_rejected():
    X_train, _, y_train, _ = _split()
    for bad in (MajorityVoteClassifier(classifiers=_members(), vote="loudest"),
                MajorityVoteClassifier(classifiers=_members(), weights=[1, 1])):
        try:
            bad.fit(X_train, y_train)
        except ValueError:
            continue
        raise AssertionError(f"{bad.vote}/{bad.weights} should have been rejected")


def test_majority_vote_returns_labels_in_the_original_space():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [0.1, 0.0], [1.1, 1.0]])
    y = np.array(["benign", "malignant", "benign", "malignant"])
    ensemble = MajorityVoteClassifier(classifiers=_members()).fit(X, y)
    assert set(ensemble.predict(X)) <= {"benign", "malignant"}


def test_nested_parameters_are_addressable_for_grid_search():
    params = MajorityVoteClassifier(classifiers=_members()).get_params(deep=True)
    assert "decisiontreeclassifier__max_depth" in params
    assert "pipeline-1__clf__C" in params
