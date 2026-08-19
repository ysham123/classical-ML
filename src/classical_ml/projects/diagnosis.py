"""Diagnosing breast cancer from digitised cell nuclei.

The Breast Cancer Wisconsin dataset: 569 fine needle aspirates, 30 features computed
from the cell nuclei in each image, and a malignant or benign diagnosis. The malignant
class is the positive one throughout, because a missed malignancy costs far more than a
false alarm and the metrics should reflect that.

The project runs end to end in seven stages: a baseline, a look at the feature space
with PCA and LDA, feature selection three ways, honest validation, hyperparameter
search, an ensemble comparison, and a single held-out evaluation of the model that won.
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import loguniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.experimental import enable_halving_search_cv  # noqa: F401 - registers halving search
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    HalvingRandomSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
    validation_curve,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .. import datasets
from ..algorithms import SBS, MajorityVoteClassifier, lda_from_scratch, pca_from_scratch
from ..compat import l1_logistic_regression
from ..plotting import PALETTE, plot_decision_regions, save

PROJECT = "diagnosis"
RANDOM_STATE = 1


def _baseline(X_train, y_train, X_test, y_test) -> dict[str, float]:
    """Stage 1. Scale and fit a logistic regression, to have something to beat."""
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000, random_state=RANDOM_STATE))
    pipeline.fit(X_train, y_train)
    score = float(pipeline.score(X_test, y_test))
    print(f"  baseline logistic regression: test accuracy {score:.4f}")
    return {"baseline_test_accuracy": score}


def _feature_space(X_train, y_train, feature_names, target_names) -> dict[str, float]:
    """Stage 2. Compress 30 correlated features into 2, unsupervised and supervised."""
    scaler = StandardScaler().fit(X_train)
    X_std = scaler.transform(X_train)

    w_manual, _, explained = pca_from_scratch(X_std, n_components=2)
    sk_pca = PCA(n_components=2).fit(X_std)
    X_pca = sk_pca.transform(X_std)
    manual_pca = X_std.dot(w_manual)
    agreement = float(abs(np.corrcoef(manual_pca[:, 0], X_pca[:, 0])[0, 1]))
    print(f"  PCA: 2 components hold {explained[:2].sum():.1%} of the variance; "
          f"the from-scratch projection matches scikit-learn at r = {agreement:.4f}")

    _, _, discriminability, _, _ = lda_from_scratch(X_std, y_train, n_components=1)
    X_lda = LinearDiscriminantAnalysis(n_components=1).fit_transform(X_std, y_train)

    # A classifier fitted in the compressed space, to see how much survives the squeeze.
    pca_model = LogisticRegression(max_iter=10000, random_state=RANDOM_STATE).fit(X_pca, y_train)
    pca_cv = cross_val_score(
        make_pipeline(StandardScaler(), PCA(n_components=2),
                      LogisticRegression(max_iter=10000, random_state=RANDOM_STATE)),
        X_train, y_train, cv=10,
    )
    print(f"  logistic regression on 2 principal components: {pca_cv.mean():.4f} +/- {pca_cv.std():.4f} "
          f"(10-fold, versus 30 raw features)")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    axes[0].bar(range(1, 31), explained, alpha=0.65, color=PALETTE[0], label="individual")
    axes[0].step(range(1, 31), np.cumsum(explained), where="mid", color=PALETTE[1], label="cumulative")
    axes[0].set(xlabel="principal component", ylabel="explained variance ratio",
                title="30 features, most of the variance in a few directions")
    axes[0].legend(loc="center right")

    plot_decision_regions(X_pca, y_train, pca_model, ax=axes[1], class_names=list(target_names))
    axes[1].set(xlabel="PC 1", ylabel="PC 2", title="Logistic regression in PCA space")
    axes[1].legend(loc="lower right")

    for index, name in enumerate(target_names):
        axes[2].hist(X_lda[y_train == index].ravel(), bins=30, alpha=0.65,
                     color=PALETTE[index], label=name)
    axes[2].set(xlabel="linear discriminant 1", ylabel="count",
                title="LDA: one supervised direction separates the classes")
    axes[2].legend()
    fig.tight_layout()
    save(fig, PROJECT, "feature_space")

    return {
        "pca_variance_first_two": float(explained[:2].sum()),
        "pca_manual_vs_sklearn_correlation": agreement,
        "pca_logreg_cv_accuracy": float(pca_cv.mean()),
        "lda_discriminability_first": float(discriminability[0]),
    }


def _feature_selection(X_train, y_train, X_test, y_test, feature_names) -> dict[str, float]:
    """Stage 3. Three ways to find the features that matter, with three different biases."""
    scaler = StandardScaler().fit(X_train)
    X_train_std, X_test_std = scaler.transform(X_train), scaler.transform(X_test)

    # An L1 penalty as an embedded method: sparsity falls out of the optimisation. The
    # path stops at C = 100 because these classes are linearly separable, and liblinear's
    # coordinate descent takes minutes to converge once the penalty is weaker than that.
    weights, params, sparsity = [], [], []
    for exponent in np.arange(-4.0, 3.0):
        model = l1_logistic_regression(C=10.0**exponent, random_state=0, max_iter=10000)
        model.fit(X_train_std, y_train)
        weights.append(model.coef_[0])
        params.append(10.0**exponent)
        sparsity.append(int(np.count_nonzero(model.coef_[0])))
    weights = np.array(weights)
    at_c_one = sparsity[params.index(1.0)]
    print(f"  L1 penalty at C = 1: {at_c_one} of 30 coefficients stay nonzero")

    # A wrapper method: sequential backward selection scored by a 5-nearest-neighbor model.
    knn = KNeighborsClassifier(n_neighbors=5)
    sbs = SBS(knn, k_features=1, random_state=RANDOM_STATE).fit(X_train_std, y_train)
    sizes = [len(subset) for subset in sbs.subsets_]
    best_score = max(sbs.scores_)
    smallest_best = min((s for s, sc in zip(sbs.subsets_, sbs.scores_) if sc == best_score), key=len)

    knn.fit(X_train_std, y_train)
    all_features = float(knn.score(X_test_std, y_test))
    knn.fit(X_train_std[:, smallest_best], y_train)
    subset_only = float(knn.score(X_test_std[:, smallest_best], y_test))
    print(f"  SBS: {len(smallest_best)} of 30 features reach the best validation score; "
          f"KNN test accuracy {all_features:.4f} -> {subset_only:.4f}")
    print(f"    kept: {', '.join(feature_names[i] for i in smallest_best)}")

    # A filter method: impurity decrease averaged over a forest.
    forest = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=2)
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    order = np.argsort(importances)[::-1]
    print(f"  most important feature by impurity decrease: {feature_names[order[0]]} "
          f"({importances[order[0]]:.3f})")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    shades = plt.cm.viridis(np.linspace(0.05, 0.9, weights.shape[1]))
    for column in range(weights.shape[1]):
        axes[0].plot(params, weights[:, column], linewidth=1.0, color=shades[column], alpha=0.85)
    axes[0].axhline(0, color="#666666", linestyle="--", linewidth=0.8)
    axes[0].set(xscale="log", xlabel="C (inverse regularization strength)", ylabel="weight coefficient",
                title=f"L1 path: {at_c_one} features survive at C = 1")

    axes[1].plot(sizes, sbs.scores_, marker="o", markersize=3, color=PALETTE[0])
    axes[1].axvline(len(smallest_best), color=PALETTE[1], linestyle="--", linewidth=1,
                    label=f"{len(smallest_best)} features")
    axes[1].set(xlabel="number of features", ylabel="validation accuracy",
                title="Sequential backward selection (KNN)")
    axes[1].legend()

    top = order[:12]
    axes[2].barh(range(len(top)), importances[top][::-1], color=PALETTE[0], height=0.7)
    axes[2].set_yticks(range(len(top)), [feature_names[i] for i in top][::-1], fontsize=7)
    axes[2].set(xlabel="mean impurity decrease", title="Random forest importance (top 12)")
    axes[2].grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, PROJECT, "feature_selection")

    return {
        "l1_nonzero_coefficients_at_c1": at_c_one,
        "sbs_smallest_optimal_subset": len(smallest_best),
        "sbs_best_validation_accuracy": float(best_score),
        "knn_test_accuracy_all_features": all_features,
        "knn_test_accuracy_selected_subset": subset_only,
        "top_feature_by_importance": feature_names[order[0]],
    }


def _validation(X_train, y_train) -> dict[str, float]:
    """Stage 4. Cross-validation, then learning and validation curves to read the fit."""
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000, random_state=RANDOM_STATE))

    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipeline, X_train, y_train, cv=folds, n_jobs=1)
    print(f"  stratified 10-fold cross-validation: {scores.mean():.4f} +/- {scores.std():.4f}")

    train_sizes, train_scores, valid_scores = learning_curve(
        pipeline, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=10
    )
    param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    v_train, v_valid = validation_curve(
        pipeline, X_train, y_train, param_name="logisticregression__C", param_range=param_range, cv=10
    )
    best_c = float(param_range[int(np.argmax(v_valid.mean(axis=1)))])
    gap = float(train_scores.mean(axis=1)[-1] - valid_scores.mean(axis=1)[-1])
    print(f"  learning curve gap at full training size: {gap:.4f}; validation curve peaks at C = {best_c}")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    for ax, x_values, train_s, valid_s, xlabel, title in (
        (axes[0], train_sizes, train_scores, valid_scores, "training examples", "Learning curve"),
        (axes[1], param_range, v_train, v_valid, "C (inverse regularization strength)", "Validation curve"),
    ):
        for values, colour, style, label in (
            (train_s, PALETTE[0], "-", "training"),
            (valid_s, PALETTE[2], "--", "validation"),
        ):
            mean, std = values.mean(axis=1), values.std(axis=1)
            ax.plot(x_values, mean, color=colour, linestyle=style, marker="o", markersize=3, label=label)
            ax.fill_between(x_values, mean + std, mean - std, alpha=0.15, color=colour)
        ax.set(xlabel=xlabel, ylabel="accuracy", ylim=(0.9, 1.005), title=title)
        ax.legend(loc="lower right")
    axes[1].set_xscale("log")
    fig.tight_layout()
    save(fig, PROJECT, "validation_curves")

    return {
        "cv_accuracy": float(scores.mean()),
        "cv_std": float(scores.std()),
        "learning_curve_gap": gap,
        "validation_curve_best_C": best_c,
    }


def _tuning(X_train, y_train) -> tuple[dict, dict[str, float]]:
    """Stage 5. Three searches over one SVM parameter space, then nested CV to choose a family.

    The searches fit a plain ``SVC``: Platt scaling for ``predict_proba`` costs an extra
    internal cross-validation on every fit, and only the single winning model needs it.
    """
    pipeline = make_pipeline(StandardScaler(), SVC(random_state=RANDOM_STATE))
    grid_values = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    param_grid = [
        {"svc__C": grid_values, "svc__kernel": ["linear"]},
        {"svc__C": grid_values, "svc__gamma": grid_values, "svc__kernel": ["rbf"]},
    ]
    distributions = {
        "svc__C": loguniform(0.0001, 1000.0),
        "svc__gamma": loguniform(0.0001, 1000.0),
        "svc__kernel": ["linear", "rbf"],
    }

    searches = {
        "grid": GridSearchCV(pipeline, param_grid, scoring="accuracy", cv=10, n_jobs=2),
        "randomized": RandomizedSearchCV(pipeline, distributions, n_iter=40, scoring="accuracy", cv=10,
                                         random_state=RANDOM_STATE, n_jobs=2),
        "halving": HalvingRandomSearchCV(pipeline, distributions, n_candidates="exhaust", factor=2,
                                         resource="n_samples", random_state=RANDOM_STATE, n_jobs=2),
    }

    metrics: dict[str, float] = {}
    timings: dict[str, float] = {}
    for name, search in searches.items():
        started = time.perf_counter()
        search.fit(X_train, y_train)
        timings[name] = time.perf_counter() - started
        metrics[f"{name}_search_cv_accuracy"] = float(search.best_score_)
        metrics[f"{name}_search_seconds"] = round(timings[name], 2)
        print(f"  {name + ' search':<19} cv accuracy {search.best_score_:.4f} in {timings[name]:.1f}s")
    metrics["grid_search_best_params"] = str(searches["grid"].best_params_)

    # Nested cross-validation: the inner loop tunes, the outer loop estimates, and the two
    # never see the same data, so the comparison between families is not optimistic.
    families = {
        "svm": (make_pipeline(StandardScaler(), SVC(random_state=RANDOM_STATE)),
                [{"svc__C": grid_values, "svc__kernel": ["linear"]},
                 {"svc__C": grid_values, "svc__gamma": grid_values, "svc__kernel": ["rbf"]}]),
        "decision_tree": (DecisionTreeClassifier(random_state=0),
                          [{"max_depth": [1, 2, 3, 4, 5, 6, 7, None]}]),
        "random_forest": (RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
                          [{"n_estimators": [100, 300], "max_features": ["sqrt", 0.5]}]),
        "knn": (make_pipeline(StandardScaler(), KNeighborsClassifier()),
                [{"kneighborsclassifier__n_neighbors": [1, 3, 5, 9, 15]}]),
    }
    nested = {}
    for name, (estimator, grid) in families.items():
        inner = GridSearchCV(estimator, grid, scoring="accuracy", cv=2, n_jobs=2)
        outer = cross_val_score(inner, X_train, y_train, scoring="accuracy", cv=5)
        nested[name] = (float(outer.mean()), float(outer.std()))
        metrics[f"nested_cv_{name}"] = float(outer.mean())
        print(f"  nested CV {name:<15} {outer.mean():.4f} +/- {outer.std():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    names = list(searches)
    axes[0].bar(names, [metrics[f"{n}_search_cv_accuracy"] for n in names], color=PALETTE[0], width=0.55)
    for index, name in enumerate(names):
        axes[0].text(index, metrics[f"{name}_search_cv_accuracy"] + 0.0008,
                     f"{metrics[f'{name}_search_cv_accuracy']:.4f}\n{timings[name]:.1f}s",
                     ha="center", fontsize=8)
    axes[0].set(ylim=(0.95, 1.0), ylabel="best cross-validated accuracy",
                title="Same parameter space, three search strategies")
    axes[0].grid(axis="x", visible=False)

    order = sorted(nested, key=lambda k: nested[k][0])
    axes[1].barh(order, [nested[k][0] for k in order], xerr=[nested[k][1] for k in order],
                 color=PALETTE[0], height=0.6, error_kw={"ecolor": "#444444", "capsize": 3})
    axes[1].set(xlim=(0.85, 1.0), xlabel="nested CV accuracy", title="Algorithm families, nested 5x2 CV")
    axes[1].grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, PROJECT, "tuning")

    return searches["grid"].best_params_, metrics


def _ensembles(X_train, y_train, X_test, y_test) -> dict[str, float]:
    """Stage 6. Does combining models beat the best single one on this dataset?"""
    members = [
        make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=10000, random_state=RANDOM_STATE)),
        DecisionTreeClassifier(max_depth=3, criterion="entropy", random_state=0),
        make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    ]
    tree = DecisionTreeClassifier(criterion="entropy", random_state=RANDOM_STATE)
    candidates = {
        "logistic_regression": members[0],
        "decision_tree": members[1],
        "knn": members[2],
        "majority_vote": MajorityVoteClassifier(classifiers=members, vote="probability"),
        "bagging": BaggingClassifier(estimator=tree, n_estimators=300, random_state=RANDOM_STATE, n_jobs=2),
        "adaboost": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, random_state=0),
                                       n_estimators=300, learning_rate=0.1, random_state=RANDOM_STATE),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3,
                                                        random_state=RANDOM_STATE),
    }

    metrics: dict[str, float] = {}
    summary = {}
    for name, model in candidates.items():
        scores = cross_val_score(model, X_train, y_train, cv=10, scoring="roc_auc", n_jobs=1)
        model.fit(X_train, y_train)
        test_auc = float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
        summary[name] = (float(scores.mean()), float(scores.std()), test_auc)
        metrics[f"{name}_cv_roc_auc"] = float(scores.mean())
        metrics[f"{name}_test_roc_auc"] = test_auc
        print(f"  {name:<20} cv ROC AUC {scores.mean():.4f} +/- {scores.std():.4f}   test {test_auc:.4f}")

    order = sorted(summary, key=lambda k: summary[k][0])
    fig, ax = plt.subplots(figsize=(7, 4))
    positions = np.arange(len(order))
    ax.barh(positions - 0.19, [summary[k][0] for k in order], height=0.36, color=PALETTE[0],
            xerr=[summary[k][1] for k in order], error_kw={"ecolor": "#444444", "capsize": 2},
            label="10-fold CV (training half)")
    ax.barh(positions + 0.19, [summary[k][2] for k in order], height=0.36, color=PALETTE[2],
            label="held-out test set")
    ax.set_yticks(positions, [name.replace("_", " ") for name in order])
    ax.set(xlim=(0.9, 1.005), xlabel="ROC AUC", title="Single models against ensembles")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, PROJECT, "ensembles")

    metrics["best_ensemble_by_cv"] = max(summary, key=lambda k: summary[k][0])
    return metrics


def _final_evaluation(model, X_test, y_test, target_names) -> dict[str, float]:
    """Stage 7. One pass over the held-out set with the model chosen by cross-validation."""
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = matrix.ravel()

    scores = {
        "final_test_accuracy": float((y_pred == y_test).mean()),
        "final_precision": float(precision_score(y_test, y_pred)),
        "final_recall": float(recall_score(y_test, y_pred)),
        "final_f1": float(f1_score(y_test, y_pred)),
        "final_mcc": float(matthews_corrcoef(y_test, y_pred)),
        "final_roc_auc": float(roc_auc_score(y_test, y_score)),
        "final_average_precision": float(average_precision_score(y_test, y_score)),
        "final_false_negatives": int(fn),
        "final_false_positives": int(fp),
    }
    print(f"  held-out set: {tp} malignant caught, {fn} missed, {fp} false alarms out of {len(y_test)} cases")
    print("  " + ", ".join(f"{k.replace('final_', '')} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                           for k, v in scores.items()))

    fpr, tpr, _ = roc_curve(y_test, y_score)
    precision, recall, _ = precision_recall_curve(y_test, y_score)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
    ConfusionMatrixDisplay(matrix, display_labels=target_names).plot(
        ax=axes[0], cmap="Blues", colorbar=False, values_format="d"
    )
    axes[0].set_title("Confusion matrix")
    axes[0].grid(False)
    axes[1].plot(fpr, tpr, color=PALETTE[0], linewidth=1.8, label=f"AUC = {scores['final_roc_auc']:.4f}")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=0.9, label="random guessing")
    axes[1].set(xlabel="false positive rate", ylabel="true positive rate", title="ROC curve")
    axes[1].legend(loc="lower right")
    axes[2].plot(recall, precision, color=PALETTE[1], linewidth=1.8,
                 label=f"average precision = {scores['final_average_precision']:.4f}")
    axes[2].set(xlabel="recall", ylabel="precision", ylim=(0.5, 1.02), title="Precision against recall")
    axes[2].legend(loc="lower left")
    fig.tight_layout()
    save(fig, PROJECT, "final_evaluation")
    return scores


def run() -> dict[str, float]:
    """Run all seven stages and return every number they produced."""
    data = datasets.breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.X, data.y, test_size=0.2, stratify=data.y, random_state=RANDOM_STATE
    )
    print(f"  {len(data.y)} cases, {data.X.shape[1]} features, "
          f"{int(data.y.sum())} malignant ({data.y.mean():.1%})")
    print(f"  split: {len(y_train)} training, {len(y_test)} held out\n")

    metrics: dict[str, float] = {
        "n_samples": int(len(data.y)),
        "n_features": int(data.X.shape[1]),
        "malignant_fraction": float(data.y.mean()),
    }
    metrics.update(_baseline(X_train, y_train, X_test, y_test))
    metrics.update(_feature_space(X_train, y_train, data.feature_names, data.target_names))
    metrics.update(_feature_selection(X_train, y_train, X_test, y_test, data.feature_names))
    metrics.update(_validation(X_train, y_train))
    best_params, tuning_metrics = _tuning(X_train, y_train)
    metrics.update(tuning_metrics)
    metrics.update(_ensembles(X_train, y_train, X_test, y_test))

    # Refit the configuration the grid search chose. An SVM has no native probabilities,
    # and the ROC and precision-recall curves need scores that behave like ones, so the
    # winning SVC is wrapped in a calibrator fitted by internal cross-validation.
    svc_params = {key.removeprefix("svc__"): value for key, value in best_params.items()}
    final_model = make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(SVC(random_state=RANDOM_STATE, **svc_params), ensemble=False, cv=5),
    )
    final_model.fit(X_train, y_train)
    metrics.update(_final_evaluation(final_model, X_test, y_test, data.target_names))
    return metrics
