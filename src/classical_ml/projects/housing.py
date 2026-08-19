"""Predicting house prices in Ames, Iowa.

2,930 residential sales between 2006 and 2010, described by size, quality and condition.
The project starts with the exploratory work that decides whether a linear model is
even reasonable here, fits ordinary least squares twice (once by gradient descent to
show the mechanics, once in closed form), then works through the failure modes a linear
fit runs into on real housing data: outliers, heteroscedastic residuals, correlated
predictors, and a relationship that is not straight to begin with.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, RANSACRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor

from .. import datasets
from ..algorithms import LinearRegressionGD
from ..plotting import PALETTE, save

PROJECT = "housing"



def _scatterplot_matrix(frame, columns: list[str]) -> None:
    """Pairwise scatter plots with histograms down the diagonal."""
    size = len(columns)
    fig, axes = plt.subplots(size, size, figsize=(2.0 * size, 1.9 * size))
    for row in range(size):
        for col in range(size):
            ax = axes[row, col]
            if row == col:
                ax.hist(frame[columns[row]], bins=30, color=PALETTE[0], edgecolor="white", linewidth=0.3)
            else:
                ax.scatter(frame[columns[col]], frame[columns[row]], s=3, alpha=0.25, color=PALETTE[0],
                           edgecolor="none")
            if row == size - 1:
                ax.set_xlabel(columns[col], fontsize=7)
            if col == 0:
                ax.set_ylabel(columns[row], fontsize=7)
            ax.tick_params(labelsize=6)
    fig.suptitle("Ames Housing: pairwise relationships", y=0.995)
    fig.tight_layout()
    save(fig, PROJECT, "scatterplot_matrix")


def _correlation_heatmap(frame, columns: list[str]) -> float:
    """Pearson correlations between every pair of columns; returns the top one with price."""
    corr = np.corrcoef(frame[columns].values.T)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)), columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(columns)), columns, fontsize=7)
    for row in range(len(columns)):
        for col in range(len(columns)):
            ax.text(col, row, f"{corr[row, col]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(corr[row, col]) > 0.6 else "#222222")
    ax.set_title("Correlation matrix")
    ax.grid(False)
    fig.colorbar(image, ax=ax, shrink=0.8)
    save(fig, PROJECT, "correlation_heatmap")

    price_row = corr[columns.index("SalePrice")]
    others = [(columns[i], price_row[i]) for i in range(len(columns)) if columns[i] != "SalePrice"]
    return max(others, key=lambda pair: abs(pair[1]))[1]


def _regression_line(ax, X, y, model, xlabel: str, ylabel: str, title: str) -> None:
    ax.scatter(X, y, s=8, alpha=0.35, color=PALETTE[0], edgecolor="none")
    order = np.argsort(X.ravel())
    ax.plot(X.ravel()[order], model.predict(X)[order], color=PALETTE[1], linewidth=2)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)


def run() -> dict[str, float]:
    """Fit and compare eight regressors on Ames Housing sale prices."""
    frame = datasets.ames_housing()
    columns = list(frame.columns)
    print(f"  Ames Housing: {frame.shape[0]} rows, {frame.shape[1]} columns after dropping missing values")

    _scatterplot_matrix(frame, columns)
    strongest = _correlation_heatmap(frame, columns)
    metrics: dict[str, float] = {"strongest_correlation_with_price": float(strongest)}

    X = frame[["Gr Liv Area"]].values
    y = frame["SalePrice"].values

    # --- gradient descent versus the closed form ------------------------------------
    X_std = (X - X.mean()) / X.std()
    y_std = (y - y.mean()) / y.std()
    gd = LinearRegressionGD(eta=0.1, n_iter=20).fit(X_std, y_std)
    slr = LinearRegression().fit(X, y)
    print(f"  gradient descent: slope {gd.w_[0]:.3f} in standardized units after {gd.n_iter} epochs")
    print(f"  closed form:      slope {slr.coef_[0]:.3f} USD per square foot, intercept {slr.intercept_:.2f}")

    ransac = RANSACRegressor(LinearRegression(), max_trials=100, min_samples=0.95,
                             residual_threshold=None, random_state=123).fit(X, y)
    inlier_mask = ransac.inlier_mask_
    print(f"  RANSAC:           slope {ransac.estimator_.coef_[0]:.3f}, "
          f"{inlier_mask.sum()} of {len(y)} points treated as inliers")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(range(1, gd.n_iter + 1), gd.losses_, marker="o", color=PALETTE[0])
    axes[0].set(xlabel="epoch", ylabel="mean squared error", title="Gradient descent convergence")
    _regression_line(axes[1], X, y, slr, "Gr Liv Area [sq ft]", "SalePrice [USD]", "Ordinary least squares")
    axes[2].scatter(X[inlier_mask], y[inlier_mask], s=8, alpha=0.4, color=PALETTE[0], label="inliers")
    axes[2].scatter(X[~inlier_mask], y[~inlier_mask], s=8, alpha=0.6, color=PALETTE[1], label="outliers")
    line_x = np.arange(X.min(), X.max(), 10).reshape(-1, 1)
    axes[2].plot(line_x, ransac.predict(line_x), color="#222222", linewidth=1.8)
    axes[2].set(xlabel="Gr Liv Area [sq ft]", ylabel="SalePrice [USD]", title="RANSAC ignores the outliers")
    axes[2].legend(loc="upper left")
    fig.tight_layout()
    save(fig, PROJECT, "linear_fits")

    metrics.update({
        "gd_standardized_slope": float(gd.w_[0]),
        "ols_slope_usd_per_sqft": float(slr.coef_[0]),
        "ols_intercept": float(slr.intercept_),
        "ransac_slope_usd_per_sqft": float(ransac.estimator_.coef_[0]),
        "ransac_inlier_fraction": float(inlier_mask.mean()),
    })

    # --- multiple regression and residual diagnostics -------------------------------
    target = "SalePrice"
    features = frame.columns[frame.columns != target]
    X_all = frame[features].values
    y_all = frame[target].values
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=123)

    model = LinearRegression().fit(X_train, y_train)
    y_train_pred, y_test_pred = model.predict(X_train), model.predict(X_test)
    print(f"  multiple regression: R2 train {r2_score(y_train, y_train_pred):.3f}, "
          f"test {r2_score(y_test, y_test_pred):.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, pred, true, title in (
        (axes[0], y_train_pred, y_train, "Training residuals"),
        (axes[1], y_test_pred, y_test, "Test residuals"),
    ):
        residuals = pred - true
        ax.scatter(pred, residuals, s=10, alpha=0.35, color=PALETTE[0], edgecolor="none")
        ax.hlines(0, xmin=pred.min() - 10000, xmax=pred.max() + 10000, color="#222222", linewidth=1)
        ax.set(xlabel="predicted price [USD]", title=f"{title} (R2 = {r2_score(true, pred):.3f})")
    axes[0].set_ylabel("residual [USD]")
    fig.tight_layout()
    save(fig, PROJECT, "residuals")

    metrics.update({
        "ols_train_r2": float(r2_score(y_train, y_train_pred)),
        "ols_test_r2": float(r2_score(y_test, y_test_pred)),
        "ols_train_mae": float(mean_absolute_error(y_train, y_train_pred)),
        "ols_test_mae": float(mean_absolute_error(y_test, y_test_pred)),
        "ols_test_mse": float(mean_squared_error(y_test, y_test_pred)),
    })

    # --- regularized linear models ---------------------------------------------------
    regularized = {
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=1.0),
        "elastic_net": ElasticNet(alpha=1.0, l1_ratio=0.5),
    }
    for name, estimator in regularized.items():
        estimator.fit(X_train, y_train)
        score = float(r2_score(y_test, estimator.predict(X_test)))
        metrics[f"{name}_test_r2"] = score
        print(f"  {name:<12} test R2 {score:.3f}")

    # --- polynomial terms ------------------------------------------------------------
    X_quality = frame[["Overall Qual"]].values
    y_quality = frame["SalePrice"].values
    linear = LinearRegression().fit(X_quality, y_quality)
    fit_range = np.arange(X_quality.min() - 1, X_quality.max() + 2, 0.1).reshape(-1, 1)

    fig, ax = plt.subplots(figsize=(5.8, 4))
    ax.scatter(X_quality, y_quality, s=10, alpha=0.25, color="#888888", edgecolor="none", label="training points")
    ax.plot(fit_range, linear.predict(fit_range), color=PALETTE[0], linewidth=1.8,
            label=f"linear (R2 = {r2_score(y_quality, linear.predict(X_quality)):.3f})")
    for degree, colour in ((2, PALETTE[1]), (3, PALETTE[2])):
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X_quality)
        poly_model = LinearRegression().fit(X_poly, y_quality)
        score = float(r2_score(y_quality, poly_model.predict(X_poly)))
        metrics[f"polynomial_degree_{degree}_r2"] = score
        ax.plot(fit_range, poly_model.predict(poly.transform(fit_range)), color=colour, linestyle="--",
                linewidth=1.8, label=f"degree {degree} (R2 = {score:.3f})")
    metrics["polynomial_degree_1_r2"] = float(r2_score(y_quality, linear.predict(X_quality)))
    ax.set(xlabel="Overall Qual", ylabel="SalePrice [USD]", title="Polynomial terms on an ordinal feature")
    ax.legend(loc="upper left", fontsize=7)
    save(fig, PROJECT, "polynomial_regression")

    # --- tree-based regressors -------------------------------------------------------
    tree = DecisionTreeRegressor(max_depth=3).fit(X, y)
    forest = RandomForestRegressor(n_estimators=1000, criterion="squared_error", random_state=1,
                                   n_jobs=2).fit(X_train, y_train)
    forest_train_r2 = float(r2_score(y_train, forest.predict(X_train)))
    forest_test_r2 = float(r2_score(y_test, forest.predict(X_test)))
    print(f"  random forest: R2 train {forest_train_r2:.3f}, test {forest_test_r2:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    _regression_line(axes[0], X, y, tree, "Gr Liv Area [sq ft]", "SalePrice [USD]",
                     "Decision tree regressor (depth 3)")
    axes[1].scatter(forest.predict(X_train), forest.predict(X_train) - y_train, s=8, alpha=0.3,
                    color=PALETTE[0], label="training")
    axes[1].scatter(forest.predict(X_test), forest.predict(X_test) - y_test, s=8, alpha=0.5,
                    color=PALETTE[1], label="test")
    axes[1].hlines(0, xmin=0, xmax=y.max(), color="#222222", linewidth=1)
    axes[1].set(xlabel="predicted price [USD]", ylabel="residual [USD]",
                title=f"Random forest residuals (test R2 = {forest_test_r2:.3f})")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    save(fig, PROJECT, "tree_regressors")

    metrics.update({
        "decision_tree_r2_gr_liv_area": float(r2_score(y, tree.predict(X))),
        "random_forest_train_r2": forest_train_r2,
        "random_forest_test_r2": forest_test_r2,
    })
    return metrics
