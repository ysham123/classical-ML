"""Shared matplotlib helpers.

``plot_decision_regions`` is the utility the book introduces in chapter 2 and then
reuses for every 2D classifier; the rest is a small house style so the figures from
all ten chapters look like they belong to one report.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures are written to disk, never shown interactively

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from .paths import output_path

PALETTE: tuple[str, ...] = ("#3b6ea5", "#d1495b", "#4e8f6d", "#7d5ba6", "#e0a458")
MARKERS: tuple[str, ...] = ("o", "s", "^", "v", "D")


def use_house_style() -> None:
    """Apply a restrained plot style: light grid, no boxed axes, consistent type."""
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 130,
            "savefig.bbox": "tight",
            "font.size": 9,
            "font.family": "sans-serif",
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 1.6,
            "figure.autolayout": False,
        }
    )


def save(fig: plt.Figure, project: str, name: str) -> Path:
    """Write ``fig`` to ``outputs/<project>/<name>.png`` and close it."""
    path = output_path(project, f"{name}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_decision_regions(
    X: np.ndarray,
    y: np.ndarray,
    classifier,
    *,
    test_idx: Sequence[int] | None = None,
    resolution: float = 0.02,
    ax: plt.Axes | None = None,
    class_names: Sequence[str] | None = None,
) -> plt.Axes:
    """Shade the decision surface of a fitted classifier over two features.

    Args:
        X: Feature matrix with exactly two columns.
        y: Integer class labels.
        classifier: Any fitted object exposing ``predict``.
        test_idx: Row indices to circle as held-out examples.
        resolution: Grid step of the mesh the classifier is evaluated on.
        ax: Axes to draw on; a new figure is created when omitted.
        class_names: Legend labels, defaulting to the raw class values.
    """
    if X.shape[1] != 2:
        raise ValueError(f"decision regions need exactly 2 features, got {X.shape[1]}")

    ax = ax or plt.subplots(figsize=(5, 4))[1]
    classes = np.unique(y)
    cmap = ListedColormap(PALETTE[: len(classes)])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution),
    )
    grid = np.array([xx1.ravel(), xx2.ravel()]).T
    lab = np.asarray(classifier.predict(grid)).reshape(xx1.shape)

    ax.contourf(xx1, xx2, lab, alpha=0.25, cmap=cmap)
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(classes):
        label = class_names[idx] if class_names is not None else f"class {cl}"
        ax.scatter(
            X[y == cl, 0],
            X[y == cl, 1],
            alpha=0.85,
            c=PALETTE[idx % len(PALETTE)],
            marker=MARKERS[idx % len(MARKERS)],
            edgecolor="white",
            linewidth=0.4,
            s=32,
            label=label,
        )

    if test_idx is not None:
        X_test = X[list(test_idx), :]
        ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            facecolors="none",
            edgecolor="#222222",
            linewidth=0.8,
            alpha=0.9,
            s=90,
            label="test set",
        )

    ax.grid(False)
    return ax
