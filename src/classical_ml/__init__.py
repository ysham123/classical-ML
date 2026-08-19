"""Three classical machine learning projects, plus the algorithms behind them.

Worked from *Machine Learning with PyTorch and Scikit-Learn* (Raschka, Liu, Mirjalili),
chapters 2 to 11. Each project in ``classical_ml.projects`` runs end to end from raw
data to a held-out evaluation and writes its figures to ``outputs/<project>/``. The
models the book builds by hand live in ``classical_ml.algorithms``, where the test suite
checks them against their scikit-learn equivalents.
"""

__version__ = "1.0.0"

PROJECTS = {
    "diagnosis": "Diagnosing breast cancer from cell nucleus measurements",
    "sentiment": "Classifying the sentiment of 50,000 IMDb movie reviews",
    "housing": "Predicting house prices in Ames, Iowa",
}

REQUIRES_DOWNLOAD = {"sentiment", "housing"}
