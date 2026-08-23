"""Four classical machine learning projects, plus the algorithms behind them.

The first three are worked from *Machine Learning with PyTorch and Scikit-Learn*
(Raschka, Liu, Mirjalili), chapters 2 to 11. The fourth extends past the book into time
series analysis and stochastic modeling. Each project in ``classical_ml.projects`` runs
end to end from raw data to a held-out evaluation and writes its figures to
``outputs/<project>/``. The models built by hand live in ``classical_ml.algorithms``,
where the test suite checks them against their library equivalents.
"""

__version__ = "1.0.0"

PROJECTS = {
    "diagnosis": "Diagnosing breast cancer from cell nucleus measurements",
    "sentiment": "Classifying the sentiment of 50,000 IMDb movie reviews",
    "housing": "Predicting house prices in Ames, Iowa",
    "timeseries": "Forecasting Mauna Loa CO2 and simulating stochastic processes",
}

REQUIRES_DOWNLOAD = {"sentiment", "housing"}
