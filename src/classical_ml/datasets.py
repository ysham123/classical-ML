"""Dataset loaders with an on-disk cache.

Three of these datasets (Iris, Wine, Breast Cancer Wisconsin) ship inside
scikit-learn, so they are loaded from there and reshaped into the matrices each project
and test expects. The two that ship nowhere (Ames Housing and the IMDb review corpus)
are downloaded once into ``data/`` and reused afterwards; ``make data`` fetches both up
front. Setting ``CLASSICAL_ML_OFFLINE=1`` turns a missing download into a clear error
instead of a network call, which is how CI runs.
"""

from __future__ import annotations

import os
import tarfile
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from statsmodels.datasets import co2

from .paths import data_path

AMES_URL = "https://raw.githubusercontent.com/rasbt/machine-learning-book/main/ch09/AmesHousing.txt"
IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

WINE_COLUMNS = [
    "Class label", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium",
    "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins",
    "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline",
]

AMES_COLUMNS = [
    "Overall Qual", "Overall Cond", "Gr Liv Area", "Central Air", "Total Bsmt SF", "SalePrice",
]


class DatasetUnavailableError(RuntimeError):
    """Raised when a dataset is neither cached nor downloadable."""


@dataclass(frozen=True)
class Dataset:
    """A feature matrix with its labels and human-readable names."""

    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    target_names: list[str]


def _download(url: str, destination, description: str):
    """Fetch ``url`` into ``destination`` unless it is already cached."""
    if destination.exists():
        return destination
    if os.environ.get("CLASSICAL_ML_OFFLINE"):
        raise DatasetUnavailableError(
            f"{description} is missing at {destination} and downloads are disabled. Run `make data`."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {description} from {url}")
    tmp = destination.with_suffix(destination.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(destination)
    return destination


# --------------------------------------------------------------------------------------
# Datasets bundled with scikit-learn
# --------------------------------------------------------------------------------------

def iris_two_class() -> Dataset:
    """Setosa versus versicolor on sepal length and petal length (chapter 2)."""
    raw = load_iris()
    X = raw.data[:100, [0, 2]]
    y = np.where(raw.target[:100] == 0, 0, 1)
    return Dataset(X, y, ["sepal length [cm]", "petal length [cm]"], ["setosa", "versicolor"])



def iris_versicolor_virginica() -> Dataset:
    """Versicolor versus virginica on sepal width and petal length (chapter 7).

    The two harder-to-separate species, on the two features that leave enough overlap
    for an ensemble to beat its members.
    """
    raw = load_iris()
    X = raw.data[50:, [1, 2]]
    y = (raw.target[50:] == 2).astype(int)
    return Dataset(X, y, ["sepal width [cm]", "petal length [cm]"], ["versicolor", "virginica"])


def wine_frame() -> pd.DataFrame:
    """The Wine dataset as a DataFrame with the book's column names (chapters 4 and 7)."""
    raw = load_wine()
    frame = pd.DataFrame(raw.data, columns=WINE_COLUMNS[1:])
    frame.insert(0, "Class label", raw.target + 1)
    return frame


def breast_cancer() -> Dataset:
    """Breast Cancer Wisconsin with malignant encoded as the positive class (chapter 6).

    scikit-learn labels benign as 1; the book labels malignant as 1, and every
    precision/recall number in chapter 6 assumes the book's convention.
    """
    raw = load_breast_cancer()
    y = 1 - raw.target
    return Dataset(raw.data, y, list(raw.feature_names), ["benign", "malignant"])


def mauna_loa_co2() -> pd.Series:
    """Weekly Mauna Loa CO2 readings, 1958 to 2001, resampled to a monthly series.

    Ships inside statsmodels rather than scikit-learn, so no download is involved. The
    weekly readings have gaps where a measurement was missed; resampling to monthly
    means and interpolating the handful of remaining gaps leaves a clean series with a
    trend and a strong yearly cycle, which is the point of using it.
    """
    weekly = co2.load_pandas().data["co2"]
    monthly = weekly.resample("MS").mean().interpolate("linear")
    monthly.name = "co2"
    return monthly


# --------------------------------------------------------------------------------------
# Datasets downloaded on demand
# --------------------------------------------------------------------------------------

def ames_housing(columns: list[str] | None = None) -> pd.DataFrame:
    """The Ames Housing dataset, restricted to the six columns used in chapter 9."""
    path = _download(AMES_URL, data_path("AmesHousing.txt"), "Ames Housing")
    frame = pd.read_csv(path, sep="\t")
    frame = frame[columns or AMES_COLUMNS]
    if "Central Air" in frame.columns:
        frame = frame.assign(**{"Central Air": frame["Central Air"].map({"N": 0, "Y": 1})})
    return frame.dropna(axis=0).reset_index(drop=True)


def imdb_reviews() -> pd.DataFrame:
    """50,000 labelled IMDb reviews, assembled once into ``data/movie_data.csv``.

    The upstream archive is 84 MB of individual text files; walking it takes about a
    minute, so the shuffled DataFrame is cached as a single CSV afterwards.
    """
    csv_path = data_path("movie_data.csv")
    if csv_path.exists():
        return pd.read_csv(csv_path, encoding="utf-8").fillna("")

    archive = _download(IMDB_URL, data_path("aclImdb_v1.tar.gz"), "IMDb movie reviews")
    root = data_path("aclImdb")
    if not root.exists():
        print("extracting aclImdb_v1.tar.gz")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(data_path())

    labels = {"pos": 1, "neg": 0}
    rows = []
    for split in ("test", "train"):
        for sentiment in ("pos", "neg"):
            directory = root / split / sentiment
            for file in sorted(directory.iterdir()):
                rows.append((file.read_text(encoding="utf-8"), labels[sentiment]))

    frame = pd.DataFrame(rows, columns=["review", "sentiment"])
    frame = frame.reindex(np.random.RandomState(0).permutation(frame.index)).reset_index(drop=True)
    frame.to_csv(csv_path, index=False, encoding="utf-8")
    return frame
