"""Filesystem locations shared by every chapter.

Both directories can be relocated with the ``CLASSICAL_ML_DATA`` and
``CLASSICAL_ML_OUTPUTS`` environment variables, which is what CI uses to keep
downloaded corpora out of the checkout.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("CLASSICAL_ML_DATA", PROJECT_ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("CLASSICAL_ML_OUTPUTS", PROJECT_ROOT / "outputs"))


def data_path(*parts: str) -> Path:
    """Return a path inside the data cache, creating the cache directory if needed."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR.joinpath(*parts)


def output_path(project: str, *parts: str) -> Path:
    """Return a path inside ``outputs/<project>/``, creating the directory if needed."""
    directory = OUTPUT_DIR / project
    directory.mkdir(parents=True, exist_ok=True)
    return directory.joinpath(*parts)
