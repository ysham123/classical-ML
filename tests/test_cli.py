"""The command line surface: listing, validation and the project registry."""

from __future__ import annotations

import importlib

import pytest

from classical_ml import PROJECTS, REQUIRES_DOWNLOAD
from classical_ml.__main__ import main
from classical_ml.report import format_metrics, run_project


def test_every_registered_project_has_a_module_with_run():
    for name in PROJECTS:
        module = importlib.import_module(f"classical_ml.projects.{name}")
        assert callable(getattr(module, "run", None)), f"{name} has no run()"
        assert module.PROJECT == name


def test_projects_needing_a_download_are_declared():
    assert REQUIRES_DOWNLOAD <= set(PROJECTS)
    assert "diagnosis" not in REQUIRES_DOWNLOAD  # ships with scikit-learn


def test_list_flag_prints_all_projects(capsys):
    assert main(["--list"]) == 0
    printed = capsys.readouterr().out
    for name, description in PROJECTS.items():
        assert name in printed
        assert description in printed


def test_unknown_project_is_rejected():
    assert main(["nonsense"]) == 2
    assert main([]) == 1


def test_run_project_rejects_unknown_names():
    with pytest.raises(KeyError):
        run_project("nonsense")


def test_format_metrics_is_compact():
    line = format_metrics({"accuracy": 0.987654321, "epochs": 12, "model": "svm"})
    assert line == "accuracy=0.9877, epochs=12, model=svm"


def test_format_metrics_respects_the_limit():
    assert format_metrics({"a": 1, "b": 2, "c": 3}, limit=2) == "a=1, b=2"
