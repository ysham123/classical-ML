VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: help setup data run test lint clean diagnosis sentiment housing timeseries

help:
	@echo "setup      create the virtual environment and install the package"
	@echo "data       download the Ames Housing and IMDb datasets into data/"
	@echo "run        run all four projects and refresh outputs/"
	@echo "diagnosis  run the breast cancer project only (no download needed)"
	@echo "sentiment  run the IMDb project only"
	@echo "housing    run the Ames Housing project only"
	@echo "timeseries run the time series and stochastic modeling project only (no download needed)"
	@echo "test       run the test suite"
	@echo "lint       run ruff over the package and tests"
	@echo "clean      remove generated figures and caches"

setup:
	python3 -m venv $(VENV)
	$(PIP) install --quiet --upgrade pip
	$(PIP) install --quiet -e ".[dev]"

data:
	$(PYTHON) -c "from classical_ml import datasets; datasets.ames_housing(); datasets.imdb_reviews(); print('datasets cached')"

run:
	$(PYTHON) -m classical_ml --all

diagnosis sentiment housing timeseries:
	$(PYTHON) -m classical_ml $@

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests

clean:
	rm -rf outputs/diagnosis outputs/sentiment outputs/housing .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
