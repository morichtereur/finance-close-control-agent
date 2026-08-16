.PHONY: help install data demo eval test lint format typecheck check clean

PY ?= python

help:
	@echo "install    install the package with dev extras"
	@echo "data       generate the synthetic dataset and build the policy index"
	@echo "demo       run one exception end to end with the mock provider"
	@echo "eval       run the labelled evaluation and write results/benchmark.csv"
	@echo "check      lint + typecheck + tests"

install:
	$(PY) -m pip install -e ".[dev]"

data:
	$(PY) -m fcca.generate_data
	$(PY) -m fcca.ingest_policies

demo:
	$(PY) -m fcca.run_case --exception EXC-0001 --provider mock

eval:
	$(PY) -m fcca.evaluate --provider mock

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

format:
	$(PY) -m ruff format .

typecheck:
	$(PY) -m mypy

check: lint typecheck test

clean:
	rm -rf data/raw data/processed data/evaluation .pytest_cache .ruff_cache .mypy_cache
