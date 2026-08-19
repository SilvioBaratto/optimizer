.PHONY: lint format typecheck test coverage coverage-report all clean contract-snapshots

lint:
	ruff check optimizer/ tests/
	ruff format --check optimizer/ tests/

format:
	ruff format optimizer/ tests/

typecheck:
	mypy optimizer/

test:
	pytest tests/ -v --cov=optimizer --cov-report=term-missing --cov-fail-under=90

coverage:
	pytest tests/ --cov=optimizer --cov-report=html --cov-fail-under=90
	@echo "Coverage report generated in htmlcov/index.html"

coverage-report:
	pytest tests/ --cov=optimizer --cov-branch --cov-report=xml --cov-fail-under=90
	python scripts/check_branch_coverage.py coverage.xml 0.80

all: lint typecheck test

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache .coverage htmlcov/ coverage.xml dist/
	find . -type d -name '*.egg-info' -exec rm -rf {} +

contract-snapshots:
	python ingestion/tests/contract/emit_schemas.py
