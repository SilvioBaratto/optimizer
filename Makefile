.PHONY: sync lint format typecheck test coverage coverage-report all clean contract-snapshots

# uv workspace: one venv for portopt-core + ingestion + packages/portopt-db.
sync:
	uv sync --all-packages --all-extras

lint:
	uv run ruff check optimizer/ tests/
	uv run ruff format --check optimizer/ tests/

format:
	uv run ruff format optimizer/ tests/

typecheck:
	uv run mypy optimizer/

test:
	uv run pytest tests/ -v --cov=optimizer --cov-report=term-missing --cov-fail-under=90

coverage:
	uv run pytest tests/ --cov=optimizer --cov-report=html --cov-fail-under=90
	@echo "Coverage report generated in htmlcov/index.html"

coverage-report:
	uv run pytest tests/ --cov=optimizer --cov-branch --cov-report=xml --cov-fail-under=90
	uv run python scripts/check_branch_coverage.py coverage.xml 0.80

all: lint typecheck test

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache .coverage htmlcov/ coverage.xml dist/
	find . -type d -name '*.egg-info' -exec rm -rf {} +

contract-snapshots:
	uv run python ingestion/tests/contract/emit_schemas.py
