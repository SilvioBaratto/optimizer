.PHONY: setup sync lint format typecheck test coverage coverage-report all clean contract-snapshots

# One-command onboarding: delegate to the thin POSIX front door (no logic here).
setup:
	./setup.sh

# uv workspace: one venv for portopt-core + ingestion + packages/portopt-db.
sync:
	uv sync --all-packages --all-extras

lint:
	uv run ruff check packages/portopt-core/optimizer/ tests/
	uv run ruff format --check packages/portopt-core/optimizer/ tests/
	uv run ruff check packages/portopt-db/ ingestion/ fund/
	uv run ruff format --check packages/portopt-db/ ingestion/ fund/

format:
	uv run ruff format packages/portopt-core/optimizer/ tests/ packages/portopt-db/ ingestion/ fund/

typecheck:
	uv run mypy packages/portopt-core/optimizer/
	cd packages/portopt-db && uv run mypy src/portopt_db
	cd ingestion && uv run mypy app

test:
	uv run pytest tests/ -v --cov=optimizer --cov-report=term-missing --cov-fail-under=90

coverage:
	uv run pytest tests/ --cov=optimizer --cov-report=html --cov-fail-under=90
	@echo "Coverage report generated in htmlcov/index.html"

coverage-report:
	uv run pytest tests/ --cov=optimizer --cov-branch --cov-report=xml --cov-fail-under=90

all: lint typecheck test

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache .coverage htmlcov/ coverage.xml dist/
	find . -type d -name '*.egg-info' -exec rm -rf {} +

contract-snapshots:
	uv run python ingestion/tests/contract/emit_schemas.py
