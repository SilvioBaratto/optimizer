.PHONY: setup sync lint format typecheck pyright test test-db test-ingestion test-fund coverage coverage-report all clean

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

pyright:
	uv run pyright

test:
	uv run pytest tests/ -v --cov=optimizer --cov-report=term-missing --cov-fail-under=90

# The member suites run from their own directory so their pytest config applies;
# ingestion's pytest.ini keeps the docker-compose deployment tests out of this run.
test-db:
	cd packages/portopt-db && uv run pytest -v --tb=short --cov=portopt_db --cov-report=term-missing --cov-fail-under=90

test-ingestion:
	cd ingestion && uv run pytest tests/ -v --tb=short --cov=app --cov-report=term-missing --cov-fail-under=80

test-fund:
	cd fund && uv run pytest tests/ -v --tb=short --cov=fund --cov-report=term-missing --cov-fail-under=80

coverage:
	uv run pytest tests/ --cov=optimizer --cov-report=html --cov-fail-under=90
	@echo "Coverage report generated in htmlcov/index.html"

coverage-report:
	uv run pytest tests/ --cov=optimizer --cov-branch --cov-report=xml --cov-fail-under=90

# Local mirror of the CI jobs (minus pip-audit and the Python version matrix), so a
# green `make all` means every workspace member passed, not just portopt-core.
all: lint typecheck pyright test test-db test-ingestion test-fund

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache .coverage htmlcov/ coverage.xml dist/
	find . -type d -name '*.egg-info' -exec rm -rf {} +
