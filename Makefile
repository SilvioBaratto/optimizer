.PHONY: lint format typecheck test coverage coverage-report all clean cycle2-gate contract-snapshots

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
	python api/tests/contract/emit_schemas.py

cycle2-gate:
	python -c "import research"
	python -c "import research.optimization; import research.factors; import research.reporting"
	ruff check research/optimization/__init__.py research/factors/__init__.py research/reporting/__init__.py
	mypy research/ --ignore-missing-imports 2>&1 | grep "^Found" || true
	@! grep -rnE "^from api|^import api" research/optimization/ research/factors/ research/reporting/
	@! grep -rnE "^from research\.data" research/optimization/ research/factors/ research/reporting/
	@! grep -rnE "^from research\.(pipeline|cli)" research/optimization/ research/factors/ research/reporting/
	python -c "import research._optimization, research._factors, research._report, research._backtest_plots, research._display"
	pytest tests/research/test_cycle2_boundary_gate.py -v
