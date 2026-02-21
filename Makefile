.PHONY: lint format typecheck test coverage all clean

lint:
	ruff check optimizer/ tests/
	ruff format --check optimizer/ tests/

format:
	ruff format optimizer/ tests/

typecheck:
	mypy optimizer/

test:
	pytest tests/ -v --cov=optimizer --cov-report=term-missing

coverage:
	pytest tests/ --cov=optimizer --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

all: lint typecheck test

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache .coverage htmlcov/ coverage.xml dist/
	find . -type d -name '*.egg-info' -exec rm -rf {} +
