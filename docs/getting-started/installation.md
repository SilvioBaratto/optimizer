# Installation

## Basic Install

```bash
pip install -e .
```

This installs the optimizer library with all runtime dependencies: numpy, pandas, scipy, scikit-learn, skfolio, hmmlearn, and arch.

## Development Install

```bash
pip install -e ".[dev]"
```

This includes test, lint, typecheck, and docs dependencies — everything needed for development and CI.

## Optional Dependencies

| Group | Install Command | Includes |
|-------|----------------|----------|
| `test` | `pip install -e ".[test]"` | pytest, pytest-cov, hypothesis |
| `lint` | `pip install -e ".[lint]"` | ruff, pip-audit |
| `typecheck` | `pip install -e ".[typecheck]"` | mypy |
| `docs` | `pip install -e ".[docs]"` | mkdocs-material, mkdocstrings |
| `dmm` | `pip install -e ".[dmm]"` | torch, pyro-ppl |
| `dev` | `pip install -e ".[dev]"` | All of the above + pre-commit |

## Requirements

- Python >= 3.10
- numpy, pandas, scipy, scikit-learn, skfolio, hmmlearn, arch

## Verifying the Installation

After installing, verify the library loads correctly:

```python
import optimizer
from optimizer.optimization import MeanRiskConfig, build_mean_risk
from optimizer.pipeline import run_full_pipeline

# Quick sanity check
config = MeanRiskConfig.for_max_sharpe()
optimizer = build_mean_risk(config)
print(f"Optimizer ready: {type(optimizer).__name__}")
```

## Deep Markov Model (Optional)

The DMM module (`optimizer.moments._dmm`) requires PyTorch and Pyro, which are **not** declared in the standard dependencies due to their size. Install them separately:

```bash
pip install -e ".[dmm]"
```

The DMM module is imported conditionally — if torch/pyro are not installed, the rest of the library works normally and DMM-related imports are silently skipped.

## Running Tests

```bash
# All tests
pytest tests/ -v

# Single module
pytest tests/rebalancing/ -v

# Single test
pytest -k "test_name"
```

## Linting and Type Checking

```bash
# Lint
ruff check optimizer/ tests/

# Lint + auto-fix
ruff check . --fix

# Type check
mypy optimizer/
```
