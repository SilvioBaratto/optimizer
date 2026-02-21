# Installation

## Basic Install

```bash
pip install -e .
```

## Development Install

```bash
pip install -e ".[dev]"
```

This includes test, lint, typecheck, and docs dependencies.

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
