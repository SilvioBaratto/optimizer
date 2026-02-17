# Portfolio Optimizer

[![CI](https://github.com/SilvioBaratto/optimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/SilvioBaratto/optimizer/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-BSD--3--Clause-green)

Quantitative portfolio construction and optimization platform built on [skfolio](https://skfolio.org/) and scikit-learn.

## Architecture

```
optimizer/          Pure-Python optimization library (DB-agnostic, sklearn/skfolio-based)
  preprocessing/    Return data cleaning (validation, outlier treatment, sector imputation)
  pre_selection/    Asset filtering pipeline (completeness, variance, correlation, dominance)
  moments/          Expected return and covariance estimation
  views/            Black-Litterman, Entropy Pooling, Opinion Pooling
  optimization/     Mean-risk, risk budgeting, HRP/HERC/NCO, stacking
  synthetic/        Vine copula scenario generation and stress testing
  validation/       Walk-forward, combinatorial purged CV, randomized CV
  scoring/          Performance scoring for model selection
  tuning/           Grid search and randomized search with temporal CV
  rebalancing/      Calendar and threshold-based rebalancing
  pipeline/         End-to-end orchestration (prices → validated weights)

api/                FastAPI backend (PostgreSQL, BAML, Trading 212 integration)
cli/                Typer CLI (data fetching, universe management, macro regime analysis)
tests/              Test suite for the optimizer library
theory/             LaTeX/Markdown theoretical documentation
```

## Quick Start — Optimizer Library

```bash
pip install -e ".[dev]"
```

```python
import pandas as pd
from optimizer.optimization import MeanRiskConfig, build_mean_risk
from optimizer.validation import WalkForwardConfig
from optimizer.pipeline import run_full_pipeline

# Load price data (DatetimeIndex, one column per asset)
prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)

# Configure and run
optimizer = build_mean_risk(MeanRiskConfig.for_max_sharpe())
cv_config = WalkForwardConfig.for_quarterly_rolling()

result = run_full_pipeline(
    prices=prices,
    optimizer=optimizer,
    cv_config=cv_config,
)

print(result.weights)
print(result.summary)
```

## Quick Start — API + CLI

```bash
# Start PostgreSQL
docker compose up -d

# Set up the API
cd api
pip install -r requirements.txt
cp .env.example .env              # Edit with your API keys
alembic upgrade head
uvicorn app.main:app --reload     # http://localhost:8000

# CLI (from project root)
pip install -r requirements.txt
python -m cli --help
python -m cli db health
python -m cli universe stats
python -m cli yfinance fetch
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check .

# Type check
mypy .
```

## Environment Variables

Copy `.env.example` to `.env` and fill in your API keys. Database defaults match `docker-compose.yml` and work out of the box.

| Variable | Description |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string |
| `FRED_API_KEY` | Federal Reserve Economic Data |
| `TRADING_212_API_KEY` | Trading 212 portfolio access |
| `TRADING_ECONOMICS_API_KEY` | Trading Economics macro data |
| `OPENAI_KEY` | Azure OpenAI (for BAML chatbot) |

## License

[BSD-3-Clause](LICENSE)
