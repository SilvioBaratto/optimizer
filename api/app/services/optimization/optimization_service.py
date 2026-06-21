"""Optimization service: maps OptimizeRequest config to optimizer library calls.

Stateless functions following Single Responsibility Principle:
  - _build_optimizer   — create optimizer instance from type key + raw config dict
  - build_pipeline     — data fetch + pipeline assembly
  - extract_results    — portfolio result extraction
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import pandas as pd
from skfolio import Population, Portfolio
from skfolio.measures import RiskMeasure
from skfolio.preprocessing import prices_to_returns
from sklearn.pipeline import Pipeline
from sqlalchemy.orm import Session

from app.services._shared import fetch_close_prices
from optimizer.optimization._config import MeanRiskConfig
from optimizer.optimization._factory import build_mean_risk
from optimizer.pre_selection._pipeline import build_preselection_pipeline

logger = logging.getLogger(__name__)

# Types that support efficient frontier computation (used by /optimize route).
FRONTIER_TYPES: frozenset[str] = frozenset({"mean_risk"})
FRONTIER_SIZE: int = 20


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_optimizer(optimizer_type: str, config: dict) -> Any:
    """Create optimizer instance from type key + raw config dict."""
    if optimizer_type == "mean_risk":
        return build_mean_risk(MeanRiskConfig(**config) if config else MeanRiskConfig())
    raise ValueError(f"Unsupported optimizer_type: {optimizer_type!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_pipeline(
    tickers: list[str],
    start_date: date,
    end_date: date,
    optimizer_type: str,
    config: dict,
    session: Session,
) -> tuple[Pipeline, pd.DataFrame]:
    """Fetch prices, convert to returns, assemble sklearn Pipeline.

    Args:
        tickers: Ticker symbols to include.
        start_date: Start of price history.
        end_date: End of price history.
        optimizer_type: Supported value: ``"mean_risk"``.
        config: Raw config dict forwarded to the optimizer factory.
        session: Active SQLAlchemy session.

    Returns:
        (pipeline, returns_df) — pipeline not yet fitted.
    """
    optimizer = _build_optimizer(optimizer_type, config)
    prices = fetch_close_prices(
        tickers, session, start_date=start_date, end_date=end_date
    )
    returns = pd.DataFrame(prices_to_returns(prices))
    pipe = Pipeline(
        [
            ("pre_selection", build_preselection_pipeline()),
            ("optimizer", optimizer),
        ]
    )
    return pipe, returns


def extract_results(fitted_pipeline: Pipeline, returns_df: pd.DataFrame) -> dict:
    """Extract weights, metrics, risk contributions, and efficient frontier."""
    prediction = fitted_pipeline.predict(returns_df)
    if isinstance(prediction, Population):
        portfolios = list(prediction)  # iterate once; Population may be a lazy iterator
        frontier = _build_frontier(portfolios)
        portfolio = max(portfolios, key=lambda p: p.annualized_sharpe_ratio)
    else:
        frontier, portfolio = None, prediction
    return _to_result(portfolio, frontier)


def _build_frontier(portfolios: list[Portfolio]) -> list[dict]:
    """Convert list of Portfolio objects to {return, risk, weights} dicts."""
    return [
        {
            "return": float(p.annualized_mean),
            "risk": float(p.annualized_standard_deviation),
            "weights": p.weights_dict,
        }
        for p in portfolios
    ]


def _to_result(portfolio: Portfolio, frontier: list[dict] | None) -> dict:
    """Build result dict from fitted Portfolio."""
    tickers = list(portfolio.weights_dict.keys())
    contribs = portfolio.contribution(measure=RiskMeasure.STANDARD_DEVIATION)
    return {
        "weights": portfolio.weights_dict,
        "metrics": {
            "annualized_return": float(portfolio.annualized_mean),
            "annualized_volatility": float(portfolio.annualized_standard_deviation),
            "annualized_sharpe_ratio": float(portfolio.annualized_sharpe_ratio),
            "annualized_sortino_ratio": float(portfolio.annualized_sortino_ratio),
            "max_drawdown": float(portfolio.max_drawdown),
        },
        "risk_contributions": dict(zip(tickers, contribs.tolist(), strict=False)),
        "efficient_frontier": frontier,
    }
