"""Cross-validation service: orchestrates run_cross_val() calls.

Stateless functions following Single Responsibility Principle:
  - build_cv        — dispatch cv_type + config dict → skfolio CV object
  - fetch_returns   — prices from DB → returns DataFrame
  - serialize_population — Population/MultiPeriodPortfolio → plain dicts
  - run_validation  — full lifecycle: fetch → build → cross-val → serialize
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import pandas as pd
from skfolio.preprocessing import prices_to_returns
from sqlalchemy.orm import Session

from app.services._price_fetcher import fetch_close_prices
from app.services.optimization_service import resolve_optimizer
from optimizer.validation import (
    CPCVConfig,
    MultipleRandomizedCVConfig,
    WalkForwardConfig,
    build_cpcv,
    build_multiple_randomized_cv,
    build_walk_forward,
    run_cross_val,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CV registry — Open/Closed: add new types without modifying callers.
# ---------------------------------------------------------------------------

_CV_REGISTRY: dict[str, Any] = {
    "walk_forward": (WalkForwardConfig, build_walk_forward),
    "cpcv": (CPCVConfig, build_cpcv),
    "multiple_randomized": (MultipleRandomizedCVConfig, build_multiple_randomized_cv),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_cv(cv_type: str, cv_config: dict[str, Any]) -> Any:
    """Map cv_type + config dict to a skfolio cross-validator.

    Args:
        cv_type: One of ``walk_forward``, ``cpcv``, ``multiple_randomized``.
        cv_config: Override fields for the matching config dataclass.

    Returns:
        A skfolio temporal cross-validator instance.

    Raises:
        ValueError: For unknown cv_type values.
    """
    entry = _CV_REGISTRY.get(cv_type)
    if entry is None:
        raise ValueError(
            f"Unknown cv_type: {cv_type!r}. Valid: {sorted(_CV_REGISTRY)}"
        )
    config_cls, factory_fn = entry
    config = config_cls(**cv_config)
    return factory_fn(config)


def fetch_returns(
    tickers: list[str],
    start_date: date,
    end_date: date,
    session: Session,
) -> pd.DataFrame:
    """Fetch close prices from DB and convert to returns.

    Args:
        tickers: Ticker symbols to fetch.
        start_date: Start of history window.
        end_date: End of history window.
        session: Active SQLAlchemy session.

    Returns:
        Returns DataFrame (observations × assets).
    """
    prices = fetch_close_prices(tickers, session, start_date=start_date, end_date=end_date)
    return pd.DataFrame(prices_to_returns(prices))


def serialize_population(population: Any) -> list[dict[str, Any]]:
    """Extract weights + measures from a Population or MultiPeriodPortfolio.

    Args:
        population: skfolio Population or MultiPeriodPortfolio from run_cross_val.

    Returns:
        List of FoldResult-compatible dicts with ``weights`` and ``measures`` keys.
    """
    folds = []
    try:
        for portfolio in population:
            folds.append(_extract_portfolio(portfolio))
    except TypeError:
        # population is a single Portfolio-like object
        folds.append(_extract_portfolio(population))
    return folds


def run_validation(
    tickers: list[str],
    start_date: date,
    end_date: date,
    cv_type: str,
    cv_config: dict[str, Any],
    optimizer_type: str,
    optimizer_config: dict[str, Any],
    session: Session,
    on_progress: Any = None,
) -> dict[str, Any]:
    """Run cross-validation and return serialized fold results + aggregate score.

    Args:
        tickers: Ticker universe.
        start_date: Start of price history.
        end_date: End of price history.
        cv_type: Cross-validation strategy key.
        cv_config: CV config overrides.
        optimizer_type: Optimizer registry key.
        optimizer_config: Optimizer factory kwargs.
        session: Active SQLAlchemy session.
        on_progress: Optional progress callback(current_fold=, total_folds=).

    Returns:
        Dict with ``fold_results`` (list) and ``aggregate_score`` (float).
    """
    returns = fetch_returns(tickers, start_date, end_date, session)
    estimator = resolve_optimizer(optimizer_type, optimizer_config)
    cv = build_cv(cv_type, cv_config)

    population = run_cross_val(estimator, returns, cv=cv)
    folds = serialize_population(population)

    if on_progress is not None:
        on_progress(current_fold=len(folds), total_folds=len(folds))

    aggregate_score = _compute_aggregate_score(population)
    return {"fold_results": folds, "aggregate_score": aggregate_score}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------




def _extract_portfolio(portfolio: Any) -> dict[str, Any]:
    """Extract weights and measures from a single Portfolio object."""
    try:
        weights = {str(k): float(v) for k, v in portfolio.weights_dict.items()}
    except Exception:
        weights = {}

    try:
        summary = portfolio.summary(formatted=False)
        measures = {str(k): float(v) for k, v in summary.items()}
    except Exception:
        measures = {}

    return {"weights": weights, "measures": measures}


def _compute_aggregate_score(population: Any) -> float:
    """Compute mean annualised Sharpe ratio across all fold portfolios."""
    sharpe_values: list[float] = []
    try:
        for portfolio in population:
            try:
                sharpe_values.append(float(portfolio.annualized_sharpe_ratio))
            except Exception:
                pass
    except TypeError:
        try:
            sharpe_values.append(float(population.annualized_sharpe_ratio))
        except Exception:
            pass

    return sum(sharpe_values) / len(sharpe_values) if sharpe_values else 0.0
