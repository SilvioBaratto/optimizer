"""Optimization service: maps OptimizeRequest config to optimizer library calls.

Stateless functions following Single Responsibility Principle:
  - resolve_optimizer  — registry dispatch
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

from app.services._price_fetcher import fetch_close_prices
from optimizer.optimization._config import (
    EqualWeightedConfig,
    HERCConfig,
    HRPConfig,
    InverseVolatilityConfig,
    MaxDiversificationConfig,
    MeanRiskConfig,
    NCOConfig,
    RiskBudgetingConfig,
    StackingConfig,
)
from optimizer.optimization._dr_cvar import DRCVaRConfig, build_dr_cvar
from optimizer.optimization._factory import (
    build_equal_weighted,
    build_herc,
    build_hrp,
    build_inverse_volatility,
    build_max_diversification,
    build_mean_risk,
    build_nco,
    build_risk_budgeting,
    build_stacking,
)
from optimizer.optimization._regime_risk import (
    RegimeRiskConfig,
    build_regime_blended_optimizer,
)
from optimizer.optimization._robust import RobustConfig, build_robust_mean_risk
from optimizer.pre_selection._pipeline import build_preselection_pipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry — Open/Closed: add types by inserting here, no if/elif chains.
# ---------------------------------------------------------------------------

OPTIMIZER_REGISTRY: dict[str, Any] = {
    "mean_risk": build_mean_risk,
    "robust_mean_risk": build_robust_mean_risk,
    "dr_cvar": build_dr_cvar,
    "hrp": build_hrp,
    "herc": build_herc,
    "nco": build_nco,
    "risk_budgeting": build_risk_budgeting,
    "max_diversification": build_max_diversification,
    "equal_weighted": build_equal_weighted,
    "inverse_volatility": build_inverse_volatility,
    "stacking": build_stacking,
    "regime_blended": build_regime_blended_optimizer,
}

# Maps optimizer_type → its frozen Config dataclass. Each builder owns the
# enum → skfolio translation, so the service must hand the builder a typed
# config rather than spreading a raw dict (which would collide with the
# builder's explicit kwargs — see issue #419).
_CONFIG_CLASSES: dict[str, type] = {
    "mean_risk": MeanRiskConfig,
    "robust_mean_risk": RobustConfig,
    "dr_cvar": DRCVaRConfig,
    "hrp": HRPConfig,
    "herc": HERCConfig,
    "nco": NCOConfig,
    "risk_budgeting": RiskBudgetingConfig,
    "max_diversification": MaxDiversificationConfig,
    "equal_weighted": EqualWeightedConfig,
    "inverse_volatility": InverseVolatilityConfig,
    "stacking": StackingConfig,
    "regime_blended": RegimeRiskConfig,
}

# Types that support efficient frontier computation.
_FRONTIER_TYPES: frozenset[str] = frozenset({"mean_risk", "robust_mean_risk"})
_FRONTIER_SIZE: int = 20


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_optimizer(optimizer_type: str, config: dict) -> Any:
    """Look up registry, return optimizer instance, raise ValueError for unknown types."""
    factory = OPTIMIZER_REGISTRY.get(optimizer_type)
    if factory is None:
        raise ValueError(f"Unknown optimizer_type: {optimizer_type!r}")
    typed_config = _build_typed_config(optimizer_type, config)
    optimizer = factory(typed_config) if typed_config is not None else factory()
    if optimizer_type in _FRONTIER_TYPES:
        optimizer.set_params(efficient_frontier_size=_FRONTIER_SIZE)
    return optimizer


def _build_typed_config(optimizer_type: str, config: dict) -> Any | None:
    """Instantiate the frozen Config dataclass for *optimizer_type* from *config*.

    Returns ``None`` when *config* is empty so the builder picks its own
    default. Unknown fields raise :class:`TypeError` via the dataclass
    constructor — callers must send only config-level primitives.
    """
    if not config:
        return None
    config_class = _CONFIG_CLASSES.get(optimizer_type)
    if config_class is None:
        return None
    return config_class(**config)


def build_pipeline(
    tickers: list[str],
    start_date: date,
    end_date: date,
    optimizer: Any,
    session: Session,
) -> tuple[Pipeline, pd.DataFrame]:
    """Fetch prices, convert to returns, assemble sklearn Pipeline.

    Returns:
        (pipeline, returns_df) — pipeline not yet fitted.
    """
    prices = fetch_close_prices(tickers, session, start_date=start_date, end_date=end_date)
    returns = pd.DataFrame(prices_to_returns(prices))
    pipe = Pipeline([
        ("pre_selection", build_preselection_pipeline()),
        ("optimizer", optimizer),
    ])
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


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------



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
        },
        "risk_contributions": dict(zip(tickers, contribs.tolist())),
        "efficient_frontier": frontier,
    }
