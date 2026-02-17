"""End-to-end portfolio pipeline orchestration.

Composes pre-selection, optimisation, validation, scoring,
hyperparameter tuning, and rebalancing into a single workflow.
"""

from optimizer.pipeline._builder import build_portfolio_pipeline
from optimizer.pipeline._config import PortfolioResult
from optimizer.pipeline._orchestrator import (
    backtest,
    optimize,
    run_full_pipeline,
    tune_and_optimize,
)

__all__ = [
    "PortfolioResult",
    "backtest",
    "build_portfolio_pipeline",
    "optimize",
    "run_full_pipeline",
    "tune_and_optimize",
]
