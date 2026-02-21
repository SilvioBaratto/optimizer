"""End-to-end portfolio pipeline orchestration.

Composes pre-selection, optimisation, validation, scoring,
hyperparameter tuning, and rebalancing into a single workflow.
"""

from optimizer.pipeline._builder import build_portfolio_pipeline
from optimizer.pipeline._config import PortfolioResult
from optimizer.pipeline._orchestrator import (
    backtest,
    compute_net_backtest_returns,
    optimize,
    run_full_pipeline,
    run_full_pipeline_with_selection,
    tune_and_optimize,
)

__all__ = [
    "PortfolioResult",
    "backtest",
    "build_portfolio_pipeline",
    "compute_net_backtest_returns",
    "optimize",
    "run_full_pipeline",
    "run_full_pipeline_with_selection",
    "tune_and_optimize",
]
