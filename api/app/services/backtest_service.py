"""Backtest service: wraps run_full_pipeline / run_full_pipeline_with_selection.

Stateless functions following Single Responsibility Principle:
  - validate_prices  — guard against empty/missing price data
  - build_optimizer  — map pipeline_config to an optimizer instance
  - extract_backtest_metrics  — convert PortfolioResult to BacktestRun fields
  - run_and_persist  — full lifecycle: fetch → validate → run → persist
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import date
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from app.repositories.execution_repository import ExecutionRepository
from app.services._price_fetcher import fetch_close_prices
from app.services.optimization_service import resolve_optimizer
from optimizer.pipeline import PortfolioResult, run_full_pipeline
from optimizer.validation import WalkForwardConfig

logger = logging.getLogger(__name__)

_DEFAULT_OPTIMIZER_TYPE = "hrp"
_ROLLING_WINDOW = 63  # ~1 quarter


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_prices(prices: pd.DataFrame, tickers: list[str]) -> None:
    """Raise ValueError with a descriptive message when price data is unusable.

    Args:
        prices: Price DataFrame (dates × tickers).
        tickers: Requested tickers.

    Raises:
        ValueError: When ``prices`` is empty or all requested tickers are absent.
    """
    if prices.empty:
        missing = ", ".join(tickers)
        raise ValueError(
            f"No price data found for requested tickers: {missing}. "
            "Ensure yfinance data has been fetched for these tickers."
        )

    absent = [t for t in tickers if t not in prices.columns]
    if absent:
        raise ValueError(
            f"Price data missing for tickers: {', '.join(absent)}. "
            "Run yfinance fetch first."
        )


def build_optimizer(pipeline_config: dict[str, Any]) -> Any:
    """Map pipeline_config dict to a skfolio optimizer instance.

    Extracts ``optimizer_type`` (default ``"hrp"``) and
    ``optimizer_config`` from *pipeline_config* and delegates to
    :func:`resolve_optimizer`.

    Args:
        pipeline_config: Dict with optional ``optimizer_type`` and
            ``optimizer_config`` keys.

    Returns:
        A configured skfolio optimizer instance.

    Raises:
        ValueError: For unknown ``optimizer_type`` values.
    """
    optimizer_type = pipeline_config.get("optimizer_type", _DEFAULT_OPTIMIZER_TYPE)
    optimizer_config = pipeline_config.get("optimizer_config", {})
    return resolve_optimizer(optimizer_type, optimizer_config)


def extract_backtest_metrics(result: PortfolioResult) -> dict[str, Any]:
    """Extract all BacktestRun columns from a PortfolioResult.

    Uses ``result.backtest`` (walk-forward) when available; falls back to
    ``result.portfolio`` (in-sample) otherwise.

    Args:
        result: Output of :func:`run_full_pipeline`.

    Returns:
        Dict with keys matching :class:`BacktestRun` columns:
        ``equity_curve``, ``drawdowns``, ``monthly_returns``,
        ``yearly_returns``, ``rolling_metrics``, ``turnover_history``,
        ``cv_fold_metrics``, ``summary_stats``.
    """
    portfolio = result.backtest if result.backtest is not None else result.portfolio
    returns_series = _returns_series(portfolio)

    return {
        "equity_curve": _equity_curve(portfolio),
        "drawdowns": _drawdowns(portfolio),
        "monthly_returns": _resample_returns(returns_series, "ME"),
        "yearly_returns": _resample_returns(returns_series, "YE"),
        "rolling_metrics": _rolling_metrics(portfolio),
        "turnover_history": _turnover_history(result),
        "cv_fold_metrics": _cv_fold_metrics(result),
        "summary_stats": _summary_stats(result),
    }


def run_and_persist(
    tickers: list[str],
    start_date: date,
    end_date: date,
    pipeline_config: dict[str, Any],
    session: Session,
    run_id: uuid.UUID,
) -> Any:
    """Fetch prices, run pipeline, persist result into BacktestRun row.

    Args:
        tickers: Ticker universe.
        start_date: Backtest start date.
        end_date: Backtest end date.
        pipeline_config: Pipeline overrides (optimizer_type, etc.).
        session: DB session (owns the transaction).
        run_id: Pre-created BacktestRun UUID to update.

    Returns:
        The updated :class:`BacktestRun` with status ``"completed"``.

    Raises:
        ValueError: When price data is empty or tickers are absent from DB.
    """
    prices = fetch_close_prices(tickers, session, start_date=start_date, end_date=end_date)
    validate_prices(prices, tickers)

    optimizer = build_optimizer(pipeline_config)
    cv_config = WalkForwardConfig() if pipeline_config.get("run_cv", True) else None

    t0 = time.monotonic()
    result = run_full_pipeline(
        prices=prices,
        optimizer=optimizer,
        cv_config=cv_config,
    )
    duration = time.monotonic() - t0

    metrics = extract_backtest_metrics(result)
    repo = ExecutionRepository(session)
    run = repo.update_backtest_status(
        run_id,
        "completed",
        {
            **metrics,
            "duration_seconds": round(duration, 4),
        },
    )
    session.commit()
    return run


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------




def _returns_series(portfolio: Any) -> pd.Series:
    """Return a 1-D returns Series from portfolio.returns_df."""
    returns_df = portfolio.returns_df
    if isinstance(returns_df, pd.DataFrame):
        return returns_df.iloc[:, 0]
    return returns_df


def _equity_curve(portfolio: Any) -> dict[str, float]:
    """Convert cumulative_returns_df to {date_str: float}."""
    series = portfolio.cumulative_returns_df
    return _series_to_dict(series)


def _drawdowns(portfolio: Any) -> dict[str, Any]:
    """Drawdown series + summary {date_str: float, max_drawdown: float}."""
    dd_series = _series_to_dict(portfolio.drawdowns_df)
    return {**dd_series, "max_drawdown": float(portfolio.max_drawdown)}


def _resample_returns(returns: pd.Series, freq: str) -> dict[str, float]:
    """Compound returns at *freq* ('ME' = month-end, 'YE' = year-end)."""
    if returns.empty:
        return {}
    compounded = (1 + returns).resample(freq).prod() - 1
    return {str(dt.date()): round(float(v), 6) for dt, v in compounded.items()}


def _rolling_metrics(portfolio: Any) -> dict[str, dict[str, float]]:
    """Rolling Sharpe and annualised volatility over _ROLLING_WINDOW periods."""
    from skfolio.measures import RatioMeasure, RiskMeasure

    sharpe = portfolio.rolling_measure(
        measure=RatioMeasure.ANNUALIZED_SHARPE_RATIO, window=_ROLLING_WINDOW
    )
    vol = portfolio.rolling_measure(
        measure=RiskMeasure.ANNUALIZED_STANDARD_DEVIATION, window=_ROLLING_WINDOW
    )
    return {
        "rolling_sharpe": _series_to_dict(sharpe),
        "rolling_vol": _series_to_dict(vol),
    }


def _turnover_history(result: PortfolioResult) -> dict[str, float]:
    """Per-period turnover from weight_history; empty when unavailable."""
    if result.weight_history is None:
        return {}
    wh: pd.DataFrame = result.weight_history.fillna(0)
    diffs = wh.diff().abs().sum(axis=1) / 2
    return _series_to_dict(diffs.iloc[1:])  # skip first NaN row


def _cv_fold_metrics(result: PortfolioResult) -> list[dict[str, Any]] | None:
    """Per-fold summary stats; None when no walk-forward backtest was run."""
    if result.backtest is None:
        return None
    folds = []
    try:
        for fold_portfolio in result.backtest:
            summary = fold_portfolio.summary(formatted=False)
            folds.append({str(k): float(v) for k, v in summary.items()})
    except Exception:
        return None
    return folds or None


def _summary_stats(result: PortfolioResult) -> dict[str, Any]:
    """Aggregate summary from result.summary + in-sample portfolio."""
    stats: dict[str, Any] = {
        str(k): float(v) for k, v in (result.summary or {}).items()
    }
    try:
        in_sample = result.portfolio.summary(formatted=False)
        stats["in_sample"] = {str(k): float(v) for k, v in in_sample.items()}
    except Exception:
        pass
    return stats


def _series_to_dict(series: pd.Series) -> dict[str, float]:
    """Convert a date-indexed pandas Series to {date_str: float}."""
    result: dict[str, float] = {}
    for idx, val in series.items():
        try:
            result[str(idx.date())] = round(float(val), 6)
        except (AttributeError, TypeError, ValueError):
            result[str(idx)] = round(float(val), 6)
    return result
