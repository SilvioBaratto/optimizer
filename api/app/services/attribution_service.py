"""Attribution service: Brinson-Fachler decomposition and factor attribution.

Two pure computation paths:
  - compute_brinson_attribution: sector-level BF decomposition from weights/returns
  - compute_factor_attribution: factor exposure decomposition from stored scores

DB-aware wrappers:
  - run_brinson_attribution: fetches prices and sector map from DB, then delegates
  - run_factor_attribution: fetches factor scores and prices from DB, then delegates
"""

from __future__ import annotations

import datetime
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from app.repositories.factor_repository import FactorRepository
from app.repositories.yfinance_repository import YFinanceRepository

logger = logging.getLogger(__name__)

_UNCLASSIFIED = "Unclassified"
_SECTOR_COVERAGE_THRESHOLD = 0.50  # max fraction of tickers allowed to be Unclassified


# ---------------------------------------------------------------------------
# Domain errors
# ---------------------------------------------------------------------------


class AttributionError(ValueError):
    """Base class for attribution computation errors."""


class MissingDataError(AttributionError):
    """Required data not found in the database."""


class InsufficientSectorCoverageError(AttributionError):
    """Too many tickers have no sector mapping."""


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BrinsonDecomposition:
    """Brinson-Fachler effects for a single sector.

    Formulas:
      allocation  = (w_p - w_b) * (r_b_sector - r_b_total)
      selection   = w_b * (r_p_sector - r_b_sector)
      interaction = (w_p - w_b) * (r_p_sector - r_b_sector)
    """

    portfolio_weight: float
    benchmark_weight: float
    portfolio_sector_return: float
    benchmark_sector_return: float
    benchmark_total_return: float

    @property
    def allocation_effect(self) -> float:
        return (self.portfolio_weight - self.benchmark_weight) * (
            self.benchmark_sector_return - self.benchmark_total_return
        )

    @property
    def selection_effect(self) -> float:
        return self.benchmark_weight * (
            self.portfolio_sector_return - self.benchmark_sector_return
        )

    @property
    def interaction_effect(self) -> float:
        return (self.portfolio_weight - self.benchmark_weight) * (
            self.portfolio_sector_return - self.benchmark_sector_return
        )

    @property
    def total_effect(self) -> float:
        return self.allocation_effect + self.selection_effect + self.interaction_effect


@dataclass(frozen=True)
class FactorAttributionResult:
    """Factor exposure and return contribution for a single factor."""

    factor_name: str
    exposure: float
    factor_return: float
    contribution: float


# ---------------------------------------------------------------------------
# Pure computation functions
# ---------------------------------------------------------------------------


def compute_brinson_attribution(
    portfolio_weights: dict[str, float],
    benchmark_weights: dict[str, float],
    sector_map: dict[str, str],
    portfolio_returns: dict[str, float],
    benchmark_returns: dict[str, float],
) -> dict[str, Any]:
    """Compute Brinson-Fachler attribution from pre-assembled inputs.

    Args:
        portfolio_weights: ticker → portfolio weight (must sum ≈ 1.0).
        benchmark_weights: ticker → benchmark weight (must sum ≈ 1.0).
        sector_map: ticker → sector name (None values mapped to 'Unclassified').
        portfolio_returns: ticker → total return over the period.
        benchmark_returns: ticker → total return over the period.

    Returns:
        Dict with ``sectors`` list and aggregate totals.

    Raises:
        InsufficientSectorCoverageError: if >20% of portfolio weight is Unclassified.
    """
    _check_sector_coverage(portfolio_weights, sector_map)

    benchmark_total = _weighted_return(benchmark_weights, benchmark_returns)
    portfolio_total = _weighted_return(portfolio_weights, portfolio_returns)

    sector_portfolio_weights = _aggregate_by_sector(portfolio_weights, sector_map)
    sector_benchmark_weights = _aggregate_by_sector(benchmark_weights, sector_map)
    sector_portfolio_returns = _sector_weighted_return(
        portfolio_weights, portfolio_returns, sector_map
    )
    sector_benchmark_returns = _sector_weighted_return(
        benchmark_weights, benchmark_returns, sector_map
    )

    all_sectors = sector_portfolio_weights.keys() | sector_benchmark_weights.keys()

    sector_rows = []
    for sector in sorted(all_sectors):
        w_p = sector_portfolio_weights.get(sector, 0.0)
        w_b = sector_benchmark_weights.get(sector, 0.0)
        r_p = sector_portfolio_returns.get(sector, 0.0)
        r_b = sector_benchmark_returns.get(sector, 0.0)

        decomp = BrinsonDecomposition(
            portfolio_weight=w_p,
            benchmark_weight=w_b,
            portfolio_sector_return=r_p,
            benchmark_sector_return=r_b,
            benchmark_total_return=benchmark_total,
        )
        sector_rows.append({
            "sector": sector,
            "portfolio_weight": w_p,
            "benchmark_weight": w_b,
            "portfolio_return": r_p,
            "benchmark_return": r_b,
            "allocation_effect": decomp.allocation_effect,
            "selection_effect": decomp.selection_effect,
            "interaction_effect": decomp.interaction_effect,
            "total_effect": decomp.total_effect,
        })

    total_allocation = sum(s["allocation_effect"] for s in sector_rows)
    total_selection = sum(s["selection_effect"] for s in sector_rows)
    total_interaction = sum(s["interaction_effect"] for s in sector_rows)

    return {
        "sectors": sector_rows,
        "total_allocation": total_allocation,
        "total_selection": total_selection,
        "total_interaction": total_interaction,
        "total_active_return": total_allocation + total_selection + total_interaction,
        "portfolio_return": portfolio_total,
        "benchmark_return": benchmark_total,
    }


def compute_factor_attribution(
    portfolio_weights: dict[str, float],
    asset_returns: dict[str, float],
    factor_exposures: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Decompose portfolio return into factor contributions and residual.

    Factor return for factor f = weighted average of exposure_i * r_i
    where weights are portfolio weights.

    Args:
        portfolio_weights: ticker → weight.
        asset_returns: ticker → period return.
        factor_exposures: ticker → {factor_name → exposure}.

    Returns:
        Dict with ``factors`` list, ``portfolio_return``, ``explained_return``, ``residual``.

    Raises:
        MissingDataError: if factor_exposures is empty.
    """
    if not factor_exposures:
        raise MissingDataError("No factor exposures provided")

    portfolio_return = _weighted_return(portfolio_weights, asset_returns)

    factor_names = {
        f for exposures in factor_exposures.values() for f in exposures
    }

    factor_rows = []
    for factor in sorted(factor_names):
        exposure = _portfolio_weighted_exposure(
            portfolio_weights, factor_exposures, factor
        )
        factor_return = _compute_factor_return(
            portfolio_weights, asset_returns, factor_exposures, factor
        )
        contribution = exposure * factor_return
        factor_rows.append(FactorAttributionResult(
            factor_name=factor,
            exposure=exposure,
            factor_return=factor_return,
            contribution=contribution,
        ))

    explained = sum(f.contribution for f in factor_rows)
    residual = portfolio_return - explained

    return {
        "factors": [
            {
                "factor_name": f.factor_name,
                "exposure": f.exposure,
                "factor_return": f.factor_return,
                "contribution": f.contribution,
            }
            for f in factor_rows
        ],
        "portfolio_return": portfolio_return,
        "explained_return": explained,
        "residual": residual,
    }


# ---------------------------------------------------------------------------
# DB-aware orchestrators
# ---------------------------------------------------------------------------


def run_brinson_attribution(
    session: Session,
    portfolio_weights: dict[str, float],
    benchmark_weights: dict[str, float],
    start_date: datetime.date,
    end_date: datetime.date,
) -> dict[str, Any]:
    """Fetch benchmark and portfolio returns from DB, run Brinson attribution.

    Args:
        session: SQLAlchemy session.
        portfolio_weights: ticker → weight.
        benchmark_weights: ticker → weight.
        start_date: Period start (inclusive).
        end_date: Period end (inclusive).

    Raises:
        MissingDataError: if benchmark data not found for any ticker.
        InsufficientSectorCoverageError: if sector coverage below threshold.
    """
    all_tickers = list(portfolio_weights.keys() | benchmark_weights.keys())

    sector_map = _fetch_sector_map(session, all_tickers)
    portfolio_returns = _fetch_ticker_returns(
        session, list(portfolio_weights), start_date, end_date
    )
    benchmark_returns = _fetch_ticker_returns(
        session, list(benchmark_weights), start_date, end_date
    )

    missing = [t for t in benchmark_weights if t not in benchmark_returns]
    if missing:
        raise MissingDataError(
            f"No price history for benchmark tickers: {missing}"
        )

    return compute_brinson_attribution(
        portfolio_weights=portfolio_weights,
        benchmark_weights=benchmark_weights,
        sector_map=sector_map,
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
    )


def run_factor_attribution(
    session: Session,
    portfolio_weights: dict[str, float],
    start_date: datetime.date,
    end_date: datetime.date,
) -> dict[str, Any]:
    """Fetch factor scores and price returns from DB, run factor attribution.

    Args:
        session: SQLAlchemy session.
        portfolio_weights: ticker → weight.
        start_date: Period start (inclusive).
        end_date: Period end (inclusive).

    Raises:
        MissingDataError: if no factor scores found for the tickers/dates.
    """
    tickers = list(portfolio_weights)

    factor_exposures = _fetch_factor_exposures(session, tickers, start_date, end_date)
    if not factor_exposures:
        raise MissingDataError(
            f"No factor scores found for tickers {tickers} "
            f"between {start_date} and {end_date}"
        )

    asset_returns = _fetch_ticker_returns(session, tickers, start_date, end_date)

    return compute_factor_attribution(
        portfolio_weights=portfolio_weights,
        asset_returns=asset_returns,
        factor_exposures=factor_exposures,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _weighted_return(
    weights: dict[str, float],
    returns: dict[str, float],
) -> float:
    return sum(w * returns.get(t, 0.0) for t, w in weights.items())


def _aggregate_by_sector(
    weights: dict[str, float],
    sector_map: dict[str, str],
) -> dict[str, float]:
    result: dict[str, float] = defaultdict(float)
    for ticker, weight in weights.items():
        sector = sector_map.get(ticker, _UNCLASSIFIED)
        result[sector] += weight
    return dict(result)


def _sector_weighted_return(
    weights: dict[str, float],
    returns: dict[str, float],
    sector_map: dict[str, str],
) -> dict[str, float]:
    sector_weighted: dict[str, float] = defaultdict(float)
    sector_total_weight: dict[str, float] = defaultdict(float)
    for ticker, weight in weights.items():
        sector = sector_map.get(ticker, _UNCLASSIFIED)
        sector_weighted[sector] += weight * returns.get(ticker, 0.0)
        sector_total_weight[sector] += weight

    result: dict[str, float] = {}
    for sector, total_w in sector_total_weight.items():
        result[sector] = sector_weighted[sector] / total_w if total_w else 0.0
    return result


def _check_sector_coverage(
    portfolio_weights: dict[str, float],
    sector_map: dict[str, str],
) -> None:
    n_tickers = len(portfolio_weights)
    if n_tickers == 0:
        return
    n_unclassified = sum(
        1 for t in portfolio_weights
        if sector_map.get(t, _UNCLASSIFIED) == _UNCLASSIFIED
    )
    fraction = n_unclassified / n_tickers
    if fraction > _SECTOR_COVERAGE_THRESHOLD:
        raise InsufficientSectorCoverageError(
            f"Sector coverage insufficient: {fraction:.1%} of portfolio tickers "
            f"have no sector mapping (threshold: {_SECTOR_COVERAGE_THRESHOLD:.0%})"
        )


def _portfolio_weighted_exposure(
    portfolio_weights: dict[str, float],
    factor_exposures: dict[str, dict[str, float]],
    factor_name: str,
) -> float:
    return sum(
        portfolio_weights.get(t, 0.0) * exposures.get(factor_name, 0.0)
        for t, exposures in factor_exposures.items()
    )


def _compute_factor_return(
    portfolio_weights: dict[str, float],
    asset_returns: dict[str, float],
    factor_exposures: dict[str, dict[str, float]],
    factor_name: str,
) -> float:
    """Factor return = weighted avg of asset_return weighted by factor exposure."""
    total_exposure = sum(
        portfolio_weights.get(t, 0.0) * abs(exposures.get(factor_name, 0.0))
        for t, exposures in factor_exposures.items()
    )
    if total_exposure == 0.0:
        return 0.0

    weighted = sum(
        portfolio_weights.get(t, 0.0)
        * abs(exposures.get(factor_name, 0.0))
        * asset_returns.get(t, 0.0)
        for t, exposures in factor_exposures.items()
    )
    return weighted / total_exposure


def _fetch_sector_map(session: Session, tickers: list[str]) -> dict[str, str]:
    """Resolve sector per ticker via the shared two-tier resolver (issue #427)."""
    from app.services._sector_resolver import resolve_sector_map

    return resolve_sector_map(session, tickers)


def _fetch_ticker_returns(
    session: Session,
    tickers: list[str],
    start_date: datetime.date,
    end_date: datetime.date,
) -> dict[str, float]:
    """Return cumulative period return for each ticker from price_history."""
    yf_repo = YFinanceRepository(session)
    returns: dict[str, float] = {}

    for ticker in tickers:
        instrument = yf_repo.get_instrument_by_yfinance_ticker(ticker)
        if instrument is None:
            continue

        history = yf_repo.get_price_history(
            instrument.id, start_date=start_date, end_date=end_date
        )
        prices = [float(ph.close) for ph in history if ph.close is not None]
        if len(prices) < 2:
            continue
        returns[ticker] = (prices[-1] / prices[0]) - 1.0

    return returns


def _fetch_factor_exposures(
    session: Session,
    tickers: list[str],
    start_date: datetime.date,
    end_date: datetime.date,
) -> dict[str, dict[str, float]]:
    """Return average factor exposure per ticker over the date range."""
    repo = FactorRepository(session)
    rows = repo.get_scores_by_tickers_and_date_range(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
    )
    if not rows:
        return {}

    df = pd.DataFrame([
        {"ticker": r.ticker, "factor_type": r.factor_type, "raw_score": r.raw_score}
        for r in rows
    ])
    # Average exposure per (ticker, factor) over the date range
    avg = df.groupby(["ticker", "factor_type"])["raw_score"].mean()

    exposures: dict[str, dict[str, float]] = defaultdict(dict)
    for idx in avg.index:
        ticker = str(idx[0])  # type: ignore[index]
        factor = str(idx[1])  # type: ignore[index]
        exposures[ticker][factor] = float(avg[idx])
    return dict(exposures)
