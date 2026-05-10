"""Factor computation service — DB fetch, compute, and persist factor scores.

Responsibilities:
  - Load fundamentals and price history from the database.
  - Call ``compute_all_factors()`` from the optimizer library.
  - Bulk-upsert computed scores into ``factor_scores`` via FactorRepository.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from app.repositories.factors.factor_repository import FactorRepository
from app.repositories.market_data.yfinance_repository import YFinanceRepository
from app.services.factors._factor_helpers import (
    _PERIOD_DATE_COL,
    _TICKER_COL,
    ProgressCallback,
    _find_instrument,
    _notify,
)
from optimizer.factors import (
    PublicationLagConfig,
    align_to_pit,
    compute_all_factors,
)

logger = logging.getLogger(__name__)


def run_factor_compute(
    session: Session,
    tickers: list[str],
    start_date: datetime.date,
    end_date: datetime.date,
    factor_config: dict[str, Any] | None = None,
    lag_config: PublicationLagConfig | None = None,
    on_progress: ProgressCallback | None = None,
) -> int:
    """Compute factor scores for *tickers* over the given date window.

    Fetches fundamentals and price history from the database, applies
    publication-lag alignment via ``align_to_pit()``, runs
    ``compute_all_factors()``, then bulk-upserts the results into
    ``factor_scores``.

    Args:
        session: SQLAlchemy session (synchronous).
        tickers: Asset ticker symbols.
        start_date: Computation window start (inclusive).
        end_date: Computation window end (inclusive).
        factor_config: Optional raw config dict (reserved for future use).
        lag_config: Publication lag configuration; defaults to quarterly lags.
        on_progress: Optional progress callback receiving ``current``,
            ``total``, and ``status`` kwargs.

    Returns:
        Number of factor score rows written to the database.
    """
    if lag_config is None:
        lag_config = PublicationLagConfig()
    if factor_config:
        logger.debug(
            "factor_config provided (reserved for future use): %s",
            list(factor_config),
        )

    _notify(on_progress, status="running", current=0, total=len(tickers))

    fundamentals = _load_fundamentals(session, tickers, end_date, lag_config)
    price_history = _load_prices(session, tickers, start_date, end_date)

    _notify(on_progress, current=len(tickers) // 2, total=len(tickers))

    factor_df = compute_all_factors(
        fundamentals=fundamentals,
        price_history=price_history,
    )

    if factor_df.empty:
        logger.warning(
            "compute_all_factors returned empty DataFrame for tickers=%s", tickers
        )
        _notify(
            on_progress, current=len(tickers), total=len(tickers), status="completed"
        )
        return 0

    rows = _build_score_rows(factor_df, as_of_date=end_date)
    factor_repo = FactorRepository(session)
    written = factor_repo.bulk_upsert_scores(rows)
    session.commit()

    _notify(on_progress, current=len(tickers), total=len(tickers), status="completed")
    logger.info("Factor compute complete: %d rows written", written)
    return written


def _load_fundamentals(
    session: Session,
    tickers: list[str],
    as_of_date: datetime.date,
    lag_config: PublicationLagConfig,
) -> pd.DataFrame:
    """Fetch financial statement data and apply PIT alignment.

    Args:
        session: SQLAlchemy session.
        tickers: Asset ticker symbols.
        as_of_date: The as-of date for point-in-time alignment.
        lag_config: Publication lag configuration.

    Returns:
        Cross-sectional DataFrame indexed by ticker; empty when no data.
    """
    yf_repo = YFinanceRepository(session)
    rows: list[dict[str, Any]] = []

    for ticker in tickers:
        instrument = _find_instrument(session, ticker)
        if instrument is None:
            continue
        statements = yf_repo.get_financial_statements(
            instrument.id,
            period_type="annual",
        )
        for stmt in statements:
            rows.append(
                {
                    _TICKER_COL: ticker,
                    _PERIOD_DATE_COL: stmt.period_date,
                    stmt.line_item: stmt.value,
                }
            )

    if not rows:
        return pd.DataFrame()

    raw = pd.DataFrame(rows)
    id_cols = [_TICKER_COL, _PERIOD_DATE_COL]
    value_cols = [c for c in raw.columns if c not in id_cols]
    if not value_cols:
        return pd.DataFrame()

    wide = raw.groupby(id_cols, as_index=False).first()
    return align_to_pit(
        data=wide,
        period_date_col=_PERIOD_DATE_COL,
        as_of_date=str(as_of_date),  # type: ignore[arg-type]
        lag_days=lag_config.annual_days,
        ticker_col=_TICKER_COL,
    )


def _load_prices(
    session: Session,
    tickers: list[str],
    start_date: datetime.date,
    end_date: datetime.date,
) -> pd.DataFrame:
    """Fetch price history and pivot to a dates x tickers matrix.

    Args:
        session: SQLAlchemy session.
        tickers: Asset ticker symbols.
        start_date: Window start (inclusive).
        end_date: Window end (inclusive).

    Returns:
        Wide pivot DataFrame with DatetimeIndex; empty when no data.
    """
    yf_repo = YFinanceRepository(session)
    price_rows: list[dict[str, Any]] = []

    for ticker in tickers:
        instrument = _find_instrument(session, ticker)
        if instrument is None:
            continue
        history = yf_repo.get_price_history(
            instrument.id,
            start_date=start_date,
            end_date=end_date,
        )
        for ph in history:
            price_rows.append({"date": ph.date, "ticker": ticker, "close": ph.close})

    if not price_rows:
        return pd.DataFrame()

    df = pd.DataFrame(price_rows)
    pivot = df.pivot(index="date", columns="ticker", values="close")
    pivot.index = pd.to_datetime(pivot.index)
    return pivot


def _build_score_rows(
    factor_df: pd.DataFrame,
    as_of_date: datetime.date,
) -> list[dict[str, Any]]:
    """Convert factor matrix (tickers x factors) to list of dicts for upsert.

    Args:
        factor_df: DataFrame with tickers as index and factor names as columns.
        as_of_date: Score date to attach to every row.

    Returns:
        List of dicts ready for ``FactorRepository.bulk_upsert_scores``.
    """
    from optimizer.factors._config import FACTOR_GROUP_MAPPING

    rows: list[dict[str, Any]] = []
    for ticker in factor_df.index:
        for factor_col in factor_df.columns:
            raw_score = factor_df.at[ticker, factor_col]
            if pd.isna(raw_score):
                continue
            rows.append(
                {
                    "ticker": str(ticker),
                    "score_date": as_of_date,
                    "factor_type": str(factor_col),
                    "factor_group": _resolve_group(factor_col, FACTOR_GROUP_MAPPING),
                    "raw_score": float(raw_score),
                    "standardized_score": None,
                    "composite_score": None,
                }
            )
    return rows


def _resolve_group(
    factor_col: str,
    group_mapping: dict[Any, Any],
) -> str:
    """Map factor column name to its group string, defaulting to 'unknown'.

    Args:
        factor_col: Factor column name string.
        group_mapping: Mapping from FactorType to FactorGroupType.

    Returns:
        Group string value or ``'unknown'`` on lookup failure.
    """
    try:
        from optimizer.factors._config import FactorType

        ft = FactorType(factor_col)
        return group_mapping[ft].value
    except (ValueError, KeyError):
        return "unknown"
