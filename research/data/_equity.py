"""Equity assembly functions — prices, volumes, fundamentals, statements.

Extracted from ``data_assembly.py``.  All functions accept a synchronous
SQLAlchemy ``Session`` and return pandas DataFrames.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

# Ensure the api package is importable from the CLI context.
_api_path = Path(__file__).parent.parent.parent / "api"
if str(_api_path) not in sys.path:
    sys.path.insert(0, str(_api_path))

from app.models.market_data.yfinance_data import (  # noqa: E402
    AnalystRecommendation,
    FinancialStatement,
    InsiderTransaction,
    PriceHistory,
    TickerProfile,
)
from app.models.universe.universe import Instrument  # noqa: E402

from ._currency import (  # noqa: E402
    build_currency_map,
    currency_dedup_rank,
    normalize_fundamentals,
    normalize_prices,
)
from ._helpers import (  # noqa: E402
    _STMT_LINE_ITEMS,
    _apply_delisting_returns,
    _build_currency_map_from_instruments,
    _build_ticker_map,
    _build_ticker_rank_map,
    _pivot_with_dedup,
    _to_float,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_asset_growth_from_statements(
    session: Session,
    ticker_map: dict[str, str],
    enrichment: dict[str, dict[str, float | None]],
) -> None:
    """Compute asset_growth from two most recent annual Total Assets values.

    Mutates *enrichment* in-place: adds ``asset_growth`` for each ticker
    where two annual Total Assets rows are available.

    Currency safety
    ~~~~~~~~~~~~~~~
    ``asset_growth = (current - prior) / abs(prior)`` is a dimensionless
    ratio.  Both numerator and denominator are in the same reporting
    currency for the same ticker, so the currency cancels regardless of
    denomination (GBP, USD, EUR, etc.).  No normalization is needed.
    """
    rows = session.execute(
        select(
            FinancialStatement.instrument_id,
            FinancialStatement.period_date,
            FinancialStatement.value,
        )
        .where(FinancialStatement.period_type == "annual")
        .where(FinancialStatement.line_item == "Total Assets")
        .where(FinancialStatement.value.isnot(None))
        .order_by(
            FinancialStatement.instrument_id,
            FinancialStatement.period_date.desc(),
        )
    ).all()

    # Group by instrument, keep two most recent values
    asset_by_inst: dict[str, list[float]] = {}
    for instrument_id, _period_date, value in rows:
        key = str(instrument_id)
        vals = asset_by_inst.setdefault(key, [])
        if len(vals) < 2:
            vals.append(float(value))

    for inst_id_hex, vals in asset_by_inst.items():
        if len(vals) == 2 and vals[1] != 0:
            ticker = ticker_map.get(inst_id_hex)
            if ticker is not None:
                growth = (vals[0] - vals[1]) / abs(vals[1])
                enrichment.setdefault(ticker, {})["asset_growth"] = growth


def _enrich_from_financial_statements(
    session: Session,
    df: pd.DataFrame,
    ticker_map: dict[str, str],
) -> pd.DataFrame:
    """Enrich a fundamentals DataFrame with data from FinancialStatement EAV.

    Queries annual financial statements for key line items, takes the latest
    period per ticker, pivots to columns, computes ``asset_growth``, and
    fills only NaN values in the existing DataFrame.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session.
    df : pd.DataFrame
        Fundamentals DataFrame indexed by yfinance ticker.
    ticker_map : dict[str, str]
        ``{instrument_id_hex: yfinance_ticker}`` mapping.

    Returns
    -------
    pd.DataFrame
        Enriched fundamentals DataFrame (same index).
    """
    if not ticker_map:
        return df

    line_item_names = list(_STMT_LINE_ITEMS.keys())

    # Sub-query: latest annual period_date per instrument + line_item
    latest_sq = (
        select(
            FinancialStatement.instrument_id,
            FinancialStatement.line_item,
            func.max(FinancialStatement.period_date).label("max_date"),
        )
        .where(FinancialStatement.period_type == "annual")
        .where(FinancialStatement.line_item.in_(line_item_names))
        .group_by(
            FinancialStatement.instrument_id,
            FinancialStatement.line_item,
        )
        .subquery()
    )

    # Main query: get values at the latest date
    rows = session.execute(
        select(
            FinancialStatement.instrument_id,
            FinancialStatement.line_item,
            FinancialStatement.value,
        )
        .join(
            latest_sq,
            (FinancialStatement.instrument_id == latest_sq.c.instrument_id)
            & (FinancialStatement.line_item == latest_sq.c.line_item)
            & (FinancialStatement.period_date == latest_sq.c.max_date),
        )
        .where(FinancialStatement.period_type == "annual")
    ).all()

    if not rows:
        logger.info("No annual financial statement rows found for enrichment.")
        return df

    # Pivot to {ticker: {target_col: value}}
    enrichment: dict[str, dict[str, float | None]] = {}
    for instrument_id, line_item, value in rows:
        ticker = ticker_map.get(str(instrument_id))
        if ticker is None:
            continue
        _, target_col = _STMT_LINE_ITEMS[line_item]
        enrichment.setdefault(ticker, {})[target_col] = _to_float(value)

    _compute_asset_growth_from_statements(session, ticker_map, enrichment)

    if not enrichment:
        return df

    enrich_df = pd.DataFrame.from_dict(enrichment, orient="index")
    enrich_df.index.name = "ticker"

    n_before = df.notna().sum().sum()

    # Combine: existing data takes precedence, enrich fills NaN only
    df = df.combine_first(enrich_df)

    n_after = df.notna().sum().sum()
    n_filled = n_after - n_before
    logger.info(
        "Enriched fundamentals with %d values from financial statements (%d tickers).",
        n_filled,
        len(enrich_df),
    )

    return df


# Column mapping: TickerProfile ORM attr → fundamentals DataFrame column
_FUNDAMENTAL_COLUMNS: list[str] = [
    "market_cap",
    "enterprise_value",
    "book_value",
    "trailing_eps",
    "operating_cashflow",
    "total_revenue",
    "ebitda",
    "gross_profits",
    "return_on_equity",
    "operating_margins",
    "profit_margins",
    "current_price",
    "dividend_yield",
    "trailing_annual_dividend_yield",
    "beta",
    "shares_outstanding",
    "total_cash",
    "total_debt",
    "free_cashflow",
    "revenue_growth",
    "earnings_growth",
]


def _dedup_fundamentals_df(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate cross-listed tickers by currency priority + completeness.

    Sorts each group of rows sharing the same yfinance_ticker by:
      1. Currency priority rank (USD < GBP < EUR < GBX < others)
      2. Number of NaN values in fundamental columns (fewer NaNs wins)

    After sorting, ``keep="first"`` always selects the deterministically
    best row.  Helper columns are dropped before returning.
    """
    df = df.copy()
    fundamental_cols = [c for c in _FUNDAMENTAL_COLUMNS if c in df.columns]

    df["_ccy_rank"] = df["_raw_currency"].map(
        lambda c: currency_dedup_rank(c if isinstance(c, str) else None)
    )
    df["_nan_count"] = df[fundamental_cols].isna().sum(axis=1)

    df = df.sort_values(["_ccy_rank", "_nan_count"])

    # Log resolved duplicates before dedup
    dup_mask = df.index.duplicated(keep=False)
    if dup_mask.any():
        dup_df = df.loc[dup_mask]
        for ticker_val in dup_df.index.unique():
            group = dup_df.loc[[ticker_val]]
            chosen_ccy = group.iloc[0]["_raw_currency"]
            logger.info(
                "Dedup %s: %d cross-listed candidates, selected currency=%s, "
                "dropped %d listing(s).",
                ticker_val,
                len(group),
                chosen_ccy,
                len(group) - 1,
            )

    df = df[~df.index.duplicated(keep="first")]
    df = df.drop(columns=["_raw_currency", "_ccy_rank", "_nan_count"])
    return df


# ---------------------------------------------------------------------------
# Public assembly functions
# ---------------------------------------------------------------------------


def assemble_prices(
    session: Session,
    include_delisted: bool = True,
    currency_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build a ``dates x tickers`` close-price DataFrame.

    Parameters
    ----------
    include_delisted : bool, default=True
        When ``True`` (default), delisted instruments are included in the
        price history up to and including their delisting date.  A synthetic
        price row is appended on the delisting date so that
        ``prices_to_returns`` produces the correct final (delisting) return.

        When ``False``, only currently active instruments are included,
        reproducing the original survivorship-biased behaviour.

    Returns
    -------
    pd.DataFrame
        Index = ``pd.DatetimeIndex``, columns = yfinance tickers.
    """
    ticker_rank_map = _build_ticker_rank_map(session, include_delisted=include_delisted)

    price_query = select(
        PriceHistory.instrument_id,
        PriceHistory.date,
        PriceHistory.close,
    ).order_by(PriceHistory.date)

    if not include_delisted:
        price_query = price_query.join(Instrument).where(
            Instrument.delisted_at.is_(None)
        )

    rows = session.execute(price_query).all()

    if not rows:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for instrument_id, row_date, close in rows:
        info = ticker_rank_map.get(str(instrument_id))
        if info is None:
            continue
        ticker, ccy_rank = info
        records.append(
            {
                "date": pd.Timestamp(row_date),
                "ticker": ticker,
                "close": _to_float(close),
                "_ccy_rank": ccy_rank,
            }
        )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Deduplicate cross-listed instruments that share the same yfinance_ticker
    # using a deterministic currency-priority tiebreaker (USD < GBP < EUR < …).
    pivoted = _pivot_with_dedup(df, "date", "ticker", "close", "assemble_prices")
    pivoted.index = pd.DatetimeIndex(pivoted.index)
    pivoted = pivoted.sort_index()

    # Append synthetic delisting-date price rows so that prices_to_returns()
    # produces the correct final return for each delisted instrument.
    if include_delisted and not pivoted.empty:
        delisting_rows = session.execute(
            select(
                Instrument.yfinance_ticker,
                Instrument.delisted_at,
                Instrument.delisting_return,
            )
            .where(Instrument.delisted_at.isnot(None))
            .where(Instrument.yfinance_ticker.isnot(None))
        ).all()

        delistings = [
            (
                yf_ticker,
                pd.Timestamp(delisted_at),
                float(dr) if dr is not None else -0.30,
            )
            for yf_ticker, delisted_at, dr in delisting_rows
            if yf_ticker in pivoted.columns
        ]
        pivoted = _apply_delisting_returns(pivoted, delistings)

    # Normalise minor-unit prices (GBX → GBP, etc.) so that ADDV
    # computation and factor construction use consistent values.
    # Prefer the caller-supplied currency_map (avoids a second DB query
    # when called from assemble_all); fall back to a direct Instrument
    # query for standalone callers.
    effective_map = (
        currency_map
        if currency_map is not None
        else _build_currency_map_from_instruments(session)
    )
    if effective_map:
        pivoted = normalize_prices(pivoted, effective_map)

    return pivoted


def assemble_volumes(
    session: Session,
    include_delisted: bool = True,
) -> pd.DataFrame:
    """Build a ``dates x tickers`` volume DataFrame.

    Parameters
    ----------
    include_delisted : bool, default=True
        When ``False``, volume data for delisted instruments is excluded.

    Returns
    -------
    pd.DataFrame
        Index = ``pd.DatetimeIndex``, columns = yfinance tickers.
    """
    ticker_rank_map = _build_ticker_rank_map(session, include_delisted=include_delisted)

    vol_query = select(
        PriceHistory.instrument_id,
        PriceHistory.date,
        PriceHistory.volume,
    ).order_by(PriceHistory.date)

    if not include_delisted:
        vol_query = vol_query.join(Instrument).where(Instrument.delisted_at.is_(None))

    rows = session.execute(vol_query).all()

    if not rows:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for instrument_id, date, volume in rows:
        info = ticker_rank_map.get(str(instrument_id))
        if info is None:
            continue
        ticker, ccy_rank = info
        records.append(
            {
                "date": pd.Timestamp(date),
                "ticker": ticker,
                "volume": _to_float(volume),
                "_ccy_rank": ccy_rank,
            }
        )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Deduplicate cross-listed instruments that share the same yfinance_ticker
    # using a deterministic currency-priority tiebreaker (USD < GBP < EUR < …).
    pivoted = _pivot_with_dedup(df, "date", "ticker", "volume", "assemble_volumes")
    pivoted.index = pd.DatetimeIndex(pivoted.index)
    pivoted = pivoted.sort_index()
    return pivoted


def assemble_fundamentals(
    session: Session,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    """Build a ``tickers x fields`` fundamentals DataFrame and sector map.

    Minor-unit currencies (GBX, ILA, ZAC) are normalised to their
    major-unit equivalents (÷100) so that downstream screening and
    factor construction receive consistent values.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, str], dict[str, str]]
        - Fundamentals DataFrame indexed by yfinance ticker.
        - ``{ticker: sector}`` mapping.
        - ``{ticker: currency_code}`` mapping (major-unit normalised).
    """
    profiles = (
        session.execute(
            select(TickerProfile).options(
                joinedload(TickerProfile.instrument).joinedload(Instrument.exchange)
            )
        )
        .scalars()
        .all()
    )

    if not profiles:
        return pd.DataFrame(), {}, {}

    # Build currency map from the already-loaded profiles (no extra query).
    currency_map = build_currency_map(list(profiles))

    fundamentals_records: list[dict[str, Any]] = []
    sector_mapping: dict[str, str] = {}

    for profile in profiles:
        instrument = profile.instrument
        if instrument is None or not instrument.yfinance_ticker:
            continue

        ticker = instrument.yfinance_ticker

        row: dict[str, Any] = {"ticker": ticker}
        for col in _FUNDAMENTAL_COLUMNS:
            row[col] = _to_float(getattr(profile, col, None))

        row["exchange"] = instrument.exchange_name
        row["_raw_currency"] = instrument.currency_code or profile.currency

        fundamentals_records.append(row)

        if profile.sector:
            sector_mapping[ticker] = profile.sector

    if not fundamentals_records:
        return pd.DataFrame(), {}, currency_map

    df = pd.DataFrame(fundamentals_records).set_index("ticker")
    # Multiple instruments can map to the same yfinance_ticker
    # (different exchanges).  Deterministic dedup: prefer listings with
    # higher-priority currencies (USD > GBP > EUR > GBX > others),
    # then prefer rows with fewer NaN fundamental columns.
    df = _dedup_fundamentals_df(df)

    # book_value from yfinance (bookValue) is per-share in listing currency
    # (GBX for LSE stocks).  Multiplied by shares_outstanding (a count),
    # the result is total book equity in listing currency.
    # normalize_fundamentals() below then divides by the minor-unit divisor
    # (÷100 for GBX) to convert total book equity to GBP.
    if "book_value" in df.columns and "shares_outstanding" in df.columns:
        df["book_value"] = df["book_value"] * df["shares_outstanding"]

    # Enrich with data from FinancialStatement EAV table
    ticker_map = _build_ticker_map(session)
    df = _enrich_from_financial_statements(session, df, ticker_map)

    # Normalise minor-unit currencies (GBX → GBP, ILA → ILS, etc.)
    # so downstream screening and factor construction see major-unit values.
    df, currency_map = normalize_fundamentals(df, currency_map)

    return df, sector_mapping, currency_map


def assemble_financial_statements(session: Session) -> pd.DataFrame:
    """Build financial statements DataFrame for screening.

    The universe screener expects columns: ``ticker``, ``period_type``,
    and optionally ``period_date``.

    Returns
    -------
    pd.DataFrame
        Rows with ``ticker``, ``statement_type``, ``period_type``,
        ``period_date`` columns.
    """
    ticker_map = _build_ticker_map(session)

    rows = session.execute(
        select(
            FinancialStatement.instrument_id,
            FinancialStatement.statement_type,
            FinancialStatement.period_type,
            FinancialStatement.period_date,
        )
    ).all()

    if not rows:
        cols = ["ticker", "statement_type", "period_type", "period_date"]
        return pd.DataFrame(columns=cols)

    records: list[dict[str, Any]] = []
    for instrument_id, stmt_type, period_type, period_date in rows:
        ticker = ticker_map.get(str(instrument_id))
        if ticker is None:
            continue
        records.append(
            {
                "ticker": ticker,
                "statement_type": stmt_type,
                "period_type": period_type,
                "period_date": period_date,
            }
        )

    return pd.DataFrame(records)


def assemble_analyst_data(session: Session) -> pd.DataFrame:
    """Build analyst recommendation DataFrame for factor construction.

    Returns
    -------
    pd.DataFrame
        Rows with ``ticker``, ``strong_buy``, ``buy``, ``hold``,
        ``sell``, ``strong_sell`` columns.
    """
    ticker_map = _build_ticker_map(session)

    rows = session.execute(
        select(
            AnalystRecommendation.instrument_id,
            AnalystRecommendation.period,
            AnalystRecommendation.strong_buy,
            AnalystRecommendation.buy,
            AnalystRecommendation.hold,
            AnalystRecommendation.sell,
            AnalystRecommendation.strong_sell,
        )
    ).all()

    if not rows:
        cols = [
            "ticker",
            "period",
            "strong_buy",
            "buy",
            "hold",
            "sell",
            "strong_sell",
        ]
        return pd.DataFrame(columns=cols)

    records: list[dict[str, Any]] = []
    for instrument_id, period, sb, b, h, s, ss in rows:
        ticker = ticker_map.get(str(instrument_id))
        if ticker is None:
            continue
        records.append(
            {
                "ticker": ticker,
                "period": period,
                "strong_buy": sb or 0,
                "buy": b or 0,
                "hold": h or 0,
                "sell": s or 0,
                "strong_sell": ss or 0,
            }
        )

    return pd.DataFrame(records)


def assemble_insider_data(session: Session) -> pd.DataFrame:
    """Build insider transaction DataFrame for factor construction.

    Returns
    -------
    pd.DataFrame
        Rows with ``ticker``, ``shares``, ``transaction_type`` columns.
    """
    ticker_map = _build_ticker_map(session)

    rows = session.execute(
        select(
            InsiderTransaction.instrument_id,
            InsiderTransaction.shares,
            InsiderTransaction.transaction_type,
            InsiderTransaction.start_date,
        )
    ).all()

    if not rows:
        cols = ["ticker", "shares", "transaction_type", "start_date"]
        return pd.DataFrame(columns=cols)

    records: list[dict[str, Any]] = []
    for instrument_id, shares, tx_type, start_date in rows:
        ticker = ticker_map.get(str(instrument_id))
        if ticker is None:
            continue
        records.append(
            {
                "ticker": ticker,
                "shares": shares or 0,
                "transaction_type": tx_type,
                "start_date": start_date,
            }
        )

    return pd.DataFrame(records)
