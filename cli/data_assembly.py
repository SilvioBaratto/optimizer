"""Assemble optimizer-ready DataFrames from database ORM rows.

This module is the glue layer between the API data layer (PostgreSQL)
and the optimizer library.  It queries the database tables and pivots /
reshapes ORM rows into the exact DataFrame shapes that
``run_full_pipeline_with_selection()`` expects.
"""

from __future__ import annotations

import datetime
import logging
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

# Ensure the api package is importable from the CLI context.
_api_path = Path(__file__).parent.parent / "api"
if str(_api_path) not in sys.path:
    sys.path.insert(0, str(_api_path))

from app.database import DatabaseManager
from app.models.macro_regime import (
    BondYield,
    BondYieldObservation,
    EconomicIndicator,
    EconomicIndicatorObservation,
    FredObservation,
    MacroNewsSummary,
    TradingEconomicsIndicator,
    TradingEconomicsObservation,
)
from app.models.universe import Instrument
from app.models.yfinance_data import (
    AnalystRecommendation,
    FinancialStatement,
    InsiderTransaction,
    PriceHistory,
    TickerProfile,
)

from cli._currency import (
    build_currency_map,
    currency_dedup_rank,
    normalize_fundamentals,
    normalize_prices,
)

logger = logging.getLogger(__name__)

# Number of trading days per year (equity convention).
_TRADING_DAYS: int = 252

# Line items to extract from the FinancialStatement EAV table.
# Mapping: DB line_item -> (statement_type, target_column_name)
_STMT_LINE_ITEMS: dict[str, tuple[str, str]] = {
    "Net Income": ("income_statement", "net_income"),
    "Gross Profit": ("income_statement", "gross_profit"),
    "Operating Income": ("income_statement", "operating_income"),
    "Total Assets": ("balance_sheet", "total_assets"),
    "Stockholders Equity": ("balance_sheet", "total_equity"),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_float(val: Any) -> float | None:
    """Coerce a DB value (Decimal / int / None) to float."""
    if val is None:
        return None
    if isinstance(val, Decimal):
        return float(val)
    return float(val)


def _build_ticker_map(
    session: Session, include_delisted: bool = True
) -> dict[str, str]:
    """Return {instrument_id_hex: yfinance_ticker} for instruments.

    Parameters
    ----------
    include_delisted : bool, default=True
        When ``False``, exclude instruments with a non-null ``delisted_at``.
    """
    stmt = (
        select(Instrument.id, Instrument.yfinance_ticker)
        .where(Instrument.yfinance_ticker.isnot(None))
        .where(Instrument.yfinance_ticker != "")
    )
    if not include_delisted:
        stmt = stmt.where(Instrument.delisted_at.is_(None))
    rows = session.execute(stmt).all()
    return {str(r[0]): r[1] for r in rows}


def _build_currency_map_from_instruments(session: Session) -> dict[str, str]:
    """Return {yfinance_ticker: currency_code} from the Instrument table.

    Lightweight query for price/volume normalisation when TickerProfile
    is not loaded.
    """
    rows = session.execute(
        select(Instrument.yfinance_ticker, Instrument.currency_code)
        .where(Instrument.yfinance_ticker.isnot(None))
        .where(Instrument.yfinance_ticker != "")
        .where(Instrument.currency_code.isnot(None))
    ).all()
    return {str(t): str(c) for t, c in rows}


def _apply_delisting_returns(
    prices: pd.DataFrame,
    delistings: list[tuple[str, pd.Timestamp, float]],
) -> pd.DataFrame:
    """Append a synthetic delisting-date price row for each delisted instrument.

    For each ``(yf_ticker, delisted_at, delisting_return)`` tuple, finds the
    last known close price on or before ``delisted_at`` and adds a synthetic
    price row at ``delisted_at`` equal to ``last_price * (1 + delisting_return)``.

    When ``prices_to_returns`` is subsequently applied, this synthetic row
    produces the correct delisting return as the final observation for that
    instrument.

    Parameters
    ----------
    prices : pd.DataFrame
        dates × tickers close-price DataFrame.
    delistings : list[tuple[str, pd.Timestamp, float]]
        ``(yf_ticker, delisted_at, delisting_return)`` for each delisted
        instrument.  Instruments not in ``prices.columns`` are silently
        skipped.

    Returns
    -------
    pd.DataFrame
        Copy of ``prices`` with synthetic delisting rows appended and sorted.
    """
    if not delistings:
        return prices

    out = prices.copy()

    for yf_ticker, delisted_ts, r in delistings:
        if yf_ticker not in out.columns:
            continue

        col = out[yf_ticker].dropna()
        if col.empty:
            continue

        # Last known price on or before the delisting date.
        before = col[col.index <= delisted_ts]
        if before.empty:
            continue

        last_price = float(before.iloc[-1])
        synthetic_price = last_price * (1.0 + r)

        # Add a new index row for the delisting date if not already present.
        if delisted_ts not in out.index:
            new_row = pd.Series(dict.fromkeys(out.columns, np.nan), name=delisted_ts)
            out = pd.concat([out, new_row.to_frame().T])
            out = out.sort_index()

        # Only write synthetic price if the cell is currently NaN.
        if pd.isna(out.loc[delisted_ts, yf_ticker]):
            out.loc[delisted_ts, yf_ticker] = synthetic_price

    return out


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
    ticker_map = _build_ticker_map(session, include_delisted=include_delisted)

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
        ticker = ticker_map.get(str(instrument_id))
        if ticker is None:
            continue
        records.append(
            {
                "date": pd.Timestamp(row_date),
                "ticker": ticker,
                "close": _to_float(close),
            }
        )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Multiple instruments can map to the same yfinance_ticker (e.g.
    # listed on different exchanges).  Use pivot_table with 'first' to
    # deduplicate gracefully instead of raising on duplicates.
    pivoted = df.pivot_table(
        index="date",
        columns="ticker",
        values="close",
        aggfunc="first",
    )
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


def assemble_delisting_returns(session: Session) -> dict[str, float]:
    """Build a ``{yfinance_ticker: delisting_return}`` mapping for delisted instruments.

    Used by ``run_full_pipeline()`` to apply the returns-space
    survivorship-bias correction after ``prices_to_returns()``.

    Returns
    -------
    dict[str, float]
        Empty when no delisted instruments exist.  Defaults to ``-0.30``
        when ``delisting_return`` is ``NULL`` in the database.
    """
    rows = session.execute(
        select(
            Instrument.yfinance_ticker,
            Instrument.delisting_return,
        )
        .where(Instrument.delisted_at.isnot(None))
        .where(Instrument.yfinance_ticker.isnot(None))
        .where(Instrument.yfinance_ticker != "")
    ).all()
    return {
        str(yf_ticker): float(dr) if dr is not None else -0.30
        for yf_ticker, dr in rows
    }


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
    ticker_map = _build_ticker_map(session, include_delisted=include_delisted)

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
        ticker = ticker_map.get(str(instrument_id))
        if ticker is None:
            continue
        records.append(
            {
                "date": pd.Timestamp(date),
                "ticker": ticker,
                "volume": _to_float(volume),
            }
        )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    pivoted = df.pivot_table(
        index="date",
        columns="ticker",
        values="volume",
        aggfunc="first",
    )
    pivoted.index = pd.DatetimeIndex(pivoted.index)
    pivoted = pivoted.sort_index()
    return pivoted


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
                joinedload(TickerProfile.instrument).joinedload(
                    Instrument.exchange
                )
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


def assemble_fundamental_history(
    session: Session,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Build a ``(period_date, ticker)`` panel from financial_statements EAV.

    Queries all historical financial statement rows for key line items,
    pivots them into a MultiIndex panel suitable for point-in-time slicing.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session.
    tickers : list[str] or None
        Restrict to these tickers.  ``None`` fetches all.

    Returns
    -------
    pd.DataFrame
        MultiIndex ``(period_date: pd.Timestamp, ticker: str)``.
        Columns: ``net_income``, ``gross_profit``, ``operating_income``,
        ``total_assets``, ``total_equity``, ``period_type``
        (``'annual'`` | ``'quarterly'``), ``asset_growth`` (float | NaN).

    Notes
    -----
    Financial statement monetary values (net_income, gross_profit,
    operating_income, total_assets, total_equity) are stored by yfinance
    in the company's **reporting currency** — GBP for UK-listed
    companies, USD for US companies, etc.  This is distinct from the
    listing quote currency: an LSE stock quoted in GBX (pence) still
    has balance-sheet data reported in GBP.

    Consequently, ``asset_growth`` — computed as the year-over-year ratio
    of ``total_assets`` values for the same ticker — is dimensionless
    and inherently currency-safe: the currency cancels in the division.
    No normalisation is applied or needed for this column.

    Callers combining ``total_assets`` (in reporting-currency units)
    with market data (market_cap, current_price) for cross-sectional
    ratios **must** ensure market data has been normalised from minor
    units to major units (e.g. via ``normalize_fundamentals()``) before
    computing ratios.
    """
    ticker_map = _build_ticker_map(session)
    if tickers is not None:
        inv = {v: k for k, v in ticker_map.items()}
        allowed_ids = {inv[t] for t in tickers if t in inv}
        ticker_map = {k: v for k, v in ticker_map.items() if k in allowed_ids}

    line_item_names = list(_STMT_LINE_ITEMS.keys())

    rows = session.execute(
        select(
            FinancialStatement.instrument_id,
            FinancialStatement.period_type,
            FinancialStatement.period_date,
            FinancialStatement.line_item,
            FinancialStatement.value,
        )
        .where(FinancialStatement.line_item.in_(line_item_names))
        .where(FinancialStatement.value.isnot(None))
        .order_by(FinancialStatement.instrument_id, FinancialStatement.period_date)
    ).all()

    if not rows:
        idx = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex([]), pd.Index([], dtype=str)],
            names=["period_date", "ticker"],
        )
        return pd.DataFrame(
            columns=[
                "net_income", "gross_profit", "operating_income",
                "total_assets", "total_equity", "period_type", "asset_growth",
            ],
            index=idx,
        )

    records: list[dict[str, Any]] = []
    for instrument_id, period_type, period_date, line_item, value in rows:
        ticker = ticker_map.get(str(instrument_id))
        if ticker is None:
            continue
        _, target_col = _STMT_LINE_ITEMS[line_item]
        records.append({
            "ticker": ticker,
            "period_date": pd.Timestamp(period_date),
            "period_type": period_type,
            "target_col": target_col,
            "value": _to_float(value),
        })

    if not records:
        idx = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex([]), pd.Index([], dtype=str)],
            names=["period_date", "ticker"],
        )
        return pd.DataFrame(
            columns=[
                "net_income", "gross_profit", "operating_income",
                "total_assets", "total_equity", "period_type", "asset_growth",
            ],
            index=idx,
        )

    raw = pd.DataFrame(records)

    # Pivot: one row per (period_date, ticker, period_type), columns = target_col
    pivoted = raw.pivot_table(
        index=["period_date", "ticker", "period_type"],
        columns="target_col",
        values="value",
        aggfunc="first",
    )
    pivoted = pivoted.reset_index()

    # Compute asset_growth per ticker from annual Total Assets (YoY)
    pivoted["asset_growth"] = np.nan
    annual_mask = pivoted["period_type"] == "annual"
    if annual_mask.any() and "total_assets" in pivoted.columns:
        annual = pivoted.loc[annual_mask].sort_values(["ticker", "period_date"])
        growth = annual.groupby("ticker")["total_assets"].pct_change()
        pivoted.loc[annual.index, "asset_growth"] = growth.values

    # Build MultiIndex (period_date, ticker)
    pivoted = pivoted.set_index(["period_date", "ticker"]).sort_index()

    # Ensure all expected columns exist
    for col in [
        "net_income", "gross_profit", "operating_income",
        "total_assets", "total_equity",
    ]:
        if col not in pivoted.columns:
            pivoted[col] = np.nan

    logger.info(
        "Assembled fundamental history panel: %d rows, %d tickers.",
        len(pivoted),
        pivoted.index.get_level_values("ticker").nunique(),
    )
    return pivoted


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


def assemble_macro_data(
    session: Session,
    country: str = "USA",
) -> pd.DataFrame:
    """Build macro DataFrame for regime classification.

    The regime classifier expects ``gdp_growth`` and/or ``yield_spread``
    columns.  The returned DataFrame is indexed by a ``DatetimeIndex``
    (derived from the most recent ``reference_date`` or bond date) so
    that the pipeline can apply point-in-time lag filtering.

    Parameters
    ----------
    country : str
        Country code as stored in the DB (e.g. ``"USA"``, ``"Germany"``).

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with ``gdp_growth`` and ``yield_spread``
        columns, indexed by date.
    """
    # GDP growth from TradingEconomics
    te_gdp = session.execute(
        select(TradingEconomicsIndicator).where(
            TradingEconomicsIndicator.country == country,
            TradingEconomicsIndicator.indicator_key == "GDP Growth Rate",
        )
    ).scalar_one_or_none()

    gdp_growth: float | None = None
    if te_gdp is not None and te_gdp.value is not None:
        gdp_growth = float(te_gdp.value)

    # Reference date from EconomicIndicator forecast row
    forecast_row = session.execute(
        select(EconomicIndicator).where(EconomicIndicator.country == country)
    ).scalar_one_or_none()

    ref_date: datetime.date | None = None
    if forecast_row is not None and forecast_row.reference_date is not None:
        ref_date = forecast_row.reference_date

    # Yield spread from bond yields (10Y - 2Y)
    bonds = (
        session.execute(select(BondYield).where(BondYield.country == country))
        .scalars()
        .all()
    )

    bond_map: dict[str, float] = {}
    bond_ref_date: datetime.date | None = None
    for bond in bonds:
        if bond.yield_value is not None:
            bond_map[bond.maturity] = float(bond.yield_value)
        if bond.reference_date is not None:
            bond_ref_date = bond.reference_date

    yield_spread: float | None = None
    lt_rate = bond_map.get("10Y")
    st_rate = bond_map.get("2Y")
    if lt_rate is not None and st_rate is not None:
        yield_spread = lt_rate - st_rate

    macro_row: dict[str, float | None] = {
        "gdp_growth": gdp_growth,
        "yield_spread": yield_spread,
    }

    # Try time-series first — if observation tables have multi-day history,
    # return that instead of the single-row snapshot.
    ts = assemble_macro_timeseries(session, country=country)
    if len(ts) >= 2:
        return ts

    # Fallback: single-row snapshot from latest-value tables.
    best_date = ref_date or bond_ref_date or datetime.date.today()
    index = pd.DatetimeIndex([pd.Timestamp(best_date)])

    return pd.DataFrame([macro_row], index=index)


def assemble_macro_timeseries(
    session: Session,
    country: str = "USA",
    start_date: datetime.date | None = None,
    end_date: datetime.date | None = None,
) -> pd.DataFrame:
    """Build a multi-row macro DataFrame from observation tables.

    Queries ``trading_economics_observations`` for GDP Growth Rate,
    ``bond_yield_observations`` for the 10Y-2Y yield spread, and
    ``economic_indicator_observations`` for IlSole forecast columns,
    producing a ``dates × indicators`` DataFrame suitable for the
    regime classifier's ``rolling(4)`` window.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session.
    country : str
        Country code as stored in the DB.
    start_date, end_date : datetime.date | None
        Optional date bounds.

    Returns
    -------
    pd.DataFrame
        Index = ``pd.DatetimeIndex``, columns include ``gdp_growth``,
        ``yield_spread``, and IlSole forecast columns when available.
        May be empty if observation tables have no data yet.
    """
    # GDP growth from TE observations
    gdp_stmt = (
        select(
            TradingEconomicsObservation.date,
            TradingEconomicsObservation.value,
        )
        .where(TradingEconomicsObservation.country == country)
        .where(TradingEconomicsObservation.indicator_key == "GDP Growth Rate")
    )
    if start_date:
        gdp_stmt = gdp_stmt.where(TradingEconomicsObservation.date >= start_date)
    if end_date:
        gdp_stmt = gdp_stmt.where(TradingEconomicsObservation.date <= end_date)
    gdp_stmt = gdp_stmt.order_by(TradingEconomicsObservation.date)
    gdp_rows = session.execute(gdp_stmt).all()

    # Bond yields from observations (10Y and 2Y)
    bond_stmt = (
        select(
            BondYieldObservation.date,
            BondYieldObservation.maturity,
            BondYieldObservation.yield_value,
        )
        .where(BondYieldObservation.country == country)
        .where(BondYieldObservation.maturity.in_(["10Y", "2Y"]))
    )
    if start_date:
        bond_stmt = bond_stmt.where(BondYieldObservation.date >= start_date)
    if end_date:
        bond_stmt = bond_stmt.where(BondYieldObservation.date <= end_date)
    bond_stmt = bond_stmt.order_by(BondYieldObservation.date)
    bond_rows = session.execute(bond_stmt).all()

    # IlSole forecast observations
    _ILSOLE_COLS = [
        "last_inflation", "inflation_6m", "inflation_10y_avg",
        "gdp_growth_6m", "earnings_12m", "eps_expected_12m",
        "peg_ratio", "lt_rate_forecast",
    ]
    ilsole_stmt = select(
        EconomicIndicatorObservation.date,
        *[getattr(EconomicIndicatorObservation, c) for c in _ILSOLE_COLS],
    ).where(EconomicIndicatorObservation.country == country)
    if start_date:
        ilsole_stmt = ilsole_stmt.where(
            EconomicIndicatorObservation.date >= start_date
        )
    if end_date:
        ilsole_stmt = ilsole_stmt.where(
            EconomicIndicatorObservation.date <= end_date
        )
    ilsole_stmt = ilsole_stmt.order_by(EconomicIndicatorObservation.date)
    ilsole_rows = session.execute(ilsole_stmt).all()

    # Build GDP series
    gdp_series = pd.Series(
        {pd.Timestamp(d): float(v) for d, v in gdp_rows if v is not None},
        dtype=float,
        name="gdp_growth",
    )

    # Build yield spread series (10Y - 2Y)
    bond_10y: dict[pd.Timestamp, float] = {}
    bond_2y: dict[pd.Timestamp, float] = {}
    for d, maturity, val in bond_rows:
        if val is None:
            continue
        ts = pd.Timestamp(d)
        if maturity == "10Y":
            bond_10y[ts] = float(val)
        elif maturity == "2Y":
            bond_2y[ts] = float(val)

    spread_dates = sorted(set(bond_10y.keys()) & set(bond_2y.keys()))
    spread_series = pd.Series(
        {d: bond_10y[d] - bond_2y[d] for d in spread_dates},
        dtype=float,
        name="yield_spread",
    )

    # Build IlSole forecast series dict
    ilsole_data: dict[str, dict[pd.Timestamp, float]] = {c: {} for c in _ILSOLE_COLS}
    for row in ilsole_rows:
        ts = pd.Timestamp(row[0])
        for i, col in enumerate(_ILSOLE_COLS):
            val = row[i + 1]
            if val is not None:
                ilsole_data[col][ts] = float(val)

    # Combine all series into DataFrame
    all_series: dict[str, pd.Series] = {
        "gdp_growth": gdp_series,
        "yield_spread": spread_series,
    }
    for col in _ILSOLE_COLS:
        if ilsole_data[col]:
            all_series[col] = pd.Series(ilsole_data[col], dtype=float, name=col)

    df = pd.DataFrame(all_series)
    df.index = pd.DatetimeIndex(df.index)
    df = df.sort_index()
    return df


def assemble_te_observations(
    session: Session,
    country: str = "USA",
    start_date: datetime.date | None = None,
) -> pd.DataFrame:
    """Build a dates x indicator_key DataFrame of Trading Economics observations.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session.
    country : str
        Country code (e.g. "USA", "Germany").
    start_date : datetime.date | None
        Optional lower bound on observation date.

    Returns
    -------
    pd.DataFrame
        Index = DatetimeIndex, columns = indicator_key strings
        (e.g. "manufacturing_pmi", "gdp_growth_rate").
    """
    stmt = select(
        TradingEconomicsObservation.date,
        TradingEconomicsObservation.indicator_key,
        TradingEconomicsObservation.value,
    ).where(TradingEconomicsObservation.country == country)

    if start_date is not None:
        stmt = stmt.where(TradingEconomicsObservation.date >= start_date)
    stmt = stmt.order_by(TradingEconomicsObservation.date)
    rows = session.execute(stmt).all()

    if not rows:
        return pd.DataFrame()

    records = [
        {"date": pd.Timestamp(d), "indicator_key": key, "value": float(val)}
        for d, key, val in rows
        if val is not None
    ]
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    pivoted = df.pivot_table(
        index="date", columns="indicator_key", values="value", aggfunc="first",
    )
    pivoted.index = pd.DatetimeIndex(pivoted.index)
    pivoted.columns.name = None
    return pivoted.sort_index()


def assemble_bond_observations(
    session: Session,
    country: str = "USA",
    start_date: datetime.date | None = None,
) -> pd.DataFrame:
    """Build a dates x maturity DataFrame of bond yield observations.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session.
    country : str
        Country code.
    start_date : datetime.date | None
        Optional lower bound.

    Returns
    -------
    pd.DataFrame
        Index = DatetimeIndex, columns = maturity strings ("2Y", "5Y", "10Y", "30Y").
    """
    stmt = select(
        BondYieldObservation.date,
        BondYieldObservation.maturity,
        BondYieldObservation.yield_value,
    ).where(BondYieldObservation.country == country)

    if start_date is not None:
        stmt = stmt.where(BondYieldObservation.date >= start_date)
    stmt = stmt.order_by(BondYieldObservation.date)
    rows = session.execute(stmt).all()

    if not rows:
        return pd.DataFrame()

    records = [
        {"date": pd.Timestamp(d), "maturity": mat, "yield_value": float(val)}
        for d, mat, val in rows
        if val is not None
    ]
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    pivoted = df.pivot_table(
        index="date", columns="maturity", values="yield_value", aggfunc="first",
    )
    pivoted.index = pd.DatetimeIndex(pivoted.index)
    pivoted.columns.name = None
    return pivoted.sort_index()


def assemble_sentiment(
    session: Session,
    start_date: datetime.date | None = None,
) -> pd.DataFrame:
    """Build a dates x country DataFrame of news sentiment scores.

    Queries ``macro_news_summaries`` for the ``sentiment_score`` field
    (continuous [-1, 1]) produced by the daily news pipeline.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session.
    start_date : datetime.date | None
        Optional lower bound on summary_date.

    Returns
    -------
    pd.DataFrame
        Index = DatetimeIndex (summary_date), columns = country strings.
        Values are sentiment scores in [-1, 1].
    """
    stmt = select(
        MacroNewsSummary.summary_date,
        MacroNewsSummary.country,
        MacroNewsSummary.sentiment_score,
    )
    if start_date is not None:
        stmt = stmt.where(MacroNewsSummary.summary_date >= start_date)
    stmt = stmt.order_by(MacroNewsSummary.summary_date)
    rows = session.execute(stmt).all()

    if not rows:
        return pd.DataFrame()

    records = [
        {"date": pd.Timestamp(d), "country": country, "sentiment_score": float(score)}
        for d, country, score in rows
        if score is not None
    ]
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    pivoted = df.pivot_table(
        index="date", columns="country", values="sentiment_score", aggfunc="first",
    )
    pivoted.index = pd.DatetimeIndex(pivoted.index)
    pivoted.columns.name = None
    return pivoted.sort_index()


# ---------------------------------------------------------------------------
# FRED time-series assembly
# ---------------------------------------------------------------------------

FRED_SERIES_IDS: list[str] = [
    # Credit & yield spreads (daily)
    "BAMLH0A0HYM2",
    "BAMLC0A0CM",
    "T10Y2Y",
    "BAA10Y",
    # Volatility (daily)
    "VIXCLS",
    # OECD CLI — amplitude adjusted (monthly)
    "USALOLITOAASTSAM",
    "DEULOLITOAASTSAM",
    "FRALOLITOAASTSAM",
    "GBRLOLITOAASTSAM",
    # US recession indicators (monthly/quarterly)
    "RECPROUSM156N",
    "JHGDPBRINDX",
    "USREC",
    # Risk-free rate proxy (daily, annualized %)
    "DGS3MO",
]


def assemble_fred_series(
    session: Session,
    series_ids: list[str] | None = None,
    start_date: datetime.date | None = None,
) -> pd.DataFrame:
    """Build a ``dates x series_id`` DataFrame of FRED observations.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session.
    series_ids : list[str] | None
        Series to include. ``None`` uses ``FRED_SERIES_IDS``.
    start_date : datetime.date | None
        Optional lower bound on observation date.

    Returns
    -------
    pd.DataFrame
        Index = ``pd.DatetimeIndex`` (daily), columns = FRED series IDs.
        Values are floats; missing observations are NaN (not forward-filled).
    """
    ids = series_ids if series_ids is not None else FRED_SERIES_IDS

    stmt = select(
        FredObservation.series_id,
        FredObservation.date,
        FredObservation.value,
    ).where(FredObservation.series_id.in_(ids))

    if start_date is not None:
        stmt = stmt.where(FredObservation.date >= start_date)

    stmt = stmt.order_by(FredObservation.date)
    rows = session.execute(stmt).all()

    if not rows:
        return pd.DataFrame(columns=ids)

    records = [
        {
            "date": pd.Timestamp(row_date),
            "series_id": sid,
            "value": float(val) if val is not None else np.nan,
        }
        for sid, row_date, val in rows
    ]

    df = pd.DataFrame(records)
    pivoted = df.pivot_table(
        index="date",
        columns="series_id",
        values="value",
        aggfunc="first",
    )
    pivoted.index = pd.DatetimeIndex(pivoted.index)
    pivoted.columns.name = None
    return pivoted.sort_index()


# ---------------------------------------------------------------------------
# Regime data merging
# ---------------------------------------------------------------------------

# Column translation maps: DB column names → classify_regime() expected names
_FRED_REGIME_MAP: dict[str, str] = {
    "T10Y2Y": "spread_2s10s",
    "BAMLH0A0HYM2": "hy_oas",
}
_TE_REGIME_MAP: dict[str, str] = {
    "manufacturing_pmi": "pmi",
}


def assemble_regime_data(
    macro_data: pd.DataFrame,
    fred_data: pd.DataFrame,
    te_observations: pd.DataFrame,
    sentiment_data: pd.DataFrame | None = None,
    sentiment_country: str = "USA",
) -> pd.DataFrame:
    """Merge macro indicators into a single DataFrame for regime classification.

    Combines columns from ``macro_data`` (GDP growth, yield spread),
    ``fred_data`` (2s10s spread, HY OAS), ``te_observations`` (PMI),
    and optionally ``sentiment_data`` into the column names that
    :func:`optimizer.factors.classify_regime` expects.

    Parameters
    ----------
    macro_data : pd.DataFrame
        GDP/yield-spread macro data (dates x columns).
    fred_data : pd.DataFrame
        FRED time-series (dates x series_ids).
    te_observations : pd.DataFrame
        Trading Economics observations (dates x indicator_key).
    sentiment_data : pd.DataFrame or None
        News sentiment (dates x country).
    sentiment_country : str
        Country column to extract from ``sentiment_data``.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with any subset of ``gdp_growth``,
        ``yield_spread``, ``pmi``, ``spread_2s10s``, ``hy_oas``,
        ``sentiment``.  Never raises on missing data.
    """
    parts: list[pd.DataFrame] = []

    # FRED columns (daily granularity — used as base index)
    fred_cols: dict[str, pd.Series] = {}
    for fred_id, regime_name in _FRED_REGIME_MAP.items():
        if fred_id in fred_data.columns:
            s = fred_data[fred_id].dropna()
            if len(s) > 0:
                fred_cols[regime_name] = s

    if fred_cols:
        fred_df = pd.DataFrame(fred_cols)
        parts.append(fred_df)

    # TE observations (monthly — forward-fill onto daily base)
    te_cols: dict[str, pd.Series] = {}
    for te_key, regime_name in _TE_REGIME_MAP.items():
        if te_key in te_observations.columns:
            s = te_observations[te_key].dropna()
            if len(s) > 0:
                te_cols[regime_name] = s

    if te_cols:
        te_df = pd.DataFrame(te_cols)
        parts.append(te_df)

    # Sentiment
    if (
        sentiment_data is not None
        and sentiment_country in sentiment_data.columns
    ):
        sent = sentiment_data[sentiment_country].dropna()
        if len(sent) > 0:
            parts.append(sent.rename("sentiment").to_frame())

    # Macro baseline columns (gdp_growth, yield_spread)
    macro_cols = [c for c in ("gdp_growth", "yield_spread") if c in macro_data.columns]
    if macro_cols:
        parts.append(macro_data[macro_cols].dropna(how="all"))

    if not parts:
        return pd.DataFrame()

    # Outer-join all parts on date index, then forward-fill
    merged = parts[0]
    for p in parts[1:]:
        merged = merged.join(p, how="outer")

    merged = merged.sort_index().ffill()

    return merged


def assemble_fx_rates(
    currency_map: dict[str, str],
    base_currency: str,
    price_index: pd.DatetimeIndex,
    *,
    cross_via_usd: bool = True,
) -> pd.DataFrame:
    """Fetch FX rates from yfinance for all foreign currencies in the map.

    Downloads exchange rate data and returns a DataFrame where each column
    holds units-of-base per one unit-of-foreign currency (the format
    expected by :class:`~optimizer.fx.FxPriceConverter`).

    Parameters
    ----------
    currency_map : dict[str, str]
        ``{yfinance_ticker: ISO_currency_code}`` mapping.
    base_currency : str
        Target base currency (e.g. ``"EUR"``).
    price_index : pd.DatetimeIndex
        Date index from the price DataFrame (used to determine the
        download range).
    cross_via_usd : bool
        Compute cross rates via USD for non-USD pairs.

    Returns
    -------
    pd.DataFrame
        Index = dates, columns = foreign currency codes, values =
        units-of-base per one unit-of-foreign.  Empty DataFrame when
        no foreign currencies are needed or download fails.
    """
    import yfinance as yf

    from optimizer.fx._rates import (
        build_fx_pair_ticker,
        compute_cross_rate,
        required_fx_currencies,
    )

    base = base_currency.upper()

    if len(price_index) == 0:
        logger.warning("price_index is empty; cannot determine FX date range.")
        return pd.DataFrame()

    foreign_ccys = required_fx_currencies(currency_map, base)

    if not foreign_ccys:
        logger.info("No foreign currencies to fetch FX rates for.")
        return pd.DataFrame(index=price_index)

    # Build yfinance ticker list and remember the mapping
    all_tickers: set[str] = set()
    pair_info: dict[str, str | tuple[str, str]] = {}

    for ccy in sorted(foreign_ccys):
        pair = build_fx_pair_ticker(ccy, base, cross_via_usd=cross_via_usd)
        if pair is None:
            continue
        pair_info[ccy] = pair
        if isinstance(pair, tuple):
            all_tickers.update(pair)
        else:
            all_tickers.add(pair)

    if not all_tickers:
        return pd.DataFrame(index=price_index)

    # Download all needed tickers in one call
    start = price_index[0] - pd.Timedelta(days=7)
    end = price_index[-1] + pd.Timedelta(days=1)
    tickers_list = sorted(all_tickers)

    logger.info(
        "Downloading FX rates for %d pair(s): %s",
        len(tickers_list),
        tickers_list,
    )

    try:
        data = yf.download(
            tickers_list,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
    except Exception:
        logger.exception("Failed to download FX rates from yfinance.")
        return pd.DataFrame(index=price_index)

    if data is None or data.empty:
        logger.warning("yfinance returned empty FX rate data.")
        return pd.DataFrame(index=price_index)

    # Extract Close prices — handle single vs multi-ticker column formats
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        # Single ticker → flat columns
        close = data[["Close"]].copy()
        close.columns = [tickers_list[0]]

    # Build rate DataFrame (units of base per one unit of foreign)
    rates: dict[str, pd.Series] = {}

    for ccy, pair in pair_info.items():
        if isinstance(pair, tuple):
            from_ticker, to_ticker = pair
            if from_ticker not in close.columns or to_ticker not in close.columns:
                logger.warning(
                    "Missing FX data for cross-rate %s: need %s and %s",
                    ccy,
                    from_ticker,
                    to_ticker,
                )
                continue
            rate = compute_cross_rate(close[from_ticker], close[to_ticker])
            rates[ccy] = rate
        else:
            if pair not in close.columns:
                logger.warning("Missing FX data for %s: %s", ccy, pair)
                continue
            downloaded = close[pair]
            if ccy.upper() == "USD":
                # Ticker is {base}USD=X → gives USD per 1 base → reciprocal
                rates[ccy] = 1.0 / downloaded
            else:
                # Ticker gives base per 1 foreign → direct
                rates[ccy] = downloaded

    if not rates:
        logger.warning("Could not compute any FX rates.")
        return pd.DataFrame(index=price_index)

    result = pd.DataFrame(rates)
    logger.info(
        "Assembled FX rates: %d currencies, %d observations.",
        len(result.columns),
        len(result),
    )
    return result


# ---------------------------------------------------------------------------
# All-in-one assembly
# ---------------------------------------------------------------------------


class DataAssembly:
    """Assembles all DataFrames from the database in a single pass.

    Attributes
    ----------
    prices : pd.DataFrame
        dates x tickers close prices.
    volumes : pd.DataFrame
        dates x tickers volume.
    fundamentals : pd.DataFrame
        tickers x fields cross-sectional data.
    sector_mapping : dict[str, str]
        ticker -> sector.
    financial_statements : pd.DataFrame
        Rows with ticker/period_type/period_date.
    analyst_data : pd.DataFrame
        Rows with ticker/strong_buy/buy/hold/sell/strong_sell.
    insider_data : pd.DataFrame
        Rows with ticker/shares/transaction_type.
    macro_data : pd.DataFrame
        gdp_growth and yield_spread.
    fred_data : pd.DataFrame
        FRED time-series (dates x series_ids).
    te_observations : pd.DataFrame
        dates x indicator_key Trading Economics observations.
    bond_observations : pd.DataFrame
        dates x maturity bond yield observations.
    sentiment_data : pd.DataFrame
        dates x country news sentiment scores.
    regime_data : pd.DataFrame
        Merged macro indicators for regime classification
        (pmi, spread_2s10s, hy_oas, sentiment, gdp_growth, yield_spread).
    fundamental_history : pd.DataFrame
        MultiIndex ``(period_date, ticker)`` panel of historical financial
        statements for point-in-time factor construction.
    include_delisted : bool
        Whether delisted instruments are included in ``prices``.
    delisting_returns : dict[str, float]
        Mapping of yfinance_ticker → terminal delisting return for each
        delisted instrument.  Used by ``run_full_pipeline()`` for the
        returns-space survivorship-bias correction.
    currency_map : dict[str, str]
        ``{yfinance_ticker: currency_code}`` mapping (major-unit normalised,
        e.g. ``"GBP"`` not ``"GBX"``).  Used to activate FX conversion in
        ``run_full_pipeline()`` and for downstream currency-aware logic.
    fx_rates : pd.DataFrame
        FX rate DataFrame (dates x currency codes).  Each column holds
        units-of-base per one unit-of-foreign (e.g. EUR per 1 GBP).
        Used by ``FxPriceConverter`` to convert local-currency prices
        to the base currency.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        fundamentals: pd.DataFrame,
        sector_mapping: dict[str, str],
        financial_statements: pd.DataFrame,
        analyst_data: pd.DataFrame,
        insider_data: pd.DataFrame,
        macro_data: pd.DataFrame,
        fred_data: pd.DataFrame | None = None,
        te_observations: pd.DataFrame | None = None,
        bond_observations: pd.DataFrame | None = None,
        sentiment_data: pd.DataFrame | None = None,
        regime_data: pd.DataFrame | None = None,
        fundamental_history: pd.DataFrame | None = None,
        include_delisted: bool = True,
        delisting_returns: dict[str, float] | None = None,
        currency_map: dict[str, str] | None = None,
        fx_rates: pd.DataFrame | None = None,
    ) -> None:
        self.prices = prices
        self.volumes = volumes
        self.fundamentals = fundamentals
        self.sector_mapping = sector_mapping
        self.financial_statements = financial_statements
        self.analyst_data = analyst_data
        self.insider_data = insider_data
        self.macro_data = macro_data
        self.fred_data = fred_data if fred_data is not None else pd.DataFrame()
        self.te_observations = te_observations if te_observations is not None else pd.DataFrame()
        self.bond_observations = bond_observations if bond_observations is not None else pd.DataFrame()
        self.sentiment_data = sentiment_data if sentiment_data is not None else pd.DataFrame()
        self.regime_data = regime_data if regime_data is not None else pd.DataFrame()
        self.fundamental_history = (
            fundamental_history if fundamental_history is not None
            else pd.DataFrame()
        )
        self.include_delisted = include_delisted
        self.delisting_returns: dict[str, float] = delisting_returns or {}
        self.currency_map: dict[str, str] = currency_map or {}
        self.fx_rates = fx_rates if fx_rates is not None else pd.DataFrame()

    @property
    def n_tickers(self) -> int:
        return len(self.prices.columns)

    @property
    def n_trading_days(self) -> int:
        return len(self.prices)

    @property
    def risk_free_rate_series(self) -> pd.Series:
        """Daily compounded risk-free rate from DGS3MO.

        Returns per-day decimal: ``(1 + annual_pct/100)^(1/252) - 1``.
        Empty Series when DGS3MO is absent from ``fred_data``.
        """
        if self.fred_data.empty or "DGS3MO" not in self.fred_data.columns:
            return pd.Series(dtype=float, name="risk_free_rate")
        raw = self.fred_data["DGS3MO"].dropna()
        return ((1 + raw / 100) ** (1.0 / _TRADING_DAYS) - 1).rename(
            "risk_free_rate"
        )

    @property
    def risk_free_rate(self) -> float:
        """Latest daily compounded risk-free rate scalar.

        Returns 0.0 when DGS3MO is unavailable.
        """
        series = self.risk_free_rate_series
        if series.empty:
            logger.warning(
                "DGS3MO not found in fred_data; using rf=0.0. "
                "Run 'python -m cli fred fetch' to populate."
            )
            return 0.0
        return float(series.iloc[-1])

    def summary(self) -> dict[str, Any]:
        rf_series = self.risk_free_rate_series
        return {
            "tickers": self.n_tickers,
            "trading_days": self.n_trading_days,
            "fundamentals_rows": len(self.fundamentals),
            "financial_statements": len(self.financial_statements),
            "analyst_records": len(self.analyst_data),
            "insider_records": len(self.insider_data),
            "sectors": len(set(self.sector_mapping.values())),
            "has_macro": len(self.macro_data) > 0,
            "fred_observations": len(self.fred_data),
            "te_observations": len(self.te_observations),
            "bond_observations": len(self.bond_observations),
            "sentiment_days": len(self.sentiment_data),
            "regime_data_rows": len(self.regime_data),
            "fundamental_history_rows": len(self.fundamental_history),
            "risk_free_rate_pct": (
                round(float(rf_series.iloc[-1]) * _TRADING_DAYS * 100, 4)
                if not rf_series.empty
                else None
            ),
            "risk_free_rate_obs": len(rf_series),
            "delisted_tickers": len(self.delisting_returns),
            "currency_map_tickers": len(self.currency_map),
            "fx_rates_currencies": (
                len(self.fx_rates.columns)
                if not self.fx_rates.empty
                else 0
            ),
            "fx_rates_observations": len(self.fx_rates),
        }


def assemble_all(
    db_manager: DatabaseManager,
    macro_country: str = "USA",
    include_delisted: bool = True,
    base_currency: str = "EUR",
) -> DataAssembly:
    """Query the database and assemble all DataFrames.

    Parameters
    ----------
    db_manager : DatabaseManager
        Initialized database manager.
    macro_country : str
        Country for macro regime data.
    include_delisted : bool, default=True
        Whether to include delisted instruments in prices and volumes.
        Pass ``False`` to reproduce the original survivorship-biased
        behaviour (e.g. for backward-compatibility checks).
    base_currency : str, default="EUR"
        Target base currency for FX rate assembly (e.g. ``"EUR"``,
        ``"USD"``, ``"GBP"``).  Must match the ``FxConfig.base_currency``
        used downstream.

    Returns
    -------
    DataAssembly
        All assembled DataFrames ready for the optimizer.
    """
    with db_manager.get_session() as session:
        logger.info("Assembling fundamentals...")
        fundamentals, sector_mapping, currency_map = assemble_fundamentals(session)

        logger.info("Assembling price data (include_delisted=%s)...", include_delisted)
        prices = assemble_prices(
            session,
            include_delisted=include_delisted,
            currency_map=currency_map,
        )

        logger.info("Assembling volume data...")
        volumes = assemble_volumes(session, include_delisted=include_delisted)

        logger.info("Assembling financial statements...")
        financial_statements = assemble_financial_statements(session)

        logger.info("Assembling analyst data...")
        analyst_data = assemble_analyst_data(session)

        logger.info("Assembling insider data...")
        insider_data = assemble_insider_data(session)

        logger.info("Assembling macro data...")
        macro_data = assemble_macro_data(session, country=macro_country)

        logger.info("Assembling FRED time-series data...")
        fred_data = assemble_fred_series(session)

        logger.info("Assembling Trading Economics observations...")
        te_observations = assemble_te_observations(session, country=macro_country)

        logger.info("Assembling bond yield observations...")
        bond_observations = assemble_bond_observations(session, country=macro_country)

        logger.info("Assembling news sentiment data...")
        sentiment_data = assemble_sentiment(session)

        logger.info("Assembling fundamental history panel...")
        fundamental_history = assemble_fundamental_history(session)

        logger.info("Assembling delisting returns...")
        delisting_returns = assemble_delisting_returns(session)

    logger.info("Assembling composite regime data...")
    regime_data = assemble_regime_data(
        macro_data, fred_data, te_observations, sentiment_data,
    )

    logger.info("Assembling FX rates...")
    if prices.empty:
        logger.warning("Prices DataFrame is empty; skipping FX rate assembly.")
        fx_rates = pd.DataFrame()
    else:
        fx_rates = assemble_fx_rates(
            currency_map=currency_map,
            base_currency=base_currency,
            price_index=prices.index,
        )

    return DataAssembly(
        prices=prices,
        volumes=volumes,
        fundamentals=fundamentals,
        sector_mapping=sector_mapping,
        financial_statements=financial_statements,
        analyst_data=analyst_data,
        insider_data=insider_data,
        macro_data=macro_data,
        fred_data=fred_data,
        te_observations=te_observations,
        bond_observations=bond_observations,
        sentiment_data=sentiment_data,
        regime_data=regime_data,
        fundamental_history=fundamental_history,
        include_delisted=include_delisted,
        delisting_returns=delisting_returns,
        currency_map=currency_map,
        fx_rates=fx_rates,
    )
