"""Assemble optimizer-ready DataFrames from database ORM rows.

This module is the glue layer between the API data layer (PostgreSQL)
and the optimizer library.  It queries the database tables and pivots /
reshapes ORM rows into the exact DataFrame shapes that
``run_full_pipeline_with_selection()`` expects.
"""

from __future__ import annotations

import logging
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

# Ensure the api package is importable from the CLI context.
_api_path = Path(__file__).parent.parent / "api"
if str(_api_path) not in sys.path:
    sys.path.insert(0, str(_api_path))

from app.database import DatabaseManager
from app.models.macro_regime import BondYield, EconomicIndicator
from app.models.universe import Instrument
from app.models.yfinance_data import (
    AnalystRecommendation,
    FinancialStatement,
    InsiderTransaction,
    PriceHistory,
    TickerProfile,
)

logger = logging.getLogger(__name__)

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


def _build_ticker_map(session: Session) -> dict[str, str]:
    """Return {instrument_id_hex: yfinance_ticker} for all instruments."""
    rows = (
        session.execute(
            select(Instrument.id, Instrument.yfinance_ticker)
            .where(Instrument.yfinance_ticker.isnot(None))
            .where(Instrument.yfinance_ticker != "")
        )
        .all()
    )
    return {str(r[0]): r[1] for r in rows}


# ---------------------------------------------------------------------------
# Public assembly functions
# ---------------------------------------------------------------------------


def assemble_prices(session: Session) -> pd.DataFrame:
    """Build a ``dates x tickers`` close-price DataFrame.

    Returns
    -------
    pd.DataFrame
        Index = ``pd.DatetimeIndex``, columns = yfinance tickers.
    """
    ticker_map = _build_ticker_map(session)

    rows = session.execute(
        select(
            PriceHistory.instrument_id,
            PriceHistory.date,
            PriceHistory.close,
        ).order_by(PriceHistory.date)
    ).all()

    if not rows:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for instrument_id, date, close in rows:
        ticker = ticker_map.get(str(instrument_id))
        if ticker is None:
            continue
        records.append({
            "date": pd.Timestamp(date),
            "ticker": ticker,
            "close": _to_float(close),
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Multiple instruments can map to the same yfinance_ticker (e.g.
    # listed on different exchanges).  Use pivot_table with 'first' to
    # deduplicate gracefully instead of raising on duplicates.
    pivoted = df.pivot_table(
        index="date", columns="ticker", values="close", aggfunc="first",
    )
    pivoted.index = pd.DatetimeIndex(pivoted.index)
    pivoted = pivoted.sort_index()
    return pivoted


def assemble_volumes(session: Session) -> pd.DataFrame:
    """Build a ``dates x tickers`` volume DataFrame.

    Returns
    -------
    pd.DataFrame
        Index = ``pd.DatetimeIndex``, columns = yfinance tickers.
    """
    ticker_map = _build_ticker_map(session)

    rows = session.execute(
        select(
            PriceHistory.instrument_id,
            PriceHistory.date,
            PriceHistory.volume,
        ).order_by(PriceHistory.date)
    ).all()

    if not rows:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for instrument_id, date, volume in rows:
        ticker = ticker_map.get(str(instrument_id))
        if ticker is None:
            continue
        records.append({
            "date": pd.Timestamp(date),
            "ticker": ticker,
            "volume": _to_float(volume),
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    pivoted = df.pivot_table(
        index="date", columns="ticker", values="volume", aggfunc="first",
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
        "Enriched fundamentals with %d values from financial statements "
        "(%d tickers).",
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


def assemble_fundamentals(
    session: Session,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Build a ``tickers x fields`` fundamentals DataFrame and sector map.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, str]]
        - Fundamentals DataFrame indexed by yfinance ticker.
        - ``{ticker: sector}`` mapping.
    """
    profiles = session.execute(
        select(TickerProfile)
        .options(joinedload(TickerProfile.instrument))
    ).scalars().all()

    if not profiles:
        return pd.DataFrame(), {}

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

        fundamentals_records.append(row)

        if profile.sector:
            sector_mapping[ticker] = profile.sector

    if not fundamentals_records:
        return pd.DataFrame(), {}

    df = pd.DataFrame(fundamentals_records).set_index("ticker")
    # Multiple instruments can map to the same yfinance_ticker
    # (different exchanges).  Keep the first (typically most complete).
    df = df[~df.index.duplicated(keep="first")]

    # Enrich with data from FinancialStatement EAV table
    ticker_map = _build_ticker_map(session)
    df = _enrich_from_financial_statements(session, df, ticker_map)

    return df, sector_mapping


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
        records.append({
            "ticker": ticker,
            "statement_type": stmt_type,
            "period_type": period_type,
            "period_date": period_date,
        })

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
            "ticker", "period", "strong_buy",
            "buy", "hold", "sell", "strong_sell",
        ]
        return pd.DataFrame(columns=cols)

    records: list[dict[str, Any]] = []
    for instrument_id, period, sb, b, h, s, ss in rows:
        ticker = ticker_map.get(str(instrument_id))
        if ticker is None:
            continue
        records.append({
            "ticker": ticker,
            "period": period,
            "strong_buy": sb or 0,
            "buy": b or 0,
            "hold": h or 0,
            "sell": s or 0,
            "strong_sell": ss or 0,
        })

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
        records.append({
            "ticker": ticker,
            "shares": shares or 0,
            "transaction_type": tx_type,
            "start_date": start_date,
        })

    return pd.DataFrame(records)


def assemble_macro_data(
    session: Session,
    country: str = "United States",
) -> pd.DataFrame:
    """Build macro DataFrame for regime classification.

    The regime classifier expects ``gdp_growth`` and/or ``yield_spread``
    columns.

    Parameters
    ----------
    country : str
        Country to pull macro indicators for.

    Returns
    -------
    pd.DataFrame
        Single-row (or multi-row) DataFrame with ``gdp_growth`` and
        ``yield_spread`` columns.
    """
    # GDP growth from EconomicIndicator (IlSole)
    indicators = session.execute(
        select(EconomicIndicator)
        .where(EconomicIndicator.country == country)
    ).scalars().all()

    gdp_growth: float | None = None
    st_rate: float | None = None
    lt_rate: float | None = None

    for ind in indicators:
        if ind.gdp_growth_qq is not None:
            gdp_growth = float(ind.gdp_growth_qq)
        if ind.st_rate is not None:
            st_rate = float(ind.st_rate)
        if ind.lt_rate is not None:
            lt_rate = float(ind.lt_rate)

    # Yield spread from bond yields if not available from indicators
    if lt_rate is None or st_rate is None:
        bonds = session.execute(
            select(BondYield)
            .where(BondYield.country == country)
        ).scalars().all()

        bond_map: dict[str, float] = {}
        for bond in bonds:
            if bond.yield_value is not None:
                bond_map[bond.maturity] = float(bond.yield_value)

        if lt_rate is None:
            lt_rate = bond_map.get("10Y")
        if st_rate is None:
            st_rate = bond_map.get("2Y")

    yield_spread: float | None = None
    if lt_rate is not None and st_rate is not None:
        yield_spread = lt_rate - st_rate

    macro_row: dict[str, float | None] = {
        "gdp_growth": gdp_growth,
        "yield_spread": yield_spread,
    }

    return pd.DataFrame([macro_row])


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
    ) -> None:
        self.prices = prices
        self.volumes = volumes
        self.fundamentals = fundamentals
        self.sector_mapping = sector_mapping
        self.financial_statements = financial_statements
        self.analyst_data = analyst_data
        self.insider_data = insider_data
        self.macro_data = macro_data

    @property
    def n_tickers(self) -> int:
        return len(self.prices.columns)

    @property
    def n_trading_days(self) -> int:
        return len(self.prices)

    def summary(self) -> dict[str, Any]:
        return {
            "tickers": self.n_tickers,
            "trading_days": self.n_trading_days,
            "fundamentals_rows": len(self.fundamentals),
            "financial_statements": len(self.financial_statements),
            "analyst_records": len(self.analyst_data),
            "insider_records": len(self.insider_data),
            "sectors": len(set(self.sector_mapping.values())),
            "has_macro": len(self.macro_data) > 0,
        }


def assemble_all(
    db_manager: DatabaseManager,
    macro_country: str = "United States",
) -> DataAssembly:
    """Query the database and assemble all DataFrames.

    Parameters
    ----------
    db_manager : DatabaseManager
        Initialized database manager.
    macro_country : str
        Country for macro regime data.

    Returns
    -------
    DataAssembly
        All assembled DataFrames ready for the optimizer.
    """
    with db_manager.get_session() as session:
        logger.info("Assembling price data...")
        prices = assemble_prices(session)

        logger.info("Assembling volume data...")
        volumes = assemble_volumes(session)

        logger.info("Assembling fundamentals...")
        fundamentals, sector_mapping = assemble_fundamentals(session)

        logger.info("Assembling financial statements...")
        financial_statements = assemble_financial_statements(session)

        logger.info("Assembling analyst data...")
        analyst_data = assemble_analyst_data(session)

        logger.info("Assembling insider data...")
        insider_data = assemble_insider_data(session)

        logger.info("Assembling macro data...")
        macro_data = assemble_macro_data(session, country=macro_country)

    return DataAssembly(
        prices=prices,
        volumes=volumes,
        fundamentals=fundamentals,
        sector_mapping=sector_mapping,
        financial_statements=financial_statements,
        analyst_data=analyst_data,
        insider_data=insider_data,
        macro_data=macro_data,
    )
