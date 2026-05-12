"""Macro-economic data assembly — GDP, yields, bonds, FRED series.

Extracted from ``data_assembly.py``.  All functions accept a synchronous
SQLAlchemy ``Session`` and return pandas DataFrames.
"""

from __future__ import annotations

import datetime
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

# Ensure the api package is importable from the CLI context.
_api_path = Path(__file__).parent.parent.parent / "api"
if str(_api_path) not in sys.path:
    sys.path.insert(0, str(_api_path))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Macro snapshot
# ---------------------------------------------------------------------------


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
    from app.models.macro.macro_regime import (
        BondYield,
        EconomicIndicator,
        TradingEconomicsIndicator,
    )

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


# ---------------------------------------------------------------------------
# Macro time-series
# ---------------------------------------------------------------------------


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
    from app.models.macro.macro_regime import (
        BondYieldObservation,
        EconomicIndicatorObservation,
        TradingEconomicsObservation,
    )

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
    ilsole_cols = [
        "last_inflation",
        "inflation_6m",
        "inflation_10y_avg",
        "gdp_growth_6m",
        "earnings_12m",
        "eps_expected_12m",
        "peg_ratio",
        "lt_rate_forecast",
    ]
    ilsole_stmt = select(
        EconomicIndicatorObservation.date,
        *[getattr(EconomicIndicatorObservation, c) for c in ilsole_cols],
    ).where(EconomicIndicatorObservation.country == country)
    if start_date:
        ilsole_stmt = ilsole_stmt.where(EconomicIndicatorObservation.date >= start_date)
    if end_date:
        ilsole_stmt = ilsole_stmt.where(EconomicIndicatorObservation.date <= end_date)
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
    ilsole_data: dict[str, dict[pd.Timestamp, float]] = {c: {} for c in ilsole_cols}
    for row in ilsole_rows:
        ts = pd.Timestamp(row[0])
        for i, col in enumerate(ilsole_cols):
            val = row[i + 1]
            if val is not None:
                ilsole_data[col][ts] = float(val)

    # Combine all series into DataFrame
    all_series: dict[str, pd.Series] = {
        "gdp_growth": gdp_series,
        "yield_spread": spread_series,
    }
    for col in ilsole_cols:
        if ilsole_data[col]:
            all_series[col] = pd.Series(ilsole_data[col], dtype=float, name=col)

    df = pd.DataFrame(all_series)
    df.index = pd.DatetimeIndex(df.index)
    df = df.sort_index()
    return df


# ---------------------------------------------------------------------------
# Trading Economics observations
# ---------------------------------------------------------------------------


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
    from app.models.macro.macro_regime import TradingEconomicsObservation

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
        index="date",
        columns="indicator_key",
        values="value",
        aggfunc="first",
    )
    pivoted.index = pd.DatetimeIndex(pivoted.index)
    pivoted.columns.name = None
    return pivoted.sort_index()


# ---------------------------------------------------------------------------
# Bond yield observations
# ---------------------------------------------------------------------------


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
    from app.models.macro.macro_regime import BondYieldObservation

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
        index="date",
        columns="maturity",
        values="yield_value",
        aggfunc="first",
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
    # US recession indicators (monthly/quarterly + daily NBER)
    "RECPROUSM156N",
    "JHGDPBRINDX",
    "USREC",
    "USRECDM",
    # Treasury yield curve (daily, annualized %)
    "DGS2",
    "DGS10",
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
    from app.models.macro.macro_regime import FredObservation

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
