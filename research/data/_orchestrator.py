"""Top-level orchestrator — FX rates and the all-in-one assembly entry point.

Extracted from ``data_assembly.py``.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

# Ensure the api package is importable from the CLI context.
_api_path = Path(__file__).parent.parent.parent / "api"
if str(_api_path) not in sys.path:
    sys.path.insert(0, str(_api_path))

if TYPE_CHECKING:
    from app.database import DatabaseManager

from ._container import DataAssembly, _compute_assembly_hash  # noqa: E402
from ._equity import (  # noqa: E402
    assemble_analyst_data,
    assemble_financial_statements,
    assemble_fundamentals,
    assemble_insider_data,
    assemble_prices,
    assemble_volumes,
)
from ._history import (  # noqa: E402
    assemble_delisting_returns,
    assemble_fundamental_history,
)
from ._macro import (  # noqa: E402
    assemble_bond_observations,
    assemble_fred_series,
    assemble_macro_data,
    assemble_te_observations,
)
from ._regime import assemble_regime_data  # noqa: E402
from ._sentiment import assemble_sentiment  # noqa: E402

logger = logging.getLogger(__name__)


def assemble_fx_rates(
    currency_map: dict[str, str],
    base_currency: str,
    price_index: pd.DatetimeIndex,
    *,
    cross_via_usd: bool = True,
) -> pd.DataFrame:
    """Fetch FX rates from yfinance for all foreign currencies in the map.

    Parameters
    ----------
    currency_map : dict[str, str]
        ``{yfinance_ticker: ISO_currency_code}`` mapping.
    base_currency : str
        Target base currency (e.g. ``"EUR"``).
    price_index : pd.DatetimeIndex
        Date range for the download.
    cross_via_usd : bool
        Compute cross rates via USD for non-USD pairs.

    Returns
    -------
    pd.DataFrame
        Columns = foreign currency codes.  Empty when no FX needed.
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
            rate = compute_cross_rate(close[from_ticker], close[to_ticker])  # pyright: ignore[reportArgumentType]
            rates[ccy] = rate
        else:
            if pair not in close.columns:
                logger.warning("Missing FX data for %s: %s", ccy, pair)
                continue
            downloaded = close[pair]
            if ccy.upper() == "USD":
                # Ticker is {base}USD=X → gives USD per 1 base → reciprocal
                rates[ccy] = 1.0 / downloaded  # pyright: ignore[reportArgumentType]
            else:
                # Ticker gives base per 1 foreign → direct
                rates[ccy] = downloaded  # pyright: ignore[reportArgumentType]

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


def assemble_all(
    db_manager: DatabaseManager,
    macro_country: str = "USA",
    include_delisted: bool = True,
    base_currency: str = "EUR",
) -> DataAssembly:
    """Query the database and assemble all DataFrames into a DataAssembly.

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

        logger.info("Assembling price data...")
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
        macro_data,
        fred_data,
        te_observations,
        sentiment_data,
    )

    logger.info("Assembling FX rates...")
    if prices.empty:
        logger.warning("Prices DataFrame is empty; skipping FX rate assembly.")
        fx_rates = pd.DataFrame()
    else:
        fx_rates = assemble_fx_rates(
            currency_map=currency_map,
            base_currency=base_currency,
            price_index=prices.index,  # pyright: ignore[reportArgumentType]
        )

    assembly = DataAssembly(
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
    assembly = dataclasses.replace(
        assembly, assembly_hash=_compute_assembly_hash(assembly)
    )
    return assembly
