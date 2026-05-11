"""DataAssembly container and content hashing.

Extracted from ``data_assembly.py``.
"""

from __future__ import annotations

import datetime
import hashlib
import logging
from typing import Any

import pandas as pd

from ._helpers import _TRADING_DAYS

logger = logging.getLogger(__name__)


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
    assembly_hash : str
        Deterministic 16-char identifier of this assembly's contents.
        Defaults to ``""``; populated by :func:`assemble_all` after
        construction.  Downstream cycles use it to identify which
        assembly produced their inputs (e.g. Cycle 5 ``report.md``).
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
        assembly_hash: str = "",
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
        self.te_observations = (
            te_observations if te_observations is not None else pd.DataFrame()
        )
        self.bond_observations = (
            bond_observations if bond_observations is not None else pd.DataFrame()
        )
        self.sentiment_data = (
            sentiment_data if sentiment_data is not None else pd.DataFrame()
        )
        self.regime_data = regime_data if regime_data is not None else pd.DataFrame()
        self.fundamental_history = (
            fundamental_history if fundamental_history is not None else pd.DataFrame()
        )
        self.include_delisted = include_delisted
        self.delisting_returns: dict[str, float] = delisting_returns or {}
        self.currency_map: dict[str, str] = currency_map or {}
        self.fx_rates = fx_rates if fx_rates is not None else pd.DataFrame()
        self.assembly_hash: str = assembly_hash

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
        return ((1 + raw / 100) ** (1.0 / _TRADING_DAYS) - 1).rename("risk_free_rate")

    @property
    def risk_free_rate(self) -> float:
        """Latest daily compounded risk-free rate scalar.

        Returns 0.0 when DGS3MO is unavailable.
        """
        series = self.risk_free_rate_series
        if series.empty:
            logger.warning(
                "DGS3MO not found in fred_data; using rf=0.0. "
                "POST /api/v1/macro-data/fred/fetch to populate."
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
                len(self.fx_rates.columns) if not self.fx_rates.empty else 0
            ),
            "fx_rates_observations": len(self.fx_rates),
            "assembly_hash": self.assembly_hash,
        }


def _compute_assembly_hash(
    assembly: DataAssembly,
    timestamp: datetime.datetime | None = None,
) -> str:
    """Return a deterministic 16-char SHA-256 prefix identifying the assembly.

    The hash digests ``(timestamp_iso, n_tickers, last_price_date)`` so two
    assemblies built from the same DB snapshot at the same wall-clock get the
    same id, while any change to coverage, breadth, or build-time mutates it.

    Parameters
    ----------
    assembly : DataAssembly
        The assembly to identify.
    timestamp : datetime.datetime | None
        Optional fixed timestamp (used in tests for determinism).  When
        ``None``, ``datetime.datetime.now(datetime.UTC)`` is used.

    Returns
    -------
    str
        16-character lowercase hex prefix of the SHA-256 digest.
    """
    ts = (
        timestamp
        if timestamp is not None
        else datetime.datetime.now(datetime.timezone.utc)
    )
    last_date = (
        assembly.prices.index.max().isoformat() if not assembly.prices.empty else ""
    )
    payload = f"{ts.isoformat()}|{assembly.n_tickers}|{last_date}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]
