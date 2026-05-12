"""DataAssembly container — frozen value object for the data-assembly stage."""

from __future__ import annotations

import datetime
import hashlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd

from ._helpers import _TRADING_DAYS

logger = logging.getLogger(__name__)

__all__ = ["DataAssembly", "_compute_assembly_hash"]


@dataclass(frozen=True, slots=True, eq=False)
class DataAssembly:
    """Frozen value object holding all data-assembly outputs.

    All fields are set at construction time; use ``dataclasses.replace()``
    to derive a new instance with updated fields (e.g. ``assembly_hash``).
    """

    prices: pd.DataFrame
    volumes: pd.DataFrame
    fundamentals: pd.DataFrame
    sector_mapping: Mapping[str, str]
    financial_statements: pd.DataFrame
    analyst_data: pd.DataFrame
    insider_data: pd.DataFrame
    macro_data: pd.DataFrame
    fred_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    te_observations: pd.DataFrame = field(default_factory=pd.DataFrame)
    bond_observations: pd.DataFrame = field(default_factory=pd.DataFrame)
    sentiment_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    regime_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    fundamental_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    include_delisted: bool = True
    delisting_returns: Mapping[str, float] = field(default_factory=dict)
    currency_map: Mapping[str, str] = field(default_factory=dict)
    fx_rates: pd.DataFrame = field(default_factory=pd.DataFrame)
    assembly_hash: str = ""

    def __hash__(self) -> int:
        return hash(self.assembly_hash) if self.assembly_hash else id(self)

    @property
    def n_tickers(self) -> int:
        """Number of tickers (columns) in the price DataFrame."""
        return len(self.prices.columns)

    @property
    def n_trading_days(self) -> int:
        """Number of trading days (rows) in the price DataFrame."""
        return len(self.prices)

    @property
    def risk_free_rate_series(self) -> pd.Series:
        """Daily compounded risk-free rate from DGS3MO.

        Returns per-day decimal: ``(1 + annual_pct/100)^(1/252) - 1``.
        Returns an empty Series when DGS3MO is absent from ``fred_data``.
        """
        if self.fred_data.empty or "DGS3MO" not in self.fred_data.columns:
            return pd.Series(dtype=float, name="risk_free_rate")
        raw = self.fred_data["DGS3MO"].dropna()
        return ((1 + raw / 100) ** (1.0 / _TRADING_DAYS) - 1).rename("risk_free_rate")

    @property
    def risk_free_rate(self) -> float:
        """Latest daily compounded risk-free rate scalar (0.0 when unavailable)."""
        series = self.risk_free_rate_series
        if series.empty:
            logger.warning(
                "DGS3MO not found in fred_data; using rf=0.0. "
                "POST /api/v1/macro-data/fred/fetch to populate."
            )
            return 0.0
        return float(series.iloc[-1])

    def summary(self) -> dict[str, str | int | float | bool | None]:
        """Return a flat summary dict of key assembly metrics."""
        rf = self.risk_free_rate_series
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
                round(float(rf.iloc[-1]) * _TRADING_DAYS * 100, 4)
                if not rf.empty
                else None
            ),
            "risk_free_rate_obs": len(rf),
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

    Parameters
    ----------
    assembly:
        The assembly to identify.
    timestamp:
        Optional fixed timestamp (used in tests for determinism). When
        ``None``, ``datetime.datetime.now(datetime.timezone.utc)`` is used.

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
