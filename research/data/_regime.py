"""Regime data merging — combine macro indicators for regime classification.

Extracted from ``data_assembly.py``.
"""

from __future__ import annotations

import logging
import warnings

import pandas as pd

logger = logging.getLogger(__name__)


# Column translation maps: DB column names → classify_regime() expected names
_FRED_REGIME_MAP: dict[str, str] = {
    "T10Y2Y": "spread_2s10s",
    "BAMLH0A0HYM2": "hy_oas",
    "A191RL1Q225SBEA": "gdp_growth",
}
_TE_REGIME_MAP: dict[str, str] = {
    "manufacturing_pmi": "pmi",
}

# Regime columns that MUST have at least one valid observation. When such a
# column has ``last_valid_index() is None`` after merging, the assembler
# raises ValueError instead of silently warning, because downstream regime
# classification cannot recover from missing data.
_REQUIRED_REGIME_COLUMNS: frozenset[str] = frozenset({"gdp_growth"})


def assemble_regime_data(
    macro_data: pd.DataFrame,
    fred_data: pd.DataFrame,
    te_observations: pd.DataFrame,
    sentiment_data: pd.DataFrame | None = None,
    sentiment_country: str = "USA",
    fill_limit: int = 45,
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
    fill_limit : int
        Maximum number of consecutive rows to forward-fill.  Prevents
        stale monthly observations from propagating indefinitely into
        the future.  A ``UserWarning`` is emitted for each column whose
        last value is still NaN after forward-filling (data older than
        ``fill_limit`` days).  Default is 45 (~1.5 months).

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with any subset of ``gdp_growth``,
        ``yield_spread``, ``pmi``, ``spread_2s10s``, ``hy_oas``,
        ``sentiment``.  Columns with data older than ``fill_limit``
        consecutive rows will contain trailing NaN values.

    Raises
    ------
    ValueError
        If a column listed in ``_REQUIRED_REGIME_COLUMNS`` ends up in
        the merged frame with no valid observations
        (``last_valid_index() is None``).
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
    if sentiment_data is not None and sentiment_country in sentiment_data.columns:
        sent = sentiment_data[sentiment_country].dropna()
        if len(sent) > 0:
            parts.append(sent.rename("sentiment").to_frame())

    # Macro baseline columns (gdp_growth, yield_spread) — skip any column
    # already populated from FRED or TE to prevent duplicate-column join errors.
    already_covered = set(fred_cols.keys()) | set(te_cols.keys())
    macro_cols = [
        c
        for c in ("gdp_growth", "yield_spread")
        if c in macro_data.columns and c not in already_covered
    ]
    if macro_cols:
        parts.append(macro_data[macro_cols].dropna(how="all"))

    if not parts:
        return pd.DataFrame()

    # Validate / coerce DatetimeIndex on all parts before joining.
    # A non-datetime index (e.g. integer years or string dates) causes a
    # silent all-NaN column after the outer join because indices never align.
    for i, p in enumerate(parts):
        if not isinstance(p.index, pd.DatetimeIndex):
            warnings.warn(
                f"Part {i} has a {type(p.index).__name__} index; "
                "attempting coercion to DatetimeIndex.",
                UserWarning,
                stacklevel=2,
            )
            try:
                parts[i] = p.set_index(pd.to_datetime(p.index))
            except Exception as exc:
                raise ValueError(
                    f"Part {i} index could not be coerced to DatetimeIndex: {exc}"
                ) from exc

    # Outer-join all parts on date index, then forward-fill
    merged = parts[0]
    for p in parts[1:]:
        merged = merged.join(p, how="outer")

    merged = merged.sort_index()
    # Capture last actual observation dates before filling (post-fill dates
    # would reflect the last *filled* row, not the last real measurement)
    last_actual = {col: merged[col].last_valid_index() for col in merged.columns}

    merged = merged.ffill(limit=fill_limit)

    # Warn for any column still NaN at the tail (data too stale to fill).
    # Required columns with NO observation ever raise instead, because regime
    # classification cannot proceed without them.
    for col in merged.columns:
        if pd.isna(merged[col].iloc[-1]):
            trailing_nans = int(merged[col].isna()[::-1].cumprod().sum())
            last_valid = last_actual[col]
            if col in _REQUIRED_REGIME_COLUMNS and last_valid is None:
                raise ValueError(
                    f"Required regime column '{col}' has no valid observations; "
                    "cannot classify regime."
                )
            last_valid_str = (
                last_valid.strftime("%Y-%m-%d") if last_valid is not None else "never"
            )
            warnings.warn(
                f"Regime data column '{col}' has {trailing_nans} trailing NaN rows "
                f"after forward-filling (fill_limit={fill_limit}). "
                f"Last valid observation: {last_valid_str}. "
                "Regime classification may fall back to UNKNOWN for recent dates.",
                UserWarning,
                stacklevel=2,
            )

    return merged
