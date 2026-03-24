"""Regime-conditional subperiod Sharpe validation.

Splits out-of-sample portfolio returns by macro regime (from
:func:`~optimizer.factors._regime.classify_regime_composite`) and computes
per-regime and per-subperiod performance statistics.  Flags strategies where
alpha is concentrated in a single regime rather than being persistent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from optimizer.exceptions import DataError
from optimizer.factors._config import MacroRegime, RegimeThresholdConfig
from optimizer.factors._regime import classify_regime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeValidationConfig:
    """Immutable configuration for regime-conditional Sharpe analysis.

    Parameters
    ----------
    min_regime_obs : int
        Minimum trading days required to compute meaningful statistics
        for a subperiod or regime aggregate.  Subperiods shorter than
        this produce ``NaN`` metrics.
    single_regime_alpha_threshold : float
        Fraction of total positive alpha above which a single regime
        is flagged as concentrated (acceptance criterion 4).
    trading_days_per_year : int
        Annualization constant for Sharpe, return, and volatility.
    risk_free_rate : float
        Annual risk-free rate for Sharpe ratio computation.
    include_unknown_regime : bool
        Whether to include ``MacroRegime.UNKNOWN`` periods in the
        per-regime breakdown.  Default ``False`` since UNKNOWN days
        are typically data gaps.
    """

    min_regime_obs: int = 21
    single_regime_alpha_threshold: float = 0.80
    trading_days_per_year: int = 252
    risk_free_rate: float = 0.0
    include_unknown_regime: bool = False

    def __post_init__(self) -> None:
        if self.min_regime_obs < 1:
            raise ValueError(f"min_regime_obs must be >= 1, got {self.min_regime_obs}")
        if not (0.0 < self.single_regime_alpha_threshold <= 1.0):
            raise ValueError(
                f"single_regime_alpha_threshold must be in (0.0, 1.0], "
                f"got {self.single_regime_alpha_threshold}"
            )
        if self.trading_days_per_year <= 0:
            raise ValueError(
                f"trading_days_per_year must be > 0, got {self.trading_days_per_year}"
            )
        if self.risk_free_rate < 0.0:
            raise ValueError(
                f"risk_free_rate must be >= 0.0, got {self.risk_free_rate}"
            )

    @classmethod
    def for_standard(cls) -> RegimeValidationConfig:
        """Standard defaults."""
        return cls()

    @classmethod
    def for_strict(cls) -> RegimeValidationConfig:
        """Tighter thresholds for production use."""
        return cls(min_regime_obs=63, single_regime_alpha_threshold=0.70)

    @classmethod
    def for_research(cls) -> RegimeValidationConfig:
        """Include unknown regime periods for diagnostics."""
        return cls(include_unknown_regime=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(value: object) -> float | None:
    """Convert to float, returning ``None`` for NaN."""
    try:
        f = float(value)  # type: ignore[arg-type]
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class RegimeValidationResult:
    """Result of regime-conditional subperiod Sharpe analysis.

    Attributes
    ----------
    per_regime_metrics : pd.DataFrame
        Index is regime name strings.  Columns:
        ``obs``, ``coverage_pct``, ``ann_return``, ``ann_vol``,
        ``sharpe``, ``max_drawdown``, ``obs_sufficient``.
    per_subperiod_metrics : pd.DataFrame
        One row per contiguous regime block.  Columns:
        ``start``, ``end``, ``regime``, ``obs``, ``ann_return``,
        ``ann_vol``, ``sharpe``, ``max_drawdown``.
    regime_alpha_concentration : pd.Series
        Fraction of total positive alpha attributable to each regime.
    concentrated_regimes : list[str]
        Regimes exceeding ``single_regime_alpha_threshold``.
    regime_timeline : pd.Series
        DatetimeIndex → regime name string for every OOS observation.
    total_obs : int
        Total number of OOS observations.
    n_regimes_observed : int
        Distinct regimes with at least one observation.
    """

    per_regime_metrics: pd.DataFrame
    per_subperiod_metrics: pd.DataFrame
    regime_alpha_concentration: pd.Series
    concentrated_regimes: list[str] = field(default_factory=list)
    regime_timeline: pd.Series = field(default_factory=lambda: pd.Series(dtype=str))
    total_obs: int = 0
    n_regimes_observed: int = 0

    def to_attribution_dict(self) -> dict[str, object]:
        """Serialize performance attribution across regimes.

        Returns a dict suitable for frontend charting (ECharts) or CLI
        reporting.  Structure::

            {
              "regimes": [
                {
                  "regime": "expansion",
                  "obs": 120,
                  "coverage_pct": 0.667,
                  "ann_return": 0.52,
                  "ann_vol": 0.16,
                  "sharpe": 3.2,
                  "max_drawdown": -0.03,
                  "alpha_concentration": 0.95,
                  "is_concentrated": True,
                },
                ...
              ],
              "subperiods": [
                {
                  "start": "2024-01-02",
                  "end": "2024-03-28",
                  "regime": "expansion",
                  "obs": 60,
                  "ann_return": 0.50,
                  "ann_vol": 0.16,
                  "sharpe": 3.1,
                  "max_drawdown": -0.02,
                },
                ...
              ],
              "summary": {
                "total_obs": 180,
                "n_regimes_observed": 2,
                "concentrated_regimes": ["expansion"],
                "has_concentration_warning": True,
              },
            }
        """
        regime_rows = []
        for regime_name in self.per_regime_metrics.index:
            row = self.per_regime_metrics.loc[regime_name]
            regime_rows.append(
                {
                    "regime": str(regime_name),
                    "obs": int(row["obs"]),
                    "coverage_pct": float(row["coverage_pct"]),
                    "ann_return": _safe_float(row["ann_return"]),
                    "ann_vol": _safe_float(row["ann_vol"]),
                    "sharpe": _safe_float(row["sharpe"]),
                    "max_drawdown": _safe_float(row["max_drawdown"]),
                    "alpha_concentration": float(
                        self.regime_alpha_concentration.get(regime_name, 0.0)
                    ),
                    "is_concentrated": str(regime_name) in self.concentrated_regimes,
                }
            )

        subperiod_rows = []
        for _, sp in self.per_subperiod_metrics.iterrows():
            subperiod_rows.append(
                {
                    "start": (
                        str(sp["start"].date())
                        if hasattr(sp["start"], "date")
                        else str(sp["start"])
                    ),
                    "end": (
                        str(sp["end"].date())
                        if hasattr(sp["end"], "date")
                        else str(sp["end"])
                    ),
                    "regime": str(sp["regime"]),
                    "obs": int(sp["obs"]),
                    "ann_return": _safe_float(sp["ann_return"]),
                    "ann_vol": _safe_float(sp["ann_vol"]),
                    "sharpe": _safe_float(sp["sharpe"]),
                    "max_drawdown": _safe_float(sp["max_drawdown"]),
                }
            )

        return {
            "regimes": regime_rows,
            "subperiods": subperiod_rows,
            "summary": {
                "total_obs": self.total_obs,
                "n_regimes_observed": self.n_regimes_observed,
                "concentrated_regimes": list(self.concentrated_regimes),
                "has_concentration_warning": len(self.concentrated_regimes) > 0,
            },
        }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_regime_series(
    macro_data: pd.DataFrame,
    thresholds: RegimeThresholdConfig | None,
) -> pd.Series:
    """Build a point-in-time regime series from macro data.

    Each date's classification uses only data up to that date to avoid
    look-ahead bias.  Uses :func:`classify_regime` which dispatches to
    the composite classifier when PMI/spread/HY columns are present and
    falls back to the GDP heuristic otherwise.
    """
    regimes: dict[pd.Timestamp, str] = {}
    for i, date in enumerate(macro_data.index):
        window = macro_data.iloc[: i + 1]
        regime = classify_regime(window, thresholds=thresholds)
        regimes[date] = regime.value
    return pd.Series(regimes, name="regime")


def _compute_period_metrics(
    returns_slice: pd.Series,
    trading_days: int,
    rf_daily: float,
) -> dict[str, float]:
    """Compute annualized return, volatility, Sharpe, and max drawdown."""
    n = len(returns_slice)
    if n == 0:
        return {
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
        }
    mean_daily = returns_slice.mean()
    std_daily = returns_slice.std(ddof=1) if n > 1 else 0.0

    ann_return = float((1.0 + mean_daily) ** trading_days - 1.0)
    ann_vol = float(std_daily * np.sqrt(trading_days))
    sharpe = (
        float((mean_daily - rf_daily) / std_daily) * np.sqrt(trading_days)
        if std_daily > 0
        else np.nan
    )

    # Max drawdown from cumulative returns
    cum = (1.0 + returns_slice).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0

    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


def _identify_subperiods(
    regime_timeline: pd.Series,
) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Find contiguous blocks of the same regime.

    Returns list of ``(start_date, end_date, regime_name)`` tuples.
    """
    if len(regime_timeline) == 0:
        return []

    blocks: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    values = regime_timeline.values
    index = regime_timeline.index

    block_start = index[0]
    current_regime = values[0]

    for i in range(1, len(values)):
        if values[i] != current_regime:
            blocks.append((block_start, index[i - 1], current_regime))
            block_start = index[i]
            current_regime = values[i]

    # Final block
    blocks.append((block_start, index[-1], current_regime))
    return blocks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_regime_validation(
    oos_returns: pd.Series,
    macro_data: pd.DataFrame,
    config: RegimeValidationConfig | None = None,
    thresholds: RegimeThresholdConfig | None = None,
) -> RegimeValidationResult:
    """Run regime-conditional subperiod Sharpe analysis.

    Parameters
    ----------
    oos_returns : pd.Series
        Daily portfolio returns indexed by ``DatetimeIndex``.
    macro_data : pd.DataFrame
        Macro indicators indexed by date, compatible with
        :func:`~optimizer.factors._regime.classify_regime_composite`.
    config : RegimeValidationConfig or None
        Validation configuration.  Defaults to standard.
    thresholds : RegimeThresholdConfig or None
        Regime classification thresholds.

    Returns
    -------
    RegimeValidationResult
        Per-regime and per-subperiod statistics with alpha
        concentration flags.

    Raises
    ------
    DataError
        If ``oos_returns`` is empty.
    """
    if config is None:
        config = RegimeValidationConfig()

    if len(oos_returns) == 0:
        raise DataError("oos_returns is empty")

    rf_daily = config.risk_free_rate / config.trading_days_per_year
    tdays = config.trading_days_per_year

    # --- Build regime timeline ---
    overlap = macro_data.index.intersection(oos_returns.index).size
    if len(macro_data) == 0 or not overlap:
        logger.warning(
            "run_regime_validation: no macro data overlaps with OOS returns; "
            "returning NaN metrics."
        )
        empty_df = pd.DataFrame(
            columns=[
                "obs",
                "coverage_pct",
                "ann_return",
                "ann_vol",
                "sharpe",
                "max_drawdown",
                "obs_sufficient",
            ]
        )
        return RegimeValidationResult(
            per_regime_metrics=empty_df,
            per_subperiod_metrics=pd.DataFrame(
                columns=[
                    "start",
                    "end",
                    "regime",
                    "obs",
                    "ann_return",
                    "ann_vol",
                    "sharpe",
                    "max_drawdown",
                ]
            ),
            regime_alpha_concentration=pd.Series(dtype=float),
            concentrated_regimes=[],
            regime_timeline=pd.Series(dtype=str),
            total_obs=len(oos_returns),
            n_regimes_observed=0,
        )

    regime_series = _build_regime_series(macro_data, thresholds)

    # Align to OOS returns index via forward-fill
    regime_timeline = regime_series.reindex(oos_returns.index, method="ffill")
    # Days before macro data starts → UNKNOWN
    regime_timeline = regime_timeline.fillna(MacroRegime.UNKNOWN.value)

    # Optionally drop UNKNOWN
    if not config.include_unknown_regime:
        mask = regime_timeline != MacroRegime.UNKNOWN.value
        regime_timeline = regime_timeline[mask]
        oos_filtered = oos_returns.reindex(regime_timeline.index)
    else:
        oos_filtered = oos_returns.reindex(regime_timeline.index)

    total_obs = len(oos_returns)

    # --- Identify subperiods ---
    subperiods = _identify_subperiods(regime_timeline)

    subperiod_rows: list[dict[str, object]] = []
    for start, end, regime_name in subperiods:
        sub_returns = oos_filtered.loc[start:end]
        n_obs = len(sub_returns)
        if n_obs >= config.min_regime_obs:
            metrics = _compute_period_metrics(sub_returns, tdays, rf_daily)
        else:
            metrics = {
                "ann_return": np.nan,
                "ann_vol": np.nan,
                "sharpe": np.nan,
                "max_drawdown": np.nan,
            }
        subperiod_rows.append(
            {
                "start": start,
                "end": end,
                "regime": regime_name,
                "obs": n_obs,
                **metrics,
            }
        )

    per_subperiod_metrics = pd.DataFrame(subperiod_rows)

    # --- Aggregate per regime ---
    observed_regimes = sorted(regime_timeline.unique())
    n_regimes_observed = len(observed_regimes)

    regime_rows: list[dict[str, object]] = []
    for regime_name in observed_regimes:
        regime_mask = regime_timeline == regime_name
        regime_returns = oos_filtered[regime_mask]
        n_obs = len(regime_returns)
        sufficient = n_obs >= config.min_regime_obs
        if sufficient:
            metrics = _compute_period_metrics(regime_returns, tdays, rf_daily)
        else:
            metrics = {
                "ann_return": np.nan,
                "ann_vol": np.nan,
                "sharpe": np.nan,
                "max_drawdown": np.nan,
            }
        regime_rows.append(
            {
                "regime": regime_name,
                "obs": n_obs,
                "coverage_pct": n_obs / total_obs if total_obs > 0 else 0.0,
                "obs_sufficient": sufficient,
                **metrics,
            }
        )

    per_regime_metrics = pd.DataFrame(regime_rows).set_index("regime")

    # --- Alpha concentration ---
    cum_return_per_regime: dict[str, float] = {}
    for regime_name in observed_regimes:
        regime_mask = regime_timeline == regime_name
        regime_returns = oos_filtered[regime_mask]
        cum_return_per_regime[regime_name] = float((1.0 + regime_returns).prod() - 1.0)

    total_positive = sum(v for v in cum_return_per_regime.values() if v > 0)

    if total_positive > 0:
        concentration = {
            r: max(v, 0.0) / total_positive for r, v in cum_return_per_regime.items()
        }
    else:
        # No positive alpha in any regime — concentration is zero everywhere
        # (no regime-dependent alpha to flag).
        concentration = dict.fromkeys(cum_return_per_regime, 0.0)

    regime_alpha_concentration = pd.Series(concentration, name="alpha_concentration")

    concentrated_regimes = [
        r for r, c in concentration.items() if c > config.single_regime_alpha_threshold
    ]

    return RegimeValidationResult(
        per_regime_metrics=per_regime_metrics,
        per_subperiod_metrics=per_subperiod_metrics,
        regime_alpha_concentration=regime_alpha_concentration,
        concentrated_regimes=concentrated_regimes,
        regime_timeline=regime_timeline,
        total_obs=total_obs,
        n_regimes_observed=n_regimes_observed,
    )
