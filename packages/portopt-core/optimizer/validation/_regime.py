"""Regime-conditional subperiod Sharpe validation.

Splits out-of-sample portfolio returns by macro regime (from
`classify_regime_composite`) and computes
per-regime and per-subperiod performance statistics.  Flags strategies where
alpha is concentrated in a single regime rather than being persistent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pandas as pd

from optimizer.exceptions import DataError
from optimizer.factors._config import MacroRegime, RegimeThresholdConfig
from optimizer.factors._regime import classify_regime

logger = logging.getLogger(__name__)

_COMPOSITE_INDICATORS = ("pmi", "spread_2s10s", "hy_oas", "sentiment")


@dataclass(frozen=True)
class RegimeValidationConfig:
    """Immutable configuration for regime-conditional Sharpe analysis.

    Attributes:
        min_regime_obs: Minimum trading days required to compute meaningful
            statistics for a subperiod or regime aggregate. Subperiods
            shorter than this produce ``NaN`` metrics.
        single_regime_alpha_threshold: Fraction of total positive alpha
            above which a single regime is flagged as concentrated.
        trading_days_per_year: Annualization constant for Sharpe, return,
            and volatility.
        risk_free_rate: Annual risk-free rate for Sharpe ratio computation.
        include_unknown_regime: Whether to include ``MacroRegime.UNKNOWN``
            periods in the per-regime breakdown. ``False`` by default
            since UNKNOWN days are typically data gaps.
        max_macro_staleness_days: Longest gap in calendar days over which the
            last macro observation is carried forward onto an OOS day. Days
            further past the latest macro row are ``MacroRegime.UNKNOWN``
            rather than inheriting a regime nobody observed. The default
            covers quarterly releases; ``None`` carries forward without
            limit.
    """

    min_regime_obs: int = 21
    single_regime_alpha_threshold: float = 0.80
    trading_days_per_year: int = 252
    risk_free_rate: float = 0.0
    include_unknown_regime: bool = False
    max_macro_staleness_days: int | None = 120

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
        if self.max_macro_staleness_days is not None and (
            self.max_macro_staleness_days < 0
        ):
            raise ValueError(
                f"max_macro_staleness_days must be >= 0 or None, "
                f"got {self.max_macro_staleness_days}"
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


def _safe_float(value: object) -> float | None:
    """Convert to float, returning ``None`` for NaN."""
    try:
        f = float(value)  # type: ignore[arg-type]
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


@dataclass
class RegimeValidationResult:
    """Result of regime-conditional subperiod Sharpe analysis.

    Attributes:
        per_regime_metrics: DataFrame indexed by regime name. Columns:
            ``obs``, ``coverage_pct``, ``ann_return``, ``ann_vol``,
            ``sharpe``, ``max_drawdown``, ``obs_sufficient``.
        per_subperiod_metrics: One row per contiguous regime block. Columns:
            ``start``, ``end``, ``regime``, ``obs``, ``ann_return``,
            ``ann_vol``, ``sharpe``, ``max_drawdown``.
        regime_alpha_concentration: Fraction of total positive alpha
            attributable to each regime, where alpha is the mean daily
            return in excess of the risk-free rate. Regimes with fewer than
            ``min_regime_obs`` observations get ``0.0``.
        concentrated_regimes: Regimes exceeding
            ``single_regime_alpha_threshold``.
        regime_timeline: DatetimeIndex to regime name string for every OOS
            observation.
        total_obs: Total number of OOS observations.
        n_regimes_observed: Distinct regimes with at least one observation.
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

        Returns:
            Dict suitable for frontend charting (ECharts) or CLI reporting.
            Structure::

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
            alpha_conc = cast(
                float, self.regime_alpha_concentration.get(regime_name, 0.0)
            )
            regime_rows.append(
                {
                    "regime": str(regime_name),
                    "obs": int(row["obs"]),
                    "coverage_pct": float(row["coverage_pct"]),
                    "ann_return": _safe_float(row["ann_return"]),
                    "ann_vol": _safe_float(row["ann_vol"]),
                    "sharpe": _safe_float(row["sharpe"]),
                    "max_drawdown": _safe_float(row["max_drawdown"]),
                    "alpha_concentration": float(alpha_conc),
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
                    "obs": int(cast(int, sp["obs"])),
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


def _build_regime_series(
    macro_data: pd.DataFrame,
    thresholds: RegimeThresholdConfig | None,
) -> pd.Series:
    """Build a point-in-time regime series from macro data.

    Each date's classification uses only data up to that date to avoid
    look-ahead bias.  Uses `classify_regime` which dispatches to
    the composite classifier when PMI/spread/HY columns are present and
    falls back to the GDP heuristic otherwise.

    The composite indicators are forward-filled first: the composite scorer
    reads only the last row and scores a missing value as neutral, so a
    monthly PMI on a daily frame would otherwise flip the regime on every
    non-release day.  Forward-filling only looks backwards, so it stays
    point-in-time.  The GDP and yield-spread fallbacks work on the
    ``dropna()`` series and need the native release cadence, so they are
    left untouched.
    """
    carried = [c for c in _COMPOSITE_INDICATORS if c in macro_data.columns]
    if carried:
        macro_data = macro_data.copy()
        macro_data[carried] = macro_data[carried].ffill()

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
    n = len(returns_slice)
    if n == 0:
        return {
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
        }
    mean_daily = returns_slice.mean()
    valid = returns_slice.dropna()
    # Volatility is undefined for a single observation, and annualising one
    # day's return only extrapolates it, so both stay NaN rather than 0.0 / a
    # four-digit percentage.
    if len(valid) < 2:
        ann_return = np.nan
        ann_vol = np.nan
        sharpe = np.nan
    else:
        std_daily = returns_slice.std(ddof=1)
        # Realised CAGR, not (1 + mean)^T - 1: compounding the arithmetic mean
        # ignores volatility drag and overstates returns, most in the high-vol
        # regimes this module exists to scrutinise.
        growth = float(np.prod(1.0 + valid.to_numpy()))
        if growth <= 0.0:
            ann_return = -1.0
        else:
            ann_return = float(growth ** (trading_days / len(valid)) - 1.0)
        ann_vol = float(std_daily * np.sqrt(trading_days))
        sharpe = (
            float((mean_daily - rf_daily) / std_daily) * np.sqrt(trading_days)
            if std_daily > 0
            else np.nan
        )

    cum = (1.0 + returns_slice).cumprod()
    # The running peak starts at the initial equity of 1.0 so that a slice
    # opening with losses registers them as a drawdown.
    running_max = np.maximum(cum.cummax(), 1.0)
    drawdown = cum / running_max - 1.0
    max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0

    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


def _empty_result(total_obs: int) -> RegimeValidationResult:
    """Build a well-formed empty result (no regimes to report).

    Shared by the no-macro-overlap branch and the all-observations-filtered
    branch so both return identical, correctly-typed empty frames instead of
    crashing on an empty ``set_index`` / aggregation.
    """
    return RegimeValidationResult(
        per_regime_metrics=pd.DataFrame(
            columns=[
                "obs",
                "coverage_pct",
                "ann_return",
                "ann_vol",
                "sharpe",
                "max_drawdown",
                "obs_sufficient",
            ]
        ),
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
        total_obs=total_obs,
        n_regimes_observed=0,
    )


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
    index = cast(pd.DatetimeIndex, regime_timeline.index)

    block_start = index[0]
    current_regime = values[0]

    for i in range(1, len(values)):
        if values[i] != current_regime:
            blocks.append((block_start, index[i - 1], current_regime))
            block_start = index[i]
            current_regime = values[i]

    blocks.append((block_start, index[-1], current_regime))
    return blocks


def run_regime_validation(
    oos_returns: pd.Series,
    macro_data: pd.DataFrame,
    config: RegimeValidationConfig | None = None,
    thresholds: RegimeThresholdConfig | None = None,
) -> RegimeValidationResult:
    """Run regime-conditional subperiod Sharpe analysis.

    Args:
        oos_returns: Daily portfolio returns indexed by ``DatetimeIndex``.
        macro_data: Macro indicators indexed by date, compatible with
            `classify_regime_composite`. Rows may arrive in any order; they
            are sorted chronologically and, for a repeated date, the last
            row wins.
        config: Validation configuration. Defaults to standard.
        thresholds: Regime classification thresholds.

    Returns:
        Per-regime and per-subperiod statistics with alpha concentration
        flags.

    Raises:
        DataError: If ``oos_returns`` is empty.
    """
    if config is None:
        config = RegimeValidationConfig()

    if len(oos_returns) == 0:
        raise DataError("oos_returns is empty")

    rf_daily = config.risk_free_rate / config.trading_days_per_year
    tdays = config.trading_days_per_year

    # The point-in-time window below is positional, so row order has to be
    # chronological; a stable sort keeps "last row wins" meaningful for
    # repeated dates.
    macro_data = macro_data.sort_index(kind="stable")
    macro_data = macro_data.loc[~macro_data.index.duplicated(keep="last")]

    overlap = macro_data.index.intersection(oos_returns.index).size
    if len(macro_data) == 0 or not overlap:
        logger.warning(
            "run_regime_validation: no macro data overlaps with OOS returns; "
            "returning NaN metrics."
        )
        return _empty_result(len(oos_returns))

    regime_series = _build_regime_series(macro_data, thresholds)

    tolerance = (
        None
        if config.max_macro_staleness_days is None
        else pd.Timedelta(days=config.max_macro_staleness_days)
    )
    regime_timeline = regime_series.reindex(
        oos_returns.index, method="ffill", tolerance=tolerance
    )
    stale = (oos_returns.index >= regime_series.index[0]) & regime_timeline.isna()
    if stale.any():
        logger.warning(
            "run_regime_validation: %d OOS observations are more than %s days "
            "past the latest macro observation (stale macro data); classified "
            "as UNKNOWN.",
            int(stale.sum()),
            config.max_macro_staleness_days,
        )
    # Days before macro data starts, or too far past its latest row → UNKNOWN
    regime_timeline = regime_timeline.fillna(MacroRegime.UNKNOWN.value)

    if not config.include_unknown_regime:
        mask = regime_timeline != MacroRegime.UNKNOWN.value
        regime_timeline = regime_timeline[mask]
        oos_filtered = oos_returns.reindex(regime_timeline.index)
    else:
        oos_filtered = oos_returns.reindex(regime_timeline.index)

    total_obs = len(oos_returns)

    # Every overlapping observation classified as UNKNOWN and dropped — no
    # regimes remain to aggregate. Return a well-formed empty result rather
    # than crashing on an empty per-regime aggregation.
    if len(regime_timeline) == 0:
        logger.warning(
            "run_regime_validation: all overlapping observations classified as "
            "UNKNOWN and excluded; returning empty metrics."
        )
        return _empty_result(total_obs)

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

    observed_regimes = sorted(regime_timeline.unique())
    n_regimes_observed = len(observed_regimes)

    regime_rows: list[dict[str, object]] = []
    alpha_per_day: dict[str, float] = {}
    for regime_name in observed_regimes:
        regime_mask = regime_timeline == regime_name
        regime_returns = cast(pd.Series, oos_filtered[regime_mask])
        n_obs = len(regime_returns)
        sufficient = n_obs >= config.min_regime_obs
        # Alpha is the mean daily excess return, so a regime's weight does not
        # grow with the number of days it lasts; under-sampled regimes carry
        # NaN metrics and so take no share.
        mean_excess = float((regime_returns - rf_daily).mean()) if sufficient else 0.0
        alpha_per_day[regime_name] = mean_excess if mean_excess > 0 else 0.0
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

    total_positive = sum(alpha_per_day.values())

    if total_positive > 0:
        concentration = {r: v / total_positive for r, v in alpha_per_day.items()}
    else:
        # No positive alpha in any regime — concentration is zero everywhere
        # (no regime-dependent alpha to flag).
        concentration = dict.fromkeys(alpha_per_day, 0.0)

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
