"""Configuration for the pre-selection pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SelectKMeasure(str, Enum):
    """Serialisable performance measure for :class:`SelectKExtremes`.

    Maps to a member of one of skfolio's measure enums
    (:class:`skfolio.measures.RatioMeasure`,
    :class:`skfolio.measures.PerfMeasure` or
    :class:`skfolio.measures.RiskMeasure`).  The factory resolves the string
    to the concrete skfolio enum member so the config stays serialisable.

    ``SHARPE_RATIO`` reproduces skfolio's default ranking measure.  ``MEAN``
    ranks purely on average return (the behaviour the legacy docstring
    described).  Risk measures rank by risk, so ``highest=False`` selects the
    *lowest*-risk assets.
    """

    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    MEAN = "mean"
    ANNUALIZED_MEAN = "annualized_mean"
    VARIANCE = "variance"
    STANDARD_DEVIATION = "standard_deviation"
    SEMI_DEVIATION = "semi_deviation"
    CVAR = "cvar"
    MAX_DRAWDOWN = "max_drawdown"


@dataclass(frozen=True)
class PreSelectionConfig:
    """Immutable configuration for the pre-selection pipeline.

    All parameters map 1:1 to transformer/selector constructor arguments,
    making the config serialisable and suitable for hyperparameter sweeps.

    Parameters
    ----------
    max_abs_return : float
        Maximum absolute return before treating as data error (DataValidator).
    winsorize_threshold : float
        Z-score threshold for winsorisation (OutlierTreater).
    remove_threshold : float
        Z-score threshold for removal as data error (OutlierTreater).
    outlier_method : str
        Outlier detection approach. Currently only ``"time_series"`` is
        supported (per-column z-scores).
    imputation_fallback : str
        Fallback when sector data is unavailable. ``"global_mean"`` uses the
        cross-sectional mean across all assets.
    zero_variance_threshold : float
        Variance below which an asset is treated as constant and dropped
        (``DropZeroVariance``).  Must be non-negative.
    drop_internal_nan : bool
        Forwarded to ``SelectComplete.drop_assets_with_internal_nan``.  When
        ``True``, assets carrying NaNs *inside* their history (not just
        leading/trailing gaps) are also removed.
    correlation_threshold : float
        Pairwise correlation above which an asset is dropped
        (``DropCorrelated``).
    correlation_absolute : bool
        If ``True``, use absolute correlation values.
    top_k : int or None
        If set, keep only the *k* assets ranked most extreme on
        ``select_k_measure`` via ``SelectKExtremes``.
    top_k_highest : bool
        Select assets with the highest measure when ``True``, lowest when
        ``False``.
    select_k_measure : SelectKMeasure
        Performance measure used by ``SelectKExtremes`` to rank assets.
        Defaults to Sharpe ratio (skfolio's own default).
    use_pareto : bool
        If ``True``, apply ``SelectNonDominated`` Pareto filter.
    pareto_min_assets : int or None
        Minimum number of assets to retain after Pareto filtering.
    pareto_threshold : float
        Domination threshold forwarded to ``SelectNonDominated.threshold``
        (skfolio default ``-0.5``).
    use_non_expiring : bool
        If ``True``, apply ``SelectNonExpiring`` to remove soon-expiring
        assets.
    expiration_lookahead : int or None
        Number of calendar days to look ahead for expiring assets,
        forwarded to ``SelectNonExpiring`` as a ``timedelta``.
    is_log_normal : bool
        Whether returns are assumed log-normal for multi-period scaling
        (deferred to Chapter 2, stored here for completeness).
    """

    max_abs_return: float = 10.0
    winsorize_threshold: float = 3.0
    remove_threshold: float = 10.0
    outlier_method: str = "time_series"
    imputation_fallback: str = "global_mean"
    zero_variance_threshold: float = 1e-8
    drop_internal_nan: bool = False
    correlation_threshold: float = 0.95
    correlation_absolute: bool = False
    top_k: int | None = None
    top_k_highest: bool = True
    select_k_measure: SelectKMeasure = SelectKMeasure.SHARPE_RATIO
    use_pareto: bool = False
    pareto_min_assets: int | None = None
    pareto_threshold: float = -0.5
    use_non_expiring: bool = False
    expiration_lookahead: int | None = None
    is_log_normal: bool = True

    def __post_init__(self) -> None:
        if self.winsorize_threshold >= self.remove_threshold:
            raise ValueError(
                f"winsorize_threshold ({self.winsorize_threshold}) must be "
                f"less than remove_threshold ({self.remove_threshold})"
            )
        if not (0.0 < self.correlation_threshold <= 1.0):
            raise ValueError(
                f"correlation_threshold must be in (0, 1], "
                f"got {self.correlation_threshold}"
            )
        if self.max_abs_return <= 0:
            raise ValueError(
                f"max_abs_return must be positive, got {self.max_abs_return}"
            )
        if self.zero_variance_threshold < 0:
            raise ValueError(
                "zero_variance_threshold must be non-negative, "
                f"got {self.zero_variance_threshold}"
            )
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")
        if self.pareto_min_assets is not None and self.pareto_min_assets <= 0:
            raise ValueError(
                f"pareto_min_assets must be positive, got {self.pareto_min_assets}"
            )
        if self.expiration_lookahead is not None and self.expiration_lookahead <= 0:
            raise ValueError(
                "expiration_lookahead must be positive, "
                f"got {self.expiration_lookahead}"
            )
        # Coerce a plain string into the enum so callers can pass either.
        if not isinstance(self.select_k_measure, SelectKMeasure):
            object.__setattr__(
                self, "select_k_measure", SelectKMeasure(self.select_k_measure)
            )

    # -- factory methods -----------------------------------------------------

    @classmethod
    def for_daily_annual(cls) -> PreSelectionConfig:
        """Sensible defaults for daily returns over a ~1-year horizon."""
        return cls(
            max_abs_return=10.0,
            winsorize_threshold=3.0,
            remove_threshold=10.0,
            correlation_threshold=0.95,
            is_log_normal=True,
        )

    @classmethod
    def for_conservative(cls) -> PreSelectionConfig:
        """Tighter filters for a more conservative universe."""
        return cls(
            max_abs_return=5.0,
            winsorize_threshold=2.5,
            remove_threshold=8.0,
            correlation_threshold=0.85,
            top_k=50,
            top_k_highest=True,
            is_log_normal=True,
        )

    @classmethod
    def for_low_volatility(cls) -> PreSelectionConfig:
        """Screen down to the lowest-volatility assets.

        Ranks on standard deviation and keeps the *lowest* (``top_k_highest``
        ``False``), a common defensive-tilt pre-filter.
        """
        return cls(
            max_abs_return=5.0,
            winsorize_threshold=2.5,
            remove_threshold=8.0,
            correlation_threshold=0.90,
            top_k=50,
            top_k_highest=False,
            select_k_measure=SelectKMeasure.STANDARD_DEVIATION,
            is_log_normal=True,
        )
