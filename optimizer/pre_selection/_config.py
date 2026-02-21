"""Configuration for the pre-selection pipeline."""

from __future__ import annotations

from dataclasses import dataclass


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
    correlation_threshold : float
        Pairwise correlation above which an asset is dropped
        (``DropCorrelated``).
    correlation_absolute : bool
        If ``True``, use absolute correlation values.
    top_k : int or None
        If set, keep only the *k* assets with the highest (or lowest) mean
        return via ``SelectKExtremes``.
    top_k_highest : bool
        Select assets with the highest mean when ``True``, lowest when
        ``False``.
    use_pareto : bool
        If ``True``, apply ``SelectNonDominated`` Pareto filter.
    pareto_min_assets : int or None
        Minimum number of assets to retain after Pareto filtering.
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
    correlation_threshold: float = 0.95
    correlation_absolute: bool = False
    top_k: int | None = None
    top_k_highest: bool = True
    use_pareto: bool = False
    pareto_min_assets: int | None = None
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
