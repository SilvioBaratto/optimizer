"""Serialisable configuration for the return-cleaning pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ImputerStrategy(str, Enum):
    """NaN-imputation strategy for the cleaning pipeline."""

    NONE = "none"
    SECTOR = "sector"
    REGRESSION = "regression"


@dataclass(frozen=True)
class CleaningConfig:
    """Frozen, serialisable config for `make_cleaning_pipeline`.

    Assembles the module's per-asset (axis=0) time-series transformers into a
    single ``sklearn.pipeline.Pipeline`` operating on a return DataFrame:
    validate -> outlier-treat -> impute.  Every field is a primitive or enum,
    so the config round-trips and is grid-searchable; the non-serialisable
    ``sector_mapping`` is injected as a factory keyword, never stored here.

    Attributes:
        validate: Prepend a `DataValidator` step.
        max_abs_return: ``DataValidator`` threshold — returns beyond
            ``|max_abs_return|`` (and infinities) become NaN.
        treat_outliers: Insert an `OutlierTreater` step.
        winsorize_threshold: ``OutlierTreater`` z-score boundary between
            normal and winsorised.
        remove_threshold: ``OutlierTreater`` z-score boundary between
            winsorised and NaN-removed.
        imputer: Final imputation step. ``NONE`` leaves NaN in place (the
            pipeline step is omitted), ``SECTOR`` uses `SectorImputer`,
            ``REGRESSION`` uses `RegressionImputer`. Either imputer fills
            interior gaps only; pre-listing and post-delisting NaN stays.
        n_neighbors: ``RegressionImputer`` neighbour count (REGRESSION only).
        min_train_periods: ``RegressionImputer`` minimum complete rows before
            falling back (REGRESSION only).

    Raises:
        ValueError: If ``max_abs_return`` is not positive, ``winsorize_threshold``
            is not positive and strictly below ``remove_threshold``,
            ``n_neighbors`` is negative, or ``min_train_periods`` does not exceed
            ``n_neighbors + 1`` (the OLS coefficient count).
    """

    validate: bool = True
    max_abs_return: float = 10.0
    treat_outliers: bool = True
    winsorize_threshold: float = 3.0
    remove_threshold: float = 10.0
    imputer: ImputerStrategy = ImputerStrategy.NONE
    n_neighbors: int = 5
    min_train_periods: int = 60

    def __post_init__(self) -> None:
        # The comparisons are written as negated ``>`` / ``<`` so NaN is rejected
        # too; a NaN bound would otherwise pass silently and NaN every cell.
        if not self.max_abs_return > 0:
            raise ValueError(
                f"max_abs_return must be positive, got {self.max_abs_return}"
            )
        if not self.winsorize_threshold > 0:
            raise ValueError(
                f"winsorize_threshold must be positive, got {self.winsorize_threshold}"
            )
        if not self.winsorize_threshold < self.remove_threshold:
            raise ValueError(
                f"winsorize_threshold ({self.winsorize_threshold}) must be "
                f"less than remove_threshold ({self.remove_threshold})"
            )
        if self.n_neighbors < 0:
            raise ValueError(
                f"n_neighbors must be non-negative, got {self.n_neighbors}"
            )
        # The OLS has n_neighbors + 1 coefficients and needs strictly more
        # complete rows than that to be determined.
        if self.min_train_periods <= self.n_neighbors + 1:
            raise ValueError(
                f"min_train_periods ({self.min_train_periods}) must exceed "
                f"n_neighbors + 1 ({self.n_neighbors + 1})"
            )
