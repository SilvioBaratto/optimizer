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
            ``REGRESSION`` uses `RegressionImputer`.
        n_neighbors: ``RegressionImputer`` neighbour count (REGRESSION only).
        min_train_periods: ``RegressionImputer`` minimum complete rows before
            falling back (REGRESSION only).
    """

    validate: bool = True
    max_abs_return: float = 10.0
    treat_outliers: bool = True
    winsorize_threshold: float = 3.0
    remove_threshold: float = 10.0
    imputer: ImputerStrategy = ImputerStrategy.NONE
    n_neighbors: int = 5
    min_train_periods: int = 60
