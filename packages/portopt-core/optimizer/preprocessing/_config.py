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
    """Frozen, serialisable config for :func:`make_cleaning_pipeline`.

    Assembles the module's per-asset (axis=0) time-series transformers into a
    single ``sklearn.pipeline.Pipeline`` operating on a return DataFrame:
    validate -> outlier-treat -> impute.  Every field is a primitive or enum,
    so the config round-trips and is grid-searchable; the non-serialisable
    ``sector_mapping`` is injected as a factory keyword, never stored here.

    Parameters
    ----------
    validate : bool, default=True
        Prepend a :class:`~optimizer.preprocessing.DataValidator` step.
    max_abs_return : float, default=10.0
        ``DataValidator`` threshold — returns beyond ``|max_abs_return|`` (and
        infinities) become NaN.
    treat_outliers : bool, default=True
        Insert an :class:`~optimizer.preprocessing.OutlierTreater` step.
    winsorize_threshold : float, default=3.0
        ``OutlierTreater`` z-score boundary between normal and winsorised.
    remove_threshold : float, default=10.0
        ``OutlierTreater`` z-score boundary between winsorised and NaN-removed.
    imputer : ImputerStrategy, default=NONE
        Final imputation step.  ``NONE`` leaves NaN in place (the pipeline
        step is omitted), ``SECTOR`` uses
        :class:`~optimizer.preprocessing.SectorImputer`, ``REGRESSION`` uses
        :class:`~optimizer.preprocessing.RegressionImputer`.
    n_neighbors : int, default=5
        ``RegressionImputer`` neighbour count (REGRESSION only).
    min_train_periods : int, default=60
        ``RegressionImputer`` minimum complete rows before falling back
        (REGRESSION only).
    """

    validate: bool = True
    max_abs_return: float = 10.0
    treat_outliers: bool = True
    winsorize_threshold: float = 3.0
    remove_threshold: float = 10.0
    imputer: ImputerStrategy = ImputerStrategy.NONE
    n_neighbors: int = 5
    min_train_periods: int = 60
