"""Bridge factor-mimicking returns to a skfolio factor-model prior.

This module turns observed factor-return panels (e.g. the output of
:func:`build_all_factor_mimicking_portfolios`) into a fitted
:class:`skfolio.prior.TimeSeriesFactorModel` — the skfolio prior that
regresses asset returns on factor returns and rebuilds ``mu``/``covariance``
from the estimated loadings.  It follows the module convention: a frozen
serialisable config (:class:`FactorPriorConfig`) plus a factory
(:func:`build_time_series_factor_model`); non-serialisable estimator
instances (loading-matrix / inner-prior estimators, factor-family arrays)
are passed as factory ``**kwargs``.

skfolio 1.0 notes (verified against the installed 1.0.6):
- ``TimeSeriesFactorModel.fit(X, *, factors=...)`` — ``factors`` is
  **keyword-only and required** (NOT the legacy ``fit(X, y)`` positional
  form).  Both ``X`` and ``factors`` must be finite: NaN raises ``ValueError``.
- The fitted container attribute is ``return_distribution_`` (``mu``,
  ``covariance``, ``returns``, ...), consistent with the rest of the library.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from optimizer.exceptions import ConfigurationError, DataError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from skfolio.prior import BasePrior, TimeSeriesFactorModel
    from skfolio.prior._time_series_factor_model import BaseLoadingMatrix

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FactorPriorConfig:
    """Immutable configuration for a time-series factor-model prior.

    Holds only the serialisable, primitive constructor arguments of
    :class:`skfolio.prior.TimeSeriesFactorModel`.  Non-serialisable objects
    (the loading-matrix estimator, the inner factor prior, and the
    factor-family labels) are supplied at build time as factory ``**kwargs``.

    Parameters
    ----------
    higham : bool, default False
        When ``True``, use Higham's nearest-correlation-matrix projection to
        repair a non-PSD reconstructed covariance instead of clipping
        eigenvalues.  Forwarded to ``TimeSeriesFactorModel(higham=...)``.
    max_iteration : int, default 100
        Maximum iterations for the Higham projection.  Must be positive.
        Forwarded to ``TimeSeriesFactorModel(max_iteration=...)``.
    min_observations : int, default 12
        Caller-side hint: minimum number of aligned, fully-finite periods
        required by :func:`fit_factor_prior` before it will fit.  Not a
        skfolio constructor argument.
    """

    higham: bool = False
    max_iteration: int = 100
    min_observations: int = 12

    def __post_init__(self) -> None:
        if self.max_iteration <= 0:
            raise ConfigurationError("max_iteration must be positive")
        if self.min_observations < 2:
            raise ConfigurationError("min_observations must be >= 2")

    @classmethod
    def for_default(cls) -> FactorPriorConfig:
        """Default preset (eigenvalue clipping for PSD repair)."""
        return cls()

    @classmethod
    def for_higham(cls) -> FactorPriorConfig:
        """Preset using Higham's nearest-correlation projection."""
        return cls(higham=True)


def build_time_series_factor_model(
    config: FactorPriorConfig | None = None,
    *,
    loading_matrix_estimator: BaseLoadingMatrix | None = None,
    factor_prior_estimator: BasePrior | None = None,
    factor_families: Sequence[object] | None = None,
) -> TimeSeriesFactorModel:
    """Build an unfitted :class:`skfolio.prior.TimeSeriesFactorModel`.

    Parameters
    ----------
    config : FactorPriorConfig or None
        Serialisable configuration.  ``None`` uses
        :meth:`FactorPriorConfig.for_default`.
    loading_matrix_estimator : BaseLoadingMatrix or None
        Estimator producing the asset-on-factor loading matrix (e.g.
        ``skfolio.prior.LoadingMatrixRegression``).  ``None`` defers to the
        skfolio default.
    factor_prior_estimator : BasePrior or None
        Inner prior estimating the factors' own return distribution.
        ``None`` defers to the skfolio default (``EmpiricalPrior``).
    factor_families : sequence or None
        Optional family label per factor (aligned to the ``factors`` columns)
        used by skfolio to group factors.  ``None`` treats each factor
        independently.

    Returns
    -------
    TimeSeriesFactorModel
        An unfitted estimator; call :meth:`fit` with ``factors=`` keyword.
    """
    from skfolio.prior import TimeSeriesFactorModel

    if config is None:
        config = FactorPriorConfig.for_default()

    kwargs: dict[str, object] = {
        "higham": config.higham,
        "max_iteration": config.max_iteration,
    }
    if loading_matrix_estimator is not None:
        kwargs["loading_matrix_estimator"] = loading_matrix_estimator
    if factor_prior_estimator is not None:
        kwargs["factor_prior_estimator"] = factor_prior_estimator
    if factor_families is not None:
        kwargs["factor_families"] = list(factor_families)

    return TimeSeriesFactorModel(**kwargs)


def fit_factor_prior(
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    config: FactorPriorConfig | None = None,
    *,
    loading_matrix_estimator: BaseLoadingMatrix | None = None,
    factor_prior_estimator: BasePrior | None = None,
    factor_families: Sequence[object] | None = None,
) -> TimeSeriesFactorModel:
    """Fit a time-series factor-model prior from return panels.

    Aligns *asset_returns* and *factor_returns* on their common date index,
    drops any period that is not fully finite across both frames (skfolio
    rejects NaN), and fits
    :class:`skfolio.prior.TimeSeriesFactorModel` via the keyword-only
    ``factors=`` argument.  The fitted model exposes ``return_distribution_``
    (``mu``, ``covariance``, ...) for downstream optimisation / Black-Litterman.

    Parameters
    ----------
    asset_returns : pd.DataFrame
        Dates × assets matrix of asset (linear) returns — the regression
        target ``X``.
    factor_returns : pd.DataFrame
        Dates × factors matrix of factor returns, e.g. the output of
        :func:`build_all_factor_mimicking_portfolios` — the regressors
        passed as ``factors=``.
    config : FactorPriorConfig or None
        Serialisable configuration.  ``None`` uses the default preset;
        also supplies ``min_observations``.
    loading_matrix_estimator, factor_prior_estimator, factor_families
        Non-serialisable overrides forwarded to
        :func:`build_time_series_factor_model`.

    Returns
    -------
    TimeSeriesFactorModel
        The fitted prior.

    Raises
    ------
    DataError
        If either frame is empty, they share no dates, or fewer than
        ``config.min_observations`` fully-finite aligned periods remain.
    """
    if config is None:
        config = FactorPriorConfig.for_default()

    if asset_returns.empty or factor_returns.empty:
        raise DataError("asset_returns and factor_returns must both be non-empty")

    common = asset_returns.index.intersection(factor_returns.index)
    if len(common) == 0:
        raise DataError("asset_returns and factor_returns share no common dates")

    x = asset_returns.loc[common]
    f = factor_returns.loc[common]

    finite = x.notna().all(axis=1).to_numpy() & f.notna().all(axis=1).to_numpy()
    n_finite = int(np.count_nonzero(finite))
    if n_finite < config.min_observations:
        raise DataError(
            "fit_factor_prior requires at least "
            f"{config.min_observations} fully-finite aligned periods, "
            f"got {n_finite}"
        )

    x = x.loc[finite]
    f = f.loc[finite]

    model = build_time_series_factor_model(
        config,
        loading_matrix_estimator=loading_matrix_estimator,
        factor_prior_estimator=factor_prior_estimator,
        factor_families=factor_families,
    )
    model.fit(x, factors=f)
    return model
