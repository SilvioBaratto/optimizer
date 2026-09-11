"""Configuration for cross-sectional linear regression.

Wraps :class:`skfolio.linear_model.CSLinearRegression` and
:class:`skfolio.linear_model.CSLinearRegressorWrapper` (skfolio 1.0.6).
``weighted`` and ``min_observations`` are caller-side hints (consumed by
downstream IC computation); they are not constructor arguments of the
underlying skfolio classes.
"""

from __future__ import annotations

from dataclasses import dataclass

from optimizer.exceptions import ConfigurationError


@dataclass(frozen=True)
class CSLinearRegressionConfig:
    """Immutable configuration for a CS linear regression estimator.

    Parameters
    ----------
    fit_intercept : bool
        Whether to fit a per-period intercept. Default ``True``.
        Forwarded to :class:`skfolio.linear_model.CSLinearRegression`.
    weighted : bool
        Caller hint signalling that ``cs_weights`` will be supplied at
        ``fit`` time. Not forwarded to skfolio (no constructor arg).
    min_observations : int
        Minimum non-NaN cross-sectional observations per period required
        before downstream consumers (e.g. factor IC) accept the period.
        Not forwarded to skfolio.
    """

    fit_intercept: bool = True
    weighted: bool = False
    min_observations: int = 10

    def __post_init__(self) -> None:
        if self.min_observations < 0:
            raise ConfigurationError("min_observations must be non-negative")

    @classmethod
    def for_default(cls) -> CSLinearRegressionConfig:
        """Default OLS preset with intercept."""
        return cls()

    @classmethod
    def for_weighted(cls) -> CSLinearRegressionConfig:
        """Weighted-fit preset for downstream IC weighting."""
        return cls(weighted=True)


@dataclass(frozen=True)
class CSLinearRegressorWrapperConfig:
    """Immutable configuration for a wrapped per-period sklearn regressor.

    Adapts any scikit-learn ``Regressor`` (Ridge, Lasso, tree-based, ...) to
    the cross-sectional contract via
    :class:`skfolio.linear_model.CSLinearRegressorWrapper`: the wrapped
    regressor is fit independently for each observation ``t``.

    The regressor instance itself is **not** a serialisable primitive, so it
    is passed to the factory as a ``**kwargs`` argument rather than stored on
    the (frozen, serialisable) config.

    Parameters
    ----------
    n_jobs : int
        Number of parallel jobs for the per-period fits. Forwarded to
        :class:`skfolio.linear_model.CSLinearRegressorWrapper`. ``-1`` uses
        all processors. Default ``1``.
    weighted : bool
        Caller hint signalling that ``cs_weights`` will be supplied at
        ``fit`` time. Not forwarded to skfolio.
    min_observations : int
        Minimum non-NaN cross-sectional observations per period required
        before downstream consumers accept the period. Not forwarded to
        skfolio.
    """

    n_jobs: int = 1
    weighted: bool = False
    min_observations: int = 10

    def __post_init__(self) -> None:
        if self.n_jobs == 0:
            raise ConfigurationError("n_jobs must be non-zero (>=1 or -1)")
        if self.n_jobs < -1:
            raise ConfigurationError("n_jobs must be >= -1")
        if self.min_observations < 0:
            raise ConfigurationError("min_observations must be non-negative")

    @classmethod
    def for_default(cls) -> CSLinearRegressorWrapperConfig:
        """Default single-process preset."""
        return cls()

    @classmethod
    def for_weighted(cls) -> CSLinearRegressorWrapperConfig:
        """Weighted-fit preset for downstream IC weighting."""
        return cls(weighted=True)
