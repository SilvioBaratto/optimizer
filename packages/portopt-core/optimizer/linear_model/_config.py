"""Configuration for cross-sectional linear regression.

Wraps `CSLinearRegression` and `CSLinearRegressorWrapper` (skfolio 1.0.6).
``weighted`` and ``min_observations`` are not constructor arguments of the
underlying skfolio classes. ``CSLinearRegressionConfig.min_observations`` is
enforced by ``compute_ic_series(use_cs_regression=True)``; ``weighted`` and the
wrapper config's two hint fields are informational for callers that fit the
estimator themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

from optimizer.exceptions import ConfigurationError


@dataclass(frozen=True)
class CSLinearRegressionConfig:
    """Immutable configuration for a CS linear regression estimator.

    Args:
        fit_intercept: Whether to fit a per-period intercept. Default ``True``.
            Forwarded to `CSLinearRegression`.
        weighted: Informational flag signalling that the caller will supply
            ``cs_weights`` at ``fit`` time. Not forwarded to skfolio (no
            constructor arg) and not read by ``compute_ic_series``, which always
            passes its own 0/1 validity mask.
        min_observations: Minimum non-NaN cross-sectional observations per period.
            ``compute_ic_series(use_cs_regression=True)`` drops periods below
            ``max(its min_observations argument, this value)``. Not forwarded to
            skfolio.
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
        """Preset that sets only the informational ``weighted`` flag."""
        return cls(weighted=True)


@dataclass(frozen=True)
class CSLinearRegressorWrapperConfig:
    """Immutable configuration for a wrapped per-period sklearn regressor.

    Adapts any scikit-learn ``Regressor`` (Ridge, Lasso, tree-based, ...) to
    the cross-sectional contract via `CSLinearRegressorWrapper`: the wrapped
    regressor is fit independently for each observation ``t``.

    The regressor instance itself is **not** a serialisable primitive, so it
    is passed to the factory as a ``**kwargs`` argument rather than stored on
    the (frozen, serialisable) config.

    Args:
        n_jobs: Number of parallel jobs for the per-period fits. Forwarded to
            `CSLinearRegressorWrapper`. ``-1`` uses
            all processors. Default ``1``.
        weighted: Informational flag signalling that the caller will supply
            ``cs_weights`` at ``fit`` time. Not forwarded to skfolio and not read
            by any library routine.
        min_observations: Minimum non-NaN cross-sectional observations per period
            the caller intends to require. Not forwarded to skfolio and not
            enforced by any library routine.
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
        """Preset that sets only the informational ``weighted`` flag."""
        return cls(weighted=True)
