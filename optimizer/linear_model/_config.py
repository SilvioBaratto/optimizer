"""Configuration for cross-sectional linear regression.

Wraps :class:`skfolio.linear_model.CSLinearRegression`. ``weighted`` and
``min_observations`` are caller-side hints (consumed by downstream IC
computation); they are not constructor arguments of the underlying
skfolio class in 0.20.1.
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
