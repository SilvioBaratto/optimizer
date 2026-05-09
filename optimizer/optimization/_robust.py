"""RobustMeanRisk configuration and factory.

Both ``mu_uncertainty_set_config=None`` AND
``covariance_uncertainty_set_config=None`` recovers plain ``MeanRisk``
exactly. Uncertainty sets use ``confidence_level`` (e.g. 0.95); the
chi-squared-derived kappa scaling is internal to ``MeanRisk`` and not
exposed in the Config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from skfolio.optimization import MeanRisk
from skfolio.prior._base import BasePrior

from optimizer.optimization._config import MeanRiskConfig
from optimizer.optimization._factory import build_mean_risk
from optimizer.uncertainty_set._config import (
    CovarianceUncertaintySetConfig,
    CovarianceUncertaintySetType,
    MuUncertaintySetConfig,
)
from optimizer.uncertainty_set._factory import (
    build_covariance_uncertainty_set,
    build_mu_uncertainty_set,
)


@dataclass(frozen=True)
class RobustMeanRiskConfig:
    """Immutable configuration for the robust ``MeanRisk`` family.

    Parameters
    ----------
    mean_risk_config : MeanRiskConfig
        Forwarded to :func:`build_mean_risk`. Default
        :class:`MeanRiskConfig` (minimum-variance).
    mu_uncertainty_set_config : MuUncertaintySetConfig or None
        Mu uncertainty-set configuration. ``None`` disables it.
    covariance_uncertainty_set_config : CovarianceUncertaintySetConfig or None
        Covariance uncertainty-set configuration. ``None`` disables it.
    """

    mean_risk_config: MeanRiskConfig = field(default_factory=MeanRiskConfig)
    mu_uncertainty_set_config: MuUncertaintySetConfig | None = None
    covariance_uncertainty_set_config: CovarianceUncertaintySetConfig | None = None

    @classmethod
    def for_conservative(cls) -> RobustMeanRiskConfig:
        """High-confidence preset (99%) — wide ellipsoidal uncertainty."""
        return cls(
            mu_uncertainty_set_config=MuUncertaintySetConfig.for_empirical(
                confidence_level=0.99
            ),
        )

    @classmethod
    def for_moderate(
        cls, *, uncertainty_level: float = 0.95
    ) -> RobustMeanRiskConfig:
        """Moderate confidence preset (default 95%).

        Parameters
        ----------
        uncertainty_level : float, default=0.95
            Confidence level forwarded to
            :meth:`MuUncertaintySetConfig.for_empirical`.  Keyword-only.
        """
        return cls(
            mu_uncertainty_set_config=MuUncertaintySetConfig.for_empirical(
                confidence_level=uncertainty_level
            ),
        )

    @classmethod
    def for_aggressive(cls) -> RobustMeanRiskConfig:
        """Low-confidence preset (90%) — narrow ellipsoidal uncertainty."""
        return cls(
            mu_uncertainty_set_config=MuUncertaintySetConfig.for_empirical(
                confidence_level=0.90
            ),
        )

    @classmethod
    def for_bootstrap_covariance(cls) -> RobustMeanRiskConfig:
        """Stationary-bootstrap covariance uncertainty preset."""
        return cls(
            covariance_uncertainty_set_config=CovarianceUncertaintySetConfig(
                kind=CovarianceUncertaintySetType.BOOTSTRAP,
            ),
        )


def build_robust_mean_risk(
    config: RobustMeanRiskConfig | None = None,
    *,
    prior_estimator: BasePrior | None = None,
    **kwargs: Any,
) -> MeanRisk:
    """Build a robust ``MeanRisk`` optimiser from *config*.

    When both uncertainty configs are ``None``, the returned estimator
    is identical (in fitted weights) to ``build_mean_risk(config.mean_risk_config)``.

    Parameters
    ----------
    config : RobustMeanRiskConfig or None
        Robust configuration. ``None`` triggers default.
    prior_estimator : BasePrior or None
        Forwarded to :func:`build_mean_risk`.
    **kwargs
        Additional kwargs forwarded to the wrapped ``MeanRisk``.

    Returns
    -------
    MeanRisk
        A fitted-ready skfolio optimiser with optional uncertainty-set
        estimators attached.
    """
    if config is None:
        config = RobustMeanRiskConfig()

    if config.mu_uncertainty_set_config is not None:
        kwargs.setdefault(
            "mu_uncertainty_set_estimator",
            build_mu_uncertainty_set(config.mu_uncertainty_set_config),
        )
    if config.covariance_uncertainty_set_config is not None:
        kwargs.setdefault(
            "covariance_uncertainty_set_estimator",
            build_covariance_uncertainty_set(
                config.covariance_uncertainty_set_config
            ),
        )

    return build_mean_risk(
        config.mean_risk_config,
        prior_estimator=prior_estimator,
        **kwargs,
    )


__all__ = ["RobustMeanRiskConfig", "build_robust_mean_risk"]
