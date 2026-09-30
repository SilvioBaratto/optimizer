"""StackingOptimization configuration and factory.

Base estimators are passed as a factory kwarg, not stored in Config —
they are not serialisable. The Config holds the meta-optimizer preset
and CV / quantile settings only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from skfolio.optimization import StackingOptimization
from skfolio.optimization._base import BaseOptimization
from skfolio.prior._base import BasePrior

from optimizer.exceptions import ConfigurationError
from optimizer.moments._config import MomentEstimationConfig
from optimizer.moments._factory import build_prior
from optimizer.optimization._config import MeanRiskConfig
from optimizer.optimization._factory import build_mean_risk


@dataclass(frozen=True)
class StackingConfig:
    """Immutable configuration for StackingOptimization.

    Base estimators are non-serialisable — pass them as the
    ``estimators`` factory kwarg.

    Attributes:
        prior_config: Inner prior configuration for the final estimator's prior.
        quantile: Quantile threshold passed to the wrapped optimizer.
        final_default: Preset used to build the default final_estimator when
            the final_estimator factory kwarg is omitted.
    """

    prior_config: MomentEstimationConfig | None = None
    quantile: float = 0.5
    final_default: MeanRiskConfig = field(default_factory=MeanRiskConfig)

    @classmethod
    def for_min_variance_final(cls) -> StackingConfig:
        """Min-variance final estimator preset."""
        return cls(final_default=MeanRiskConfig.for_min_variance())

    @classmethod
    def for_max_sharpe_final(cls) -> StackingConfig:
        """Max-Sharpe final estimator preset."""
        return cls(final_default=MeanRiskConfig.for_max_sharpe())


def build_stacking(
    config: StackingConfig | None = None,
    *,
    estimators: list[tuple[str, BaseOptimization]] | None = None,
    final_estimator: BaseOptimization | None = None,
    prior_estimator: BasePrior | None = None,
    **kwargs: Any,
) -> StackingOptimization:
    """Build a StackingOptimization from config.

    Args:
        config: Stacking configuration. None triggers default.
        estimators: Base estimators as [(name, estimator), ...]. Required —
            non-serialisable, must be supplied at factory time.
        final_estimator: Meta-optimizer. When None, build_mean_risk is called
            with config.final_default.
        prior_estimator: Optional prior built from config.prior_config and
            forwarded to the auto-built final_estimator (only used when
            final_estimator is omitted).
        **kwargs: Additional kwargs forwarded to StackingOptimization.

    Returns:
        A fitted-ready ensemble estimator.

    Raises:
        ConfigurationError: If estimators is omitted.
    """
    if estimators is None:
        raise ConfigurationError(
            "estimators kwarg is required (non-serialisable, not on Config)"
        )
    if config is None:
        config = StackingConfig()

    if final_estimator is None:
        if prior_estimator is None and config.prior_config is not None:
            prior_estimator = build_prior(config.prior_config)
        final_estimator = build_mean_risk(
            config.final_default,
            prior_estimator=prior_estimator,
        )

    return StackingOptimization(
        estimators=estimators,
        final_estimator=final_estimator,
        quantile=config.quantile,
        **kwargs,
    )


__all__ = ["StackingConfig", "build_stacking"]
