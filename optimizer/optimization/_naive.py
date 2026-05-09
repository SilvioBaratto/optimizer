"""Naive benchmark optimizer configurations and factories.

EqualWeighted, InverseVolatility, and Random — three lightweight
baselines used as comparison points for convex/hierarchical optimizers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from skfolio.moments import EWCovariance
from skfolio.optimization import EqualWeighted, InverseVolatility, Random
from skfolio.prior import EmpiricalPrior
from skfolio.prior._base import BasePrior

from optimizer.moments._config import MomentEstimationConfig
from optimizer.moments._factory import build_prior

# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EqualWeightedConfig:
    """Immutable configuration for :class:`EqualWeighted`.

    The optimizer has no tunable parameters; this Config exists only
    to keep the build pattern uniform across the optimization module.
    """


@dataclass(frozen=True)
class InverseVolatilityConfig:
    """Immutable configuration for :class:`InverseVolatility`.

    Parameters
    ----------
    prior_config : MomentEstimationConfig or None
        Inner prior configuration. ``None`` defers to skfolio default
        (``EmpiricalPrior`` with ``EmpiricalCovariance``).
    ew_half_life : float or None
        When set, the factory composes
        ``EmpiricalPrior(covariance_estimator=EWCovariance(half_life=...))``
        and passes it as ``prior_estimator`` (non-serialisable).
        Mutually exclusive with ``prior_config`` — populated via
        :meth:`for_ew_covariance`.
    """

    prior_config: MomentEstimationConfig | None = None
    ew_half_life: float | None = None

    @classmethod
    def for_default(cls) -> InverseVolatilityConfig:
        """Default empirical-covariance preset."""
        return cls()

    @classmethod
    def for_ew_covariance(cls, half_life: float = 63) -> InverseVolatilityConfig:
        """Exponentially-weighted covariance preset (half-life in days)."""
        return cls(ew_half_life=half_life)


@dataclass(frozen=True)
class RandomConfig:
    """Immutable configuration for :class:`Random`.

    Note: ``skfolio.optimization.Random`` (0.20.1) draws a single
    portfolio from a Dirichlet distribution and has NO ``n_portfolios``
    or ``random_state`` constructor parameter. The fields below are
    reserved for forward compatibility and currently have no runtime
    effect.

    Parameters
    ----------
    n_portfolios : int
        Reserved sample-count field. Default ``100`` per spec.
    random_state : int or None
        Reserved RNG seed. Default ``None``.
    """

    n_portfolios: int = 100
    random_state: int | None = None

    @classmethod
    def for_n_portfolios(
        cls,
        n_portfolios: int = 100,
        random_state: int | None = None,
    ) -> RandomConfig:
        """Random-portfolio sampling preset."""
        return cls(n_portfolios=n_portfolios, random_state=random_state)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def build_equal_weighted(
    config: EqualWeightedConfig | None = None,
    **kwargs: Any,
) -> EqualWeighted:
    """Build a skfolio :class:`EqualWeighted` from *config*."""
    _ = config
    return EqualWeighted(**kwargs)


def build_inverse_volatility(
    config: InverseVolatilityConfig | None = None,
    *,
    prior_estimator: BasePrior | None = None,
    **kwargs: Any,
) -> InverseVolatility:
    """Build a skfolio :class:`InverseVolatility` from *config*.

    When ``config.ew_half_life`` is set, an
    ``EmpiricalPrior(EWCovariance(half_life=...))`` is composed at
    factory call time (non-serialisable; not stored on the Config).
    Falls back to ``build_prior(config.prior_config)`` otherwise.
    """
    if config is None:
        config = InverseVolatilityConfig()
    if prior_estimator is None:
        prior_estimator = _resolve_inverse_volatility_prior(config)
    return InverseVolatility(prior_estimator=prior_estimator, **kwargs)


def _resolve_inverse_volatility_prior(
    config: InverseVolatilityConfig,
) -> BasePrior | None:
    """Compose the inner prior for :class:`InverseVolatility`."""
    if config.ew_half_life is not None:
        return EmpiricalPrior(
            covariance_estimator=EWCovariance(half_life=config.ew_half_life)
        )
    if config.prior_config is not None:
        return build_prior(config.prior_config)
    return None


def build_random(
    config: RandomConfig | None = None,
    **kwargs: Any,
) -> Random:
    """Build a skfolio :class:`Random` from *config*.

    Note: ``n_portfolios`` and ``random_state`` Config fields are
    reserved — skfolio 0.20.1 does NOT accept these as constructor
    arguments and they are intentionally not forwarded.
    """
    _ = config
    return Random(**kwargs)


__all__ = [
    "EqualWeightedConfig",
    "InverseVolatilityConfig",
    "RandomConfig",
    "build_equal_weighted",
    "build_inverse_volatility",
    "build_random",
]
