"""Factory functions for building synthetic data and vine copula estimators."""

from __future__ import annotations

import logging
from typing import Any

from skfolio.distribution import (
    ClaytonCopula,
    DependenceMethod,
    Gaussian,
    GaussianCopula,
    GumbelCopula,
    IndependentCopula,
    JoeCopula,
    JohnsonSU,
    NormalInverseGaussian,
    SelectionCriterion,
    StudentT,
    StudentTCopula,
    VineCopula,
)
from skfolio.distribution._base import BaseDistribution
from skfolio.prior import SyntheticData

from optimizer.exceptions import ConfigurationError
from optimizer.synthetic._config import (
    DependenceMethodType,
    SelectionCriterionType,
    SyntheticDataConfig,
    VineCopulaConfig,
)

_MARGINAL_MAP: dict[str, type[BaseDistribution]] = {
    "Gaussian": Gaussian,
    "StudentT": StudentT,
    "JohnsonSU": JohnsonSU,
    "NormalInverseGaussian": NormalInverseGaussian,
}

_COPULA_MAP: dict[str, type[BaseDistribution]] = {
    "ClaytonCopula": ClaytonCopula,
    "GaussianCopula": GaussianCopula,
    "GumbelCopula": GumbelCopula,
    "IndependentCopula": IndependentCopula,
    "JoeCopula": JoeCopula,
    "StudentTCopula": StudentTCopula,
}


def _resolve_distributions(
    names: tuple[str, ...] | None,
    mapping: dict[str, type[BaseDistribution]],
    kind: str,
) -> list[BaseDistribution] | None:
    """Resolve a tuple of distribution names to fresh estimator instances."""
    if names is None:
        return None
    instances: list[BaseDistribution] = []
    for name in names:
        if name not in mapping:
            valid = ", ".join(sorted(mapping))
            raise ConfigurationError(
                f"Unknown {kind} candidate {name!r}. Valid: {valid}"
            )
        instances.append(mapping[name]())
    return instances

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mapping dicts
# ---------------------------------------------------------------------------

_DEPENDENCE_MAP: dict[DependenceMethodType, DependenceMethod] = {
    DependenceMethodType.KENDALL_TAU: DependenceMethod.KENDALL_TAU,
    DependenceMethodType.MUTUAL_INFORMATION: DependenceMethod.MUTUAL_INFORMATION,
    DependenceMethodType.WASSERSTEIN_DISTANCE: DependenceMethod.WASSERSTEIN_DISTANCE,
}

_SELECTION_MAP: dict[SelectionCriterionType, SelectionCriterion] = {
    SelectionCriterionType.AIC: SelectionCriterion.AIC,
    SelectionCriterionType.BIC: SelectionCriterion.BIC,
}


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def build_vine_copula(
    config: VineCopulaConfig | None = None,
    **kwargs: Any,
) -> VineCopula:
    """Build a skfolio :class:`VineCopula` from *config*.

    Parameters
    ----------
    config : VineCopulaConfig or None
        Vine copula configuration.  Defaults to
        ``VineCopulaConfig()``.
    **kwargs
        Additional keyword arguments forwarded to the
        :class:`VineCopula` constructor (for non-serialisable
        parameters such as ``marginal_candidates``,
        ``copula_candidates``, ``central_assets``).

    Returns
    -------
    VineCopula
        A fitted-ready skfolio vine copula estimator.
    """
    if config is None:
        config = VineCopulaConfig()

    kwargs.setdefault(
        "marginal_candidates",
        _resolve_distributions(
            config.marginal_candidates, _MARGINAL_MAP, "marginal"
        ),
    )
    kwargs.setdefault(
        "copula_candidates",
        _resolve_distributions(config.copula_candidates, _COPULA_MAP, "copula"),
    )
    kwargs.setdefault(
        "central_assets",
        list(config.central_assets) if config.central_assets is not None else None,
    )

    return VineCopula(
        fit_marginals=config.fit_marginals,
        max_depth=config.max_depth,
        log_transform=config.log_transform,
        dependence_method=_DEPENDENCE_MAP[config.dependence_method],
        selection_criterion=_SELECTION_MAP[config.selection_criterion],
        independence_level=config.independence_level,
        n_jobs=config.n_jobs,
        random_state=config.random_state,
        **kwargs,
    )


def build_synthetic_data(
    config: SyntheticDataConfig | None = None,
    *,
    distribution_estimator: VineCopula | None = None,
    sample_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> SyntheticData:
    """Build a skfolio :class:`SyntheticData` prior from *config*.

    Parameters
    ----------
    config : SyntheticDataConfig or None
        Synthetic data configuration.  Defaults to
        ``SyntheticDataConfig()``.
    distribution_estimator : VineCopula or None
        Pre-built distribution estimator.  When ``None``, one is
        built from ``config.vine_copula_config`` (or skfolio default).
    sample_args : dict or None
        Arguments passed to the distribution's ``sample`` method.
        Use ``{"conditioning": {"AAPL": -0.10}}`` for conditional
        stress testing.
    **kwargs
        Additional keyword arguments forwarded to the
        :class:`SyntheticData` constructor.

    Returns
    -------
    SyntheticData
        A fitted-ready skfolio prior estimator generating synthetic
        return scenarios.
    """
    if config is None:
        config = SyntheticDataConfig()

    if distribution_estimator is None and config.vine_copula_config is not None:
        distribution_estimator = build_vine_copula(config.vine_copula_config)

    return SyntheticData(
        distribution_estimator=distribution_estimator,
        n_samples=config.n_samples,
        sample_args=sample_args,
        **kwargs,
    )
