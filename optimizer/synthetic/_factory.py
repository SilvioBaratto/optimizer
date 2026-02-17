"""Factory functions for building synthetic data and vine copula estimators."""

from __future__ import annotations

from typing import Any

from skfolio.distribution import DependenceMethod, SelectionCriterion, VineCopula
from skfolio.prior import SyntheticData

from optimizer.synthetic._config import (
    DependenceMethodType,
    SelectionCriterionType,
    SyntheticDataConfig,
    VineCopulaConfig,
)

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
