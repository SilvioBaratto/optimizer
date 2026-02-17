"""Factory functions for building skfolio view integration priors."""

from __future__ import annotations

from collections.abc import Sequence

from skfolio.prior import (
    BlackLitterman,
    EntropyPooling,
    FactorModel,
    OpinionPooling,
)
from skfolio.prior._base import BasePrior

from optimizer.moments._config import MomentEstimationConfig
from optimizer.moments._factory import build_prior
from optimizer.views._config import (
    BlackLittermanConfig,
    EntropyPoolingConfig,
    OpinionPoolingConfig,
    ViewUncertaintyMethod,
)


def build_black_litterman(config: BlackLittermanConfig) -> BasePrior:
    """Build a skfolio Black-Litterman prior from *config*.

    Parameters
    ----------
    config : BlackLittermanConfig
        Black-Litterman configuration.

    Returns
    -------
    BasePrior
        A fitted-ready :class:`skfolio.prior.BlackLitterman`, optionally
        wrapped in a :class:`skfolio.prior.FactorModel`.
    """
    prior_cfg = (
        config.prior_config
        if config.prior_config is not None
        else MomentEstimationConfig.for_equilibrium_ledoitwolf()
    )
    inner_prior = build_prior(prior_cfg)

    view_confidences = (
        list(config.view_confidences)
        if config.uncertainty_method == ViewUncertaintyMethod.IDZOREK
        and config.view_confidences is not None
        else None
    )

    bl = BlackLitterman(
        views=list(config.views),
        groups=config.groups,
        prior_estimator=inner_prior,
        tau=config.tau,
        view_confidences=view_confidences,
        risk_free_rate=config.risk_free_rate,
    )

    if config.use_factor_model:
        return FactorModel(
            factor_prior_estimator=bl,
            residual_variance=config.residual_variance,
        )

    return bl


def build_entropy_pooling(config: EntropyPoolingConfig) -> EntropyPooling:
    """Build a skfolio Entropy Pooling prior from *config*.

    Parameters
    ----------
    config : EntropyPoolingConfig
        Entropy Pooling configuration.

    Returns
    -------
    EntropyPooling
        A fitted-ready :class:`skfolio.prior.EntropyPooling`.
    """
    inner_prior = build_prior(config.prior_config)

    return EntropyPooling(
        prior_estimator=inner_prior,
        mean_views=list(config.mean_views) if config.mean_views else None,
        variance_views=(
            list(config.variance_views) if config.variance_views else None
        ),
        correlation_views=(
            list(config.correlation_views)
            if config.correlation_views
            else None
        ),
        skew_views=(
            list(config.skew_views) if config.skew_views else None
        ),
        kurtosis_views=(
            list(config.kurtosis_views) if config.kurtosis_views else None
        ),
        cvar_views=(
            list(config.cvar_views) if config.cvar_views else None
        ),
        cvar_beta=config.cvar_beta,
        groups=config.groups,
        solver=config.solver,
        solver_params=config.solver_params,
    )


def build_opinion_pooling(
    estimators: Sequence[tuple[str, BasePrior]],
    config: OpinionPoolingConfig | None = None,
) -> OpinionPooling:
    """Build a skfolio Opinion Pooling prior from *config*.

    Parameters
    ----------
    estimators : list[tuple[str, BasePrior]]
        Named expert prior estimators.  Passed directly because
        estimator objects are not serialisable in a frozen dataclass.
    config : OpinionPoolingConfig or None
        Opinion Pooling configuration.  Defaults to
        ``OpinionPoolingConfig()``.

    Returns
    -------
    OpinionPooling
        A fitted-ready :class:`skfolio.prior.OpinionPooling`.
    """
    if config is None:
        config = OpinionPoolingConfig()

    common_prior = (
        build_prior(config.prior_config)
        if config.prior_config is not None
        else None
    )

    return OpinionPooling(
        estimators=list(estimators),
        opinion_probabilities=(
            list(config.opinion_probabilities)
            if config.opinion_probabilities is not None
            else None
        ),
        prior_estimator=common_prior,
        is_linear_pooling=config.is_linear_pooling,
        divergence_penalty=config.divergence_penalty,
        n_jobs=config.n_jobs,
    )
