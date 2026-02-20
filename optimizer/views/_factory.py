"""Factory functions for building skfolio view integration priors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from skfolio.prior import (
    BlackLitterman,
    EntropyPooling,
    FactorModel,
    OpinionPooling,
)
from skfolio.prior._base import BasePrior, ReturnDistribution

from optimizer.moments._config import MomentEstimationConfig
from optimizer.moments._factory import build_prior
from optimizer.views._config import (
    BlackLittermanConfig,
    EntropyPoolingConfig,
    OpinionPoolingConfig,
    ViewUncertaintyMethod,
)
from optimizer.views._uncertainty import calibrate_omega_from_track_record


class _EmpiricalOmegaBlackLitterman(BlackLitterman):
    """BlackLitterman variant using a pre-computed empirical omega matrix.

    Extends :class:`skfolio.prior.BlackLitterman` by accepting a
    diagonal omega matrix calibrated from a forecast error track record.
    After the parent ``fit()`` completes (handling view parsing and prior
    estimation), the posterior mean and covariance are recomputed with the
    empirical omega in place of the He-Litterman or Idzorek-derived one.
    """

    def __init__(
        self,
        empirical_omega: npt.NDArray[np.float64],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.empirical_omega = empirical_omega

    def fit(
        self, X: npt.ArrayLike, y: object = None, **fit_params: Any
    ) -> _EmpiricalOmegaBlackLitterman:
        """Fit prior then recompute posterior with empirical omega."""
        # Let the parent handle view parsing, prior fitting, and validation.
        super().fit(X, y, **fit_params)

        prior_mu = self.prior_estimator_.return_distribution_.mu
        prior_cov = self.prior_estimator_.return_distribution_.covariance
        prior_returns = self.prior_estimator_.return_distribution_.returns

        omega: npt.NDArray[np.float64] = np.diag(
            np.diag(self.empirical_omega.astype(np.float64))
        )
        P = self.picking_matrix_
        q = self.views_

        _v = self.tau * prior_cov @ P.T
        _a = P @ _v + omega
        _b = q - P @ prior_mu
        posterior_mu = (
            prior_mu + _v @ np.linalg.solve(_a, _b) + self.risk_free_rate
        )
        posterior_cov = (
            prior_cov
            + self.tau * prior_cov
            - _v @ np.linalg.solve(_a, _v.T)
        )
        self.return_distribution_ = ReturnDistribution(
            mu=posterior_mu,
            covariance=posterior_cov,
            returns=prior_returns,
        )
        return self


def build_black_litterman(
    config: BlackLittermanConfig,
    view_history: pd.DataFrame | None = None,
    return_history: pd.DataFrame | None = None,
    omega: npt.NDArray[np.float64] | None = None,
) -> BasePrior:
    """Build a skfolio Black-Litterman prior from *config*.

    Parameters
    ----------
    config : BlackLittermanConfig
        Black-Litterman configuration.
    view_history : pd.DataFrame or None
        Historical forecasted Q values (dates × views).  Required when
        ``config.uncertainty_method`` is ``EMPIRICAL_TRACK_RECORD`` and
        ``omega`` is not pre-supplied.
    return_history : pd.DataFrame or None
        Realised returns aligned to each view (dates × views).  Required
        together with ``view_history`` for empirical omega calibration.
    omega : ndarray of shape (n_views, n_views) or None
        Pre-computed diagonal omega matrix.  When provided and method is
        ``EMPIRICAL_TRACK_RECORD``, used directly (skipping the history
        computation).

    Returns
    -------
    BasePrior
        A fitted-ready :class:`skfolio.prior.BlackLitterman` (or
        :class:`_EmpiricalOmegaBlackLitterman` for the empirical method),
        optionally wrapped in a :class:`skfolio.prior.FactorModel`.
    """
    prior_cfg = (
        config.prior_config
        if config.prior_config is not None
        else MomentEstimationConfig.for_equilibrium_ledoitwolf()
    )
    inner_prior = build_prior(prior_cfg)

    shared_kwargs: dict[str, Any] = dict(
        views=list(config.views),
        groups=config.groups,
        prior_estimator=inner_prior,
        tau=config.tau,
        risk_free_rate=config.risk_free_rate,
    )

    if config.uncertainty_method == ViewUncertaintyMethod.EMPIRICAL_TRACK_RECORD:
        if omega is None:
            if view_history is None or return_history is None:
                raise ValueError(
                    "EMPIRICAL_TRACK_RECORD requires either a pre-computed "
                    "'omega' array or both 'view_history' and 'return_history'"
                )
            omega = calibrate_omega_from_track_record(view_history, return_history)
        bl: BasePrior = _EmpiricalOmegaBlackLitterman(
            empirical_omega=omega, **shared_kwargs
        )
    elif config.uncertainty_method == ViewUncertaintyMethod.IDZOREK:
        view_confidences = (
            list(config.view_confidences)
            if config.view_confidences is not None
            else None
        )
        bl = BlackLitterman(view_confidences=view_confidences, **shared_kwargs)
    else:
        bl = BlackLitterman(**shared_kwargs)

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
