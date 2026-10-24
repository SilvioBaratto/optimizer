"""Factory functions for building skfolio view integration priors."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from skfolio.prior import (
    BlackLitterman,
    EmpiricalPrior,
    EntropyPooling,
    OpinionPooling,
    TimeSeriesFactorModel,
)
from skfolio.prior._base import BasePrior, ReturnDistribution
from sklearn.base import clone

from optimizer.exceptions import ConfigurationError
from optimizer.moments._config import MomentEstimationConfig
from optimizer.moments._factory import build_prior
from optimizer.views._config import (
    BlackLittermanConfig,
    EntropyPoolingConfig,
    OpinionPoolingConfig,
    ViewUncertaintyMethod,
)
from optimizer.views._uncertainty import calibrate_omega_from_track_record

logger = logging.getLogger(__name__)


class _EmpiricalOmegaBlackLitterman(BlackLitterman):
    """BlackLitterman variant using a pre-computed empirical omega matrix.

    Extends `BlackLitterman` by accepting a
    diagonal omega matrix calibrated from a forecast error track record.
    After the parent ``fit()`` completes (handling view parsing and prior
    estimation), the posterior mean and covariance are recomputed with the
    empirical omega in place of the He-Litterman or Idzorek-derived one.
    """

    # sklearn derives get_params()/clone() from the explicit __init__ signature and
    # skips **kwargs, so every parent parameter must be spelled out here.
    def __init__(
        self,
        views: npt.ArrayLike,
        empirical_omega: npt.NDArray[np.float64],
        groups: dict[str, list[str]] | npt.ArrayLike | None = None,
        prior_estimator: BasePrior | None = None,
        tau: float = 0.05,
        view_confidences: npt.ArrayLike | None = None,
        risk_free_rate: float = 0,
    ) -> None:
        super().__init__(
            views=views,
            groups=groups,
            prior_estimator=prior_estimator,
            tau=tau,
            view_confidences=view_confidences,
            risk_free_rate=risk_free_rate,
        )
        self.empirical_omega = empirical_omega

    def fit(
        self, X: npt.ArrayLike, y: object = None, **fit_params: Any
    ) -> _EmpiricalOmegaBlackLitterman:
        """Fit prior then recompute posterior with empirical omega."""
        # Let the parent handle view parsing, prior fitting, and validation.
        # Parent ignores y but its stub types it None; forwarding a value trips pyright.
        super().fit(X, y, **fit_params)  # pyright: ignore[reportArgumentType]

        prior_mu = self.prior_estimator_.return_distribution_.mu
        prior_cov = self.prior_estimator_.return_distribution_.covariance
        prior_returns = self.prior_estimator_.return_distribution_.returns

        P = self.picking_matrix_
        q = self.views_

        n_views = P.shape[0]
        empirical_omega = np.asarray(self.empirical_omega, dtype=np.float64)
        if empirical_omega.shape != (n_views, n_views):
            raise ValueError(
                "empirical_omega must be square of shape "
                f"(n_views, n_views) = ({n_views}, {n_views}), got "
                f"{empirical_omega.shape}"
            )
        omega: npt.NDArray[np.float64] = np.diag(np.diag(empirical_omega))

        _v = self.tau * prior_cov @ P.T
        _a = P @ _v + omega
        _b = q - P @ prior_mu
        posterior_mu = prior_mu + _v @ np.linalg.solve(_a, _b) + self.risk_free_rate
        posterior_cov = (
            prior_cov + self.tau * prior_cov - _v @ np.linalg.solve(_a, _v.T)
        )
        self.return_distribution_ = ReturnDistribution(
            mu=posterior_mu,
            covariance=posterior_cov,
            returns=prior_returns,
            sample_weight=self.prior_estimator_.return_distribution_.sample_weight,
        )
        return self


class AvailableAssetViewsPrior(BasePrior):
    """Black-Litterman prior that applies only the views on assets it is fitted on.

    The views are built once for a fixed set of assets, but a pipeline's
    pre-selection drops assets at every fit, and skfolio rejects a view naming
    an asset that is absent from ``X``. An asset that is not held cannot carry a
    view, so at fit the views on absent assets are set aside and the rest are
    applied unchanged. With none left the result is the unviewed posterior.

    Args:
        black_litterman: Template prior holding every view.
        view_assets: Asset named by each view of ``black_litterman``, in the
            same order.
    """

    def __init__(
        self, black_litterman: BlackLitterman, view_assets: tuple[str, ...]
    ) -> None:
        self.black_litterman = black_litterman
        self.view_assets = view_assets

    def get_metadata_routing(self) -> Any:
        """Route fit metadata exactly as the wrapped prior would."""
        return self.black_litterman.get_metadata_routing()

    def fit(
        self, X: Any, y: None = None, **fit_params: Any
    ) -> AvailableAssetViewsPrior:
        """Fit the prior on the views whose asset is a column of ``X``.

        Args:
            X: Asset returns with the asset names as columns.
            y: Ignored; present for sklearn API compatibility.
            **fit_params: Forwarded to the underlying estimators.

        Returns:
            The fitted prior.
        """
        template = self.black_litterman
        present = set(getattr(X, "columns", ()))
        keep = [i for i, asset in enumerate(self.view_assets) if asset in present]

        fitted: BasePrior
        if keep:
            params: dict[str, Any] = {"views": [template.views[i] for i in keep]}
            if template.view_confidences is not None:
                params["view_confidences"] = [
                    template.view_confidences[i] for i in keep
                ]
            fitted = clone(template).set_params(**params)
            fitted.fit(X, y, **fit_params)
            self.return_distribution_ = fitted.return_distribution_
        else:
            inner = template.prior_estimator
            fitted = (
                cast("BasePrior", clone(inner))
                if inner is not None
                else EmpiricalPrior()
            )
            fitted.fit(X, y, **fit_params)
            base = fitted.return_distribution_
            # What BlackLitterman computes when no view constrains the posterior.
            self.return_distribution_ = ReturnDistribution(
                mu=base.mu + template.risk_free_rate,
                covariance=(1.0 + template.tau) * base.covariance,
                returns=base.returns,
            )
        self.n_features_in_ = int(np.shape(X)[1])
        if hasattr(fitted, "feature_names_in_"):
            self.feature_names_in_ = fitted.feature_names_in_
        return self


def build_black_litterman(
    config: BlackLittermanConfig,
    view_history: pd.DataFrame | None = None,
    return_history: pd.DataFrame | None = None,
    omega: npt.NDArray[np.float64] | None = None,
    *,
    market_weights: npt.NDArray[np.float64] | None = None,
) -> BasePrior:
    """Build a skfolio Black-Litterman prior from *config*.

    Args:
        config: Black-Litterman configuration.
        view_history: Historical forecasted Q values (dates × views). Required when
            ``config.uncertainty_method`` is ``EMPIRICAL_TRACK_RECORD`` and
            ``omega`` is not pre-supplied.
        return_history: Realised returns aligned to each view (dates × views). Required
            together with ``view_history`` for empirical omega calibration.
        omega: Pre-computed diagonal omega matrix. When provided and method is
            ``EMPIRICAL_TRACK_RECORD``, used directly (skipping the history
            computation).
        market_weights: Market-capitalisation weights, aligned to the columns the
            prior is fitted on, forwarded to the equilibrium prior. Without them
            skfolio falls back to equal weights, which is a degenerate
            (non cap-weighted) equilibrium. Only used when the inner prior's
            ``mu_estimator`` is ``EQUILIBRIUM``.

    Returns:
        A fitted-ready `BlackLitterman` (or `_EmpiricalOmegaBlackLitterman`
        for the empirical method), optionally wrapped in a
        `TimeSeriesFactorModel`.
    """
    prior_cfg = (
        config.prior_config
        if config.prior_config is not None
        else MomentEstimationConfig.for_equilibrium_ledoitwolf()
    )
    inner_prior = build_prior(prior_cfg, market_weights=market_weights)

    shared_kwargs: dict[str, Any] = {
        "views": list(config.views),
        "groups": config.groups,
        "prior_estimator": inner_prior,
        "tau": config.tau,
        "risk_free_rate": config.risk_free_rate,
    }

    if config.uncertainty_method == ViewUncertaintyMethod.EMPIRICAL_TRACK_RECORD:
        if omega is None:
            if view_history is None or return_history is None:
                raise ConfigurationError(
                    "EMPIRICAL_TRACK_RECORD requires either a pre-computed "
                    "'omega' array or both 'view_history' and 'return_history'"
                )
            omega = calibrate_omega_from_track_record(view_history, return_history)
        bl: BasePrior = _EmpiricalOmegaBlackLitterman(
            empirical_omega=omega, **shared_kwargs
        )
    elif config.uncertainty_method == ViewUncertaintyMethod.IDZOREK:
        # skfolio does np.asarray(view_confidences); an all-int input yields an int
        # array, so its 0 -> 1e-16 floor truncates back to 0 and 1/0 poisons mu.
        view_confidences = (
            [float(c) for c in config.view_confidences]
            if config.view_confidences is not None
            else None
        )
        bl = BlackLitterman(view_confidences=view_confidences, **shared_kwargs)
    else:
        bl = BlackLitterman(**shared_kwargs)

    if config.use_factor_model:
        return TimeSeriesFactorModel(
            factor_prior_estimator=bl,
        )

    return bl


def _merge_mean_views(config: EntropyPoolingConfig) -> list[str] | None:
    """Merge equality and inequality mean views into a single list.

    skfolio's EntropyPooling handles ``==``, ``>=``, and ``<=`` operators
    in view strings, so both equality and inequality views can be passed
    together in the ``mean_views`` parameter.
    """
    views = list(config.mean_views or ()) + list(config.mean_inequality_views or ())
    return views if views else None


def build_entropy_pooling(
    config: EntropyPoolingConfig,
    prior_moments: (
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None
    ) = None,
    asset_names: list[str] | None = None,
) -> EntropyPooling:
    """Build a skfolio Entropy Pooling prior from *config*.

    Args:
        config: Entropy Pooling configuration.
        prior_moments: ``(mu, cov)`` arrays from a fitted prior. Required when
            ``config.relative_mean_views`` or ``config.relative_variance_views``
            are set. Each relative view becomes ``asset == multiplier * prior``,
            with the prior mean (or variance) read from these arrays; a zero
            prior mean therefore stays zero for any multiplier.
        asset_names: Asset names corresponding to rows/columns of *prior_moments*.
            Required together with *prior_moments*.

    Returns:
        A fitted-ready `EntropyPooling`.  Its ``prior_estimator`` is ``None``
        unless ``config.prior_config`` is set, so the result can be used as an
        `OpinionPooling` expert (which forbids per-expert priors).
    """
    # OpinionPooling rejects experts that carry their own prior_estimator, so an
    # unset prior_config must stay None and fall back to skfolio's EmpiricalPrior().
    inner_prior = (
        build_prior(config.prior_config) if config.prior_config is not None else None
    )
    merged_mean_views = _merge_mean_views(config)

    variance_views = list(config.variance_views) if config.variance_views else None

    has_relative = (
        config.relative_mean_views is not None
        or config.relative_variance_views is not None
    )
    if has_relative and (prior_moments is None or asset_names is None):
        raise ConfigurationError(
            "prior_moments and asset_names are required when "
            "relative_mean_views or relative_variance_views are set"
        )

    if has_relative and prior_moments is not None and asset_names is not None:
        mu, cov = prior_moments
        name_to_idx = {name: i for i, name in enumerate(asset_names)}

        if config.relative_mean_views is not None:
            if merged_mean_views is None:
                merged_mean_views = []
            for asset, multiplier in config.relative_mean_views:
                idx = name_to_idx[asset]
                abs_val = float(mu[idx]) * multiplier
                merged_mean_views.append(f"{asset} == {abs_val}")

        if config.relative_variance_views is not None:
            if variance_views is None:
                variance_views = []
            for asset, multiplier in config.relative_variance_views:
                idx = name_to_idx[asset]
                abs_val = float(cov[idx, idx]) * multiplier
                variance_views.append(f"{asset} == {abs_val}")

    return EntropyPooling(
        prior_estimator=inner_prior,
        mean_views=merged_mean_views,
        variance_views=variance_views,
        correlation_views=(
            list(config.correlation_views) if config.correlation_views else None
        ),
        skew_views=(list(config.skew_views) if config.skew_views else None),
        kurtosis_views=(list(config.kurtosis_views) if config.kurtosis_views else None),
        value_at_risk_views=(
            list(config.value_at_risk_views) if config.value_at_risk_views else None
        ),
        cvar_views=(list(config.cvar_views) if config.cvar_views else None),
        value_at_risk_beta=config.value_at_risk_beta,
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

    Args:
        estimators: Named expert prior estimators. Passed directly because
            estimator objects are not serialisable in a frozen dataclass.
        config: Opinion Pooling configuration. Defaults to
            ``OpinionPoolingConfig()``.

    Returns:
        A fitted-ready `OpinionPooling`.
    """
    if config is None:
        config = OpinionPoolingConfig()

    common_prior = (
        build_prior(config.prior_config) if config.prior_config is not None else None
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
