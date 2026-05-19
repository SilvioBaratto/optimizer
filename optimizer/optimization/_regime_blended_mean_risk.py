"""Regime-blended MeanRisk configuration and factory.

Composes :class:`_ExternallyControlledRegimeCovariance` →
:class:`~skfolio.prior.EmpiricalPrior` →
:class:`~skfolio.prior.TimeSeriesFactorModel` →
:func:`build_mean_risk` → :func:`~optimizer.pipeline.build_portfolio_pipeline`.

Factor returns are passed as ``y`` when calling ``pipeline.fit(X, y=factor_returns)``
or :func:`~optimizer.validation.run_cross_val`.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from skfolio.model_selection import WalkForward
from skfolio.moments.covariance._base import BaseCovariance
from skfolio.prior import EmpiricalPrior, TimeSeriesFactorModel
from sklearn.pipeline import Pipeline

from optimizer.exceptions import ConfigurationError
from optimizer.optimization._config import MeanRiskConfig
from optimizer.optimization._factory import build_mean_risk
from optimizer.pipeline._builder import build_portfolio_pipeline
from optimizer.pre_selection._config import PreSelectionConfig
from optimizer.validation._config import WalkForwardConfig
from optimizer.validation._factory import build_walk_forward

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeBlendedMeanRiskConfig:
    """Immutable configuration for :func:`build_regime_blended_mean_risk`.

    Parameters
    ----------
    mean_risk_config : MeanRiskConfig
        Configuration forwarded to :func:`build_mean_risk`.
        Defaults to max-Sharpe.
    walk_forward_config : WalkForwardConfig
        Walk-forward cross-validator configuration.
        Defaults to quarterly rolling with a one-year training window.
    half_life : int
        Exponential-weighting half-life in trading days used when
        computing per-regime covariance matrices.  Defaults to 40.
    """

    mean_risk_config: MeanRiskConfig = field(
        default_factory=MeanRiskConfig.for_max_sharpe
    )
    walk_forward_config: WalkForwardConfig = field(
        default_factory=WalkForwardConfig.for_quarterly_rolling
    )
    half_life: int = 40


# ---------------------------------------------------------------------------
# Private covariance estimator
# ---------------------------------------------------------------------------


class _ExternallyControlledRegimeCovariance(BaseCovariance):
    """Regime-probability-blended EW covariance driven by external HMM output.

    For each regime *k* computes an exponentially-weighted covariance
    where observation weights equal ``ew(t) × p_k(t)``.  The per-regime
    matrices are then blended using the terminal (last-period) regime
    probabilities.

    Parameters
    ----------
    half_life : int
        EW decay half-life in trading days.
    regime_probabilities : pd.DataFrame or None
        Pre-computed HMM regime probabilities indexed by date,
        shape ``(T, n_regimes)``.  Must cover every date in the
        training window passed to :meth:`fit`.
    nearest : bool
        Passed to :class:`~skfolio.moments.covariance._base.BaseCovariance`.
    higham : bool
        Passed to :class:`~skfolio.moments.covariance._base.BaseCovariance`.
    higham_max_iteration : int
        Passed to :class:`~skfolio.moments.covariance._base.BaseCovariance`.
    """

    def __init__(
        self,
        half_life: int = 40,
        regime_probabilities: pd.DataFrame | None = None,
        *,
        nearest: bool = True,
        higham: bool = False,
        higham_max_iteration: int = 100,
    ) -> None:
        super().__init__(
            nearest=nearest,
            higham=higham,
            higham_max_iteration=higham_max_iteration,
        )
        self.half_life = half_life
        self.regime_probabilities = regime_probabilities

    def fit(
        self, X: npt.ArrayLike, y: Any = None
    ) -> _ExternallyControlledRegimeCovariance:
        """Fit the regime-blended covariance on the training window *X*.

        Parameters
        ----------
        X : DataFrame of shape (n_observations, n_assets)
            Asset return slice for the training fold.  Must be a
            :class:`pandas.DataFrame` with a :class:`~pandas.DatetimeIndex`
            so that ``regime_probabilities`` can be aligned by date.
        y : Ignored
            Present for API consistency.

        Returns
        -------
        self : _ExternallyControlledRegimeCovariance
        """
        if self.regime_probabilities is None:
            raise ConfigurationError(
                "regime_probabilities must be supplied; "
                "use build_regime_blended_mean_risk() to construct this estimator."
            )
        if not isinstance(X, pd.DataFrame):
            raise ConfigurationError(
                "_ExternallyControlledRegimeCovariance requires a pandas DataFrame "
                "with DatetimeIndex to align regime probabilities. "
                "Ensure the pipeline uses set_output(transform='pandas')."
            )

        # Align and row-normalise regime probabilities to the training window.
        aligned = self.regime_probabilities.reindex(X.index).fillna(0.0)
        overlap = self.regime_probabilities.index.intersection(X.index)
        if len(overlap) < 30:
            raise ConfigurationError(
                f"Insufficient overlap: {len(overlap)} observations "
                "(minimum 30 required). Ensure regime_probabilities "
                "covers the training window dates."
            )
        row_sums = aligned.sum(axis=1).clip(lower=1e-10)
        aligned = aligned.div(row_sums, axis=0)

        X_np = X.to_numpy(dtype=float)
        n_obs, n_assets = X_np.shape
        n_regimes = aligned.shape[1]

        # EW vector: most-recent observation gets weight 1, older get decay^k.
        decay = math.exp(-math.log(2.0) / self.half_life)
        ew = decay ** np.arange(n_obs - 1, -1, -1)  # shape (n_obs,)

        # Terminal (last-period) probabilities used to blend per-regime matrices.
        terminal_probs = aligned.iloc[-1].to_numpy()

        blended_cov = np.zeros((n_assets, n_assets))
        blended_loc = np.zeros(n_assets)

        for k in range(n_regimes):
            combined = ew * aligned.iloc[:, k].to_numpy()
            w_sum = combined.sum()
            if w_sum < 1e-10:
                continue
            w_norm = combined / w_sum

            mu_k = w_norm @ X_np  # (n_assets,)
            dev_k = X_np - mu_k  # (n_obs, n_assets)
            cov_k = (w_norm[:, None] * dev_k).T @ dev_k  # (n_assets, n_assets)

            blended_cov += terminal_probs[k] * cov_k
            blended_loc += terminal_probs[k] * mu_k

        # Set sklearn bookkeeping attributes.
        self.n_features_in_ = n_assets
        self.feature_names_in_ = np.array(X.columns.tolist())
        self.location_ = blended_loc
        self._set_covariance(blended_cov)
        return self


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_regime_blended_mean_risk(
    config: RegimeBlendedMeanRiskConfig | None = None,
    *,
    factor_returns: pd.DataFrame,
    regime_probabilities: pd.DataFrame,
    pre_selection_config: PreSelectionConfig | None = None,
    sector_mapping: dict[str, str] | None = None,
    previous_weights: np.ndarray | None = None,
) -> tuple[Pipeline, WalkForward]:
    """Build a regime-blended MeanRisk pipeline and walk-forward CV splitter.

    Composes a regime-aware prior chain:

    .. code-block:: text

        _ExternallyControlledRegimeCovariance  (baked-in HMM probs)
          └─ EmpiricalPrior
               └─ TimeSeriesFactorModel        (factor returns via y=...)
                    └─ MeanRisk
                         └─ Pipeline           (pre-selection → optimizer)

    The returned pipeline expects ``y=factor_returns`` (aligned to the
    asset return index) when calling ``pipeline.fit(X, y=factor_returns)``
    or :func:`~optimizer.validation.run_cross_val`.

    Parameters
    ----------
    config : RegimeBlendedMeanRiskConfig or None
        Configuration.  Defaults to
        :class:`RegimeBlendedMeanRiskConfig` with max-Sharpe objective and
        quarterly rolling walk-forward.
    factor_returns : pd.DataFrame
        Factor return time series indexed by date, shape
        ``(T, n_factors)``.  Pass this as ``y`` when calling
        ``pipeline.fit(X, y=factor_returns)`` or
        :func:`~optimizer.validation.run_cross_val`.
    regime_probabilities : pd.DataFrame
        Pre-computed HMM regime probabilities indexed by date,
        shape ``(T, n_regimes)``.  Baked into the covariance estimator;
        sliced to each CV fold's training window automatically during fit.
    pre_selection_config : PreSelectionConfig or None
        Pre-selection pipeline configuration.
    sector_mapping : dict[str, str] or None
        Ticker → sector mapping forwarded to
        :func:`~optimizer.pipeline.build_portfolio_pipeline`.
    previous_weights : np.ndarray or None
        Previous portfolio weights for turnover control.  Forwarded to
        :func:`build_mean_risk` via ``**kwargs``.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Pre-selection → MeanRisk(TimeSeriesFactorModel) pipeline.
    cv : skfolio.model_selection.WalkForward
        Temporal cross-validator configured from
        ``config.walk_forward_config``.

    Raises
    ------
    ConfigurationError
        If ``factor_returns`` or ``regime_probabilities`` are empty.
    """
    if config is None:
        config = RegimeBlendedMeanRiskConfig()

    if factor_returns.empty:
        raise ConfigurationError(
            "factor_returns must not be empty; provide a non-empty DataFrame."
        )
    if regime_probabilities.empty:
        raise ConfigurationError(
            "regime_probabilities must not be empty; provide a non-empty DataFrame."
        )

    # 1. Regime-blended covariance with baked-in HMM probabilities.
    regime_cov = _ExternallyControlledRegimeCovariance(
        half_life=config.half_life,
        regime_probabilities=regime_probabilities,
    )

    # 2. Empirical prior wrapping the regime covariance.
    empirical_prior = EmpiricalPrior(covariance_estimator=regime_cov)

    # 3. Factor model: empirical_prior provides regime-aware Σ; factor
    #    returns are supplied as y when pipeline.fit(X, y=factor_returns).
    factor_model = TimeSeriesFactorModel(factor_prior_estimator=empirical_prior)

    # 4. MeanRisk optimizer with optional turnover control.
    extra_kwargs: dict[str, Any] = {}
    if previous_weights is not None:
        extra_kwargs["previous_weights"] = previous_weights

    optimizer = build_mean_risk(
        config.mean_risk_config,
        prior_estimator=factor_model,
        **extra_kwargs,
    )

    # 5. Full pre-selection → optimizer pipeline.
    pipeline = build_portfolio_pipeline(
        optimizer,
        pre_selection_config=pre_selection_config,
        sector_mapping=sector_mapping,
    )

    # 6. Walk-forward cross-validator.
    cv = build_walk_forward(config.walk_forward_config)

    return pipeline, cv
