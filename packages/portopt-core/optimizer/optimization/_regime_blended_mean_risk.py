"""Regime-blended MeanRisk configuration and factory.

Composes `_ExternallyControlledRegimeCovariance` →
`EmpiricalPrior` →
`TimeSeriesFactorModel` →
`build_mean_risk` → `build_portfolio_pipeline`.

Factor returns are passed via the optimizer step's ``factors`` fit param —
``pipeline.fit(X, optimizer__factors=factor_returns)``, or through
`run_cross_val` with
``params={"factors": factor_returns}`` (skfolio 1.0 replaced the former
positional ``y`` factor argument).
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
from optimizer.pre_selection._config import PreSelectionConfig
from optimizer.pre_selection._pipeline import build_portfolio_pipeline
from optimizer.validation._config import WalkForwardConfig
from optimizer.validation._factory import build_walk_forward

logger = logging.getLogger(__name__)

# Caps how many consecutive training dates may reuse the last known regime
# probabilities. Bridges a one-day HMM lag, holiday-calendar mismatches and
# weekly probability grids while still failing loudly on a truncated series
# instead of silently blending on a stale state.
_MAX_PROBABILITY_STALENESS = 5


@dataclass(frozen=True)
class RegimeBlendedMeanRiskConfig:
    """Immutable configuration for `build_regime_blended_mean_risk`.

    Attributes:
        mean_risk_config: Configuration forwarded to `build_mean_risk`.
            Defaults to max-Sharpe.
        walk_forward_config: Walk-forward cross-validator configuration.
            Defaults to quarterly rolling with a one-year training window.
        half_life: Exponential-weighting half-life in trading days used when
            computing per-regime covariance matrices. Defaults to 40.
    """

    mean_risk_config: MeanRiskConfig = field(
        default_factory=MeanRiskConfig.for_max_sharpe
    )
    walk_forward_config: WalkForwardConfig = field(
        default_factory=WalkForwardConfig.for_quarterly_rolling
    )
    half_life: int = 40


class _ExternallyControlledRegimeCovariance(BaseCovariance):
    """Regime-probability-blended EW covariance driven by external HMM output.

    For each regime `k` computes an exponentially-weighted covariance
    where observation weights equal ``ew(t) × p_k(t)``.  The per-regime
    matrices are then blended using the terminal (last-period) regime
    probabilities.

    Args:
        half_life: EW decay half-life in trading days.
        regime_probabilities: Pre-computed HMM regime probabilities indexed by
            date, shape ``(T, n_regimes)``. Each training date uses the most
            recent probabilities at or before it, carried forward for at most
            ``_MAX_PROBABILITY_STALENESS`` consecutive dates; the last
            training date must be covered or `fit` raises.
        nearest: Passed to `BaseCovariance`.
        higham: Passed to `BaseCovariance`.
        higham_max_iteration: Passed to `BaseCovariance`.
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
        """Fit the regime-blended covariance on the training window `X`.

        Args:
            X: Asset return slice for the training fold, shape
                ``(n_observations, n_assets)``. Must be a
                `DataFrame` with a `DatetimeIndex`
                so that ``regime_probabilities`` can be aligned by date.
            y: Ignored. Present for API consistency.

        Returns:
            The fitted estimator instance.
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

        aligned = (
            self.regime_probabilities.sort_index()
            .reindex(X.index, method="ffill", limit=_MAX_PROBABILITY_STALENESS)
            .fillna(0.0)
        )
        row_sums = aligned.sum(axis=1)
        has_mass = row_sums > 1e-10
        n_covered = int(has_mass.sum())
        if n_covered < 30:
            raise ConfigurationError(
                f"Insufficient overlap: {n_covered} observations "
                "(minimum 30 required). Ensure regime_probabilities "
                "covers the training window dates."
            )
        if not has_mass.iloc[-1]:
            raise ConfigurationError(
                f"regime_probabilities has no row within "
                f"{_MAX_PROBABILITY_STALENESS} observations of the last "
                f"training date ({X.index[-1]}); the terminal regime "
                "probabilities that blend the covariance are undefined."
            )
        if n_covered < len(X):
            logger.warning(
                "regime_probabilities cover %d of %d training dates; the "
                "uncovered dates are excluded from every regime's covariance.",
                n_covered,
                len(X),
            )
        aligned = aligned.div(row_sums.clip(lower=1e-10), axis=0)

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

        self.n_features_in_ = n_assets
        self.feature_names_in_ = np.array(X.columns.tolist())
        self.location_ = blended_loc
        self._set_covariance(blended_cov)
        return self


def build_regime_blended_mean_risk(
    config: RegimeBlendedMeanRiskConfig | None = None,
    *,
    factor_returns: pd.DataFrame,
    regime_probabilities: pd.DataFrame,
    pre_selection_config: PreSelectionConfig | None = None,
    sector_mapping: dict[str, str] | None = None,
    min_sector_weights: dict[str, float] | None = None,
    sector_bands: dict[str, tuple[float, float]] | None = None,
    previous_weights: np.ndarray | None = None,
) -> tuple[Pipeline, WalkForward]:
    """Build a regime-blended MeanRisk pipeline and walk-forward CV splitter.

    Composes a regime-aware prior chain::

        _ExternallyControlledRegimeCovariance  (baked-in HMM probs)
          └─ EmpiricalPrior
               └─ TimeSeriesFactorModel        (factor returns via factors=)
                    └─ MeanRisk
                         └─ Pipeline           (pre-selection → optimizer)

    Pass factor returns (aligned to the asset return index) via the
    ``optimizer`` step's ``factors`` fit param:
    ``pipeline.fit(X, optimizer__factors=factor_returns)``, or through
    `run_cross_val` with
    ``params={"factors": factor_returns}`` (skfolio 1.0).

    Args:
        config: Configuration. Defaults to `RegimeBlendedMeanRiskConfig`
            with max-Sharpe objective and quarterly rolling walk-forward.
        factor_returns: Factor return time series indexed by date, shape
            ``(T, n_factors)``. Pass via the ``optimizer`` step's ``factors``
            fit param: ``pipeline.fit(X, optimizer__factors=...)`` or through
            `run_cross_val` with
            ``params={"factors": factor_returns}``.
        regime_probabilities: Pre-computed HMM regime probabilities indexed by
            date, shape ``(T, n_regimes)``. Baked into the covariance estimator;
            sliced to each CV fold's training window automatically during fit.
        pre_selection_config: Pre-selection pipeline configuration.
        sector_mapping: Ticker → sector mapping. Forwarded both to
            `build_portfolio_pipeline` and to `build_mean_risk`, so
            ``config.mean_risk_config.max_sector_weight`` is enforced on the
            fitted weights.
        min_sector_weights: Per-sector floor weights forwarded to
            `build_mean_risk`. Requires ``sector_mapping``.
        sector_bands: Regime-conditional per-sector ``(floor, cap)`` bands
            forwarded to `build_mean_risk`. Requires ``sector_mapping``.
        previous_weights: Previous portfolio weights for turnover control.
            Forwarded to `build_mean_risk` via ``**kwargs``.

    Returns:
        A 2-tuple of the pre-selection → MeanRisk(TimeSeriesFactorModel)
        pipeline and a temporal walk-forward cross-validator configured from
        ``config.walk_forward_config``.

    Raises:
        ConfigurationError: If ``factor_returns`` or ``regime_probabilities``
            are empty.
        ValueError: If ``min_sector_weights`` or ``sector_bands`` is given
            without ``sector_mapping``, or the floors are infeasible.
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

    regime_cov = _ExternallyControlledRegimeCovariance(
        half_life=config.half_life,
        regime_probabilities=regime_probabilities,
    )

    empirical_prior = EmpiricalPrior(covariance_estimator=regime_cov)
    factor_model = TimeSeriesFactorModel(factor_prior_estimator=empirical_prior)

    extra_kwargs: dict[str, Any] = {}
    if previous_weights is not None:
        extra_kwargs["previous_weights"] = previous_weights

    optimizer = build_mean_risk(
        config.mean_risk_config,
        prior_estimator=factor_model,
        sector_mapping=sector_mapping,
        min_sector_weights=min_sector_weights,
        sector_bands=sector_bands,
        **extra_kwargs,
    )

    pipeline = build_portfolio_pipeline(
        optimizer,
        pre_selection_config=pre_selection_config,
        sector_mapping=sector_mapping,
    )

    cv = build_walk_forward(config.walk_forward_config)

    return pipeline, cv
