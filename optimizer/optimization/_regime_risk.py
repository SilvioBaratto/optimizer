"""Markov-driven regime-blended risk measure for portfolio optimization.

Implements the probability-weighted risk objective::

    ρ_t(w) = Σ_s  γ_T(s) · ρ_s(w)

where γ_T(s) = P(z_T = s | r_{1:T}) is the current filtered state probability
from a fitted HMM, and ρ_s is the regime-specific risk measure for state s.

Usage example::

    from optimizer.moments._hmm import fit_hmm, HMMConfig
    from optimizer.optimization._regime_risk import (
        RegimeRiskConfig,
        compute_blended_risk_measure,
        build_regime_blended_optimizer,
    )

    hmm_result = fit_hmm(returns, HMMConfig(n_states=2))

    config = RegimeRiskConfig.for_calm_stress()
    risk = compute_blended_risk_measure(
        returns, weights, hmm_result, config.regime_measures
    )
    optimizer = build_regime_blended_optimizer(config, hmm_result)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd
from skfolio.optimization import MeanRisk

from optimizer.moments._hmm import HMMConfig, HMMResult
from optimizer.optimization._config import RiskMeasureType
from optimizer.optimization._factory import _RISK_MEASURE_MAP

# Minimum per-regime observations before falling back to full-sample risk.
_MIN_REGIME_SAMPLES: int = 5


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeRiskConfig:
    """Configuration for Markov-driven blended risk measure optimisation.

    Attributes
    ----------
    regime_measures : tuple[RiskMeasureType, ...]
        One :class:`RiskMeasureType` per HMM state (index = state label).
        E.g. ``(VARIANCE, CVAR)`` means state-0 uses variance and state-1
        uses CVaR.  Must match ``HMMResult.n_states``.
    hmm_config : HMMConfig
        HMM hyper-parameters used when fitting inside the pipeline.
    cvar_beta : float
        Confidence level for CVaR / ES computation (default 0.95 = 95th
        percentile).
    """

    regime_measures: tuple[RiskMeasureType, ...]
    hmm_config: HMMConfig = field(default_factory=HMMConfig)
    cvar_beta: float = 0.95

    @classmethod
    def for_calm_stress(cls, **kwargs: object) -> RegimeRiskConfig:
        """Two-regime preset: calm → variance, stress → CVaR."""
        return cls(
            regime_measures=(RiskMeasureType.VARIANCE, RiskMeasureType.CVAR),
            hmm_config=HMMConfig(n_states=2),
            **kwargs,  # type: ignore[arg-type]
        )

    @classmethod
    def for_three_regimes(cls, **kwargs: object) -> RegimeRiskConfig:
        """Three-regime preset: calm → variance, normal → MAD, stress → CVaR."""
        return cls(
            regime_measures=(
                RiskMeasureType.VARIANCE,
                RiskMeasureType.MEAN_ABSOLUTE_DEVIATION,
                RiskMeasureType.CVAR,
            ),
            hmm_config=HMMConfig(n_states=3),
            **kwargs,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Per-regime risk computation (pure, numpy-only, no skfolio fitting needed)
# ---------------------------------------------------------------------------


def _compute_regime_risk(
    portfolio_returns: npt.NDArray[np.float64],
    measure: RiskMeasureType,
    cvar_beta: float = 0.95,
) -> float:
    """Compute a scalar risk for *portfolio_returns* using *measure*.

    Supports: VARIANCE, STANDARD_DEVIATION, SEMI_VARIANCE, SEMI_DEVIATION,
    MEAN_ABSOLUTE_DEVIATION, CVAR, WORST_REALIZATION.  All other measures
    fall back to standard deviation.

    Returns 0.0 for an empty array.
    """
    n = len(portfolio_returns)
    if n == 0:
        return 0.0

    r = portfolio_returns
    ddof = 1 if n > 1 else 0

    if measure == RiskMeasureType.VARIANCE:
        return float(np.var(r, ddof=ddof))

    if measure == RiskMeasureType.STANDARD_DEVIATION:
        return float(np.std(r, ddof=ddof))

    if measure == RiskMeasureType.SEMI_VARIANCE:
        neg = r[r < 0.0]
        if len(neg) < 2:
            return 0.0
        return float(np.var(neg, ddof=1))

    if measure == RiskMeasureType.SEMI_DEVIATION:
        neg = r[r < 0.0]
        if len(neg) < 2:
            return 0.0
        return float(np.std(neg, ddof=1))

    if measure == RiskMeasureType.MEAN_ABSOLUTE_DEVIATION:
        return float(np.mean(np.abs(r - np.mean(r))))

    if measure == RiskMeasureType.CVAR:
        sorted_r = np.sort(r)
        cutoff = max(1, int(np.floor((1.0 - cvar_beta) * n)))
        return float(-np.mean(sorted_r[:cutoff]))

    if measure == RiskMeasureType.WORST_REALIZATION:
        return float(-np.min(r))

    # Fallback for unsupported measures: standard deviation
    return float(np.std(r, ddof=ddof))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_blended_risk_measure(
    returns: pd.DataFrame,
    weights: npt.NDArray[np.float64],
    hmm_result: HMMResult,
    regime_measures: Sequence[RiskMeasureType],
    cvar_beta: float = 0.95,
) -> float:
    """Compute ρ_t(w) = Σ_s γ_T(s) · ρ_s(w): probability-weighted blended risk.

    Each regime's risk ρ_s(w) is computed on the historical portfolio returns
    from periods assigned to that regime (hard Viterbi-style assignment via
    argmax of filtered probabilities).  Regimes with fewer than
    ``_MIN_REGIME_SAMPLES`` observations fall back to full-sample risk.

    The blend weights γ_T(s) are the **last row** of
    ``hmm_result.filtered_probs`` — the most-recent filtered state
    probabilities, which sum to 1 by HMM construction.

    Args:
        returns: Dates × assets matrix of linear returns.
        weights: 1-D array of portfolio weights, shape ``(n_assets,)``.
        hmm_result: Fitted HMM with ``filtered_probs`` aligned to *returns*.
        regime_measures: One :class:`RiskMeasureType` per HMM state.
        cvar_beta: CVaR confidence level (used when a regime uses CVaR).

    Returns:
        Scalar non-negative blended risk value.

    Raises:
        ValueError: If ``len(regime_measures) != hmm_result.n_states``.
        ValueError: If *weights* length does not match *returns* columns.
    """
    n_states = hmm_result.filtered_probs.shape[1]
    if len(regime_measures) != n_states:
        raise ValueError(
            f"len(regime_measures) = {len(regime_measures)} must match "
            f"hmm_result n_states = {n_states}."
        )

    n_assets = returns.shape[1]
    if len(weights) != n_assets:
        raise ValueError(
            f"weights length {len(weights)} != returns columns {n_assets}."
        )

    # Current regime probabilities (last timestep of the filtered sequence)
    gamma: npt.NDArray[np.float64] = (
        hmm_result.filtered_probs.iloc[-1].to_numpy(dtype=np.float64)
    )

    # Align on common dates
    common_idx = returns.index.intersection(hmm_result.filtered_probs.index)
    if len(common_idx) == 0:
        return 0.0

    aligned_returns = returns.loc[common_idx].to_numpy(dtype=np.float64)
    aligned_probs = hmm_result.filtered_probs.loc[common_idx].to_numpy(
        dtype=np.float64
    )

    portfolio_returns: npt.NDArray[np.float64] = aligned_returns @ weights
    regime_assignments: npt.NDArray[np.intp] = np.argmax(aligned_probs, axis=1)

    blended = 0.0
    for s in range(n_states):
        if gamma[s] == 0.0:
            continue

        mask = regime_assignments == s
        n_regime = int(mask.sum())

        p_ret = (
            portfolio_returns[mask]
            if n_regime >= _MIN_REGIME_SAMPLES
            else portfolio_returns  # fall back to full sample
        )

        risk_s = _compute_regime_risk(p_ret, regime_measures[s], cvar_beta)
        blended += gamma[s] * risk_s

    return float(blended)


def build_regime_blended_optimizer(
    config: RegimeRiskConfig,
    hmm_result: HMMResult,
    **mean_risk_kwargs: object,
) -> MeanRisk:
    """Return a MeanRisk optimizer configured for the dominant regime.

    Because skfolio's ``MeanRisk`` requires a single convex risk measure,
    this function selects the risk measure of the **dominant regime** —
    the state s* = argmax_s γ_T(s) — and builds a standard
    :class:`skfolio.optimization.MeanRisk` with that measure.

    Args:
        config: Regime risk configuration.
        hmm_result: Fitted HMM providing current regime probabilities.
        **mean_risk_kwargs: Additional keyword arguments forwarded to
            :class:`skfolio.optimization.MeanRisk` (e.g.
            ``min_weights``, ``max_weights``, ``prior_estimator``).

    Returns:
        :class:`skfolio.optimization.MeanRisk` instance ready for
        ``fit(X)`` calls.

    Raises:
        ValueError: If ``len(config.regime_measures) != hmm_result.n_states``.
    """
    n_states = hmm_result.filtered_probs.shape[1]
    if len(config.regime_measures) != n_states:
        raise ValueError(
            f"len(config.regime_measures) = {len(config.regime_measures)} must "
            f"match hmm_result n_states = {n_states}."
        )

    gamma: npt.NDArray[np.float64] = (
        hmm_result.filtered_probs.iloc[-1].to_numpy(dtype=np.float64)
    )
    dominant_state = int(np.argmax(gamma))
    dominant_measure = config.regime_measures[dominant_state]
    skfolio_measure = _RISK_MEASURE_MAP[dominant_measure]

    return MeanRisk(risk_measure=skfolio_measure, **mean_risk_kwargs)  # type: ignore[arg-type]
