"""Gaussian HMM regime-probability fitting.

Fits a Gaussian Hidden Markov Model (Baum-Welch EM) on a cross-sectional
return panel and returns **filtered** (causal, forward-pass-only) regime
probabilities suitable for feeding into
:class:`optimizer.optimization._regime_blended_mean_risk._ExternallyControlledRegimeCovariance`
via its ``regime_probabilities`` argument.

Filtered vs. smoothed
----------------------
``hmmlearn``'s ``predict_proba`` returns **smoothed** posteriors
(``P(state_t | all observations)``), which use future data at every
timestep and are unsafe for walk-forward backtesting. This module
instead hand-rolls the log-space forward recursion (no backward pass)
using the fitted model's ``startprob_``, ``transmat_``, and per-frame
emission log-likelihoods, normalising at each step to yield
``P(state_t | observations up to t)`` — the same causal probability an
online/live system would have access to at time *t*. ``hmmlearn`` 0.3.x
does not expose a public/private forward-only method, so this recursion
is implemented directly against its documented public model attributes.
"""

from __future__ import annotations

import logging
from importlib.util import find_spec
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from optimizer.exceptions import ConfigurationError, ConvergenceError
from optimizer.regime._config import HMMFeatureType, HMMRegimeConfig

logger = logging.getLogger(__name__)


def _build_features(returns: pd.DataFrame, config: HMMRegimeConfig) -> pd.DataFrame:
    """Build the cross-sectional feature panel the HMM is fit on."""
    mean_return = returns.mean(axis=1)
    features = {"mean_return": mean_return}

    if config.feature in (HMMFeatureType.RETURN_VOL, HMMFeatureType.RETURN_VOL_SKEW):
        features["realised_vol"] = (
            mean_return.rolling(config.vol_window).std().bfill()
        )

    if config.feature == HMMFeatureType.RETURN_VOL_SKEW:
        features["realised_skew"] = (
            mean_return.rolling(config.skew_window)
            .skew()
            .fillna(0.0)
        )

    return pd.DataFrame(features, index=returns.index).dropna()


def fit_hmm_regime_probabilities(
    returns: pd.DataFrame,
    config: HMMRegimeConfig | None = None,
) -> tuple[pd.DataFrame, Any]:
    """Fit a Gaussian HMM and return filtered (causal) regime probabilities.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset return panel indexed by date, shape ``(T, n_assets)``.
        Linear (simple) returns, same convention as the rest of
        ``optimizer`` (see :func:`skfolio.preprocessing.prices_to_returns`).
    config : HMMRegimeConfig or None
        Fitting configuration. Defaults to :meth:`HMMRegimeConfig.for_two_regime`.

    Returns
    -------
    regime_probabilities : pd.DataFrame
        Filtered regime probabilities indexed by date, shape
        ``(T', n_regimes)``, columns ``"regime_0", "regime_1", ...``.
        Each row sums to 1. ``T' <= T`` because rolling-window features
        drop the initial warm-up rows.
    model : hmmlearn.hmm.GaussianHMM
        The fitted model (exposed for diagnostics / regime labelling;
        not required by the blended-covariance consumer).

    Raises
    ------
    ConfigurationError
        If ``hmmlearn`` is not installed, ``returns`` is empty, or the
        usable observation count is below ``config.min_observations``.
    ConvergenceError
        If Baum-Welch EM does not converge within ``config.n_iter``.
    """
    if find_spec("hmmlearn") is None:
        raise ConfigurationError(
            "hmmlearn is required for fit_hmm_regime_probabilities(); "
            "install it with `pip install hmmlearn`."
        )
    from hmmlearn.hmm import GaussianHMM

    if config is None:
        config = HMMRegimeConfig.for_two_regime()

    if returns.empty:
        raise ConfigurationError(
            "returns must not be empty; provide a non-empty DataFrame."
        )

    features = _build_features(returns, config)
    if len(features) < config.min_observations:
        raise ConfigurationError(
            f"Insufficient observations after feature warm-up: {len(features)} "
            f"(minimum {config.min_observations} required). "
            "Provide a longer return history or reduce vol_window/skew_window."
        )

    X = features.to_numpy(dtype=float)

    model = GaussianHMM(
        n_components=config.n_regimes,
        covariance_type=config.covariance_type.value,
        n_iter=config.n_iter,
        tol=config.tol,
        random_state=config.random_state,
    )
    model.fit(X)

    if not model.monitor_.converged:
        raise ConvergenceError(
            f"GaussianHMM Baum-Welch EM did not converge within "
            f"{config.n_iter} iterations (final change "
            f"{model.monitor_.history[-1] - model.monitor_.history[-2]:.2e} "
            f"vs tol={config.tol:.2e})."
        )

    # Sort hidden states by realised_vol/mean_return mean so "regime_0" is
    # consistently the calmest state across refits (label-switching guard).
    order_key = model.means_[:, min(1, X.shape[1] - 1)]
    order = np.argsort(order_key)

    # Filtered (forward-pass-only) posteriors — causal, no look-ahead.
    # hmmlearn 0.3.x exposes no public/private forward-only method, so the
    # log-space forward recursion is hand-rolled against documented public
    # attributes (startprob_, transmat_) and the per-frame emission
    # log-likelihoods from the fitted model.
    log_frameprob = model._compute_log_likelihood(X)
    log_startprob = np.log(np.clip(model.startprob_, 1e-300, None))
    log_transmat = np.log(np.clip(model.transmat_, 1e-300, None))

    n_samples = log_frameprob.shape[0]
    log_alpha = np.empty((n_samples, config.n_regimes))
    log_alpha[0] = log_startprob + log_frameprob[0]
    for t in range(1, n_samples):
        log_alpha[t] = (
            logsumexp(log_alpha[t - 1][:, None] + log_transmat, axis=0)
            + log_frameprob[t]
        )

    # Normalise each row (filtered posterior at t): softmax over states.
    filtered = np.exp(log_alpha - logsumexp(log_alpha, axis=1, keepdims=True))
    filtered = filtered[:, order]

    columns = [f"regime_{i}" for i in range(config.n_regimes)]
    regime_probabilities = pd.DataFrame(filtered, index=features.index, columns=columns)

    return regime_probabilities, model
