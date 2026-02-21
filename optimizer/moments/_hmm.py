"""Hidden Markov Model for regime-conditional moment estimation."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn.utils.validation as skv
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from skfolio.moments.covariance._base import BaseCovariance
from skfolio.moments.expected_returns._base import BaseMu

from optimizer.exceptions import ConvergenceError, DataError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HMMConfig:
    """Configuration for the Gaussian HMM.

    Attributes
    ----------
    n_states : int
        Number of latent regimes (hidden states).
    n_iter : int
        Maximum number of Baum-Welch EM iterations.
    tol : float
        Convergence tolerance on log-likelihood improvement.
    covariance_type : str
        Covariance structure: ``"full"``, ``"diag"``, ``"tied"``, or
        ``"spherical"``.
    random_state : int or None
        Seed for reproducible initialisation.
    """

    n_states: int = 2
    n_iter: int = 100
    tol: float = 1e-4
    covariance_type: str = "full"
    random_state: int | None = None


@dataclass
class HMMResult:
    """Result of fitting a Gaussian HMM to return data.

    Attributes
    ----------
    transition_matrix : ndarray, shape (n_states, n_states)
        Row-stochastic transition probability matrix ``A[i, j] = P(z_t=j | z_{t-1}=i)``.
    regime_means : pd.DataFrame, shape (n_states, n_assets)
        Per-regime mean return vectors.  Index is integer state label
        (0, 1, ..., n_states-1); columns are asset tickers.
    regime_covariances : ndarray, shape (n_states, n_assets, n_assets)
        Per-regime covariance matrices.  Axis-0 indexes the state.
    filtered_probs : pd.DataFrame, shape (n_dates, n_states)
        Forward-only filtered state probabilities
        ``α_t(s) ∝ P(r_t | z_t=s) · Σ_{s'} A[s', s] · α_{t-1}(s')``,
        conditioned only on past and current observations ``r_{1:t}``.
        These are causal (no look-ahead bias) and suitable for online
        blending in backtests.  Rows sum to 1.0.
    smoothed_probs : pd.DataFrame, shape (n_dates, n_states)
        Baum-Welch smoothed posterior probabilities
        ``γ_t(s) = P(z_t=s | r_{1:T})`` conditioned on the **entire**
        sequence.  Use for diagnostics and regime labeling, but NOT for
        causal blending in backtests.
    log_likelihood : float
        Log-likelihood of the data under the fitted model.
    """

    transition_matrix: npt.NDArray[np.float64]
    regime_means: pd.DataFrame
    regime_covariances: npt.NDArray[np.float64]
    filtered_probs: pd.DataFrame
    smoothed_probs: pd.DataFrame
    log_likelihood: float


def fit_hmm(returns: pd.DataFrame, config: HMMConfig | None = None) -> HMMResult:
    """Fit a Gaussian HMM to a panel of asset returns.

    Uses the Baum-Welch (EM) algorithm implemented by ``hmmlearn`` to
    estimate the transition matrix, regime-conditional means and
    covariances, and the smoothed filtered probabilities.

    Parameters
    ----------
    returns : pd.DataFrame
        Dates × assets matrix of linear returns.  Rows with any NaN
        are dropped before fitting.
    config : HMMConfig or None
        Model hyper-parameters.  Defaults to ``HMMConfig()`` (2 states,
        full covariance, 100 EM iterations).

    Returns
    -------
    HMMResult
        Fitted HMM parameters and smoothed state probabilities.

    Raises
    ------
    ValueError
        If fewer than 2 assets or fewer than ``n_states + 1`` observations
        remain after dropping NaN rows.
    """
    if config is None:
        config = HMMConfig()

    if returns.shape[1] < 1:
        raise DataError(
            f"fit_hmm requires at least 1 asset column, got {returns.shape[1]}"
        )

    clean = returns.dropna()
    min_obs = config.n_states + 1
    if len(clean) < min_obs:
        raise DataError(
            f"fit_hmm requires at least {min_obs} observations after dropping "
            f"NaN rows, got {len(clean)}"
        )

    X: npt.NDArray[np.float64] = clean.to_numpy(dtype=np.float64)
    tickers = list(clean.columns)
    dates = clean.index

    model = GaussianHMM(
        n_components=config.n_states,
        covariance_type=config.covariance_type,
        n_iter=config.n_iter,
        tol=config.tol,
        random_state=config.random_state,
    )
    model.fit(X)

    # Smoothed posterior state probabilities (conditioned on entire sequence)
    smoothed = model.predict_proba(X)

    # Forward-only filtered probabilities (causal — no look-ahead bias)
    framelogprob = model._compute_log_likelihood(X)
    log_startprob = np.log(model.startprob_ + 1e-300)
    log_transmat = np.log(model.transmat_ + 1e-300)
    T_len = X.shape[0]
    n_s = config.n_states
    fwdlattice = np.zeros((T_len, n_s), dtype=np.float64)
    fwdlattice[0] = log_startprob + framelogprob[0]
    for t in range(1, T_len):
        for j in range(n_s):
            fwdlattice[t, j] = (
                logsumexp(fwdlattice[t - 1] + log_transmat[:, j]) + framelogprob[t, j]
            )
    # Normalize rows in log space to get proper probabilities
    log_norm = logsumexp(fwdlattice, axis=1, keepdims=True)
    log_filtered = fwdlattice - log_norm
    filtered = np.exp(log_filtered)

    state_labels = list(range(config.n_states))

    filtered_probs = pd.DataFrame(
        filtered,
        index=dates,
        columns=state_labels,
        dtype=float,
    )

    smoothed_probs = pd.DataFrame(
        smoothed,
        index=dates,
        columns=state_labels,
        dtype=float,
    )

    regime_means = pd.DataFrame(
        model.means_,
        index=state_labels,
        columns=tickers,
        dtype=float,
    )

    # covars_ shape depends on covariance_type; "full" → (n_states, n_feat, n_feat)
    if model.covars_ is None:
        raise ConvergenceError(
            "HMM covariance matrix is None after fitting — model did not converge"
        )
    covars: npt.NDArray[np.float64] = _expand_covars(
        model.covars_, config.covariance_type, config.n_states, len(tickers)
    )

    return HMMResult(
        transition_matrix=model.transmat_.astype(np.float64),
        regime_means=regime_means,
        regime_covariances=covars,
        filtered_probs=filtered_probs,
        smoothed_probs=smoothed_probs,
        log_likelihood=float(model.score(X)),
    )


def select_hmm_n_states(
    returns: pd.DataFrame,
    candidate_n_states: Sequence[int] = (2, 3, 4),
    criterion: str = "bic",
    hmm_config: HMMConfig | None = None,
) -> int:
    """Select the optimal number of HMM states via AIC or BIC.

    For each candidate state count, fits an HMM via :func:`fit_hmm` and
    computes the information criterion.  Returns the candidate that
    minimises the chosen criterion.

    Free parameters per state count S with d assets::

        k = S*(S-1) + S*d + S*d*(d+1)//2

    where the three terms correspond to transition matrix rows (each
    row sums to 1, so S-1 free per row), per-regime means, and
    per-regime full covariance (lower triangle).

    Parameters
    ----------
    returns : pd.DataFrame
        Dates x assets matrix of linear returns.
    candidate_n_states : Sequence[int]
        State counts to evaluate (default ``(2, 3, 4)``).
    criterion : str
        ``"aic"`` or ``"bic"`` (default ``"bic"``).
    hmm_config : HMMConfig or None
        Base HMM config.  ``n_states`` is overridden per candidate.

    Returns
    -------
    int
        The state count that minimises the chosen criterion.

    Raises
    ------
    ValueError
        If ``criterion`` is not ``"aic"`` or ``"bic"``.
    DataError
        If no candidate succeeds.
    """
    if criterion not in ("aic", "bic"):
        raise ValueError(f"criterion must be 'aic' or 'bic', got {criterion!r}")

    base_cfg = hmm_config if hmm_config is not None else HMMConfig()
    clean = returns.dropna()
    T = len(clean)
    d = clean.shape[1]

    best_score = np.inf
    best_n: int | None = None

    for n_s in candidate_n_states:
        cfg = HMMConfig(
            n_states=n_s,
            n_iter=base_cfg.n_iter,
            tol=base_cfg.tol,
            covariance_type=base_cfg.covariance_type,
            random_state=base_cfg.random_state,
        )
        try:
            result = fit_hmm(returns, cfg)
        except (DataError, ConvergenceError, ValueError):
            continue

        # Free parameters: transition + means + covariances
        k = n_s * (n_s - 1) + n_s * d + n_s * d * (d + 1) // 2
        ll = result.log_likelihood

        score = -2.0 * ll + 2.0 * k if criterion == "aic" else -2.0 * ll + k * np.log(T)

        if score < best_score:
            best_score = score
            best_n = n_s

    if best_n is None:
        raise DataError("No candidate n_states succeeded during HMM fitting")

    return best_n


def blend_moments_by_regime(
    result: HMMResult,
) -> tuple[pd.Series, pd.DataFrame]:
    """Compute probability-weighted moments from the last filtered time step.

    Uses the smoothed posterior at the final observation ``γ_T(s)`` to
    produce a single blended expected-return vector and covariance matrix:

        μ = Σ_s γ_T(s) · μ_s
        Σ = Σ_s γ_T(s) · Σ_s

    Parameters
    ----------
    result : HMMResult
        Output of :func:`fit_hmm`.

    Returns
    -------
    tuple[pd.Series, pd.DataFrame]
        ``(mu, cov)`` — blended expected returns (indexed by ticker) and
        blended covariance matrix (tickers × tickers).

    Notes
    -----
    This function computes only the within-regime weighted covariance
    ``Σ = Σ_s γ(s) · Σ_s`` and **omits** the cross-state mean-dispersion
    term ``Σ_s γ(s) · (μ_s − μ)(μ_s − μ)ᵀ`` from the full law of total
    variance.  As a result, the blended covariance will under-estimate
    total uncertainty when regime means differ materially.

    For optimizer inputs that require the full variance decomposition, use
    :class:`HMMBlendedCovariance` instead, which includes both the
    within-regime and between-regime components.
    """
    weights: npt.NDArray[np.float64] = result.filtered_probs.iloc[-1].to_numpy(
        dtype=np.float64
    )
    tickers = list(result.regime_means.columns)
    n_assets = len(tickers)
    n_states = len(weights)

    mu_arr = np.zeros(n_assets, dtype=np.float64)
    cov_arr = np.zeros((n_assets, n_assets), dtype=np.float64)

    for s in range(n_states):
        mu_arr += weights[s] * result.regime_means.iloc[s].to_numpy(dtype=np.float64)
        cov_arr += weights[s] * result.regime_covariances[s]

    mu = pd.Series(mu_arr, index=tickers)
    cov = pd.DataFrame(cov_arr, index=tickers, columns=tickers)
    return mu, cov


# ---------------------------------------------------------------------------
# skfolio-compatible estimator classes
# ---------------------------------------------------------------------------


class HMMBlendedMu(BaseMu):
    """Expected-return estimator that blends regime-conditional means via HMM.

    Fits a Gaussian HMM with :func:`fit_hmm` and computes the
    probability-weighted blended expected return vector::

        μ = Σ_s p(z_T=s | r_{1:T}) · μ_s

    where the weights are the smoothed posterior at the final observation.

    Conforms to the skfolio ``BaseMu`` API: exposes ``mu_`` (``ndarray``
    of shape ``(n_assets,)``) after ``fit``.

    Parameters
    ----------
    hmm_config : HMMConfig or None, default=None
        HMM hyper-parameters.  ``None`` falls back to ``HMMConfig()``
        (2 states, full covariance, 100 EM iterations).

    Attributes
    ----------
    mu_ : ndarray of shape (n_assets,)
        Probability-weighted blended expected return vector.
    hmm_result_ : HMMResult
        The fitted HMM result (for inspection or downstream use).
    n_features_in_ : int
        Number of assets seen during ``fit``.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Asset names seen during ``fit`` (only when input is a DataFrame
        with string column names).
    """

    def __init__(self, hmm_config: HMMConfig | None = None) -> None:
        self.hmm_config = hmm_config

    def fit(self, X: npt.ArrayLike, y: object = None) -> HMMBlendedMu:
        """Fit the HMM and compute the blended expected return vector.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Linear returns of the assets.
        y : ignored

        Returns
        -------
        self
        """
        X_arr: npt.NDArray[np.float64] = skv.validate_data(self, X)
        tickers: list[str | int] = (
            list(self.feature_names_in_)
            if hasattr(self, "feature_names_in_")
            else list(range(X_arr.shape[1]))
        )
        returns_df = pd.DataFrame(X_arr, columns=tickers)

        cfg = self.hmm_config if self.hmm_config is not None else HMMConfig()
        self.hmm_result_: HMMResult = fit_hmm(returns_df, cfg)

        mu_series, _ = blend_moments_by_regime(self.hmm_result_)
        self.mu_: npt.NDArray[np.float64] = mu_series.to_numpy(dtype=np.float64)
        return self


class HMMBlendedCovariance(BaseCovariance):
    """Covariance estimator that blends regime covariances via HMM.

    Fits a Gaussian HMM with :func:`fit_hmm` and computes the blended
    covariance matrix using the full law of total variance formula::

        Σ = Σ_s p(z_T=s | r_{1:T}) · [Σ_s + (μ_s - μ)(μ_s - μ)ᵀ]

    The second term ``(μ_s - μ)(μ_s - μ)ᵀ`` is the cross-state mean
    dispersion contribution, which the simpler blend in
    :func:`blend_moments_by_regime` omits.

    Conforms to the skfolio ``BaseCovariance`` API: exposes
    ``covariance_`` (``ndarray`` of shape ``(n_assets, n_assets)``) after
    ``fit``.

    Parameters
    ----------
    hmm_config : HMMConfig or None, default=None
        HMM hyper-parameters.  ``None`` falls back to ``HMMConfig()``.
    nearest : bool, default=True
        Project the blended covariance to the nearest positive-definite
        matrix if it is not already PSD.
    higham : bool, default=False
        Use the Higham (2002) algorithm for the PSD projection instead of
        eigenvalue clipping.
    higham_max_iteration : int, default=100
        Maximum iterations for the Higham algorithm.

    Attributes
    ----------
    covariance_ : ndarray of shape (n_assets, n_assets)
        Blended covariance matrix (with mean-dispersion term).
    hmm_result_ : HMMResult
        The fitted HMM result.
    n_features_in_ : int
        Number of assets seen during ``fit``.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Asset names seen during ``fit`` (only when input is a DataFrame
        with string column names).
    """

    def __init__(
        self,
        hmm_config: HMMConfig | None = None,
        nearest: bool = True,
        higham: bool = False,
        higham_max_iteration: int = 100,
    ) -> None:
        super().__init__(
            nearest=nearest,
            higham=higham,
            higham_max_iteration=higham_max_iteration,
        )
        self.hmm_config = hmm_config

    def fit(self, X: npt.ArrayLike, y: object = None) -> HMMBlendedCovariance:
        """Fit the HMM and compute the blended covariance matrix.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Linear returns of the assets.
        y : ignored

        Returns
        -------
        self
        """
        X_arr: npt.NDArray[np.float64] = skv.validate_data(self, X)
        tickers: list[str | int] = (
            list(self.feature_names_in_)
            if hasattr(self, "feature_names_in_")
            else list(range(X_arr.shape[1]))
        )
        returns_df = pd.DataFrame(X_arr, columns=tickers)

        cfg = self.hmm_config if self.hmm_config is not None else HMMConfig()
        self.hmm_result_: HMMResult = fit_hmm(returns_df, cfg)

        weights: npt.NDArray[np.float64] = self.hmm_result_.filtered_probs.iloc[
            -1
        ].to_numpy(dtype=np.float64)
        n_states = len(weights)
        n_assets = X_arr.shape[1]

        # Blended mu: μ = Σ_s p_s · μ_s
        mu_arr = np.zeros(n_assets, dtype=np.float64)
        for s in range(n_states):
            mu_arr += weights[s] * self.hmm_result_.regime_means.iloc[s].to_numpy(
                dtype=np.float64
            )

        # Full blended covariance: Σ = Σ_s p_s · [Σ_s + (μ_s - μ)(μ_s - μ)ᵀ]
        cov_arr = np.zeros((n_assets, n_assets), dtype=np.float64)
        for s in range(n_states):
            mu_s = self.hmm_result_.regime_means.iloc[s].to_numpy(dtype=np.float64)
            diff = mu_s - mu_arr
            cov_arr += weights[s] * (
                self.hmm_result_.regime_covariances[s] + np.outer(diff, diff)
            )

        self._set_covariance(cov_arr)
        return self


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _expand_covars(
    covars: npt.NDArray[np.float64],
    covariance_type: str,
    n_states: int,
    n_features: int,
) -> npt.NDArray[np.float64]:
    """Expand hmmlearn covariance array to shape (n_states, n_features, n_features)."""
    full: npt.NDArray[np.float64] = np.zeros(
        (n_states, n_features, n_features), dtype=np.float64
    )
    if covariance_type == "full":
        full[:] = covars
    elif covariance_type == "diag":
        for s in range(n_states):
            full[s] = np.diag(covars[s])
    elif covariance_type == "tied":
        for s in range(n_states):
            full[s] = covars
    elif covariance_type == "spherical":
        for s in range(n_states):
            full[s] = np.eye(n_features) * covars[s]
    return full
