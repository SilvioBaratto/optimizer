"""Hidden Markov Model for regime-conditional moment estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from hmmlearn.hmm import GaussianHMM


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
        Smoothed posterior state probabilities ``γ_t(s) = P(z_t=s | r_{1:T})``.
        Index is the DatetimeIndex of the input returns; columns are
        integer state labels.  Rows sum to 1.0.
    log_likelihood : float
        Log-likelihood of the data under the fitted model.
    """

    transition_matrix: npt.NDArray[np.float64]
    regime_means: pd.DataFrame
    regime_covariances: npt.NDArray[np.float64]
    filtered_probs: pd.DataFrame
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
        raise ValueError(
            "fit_hmm requires at least 1 asset column, "
            f"got {returns.shape[1]}"
        )

    clean = returns.dropna()
    min_obs = config.n_states + 1
    if len(clean) < min_obs:
        raise ValueError(
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

    # Smoothed posterior state probabilities: shape (T, n_states)
    log_gamma = model.predict_proba(X)  # hmmlearn returns posteriors directly

    state_labels = list(range(config.n_states))

    filtered_probs = pd.DataFrame(
        log_gamma,
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
        raise RuntimeError(
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
        log_likelihood=float(model.score(X)),
    )


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
    """
    weights: npt.NDArray[np.float64] = (
        result.filtered_probs.iloc[-1].to_numpy(dtype=np.float64)
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
