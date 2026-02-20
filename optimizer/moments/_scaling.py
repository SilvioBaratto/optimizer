"""Log-normal moment scaling for multi-period investment horizons."""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_lognormal_correction(
    mu: pd.Series,
    cov: pd.DataFrame,
    horizon: int,
) -> tuple[pd.Series, pd.DataFrame]:
    """Scale daily log-return moments to a multi-period horizon.

    Applies the log-normal compounding correction for expected returns
    and the delta-method approximation for the covariance matrix.

    Expected return scaling (Jensen's inequality correction):

        E[R_T] = exp(μ·T + ½·diag(Σ)·T) − 1

    Covariance scaling (delta-method approximation):

        Σ_T ≈ Σ · T

    Parameters
    ----------
    mu : pd.Series
        Daily log-return expected values, indexed by asset ticker.
    cov : pd.DataFrame
        Daily log-return covariance matrix.  Must be square and share
        the same index/columns as *mu*.
    horizon : int
        Investment horizon in trading days (e.g. 21 for monthly,
        63 for quarterly, 252 for annual).

    Returns
    -------
    tuple[pd.Series, pd.DataFrame]
        ``(mu_T, cov_T)`` — horizon-scaled expected returns and covariance.

    Raises
    ------
    ValueError
        If *horizon* is not a positive integer, or if *mu* and *cov*
        do not share the same ticker index.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be a positive integer, got {horizon}")

    if list(mu.index) != list(cov.index) or list(mu.index) != list(cov.columns):
        raise ValueError(
            "mu and cov must share the same ticker index; "
            f"mu.index={list(mu.index)}, cov.index={list(cov.index)}"
        )

    sigma2 = np.diag(cov.to_numpy(dtype=np.float64))
    mu_arr = mu.to_numpy(dtype=np.float64)
    exponent = mu_arr * horizon + 0.5 * sigma2 * horizon
    mu_t = pd.Series(np.exp(exponent) - 1.0, index=mu.index)
    cov_t = cov * horizon

    return mu_t, cov_t


def scale_moments_to_horizon(
    mu: pd.Series,
    cov: pd.DataFrame,
    daily_horizon: int,
) -> tuple[pd.Series, pd.DataFrame]:
    """Validate inputs and apply the log-normal moment correction.

    A higher-level wrapper around :func:`apply_lognormal_correction`
    that validates array shapes and non-negativity of the covariance
    diagonal before delegating to the core scaling function.

    Parameters
    ----------
    mu : pd.Series
        Daily log-return expected values, indexed by asset ticker.
    cov : pd.DataFrame
        Daily log-return covariance matrix.
    daily_horizon : int
        Investment horizon in trading days.

    Returns
    -------
    tuple[pd.Series, pd.DataFrame]
        ``(mu_T, cov_T)`` — horizon-scaled expected returns and covariance.

    Raises
    ------
    ValueError
        If inputs are not aligned, the covariance matrix is not square,
        the diagonal contains negative values, or *daily_horizon* < 1.
    """
    if cov.shape[0] != cov.shape[1]:
        raise ValueError(
            f"cov must be a square matrix, got shape {cov.shape}"
        )
    if len(mu) != cov.shape[0]:
        raise ValueError(
            f"mu length ({len(mu)}) must match cov dimension ({cov.shape[0]})"
        )

    diag = np.diag(cov.to_numpy(dtype=np.float64))
    if np.any(diag < 0):
        raise ValueError("cov diagonal contains negative values")

    return apply_lognormal_correction(mu, cov, daily_horizon)
