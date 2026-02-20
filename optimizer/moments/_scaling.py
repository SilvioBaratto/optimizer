"""Log-normal moment scaling for multi-period investment horizons."""

from __future__ import annotations

import numpy as np
import pandas as pd

_VALID_METHODS = {"exact", "linear"}


def apply_lognormal_correction(
    mu: pd.Series,
    cov: pd.DataFrame,
    horizon: int,
    method: str = "exact",
) -> tuple[pd.Series, pd.DataFrame]:
    """Scale daily log-return moments to a multi-period horizon.

    **Expected return** (Jensen's inequality correction, same for both methods):

    .. math::

        E[R_T] = \\exp(\\mu T + \\tfrac{1}{2}\\,\\mathrm{diag}(\\Sigma) T) - 1

    **Covariance — ``method="exact"``** (exact log-normal result):

    .. math::

        \\mathrm{Cov}[R_T^i, R_T^j]
        = \\exp\\!\\bigl((\\mu_i + \\mu_j)T
          + \\tfrac{1}{2}(\\sigma_i^2 + \\sigma_j^2)T\\bigr)
          \\cdot \\bigl(\\exp(\\sigma_{ij}\\,T) - 1\\bigr)

    **Covariance — ``method="linear"``** (delta-method approximation):

    .. math::

        \\Sigma_T \\approx \\Sigma \\cdot T

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
    method : {"exact", "linear"}, default "exact"
        Covariance scaling method.  ``"exact"`` applies the full
        log-normal formula; ``"linear"`` uses the simpler ``Sigma * T``
        approximation (retained for backwards compatibility).

    Returns
    -------
    tuple[pd.Series, pd.DataFrame]
        ``(mu_T, cov_T)`` — horizon-scaled expected returns and covariance.

    Raises
    ------
    ValueError
        If *horizon* is not a positive integer, if *mu* and *cov* do not
        share the same ticker index, or if *method* is not recognised.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be a positive integer, got {horizon}")

    if method not in _VALID_METHODS:
        raise ValueError(
            f"method must be one of {sorted(_VALID_METHODS)}, got {method!r}"
        )

    if list(mu.index) != list(cov.index) or list(mu.index) != list(cov.columns):
        raise ValueError(
            "mu and cov must share the same ticker index; "
            f"mu.index={list(mu.index)}, cov.index={list(cov.index)}"
        )

    sigma2 = np.diag(cov.to_numpy(dtype=np.float64))
    mu_arr = mu.to_numpy(dtype=np.float64)

    # Expected return is identical for both methods
    exponent = mu_arr * horizon + 0.5 * sigma2 * horizon
    mu_t = pd.Series(np.exp(exponent) - 1.0, index=mu.index)

    if method == "linear":
        cov_t = cov * horizon
    else:
        cov_arr = cov.to_numpy(dtype=np.float64)
        # Vectorised exact formula:
        #   Cov[R_T^i, R_T^j] = exp((mu_i+mu_j)*T + 0.5*(sigma_i^2+sigma_j^2)*T)
        #                        * (exp(sigma_ij * T) - 1)
        mu_sum = mu_arr[:, None] + mu_arr[None, :]        # (n, n)
        sigma2_sum = sigma2[:, None] + sigma2[None, :]    # (n, n)
        scale = np.exp((mu_sum + 0.5 * sigma2_sum) * horizon)
        cov_exact = scale * (np.exp(cov_arr * horizon) - 1.0)
        cov_t = pd.DataFrame(cov_exact, index=cov.index, columns=cov.columns)

    return mu_t, cov_t


def scale_moments_to_horizon(
    mu: pd.Series,
    cov: pd.DataFrame,
    daily_horizon: int,
    method: str = "exact",
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
    method : {"exact", "linear"}, default "exact"
        Covariance scaling method.  See :func:`apply_lognormal_correction`.

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

    return apply_lognormal_correction(mu, cov, daily_horizon, method=method)
