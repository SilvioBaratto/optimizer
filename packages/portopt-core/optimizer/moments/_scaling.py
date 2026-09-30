"""Log-normal moment scaling for multi-period investment horizons."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from optimizer.exceptions import ConfigurationError, DataError

logger = logging.getLogger(__name__)

_VALID_METHODS = {"exact", "linear"}


def apply_lognormal_correction(
    mu: pd.Series,
    cov: pd.DataFrame,
    horizon: int,
    method: str = "exact",
) -> tuple[pd.Series, pd.DataFrame]:
    """Scale daily log-return moments to a multi-period horizon.

    Expected return (Jensen's inequality correction, same for both methods)::

        E[R_T] = exp(mu*T + 0.5*diag(Sigma)*T) - 1

    Exact covariance (method="exact", full log-normal result)::

        Cov[R_T^i, R_T^j] = exp((mu_i+mu_j)*T + 0.5*(sigma_i^2+sigma_j^2)*T)
                             * (exp(sigma_ij*T) - 1)

    Linear approximation (method="linear", delta-method)::

        Sigma_T ~= Sigma * T

    Args:
        mu: Daily log-return expected values, indexed by asset ticker.
        cov: Daily log-return covariance matrix.  Must be square and share
            the same index/columns as ``mu``.
        horizon: Investment horizon in trading days (e.g. 21 for monthly,
            63 for quarterly, 252 for annual).
        method: Covariance scaling method.  ``"exact"`` applies the full
            log-normal formula; ``"linear"`` uses the simpler ``Sigma * T``
            approximation (retained for backwards compatibility).

    Returns:
        Horizon-scaled expected returns and covariance as ``(mu_T, cov_T)``.

    Raises:
        DataError: If ``horizon`` is not a positive integer or ``mu`` and
            ``cov`` do not share the same ticker index.
        ConfigurationError: If ``method`` is not one of
            ``{"exact", "linear"}``.
    """
    if horizon < 1:
        raise DataError(f"horizon must be a positive integer, got {horizon}")

    if method not in _VALID_METHODS:
        raise ConfigurationError(
            f"method must be one of {sorted(_VALID_METHODS)}, got {method!r}"
        )

    if list(mu.index) != list(cov.index) or list(mu.index) != list(cov.columns):
        raise DataError(
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
        mu_sum = mu_arr[:, None] + mu_arr[None, :]  # (n, n)
        sigma2_sum = sigma2[:, None] + sigma2[None, :]  # (n, n)
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

    A higher-level wrapper around ``apply_lognormal_correction`` that
    validates array shapes and non-negativity of the covariance diagonal
    before delegating to the core scaling function.

    Args:
        mu: Daily log-return expected values, indexed by asset ticker.
        cov: Daily log-return covariance matrix.
        daily_horizon: Investment horizon in trading days.
        method: Covariance scaling method.  See ``apply_lognormal_correction``.

    Returns:
        Horizon-scaled expected returns and covariance as ``(mu_T, cov_T)``.

    Raises:
        DataError: If inputs are not aligned, the covariance matrix is not
            square, the diagonal contains negative values, or
            ``daily_horizon`` < 1.
    """
    if cov.shape[0] != cov.shape[1]:
        raise DataError(f"cov must be a square matrix, got shape {cov.shape}")
    if len(mu) != cov.shape[0]:
        raise DataError(
            f"mu length ({len(mu)}) must match cov dimension ({cov.shape[0]})"
        )

    diag = np.diag(cov.to_numpy(dtype=np.float64))
    if np.any(diag < 0):
        raise DataError("cov diagonal contains negative values")

    return apply_lognormal_correction(mu, cov, daily_horizon, method=method)
