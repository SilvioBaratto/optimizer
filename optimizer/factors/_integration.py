"""Bridge factor scores to portfolio optimization inputs."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorExposureConstraints:
    """Enforceable linear constraints on portfolio factor exposure.

    Encodes the set of per-factor inequalities::

        lb_g <= sum_i w_i * z_{i,g} <= ub_g

    as a pair of matrices ready to be passed directly to
    :class:`skfolio.optimization.MeanRisk` (or any optimizer that
    accepts ``left_inequality`` / ``right_inequality``).

    Parameters
    ----------
    left_inequality : np.ndarray of shape (2 * n_factors, n_assets)
        Inequality matrix ``A`` in the constraint ``A @ w <= b``.
        Two rows per factor: ``-z`` (lower bound) and ``+z`` (upper bound).
    right_inequality : np.ndarray of shape (2 * n_factors,)
        Bound vector ``b`` in the constraint ``A @ w <= b``.
    factor_names : list[str]
        Names of the constrained factors (in the same order as the row
        pairs in ``left_inequality``).
    lower_bounds : np.ndarray of shape (n_factors,)
        Lower exposure bound per factor.
    upper_bounds : np.ndarray of shape (n_factors,)
        Upper exposure bound per factor.
    """

    left_inequality: np.ndarray
    right_inequality: np.ndarray
    factor_names: list[str]
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray


def build_factor_bl_views(
    factor_scores: pd.DataFrame,
    factor_premia: dict[str, float],
    selected_tickers: pd.Index,
) -> tuple[list[tuple[str, ...]], list[float]]:
    """Generate Black-Litterman views from factor scores.

    Creates relative views: top-scored assets outperform
    bottom-scored by the factor premium.

    Parameters
    ----------
    factor_scores : pd.DataFrame
        Tickers x factors matrix of standardized scores.
    factor_premia : dict[str, float]
        Expected premium per factor.
    selected_tickers : pd.Index
        Tickers in the portfolio.

    Returns
    -------
    tuple[list[tuple[str, ...]], list[float]]
        (views, confidences) for Black-Litterman.
    """
    scores = factor_scores.reindex(selected_tickers)
    views: list[tuple[str, ...]] = []
    confidences: list[float] = []

    for factor_name, premium in factor_premia.items():
        if factor_name not in scores.columns:
            continue

        col = scores[factor_name].dropna()
        if len(col) < 4:
            continue

        # Top quartile vs bottom quartile
        q75 = col.quantile(0.75)
        q25 = col.quantile(0.25)
        top = col[col >= q75].index.tolist()
        bottom = col[col <= q25].index.tolist()

        if top and bottom:
            views.append(tuple(top + bottom))
            confidences.append(abs(premium))

    return views, confidences


def build_factor_exposure_constraints(
    factor_scores: pd.DataFrame,
    bounds: tuple[float, float] | dict[str, tuple[float, float]],
) -> FactorExposureConstraints:
    """Build enforceable linear factor exposure constraints.

    For each factor ``g``, the constraint enforces::

        lb_g <= sum_i w_i * z_{i,g} <= ub_g

    The result is expressed as ``left_inequality @ w <= right_inequality``
    (two rows per factor) and can be passed directly to
    :class:`skfolio.optimization.MeanRisk` via its
    ``left_inequality`` / ``right_inequality`` constructor arguments.

    Parameters
    ----------
    factor_scores : pd.DataFrame
        Tickers x factors matrix of standardised factor scores.
        The tickers must match the assets used in the optimizer ``fit``.
    bounds : tuple[float, float] or dict[str, tuple[float, float]]
        Exposure bounds applied to every factor (uniform) when given as a
        single ``(lower, upper)`` tuple, or per-factor bounds when given as
        a dict mapping factor name → ``(lower, upper)``.

    Returns
    -------
    FactorExposureConstraints
        Dataclass holding ``left_inequality``, ``right_inequality``, and
        metadata.  Pass ``left_inequality`` and ``right_inequality`` as
        keyword arguments to the optimizer.

    Warns
    -----
    UserWarning
        If the equal-weight portfolio exposure lies outside ``[lb, ub]``
        for any factor (i.e. the constraint may be infeasible or very
        tight under a balanced allocation).
    """
    n_assets, n_factors = factor_scores.shape
    factor_names = list(factor_scores.columns)

    # Resolve per-factor bounds
    lower_arr = np.empty(n_factors)
    upper_arr = np.empty(n_factors)
    if isinstance(bounds, dict):
        for k, name in enumerate(factor_names):
            if name not in bounds:
                msg = f"Factor '{name}' has no entry in bounds dict."
                raise KeyError(msg)
            lb, ub = bounds[name]
            lower_arr[k] = lb
            upper_arr[k] = ub
    else:
        lb, ub = bounds
        lower_arr[:] = lb
        upper_arr[:] = ub

    # Build inequality matrices: A @ w <= b
    # lb <= z @ w  =>  -z @ w <= -lb
    # z @ w <= ub
    scores_matrix = factor_scores.to_numpy(dtype=float)  # (n_assets, n_factors)

    # Each factor contributes 2 rows
    A = np.empty((2 * n_factors, n_assets))
    b = np.empty(2 * n_factors)
    for k in range(n_factors):
        z = scores_matrix[:, k]
        A[2 * k] = -z
        b[2 * k] = -lower_arr[k]
        A[2 * k + 1] = z
        b[2 * k + 1] = upper_arr[k]

    # Feasibility warning: check equal-weight exposure
    equal_weight = np.ones(n_assets) / n_assets
    for k, name in enumerate(factor_names):
        z = scores_matrix[:, k]
        ew_exposure = float(np.dot(equal_weight, z))
        if not (lower_arr[k] <= ew_exposure <= upper_arr[k]):
            warnings.warn(
                f"Factor '{name}': equal-weight exposure {ew_exposure:.4f} "
                f"lies outside [{lower_arr[k]:.4f}, {upper_arr[k]:.4f}]. "
                "The constraint may be infeasible.",
                UserWarning,
                stacklevel=2,
            )

    return FactorExposureConstraints(
        left_inequality=A,
        right_inequality=b,
        factor_names=factor_names,
        lower_bounds=lower_arr,
        upper_bounds=upper_arr,
    )


def estimate_factor_premia(
    factor_mimicking_returns: pd.DataFrame,
) -> dict[str, float]:
    """Estimate annualized factor premia from long-short returns.

    Parameters
    ----------
    factor_mimicking_returns : pd.DataFrame
        Dates x factors matrix of factor-mimicking portfolio returns.

    Returns
    -------
    dict[str, float]
        Annualized premium per factor.
    """
    mean_daily = factor_mimicking_returns.mean()
    annualized = mean_daily * 252
    return dict(annualized)
