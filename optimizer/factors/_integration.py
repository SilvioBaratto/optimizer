"""Bridge factor scores to portfolio optimization inputs."""

from __future__ import annotations

import pandas as pd

from optimizer.factors._config import FactorIntegrationConfig


def factor_scores_to_expected_returns(
    factor_scores: pd.Series,
    betas: pd.DataFrame,
    factor_premia: dict[str, float],
    config: FactorIntegrationConfig | None = None,
) -> pd.Series:
    """Map factor scores to expected returns via linear model.

    ``E[r_i] = r_f + sum_k (beta_ik * premium_k * score_ik)``

    Parameters
    ----------
    factor_scores : pd.Series
        Composite or individual factor scores per ticker.
    betas : pd.DataFrame
        Tickers x factors matrix of factor exposures.
    factor_premia : dict[str, float]
        Expected premium per factor.
    config : FactorIntegrationConfig or None
        Integration parameters.

    Returns
    -------
    pd.Series
        Expected return per ticker (annualized).
    """
    if config is None:
        config = FactorIntegrationConfig()

    expected = pd.Series(config.risk_free_rate, index=betas.index)

    for factor_name, premium in factor_premia.items():
        if factor_name in betas.columns:
            expected = expected + betas[factor_name] * premium

    return expected


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
    bounds: tuple[float, float],
) -> list[str]:
    """Build factor exposure constraints for optimization.

    Parameters
    ----------
    factor_scores : pd.DataFrame
        Tickers x factors matrix.
    bounds : tuple[float, float]
        (lower, upper) bounds for portfolio factor exposure.

    Returns
    -------
    list[str]
        Constraint descriptions (for logging/documentation).
    """
    lower, upper = bounds
    constraints = []
    for factor_name in factor_scores.columns:
        constraints.append(
            f"{lower:.2f} <= exposure({factor_name}) <= {upper:.2f}"
        )
    return constraints


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
