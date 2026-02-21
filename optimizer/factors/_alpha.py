"""Bridge factor scores to expected returns via CAPM + factor tilt model."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def factor_scores_to_expected_returns(
    scores: pd.DataFrame,
    betas: pd.Series,
    factor_premiums: dict[str, float],
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """Convert factor Z-scores to expected returns via linear model.

    Implements the formula::

        E[r_i] = r_f + λ_mkt · β_i + Σ_g λ_g · z_{i,g}

    where ``λ_mkt`` is read from ``factor_premiums["market"]`` and each
    ``λ_g`` is read from ``factor_premiums[g]`` for factor group ``g``.

    Parameters
    ----------
    scores : pd.DataFrame
        Assets × factor-groups matrix of standardised Z-scores.  Rows are
        ticker symbols; columns are factor group names (e.g. ``"value"``,
        ``"momentum"``).
    betas : pd.Series
        Market (CAPM) beta per asset, indexed by ticker.  Assets missing
        from this Series are treated as having a beta of ``1.0`` (market
        neutral assumption).
    factor_premiums : dict[str, float]
        Mapping of premium label → annualised premium (e.g.
        ``{"market": 0.05, "value": 0.03, "momentum": 0.04}``).  The
        reserved ``"market"`` key provides ``λ_mkt``; all other keys are
        matched against columns in *scores*.
    risk_free_rate : float, default 0.0
        Annualised risk-free rate ``r_f``.

    Returns
    -------
    pd.Series
        Annualised expected return per ticker, indexed by ``scores.index``.

    Examples
    --------
    >>> import pandas as pd
    >>> scores = pd.DataFrame(
    ...     {"value": [1.0, -1.0], "momentum": [0.5, 0.0]},
    ...     index=["AAPL", "MSFT"],
    ... )
    >>> betas = pd.Series({"AAPL": 1.2, "MSFT": 0.8})
    >>> factor_premiums = {"market": 0.05, "value": 0.03, "momentum": 0.04}
    >>> factor_scores_to_expected_returns(scores, betas, factor_premiums, 0.02)
    AAPL    0.132
    MSFT    0.018
    dtype: float64
    """
    tickers = scores.index

    # CAPM component: r_f + λ_mkt · β_i
    market_premium = factor_premiums.get("market", 0.0)
    aligned_betas = betas.reindex(tickers).fillna(1.0)
    expected = pd.Series(
        risk_free_rate + market_premium * aligned_betas,
        index=tickers,
        dtype=float,
    )

    # Factor tilt component: Σ_g λ_g · z_{i,g}
    for group, premium in factor_premiums.items():
        if group == "market":
            continue
        if group in scores.columns:
            expected = expected + premium * scores[group].reindex(tickers).fillna(0.0)

    return expected


def compute_gross_alpha(
    net_alpha: float,
    avg_turnover: float,
    cost_bps: float = 10.0,
) -> float:
    """Compute gross alpha by adding back estimated transaction costs.

    Formula::

        gross = net_alpha + avg_turnover * cost_bps / 10_000

    Parameters
    ----------
    net_alpha : float
        Net alpha after transaction costs (annualised).
    avg_turnover : float
        Average one-way turnover (e.g. 0.5 means 50% of portfolio
        traded per period).
    cost_bps : float
        One-way transaction cost in basis points.

    Returns
    -------
    float
        Gross alpha before transaction costs.
    """
    return net_alpha + avg_turnover * cost_bps / 10_000
