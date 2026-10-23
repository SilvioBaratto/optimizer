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

    Args:
        scores: Assets × factor-groups matrix of standardised Z-scores. Rows are
            ticker symbols; columns are factor group names (e.g. ``"value"``,
            ``"momentum"``).
        betas: Market (CAPM) beta per asset, indexed by ticker. Assets missing
            from this Series are treated as having a beta of ``1.0`` (market
            neutral assumption).
        factor_premiums: Mapping of premium label → annualised premium (e.g.
            ``{"market": 0.05, "value": 0.03, "momentum": 0.04}``). The
            reserved ``"market"`` key provides ``λ_mkt``; all other keys are
            matched against columns in *scores*.
        risk_free_rate: Annualised risk-free rate ``r_f``.

    Returns:
        Annualised expected return per ticker, indexed by ``scores.index``.

    Examples:
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
    annualisation: int = 1,
) -> float:
    """Compute gross alpha by adding back estimated transaction costs.

    Inverts the cost deduction in ``compute_net_alpha`` when called with the
    same ``cost_bps`` and ``annualisation``::

        gross = net_alpha + avg_turnover * cost_bps / 10_000 * annualisation

    The defaults do not pair up: ``compute_net_alpha`` defaults to 252
    rebalances per year while this function defaults to 1 (the add-back of an
    ``avg_turnover`` that is already an annual figure).  Passing the result of
    ``compute_net_alpha`` here without an explicit ``annualisation`` therefore
    under-states the add-back by that factor.

    Args:
        net_alpha: Net alpha after transaction costs (annualised).
        avg_turnover: Average one-way turnover per rebalancing period (e.g. 0.5
            means half of the portfolio is replaced each period).
        cost_bps: Round-trip transaction cost in basis points, i.e. the cost of
            buying and then selling one unit.  Applied to one-way turnover this
            equals the one-way cost applied to every traded unit, buys and
            sells alike.
        annualisation: Number of rebalancing periods per year, used to scale the
            per-period cost to the annual footing of ``net_alpha``.  Always pass
            the value given to ``compute_net_alpha`` to invert it; the default
            of 1 is for an ``avg_turnover`` that is already an annual figure.

    Returns:
        Gross alpha before transaction costs.
    """
    return net_alpha + avg_turnover * cost_bps / 10_000 * annualisation
