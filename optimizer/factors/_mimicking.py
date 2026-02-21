"""Long-short factor-mimicking portfolio construction and quintile analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from optimizer.exceptions import ConfigurationError

_WEIGHTING_MODES = frozenset({"equal", "value"})


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class QuintileSpreadResult:
    """Quintile spread analysis result for a single factor.

    Attributes
    ----------
    quintile_returns : pd.DataFrame
        Dates × Q1..Qn equal-weight portfolio returns per quantile bucket.
        Q1 = bottom (lowest scores), Qn = top (highest scores).
    spread_returns : pd.Series
        Qn − Q1 long-short spread return series indexed by date.
        Equals ``quintile_returns.iloc[:, -1] - quintile_returns.iloc[:, 0]``
        element-wise.
    annualised_mean : float
        ``spread_returns.mean() * 252``.
    t_stat : float
        Two-tailed t-statistic: ``mean / (std / sqrt(T))``.
    sharpe : float
        Annualised Sharpe ratio: ``mean * sqrt(252) / std``.
    """

    quintile_returns: pd.DataFrame
    spread_returns: pd.Series
    annualised_mean: float
    t_stat: float
    sharpe: float


def _long_short_return_at(
    scores_t: pd.Series,
    returns_t: pd.Series,
    k: int,
    weighting: str,
) -> float:
    """Compute a single period's long-short factor-mimicking return.

    Parameters
    ----------
    scores_t : pd.Series
        Cross-sectional factor scores for a single date, indexed by ticker.
    returns_t : pd.Series
        Asset returns for the same date, indexed by ticker.
    k : int
        Number of assets in each leg.
    weighting : str
        ``"equal"`` or ``"value"`` weighting within each leg.

    Returns
    -------
    float
        Long return minus short return, or NaN on insufficient data.
    """
    common = scores_t.dropna().index.intersection(returns_t.dropna().index)
    n = len(common)
    if n < 2 * k:
        return float(np.nan)

    ranked = scores_t.loc[common].sort_values(ascending=False)
    long_idx = ranked.index[:k]
    short_idx = ranked.index[-k:]

    long_rets = returns_t.loc[long_idx]
    short_rets = returns_t.loc[short_idx]

    if weighting == "equal":
        return float(long_rets.mean()) - float(short_rets.mean())

    # Value weighting: proportional to absolute score magnitude
    long_w = scores_t.loc[long_idx].abs()
    short_w = scores_t.loc[short_idx].abs()
    long_w_sum = long_w.sum()
    short_w_sum = short_w.sum()
    long_ret = (
        float((long_rets * long_w).sum() / long_w_sum)
        if long_w_sum > 0
        else float(long_rets.mean())
    )
    short_ret = (
        float((short_rets * short_w).sum() / short_w_sum)
        if short_w_sum > 0
        else float(short_rets.mean())
    )
    return long_ret - short_ret


def _compute_leg_beta(
    leg_returns: pd.Series,
    market_returns: pd.Series,
) -> float:
    """Compute OLS beta of leg returns against market returns.

    Parameters
    ----------
    leg_returns : pd.Series
        Return series for the portfolio leg.
    market_returns : pd.Series
        Market return series.

    Returns
    -------
    float
        OLS beta coefficient.
    """
    common = leg_returns.dropna().index.intersection(market_returns.dropna().index)
    if len(common) < 2:
        return 1.0
    y = leg_returns.loc[common].to_numpy(dtype=np.float64)
    x = market_returns.loc[common].to_numpy(dtype=np.float64)
    cov_xy = float(np.cov(y, x, ddof=1)[0, 1])
    var_x = float(np.var(x, ddof=1))
    if var_x < 1e-20:
        return 1.0
    return cov_xy / var_x


def build_factor_mimicking_portfolios(
    scores: pd.DataFrame,
    returns: pd.DataFrame,
    quantile: float = 0.30,
    weighting: str = "equal",
    beta_neutral: bool = False,
    market_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """Build long-short factor-mimicking portfolio return time series.

    For each date the top *quantile* fraction of assets (by factor score)
    are held long and the bottom *quantile* fraction are held short.  The
    long-short return is the equal- or value-weighted long leg minus the
    corresponding short leg.

    The function handles **one factor at a time**: *scores* is a dates ×
    assets DataFrame encoding cross-sectional scores for a single factor.
    For multiple factors, call once per factor and concatenate the results::

        factor_returns = pd.concat(
            [
                build_factor_mimicking_portfolios(scores_value, returns)
                    .rename(columns={"factor_return": "value"}),
                build_factor_mimicking_portfolios(scores_mom, returns)
                    .rename(columns={"factor_return": "momentum"}),
            ],
            axis=1,
        )

    Parameters
    ----------
    scores : pd.DataFrame
        Dates × assets matrix of cross-sectional factor scores.
        Index = dates; columns = asset tickers.
    returns : pd.DataFrame
        Dates × assets matrix of asset returns, aligned with *scores*
        on the date index.  Columns may be a superset or subset of
        *scores* columns; the intersection is used.
    quantile : float, default 0.30
        Fraction of the asset universe assigned to each leg.  Must be
        in ``(0, 0.5]``.
    weighting : {"equal", "value"}, default "equal"
        Weighting scheme within each leg.
        ``"equal"`` — every asset in the leg receives the same weight.
        ``"value"``  — assets are weighted by the absolute value of
        their factor score.
    beta_neutral : bool, default False
        When ``True``, hedge the long-short portfolio against market
        beta exposure.  The hedge ratio adjusts the short-leg weight
        so that the portfolio beta is approximately zero.
    market_returns : pd.Series or None
        Market return series, required when ``beta_neutral=True``.

    Returns
    -------
    pd.DataFrame
        Dates × 1 DataFrame of long-short portfolio returns.  Column
        name is ``"factor_return"``.  Index is the intersection of
        *scores* and *returns* dates.  Missing periods (fewer than
        ``2 * k`` valid observations) are filled with NaN.

    Raises
    ------
    ValueError
        If *quantile* is outside ``(0, 0.5]`` or *weighting* is unknown.
    """
    if not (0.0 < quantile <= 0.5):
        raise ConfigurationError(f"quantile must be in (0, 0.5], got {quantile}")
    if weighting not in _WEIGHTING_MODES:
        raise ConfigurationError(
            f"weighting must be one of {sorted(_WEIGHTING_MODES)!r}, got {weighting!r}"
        )
    if beta_neutral and market_returns is None:
        raise ConfigurationError(
            "market_returns is required when beta_neutral=True"
        )

    common_dates = scores.index.intersection(returns.index)
    common_assets = scores.columns.intersection(returns.columns)

    long_ret_series: dict[object, float] = {}
    short_ret_series: dict[object, float] = {}
    ls_returns: dict[object, float] = {}
    for date in common_dates:
        scores_t = scores.loc[date, common_assets]
        returns_t = returns.loc[date, common_assets]
        valid_n = int((~scores_t.isna() & ~returns_t.isna()).sum())
        k = max(1, math.ceil(valid_n * quantile))

        if beta_neutral:
            common = scores_t.dropna().index.intersection(returns_t.dropna().index)
            n = len(common)
            if n < 2 * k:
                long_ret_series[date] = float(np.nan)
                short_ret_series[date] = float(np.nan)
                ls_returns[date] = float(np.nan)
                continue
            ranked = scores_t.loc[common].sort_values(ascending=False)
            long_idx = ranked.index[:k]
            short_idx = ranked.index[-k:]
            long_ret_series[date] = float(returns_t.loc[long_idx].mean())
            short_ret_series[date] = float(returns_t.loc[short_idx].mean())
        else:
            ls_returns[date] = _long_short_return_at(
                scores_t, returns_t, k, weighting
            )

    if beta_neutral and market_returns is not None:
        long_s = pd.Series(long_ret_series, dtype=float)
        short_s = pd.Series(short_ret_series, dtype=float)
        beta_long = _compute_leg_beta(long_s, market_returns)
        beta_short = _compute_leg_beta(short_s, market_returns)
        hedge_ratio = beta_long / beta_short if abs(beta_short) > 1e-10 else 1.0
        hedged = long_s - hedge_ratio * short_s
        return pd.DataFrame(
            {"factor_return": hedged},
            index=common_dates,
            dtype=float,
        )

    return pd.DataFrame(
        {"factor_return": ls_returns},
        index=common_dates,
        dtype=float,
    )


def compute_cross_factor_correlation(
    factor_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the Pearson correlation matrix across factor-mimicking portfolios.

    Parameters
    ----------
    factor_returns : pd.DataFrame
        Dates × factors DataFrame of long-short factor returns, as
        returned by ``build_factor_mimicking_portfolios`` (possibly
        concatenated across multiple factors).

    Returns
    -------
    pd.DataFrame
        Factors × factors symmetric correlation matrix.  Diagonal
        entries are exactly 1.0.  Computed on the rows where all
        factors have non-NaN returns (pairwise-complete otherwise).
    """
    return factor_returns.corr(method="pearson")


def compute_quintile_spread(
    scores: pd.DataFrame,
    returns: pd.DataFrame,
    n_quantiles: int = 5,
) -> QuintileSpreadResult:
    """Compute quintile portfolio returns and spread for a single factor.

    At each date assets are ranked by factor score and split into
    *n_quantiles* equal-count buckets (Q1 = lowest scores, Qn = highest).
    Each bucket return is the equal-weight average of its members.  The
    long-short spread is Qn − Q1.

    Ties in scores are broken by rank order (``method="first"``), ensuring
    every bucket is populated at every date.

    Parameters
    ----------
    scores : pd.DataFrame
        Dates × assets matrix of cross-sectional factor scores.
    returns : pd.DataFrame
        Dates × assets matrix of asset returns, aligned with *scores*.
    n_quantiles : int, default 5
        Number of equal-count buckets.  5 = quintiles, 10 = deciles.
        Must be ≥ 2.

    Returns
    -------
    QuintileSpreadResult
        See :class:`QuintileSpreadResult` for field descriptions.

    Raises
    ------
    ValueError
        If *n_quantiles* < 2.
    """
    if n_quantiles < 2:
        raise ConfigurationError(f"n_quantiles must be >= 2, got {n_quantiles}")

    labels = [f"Q{i}" for i in range(1, n_quantiles + 1)]
    common_dates = scores.index.intersection(returns.index)
    common_assets = scores.columns.intersection(returns.columns)

    rows: dict[object, dict[str, float]] = {}
    for date in common_dates:
        scores_t = scores.loc[date, common_assets].dropna()
        returns_t = returns.loc[date, common_assets]
        common = scores_t.index.intersection(returns_t.dropna().index)

        if len(common) < n_quantiles:
            rows[date] = {lbl: float(np.nan) for lbl in labels}
            continue

        scores_c = scores_t.loc[common]
        returns_c = returns_t.loc[common]

        # Rank with ties broken by order → ranks 1..n, all distinct
        ranked = scores_c.rank(method="first")
        # qcut on uniform ranks gives exactly n_quantiles balanced buckets
        q_labels = pd.qcut(
            ranked,
            n_quantiles,
            labels=labels,
        )
        row: dict[str, float] = {}
        for lbl in labels:
            bucket_assets = q_labels[q_labels == lbl].index
            row[lbl] = float(returns_c.loc[bucket_assets].mean())
        rows[date] = row

    quintile_returns = pd.DataFrame(rows, dtype=float).T
    quintile_returns.index = pd.Index(list(rows.keys()))
    # Ensure column order Q1..Qn
    quintile_returns = quintile_returns[labels]

    top_label = labels[-1]
    bot_label = labels[0]
    spread_returns = quintile_returns[top_label] - quintile_returns[bot_label]
    spread_returns.name = f"{top_label}_minus_{bot_label}"

    valid = spread_returns.dropna()
    n = len(valid)
    mean_daily = float(valid.mean()) if n > 0 else float(np.nan)
    std_daily = float(valid.std(ddof=1)) if n > 1 else float(np.nan)

    annualised_mean = mean_daily * 252 if not np.isnan(mean_daily) else float(np.nan)

    if n > 1 and std_daily > 0:
        t_stat = mean_daily / (std_daily / math.sqrt(n))
        sharpe = mean_daily * math.sqrt(252) / std_daily
    else:
        t_stat = float(np.nan)
        sharpe = float(np.nan)

    return QuintileSpreadResult(
        quintile_returns=quintile_returns,
        spread_returns=spread_returns,
        annualised_mean=annualised_mean,
        t_stat=t_stat,
        sharpe=sharpe,
    )
