"""Capital asset pricing model: CAPM report, rolling beta and security market line.

Sharpe (1964) and Lintner (1965) price an asset by its exposure to the market:

    E[r_i] - r_f = beta_i (E[r_m] - r_f).

Regressing the asset's excess return on the market's excess return estimates
``beta`` (the slope) and Jensen's (1968) ``alpha`` (the intercept), the return the
asset earned beyond what its beta explains. Every return is per period and the
risk-free rate is the per-period rate of the same periodicity.

Annualised figures scale per-period means by ``periods_per_year`` and per-period
deviations by its square root, which is exact for independent returns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from optimizer.exceptions import DataError
from optimizer.regression._config import (
    CONSTANT_NAME,
    require_periods_per_year,
    require_window,
)
from optimizer.regression._inputs import (
    annualized_mean,
    as_frame,
    as_series,
    require_same_index,
    subtract_risk_free,
)
from optimizer.regression._ols import ols

_MARKET_LABEL = "market"
_BETA_LABEL = "beta"
_MIN_ASSETS_FOR_LINE = 3


@dataclass(frozen=True)
class CapmResult:
    """The CAPM report of one asset against the market.

    Attributes:
        alpha: Jensen's alpha per period: the intercept of the regression of the
            asset's excess return on the market's excess return.
        annualized_alpha: ``alpha`` times the periods per year.
        alpha_t_statistic: t-statistic of ``alpha``.
        alpha_p_value: Two-sided p-value of ``alpha`` under a Student t
            distribution with ``n_observations - 2`` degrees of freedom.
        beta: Slope of the regression: the asset's sensitivity to the market.
        beta_std_error: Standard error of ``beta``.
        beta_t_statistic: t-statistic of ``beta``.
        r_squared: Share of the asset's excess-return variance the market explains.
        treynor_ratio: Annualised mean excess return per unit of ``beta``; infinite
            or NaN when ``beta`` is exactly zero.
        tracking_error: Annualised standard deviation of the asset's return minus
            the market's.
        information_ratio: Annualised mean of the asset's return minus the market's,
            over ``tracking_error``; NaN when the tracking error is zero and the
            active return is too, infinite when only the tracking error is.
        n_observations: Number of periods used.
    """

    alpha: float
    annualized_alpha: float
    alpha_t_statistic: float
    alpha_p_value: float
    beta: float
    beta_std_error: float
    beta_t_statistic: float
    r_squared: float
    treynor_ratio: float
    tracking_error: float
    information_ratio: float
    n_observations: int


@dataclass(frozen=True, eq=False)
class SmlResult:
    """The security market line fitted across a set of assets.

    The comparison operators are left at identity because a field holding a pandas
    object has no unambiguous equality.

    Attributes:
        table: One row per asset with ``beta``, ``annualized_mean_excess_return``,
            ``capm_expected_excess_return`` (``beta`` times ``market_premium``) and
            ``alpha_vs_sml`` (the realised minus the expected excess return, which
            equals the annualised Jensen's alpha).
        slope: Slope of the cross-sectional regression of annualised mean excess
            return on ``beta``; CAPM predicts the market premium.
        intercept: Intercept of that regression; CAPM predicts zero.
        market_premium: Annualised mean excess return of the market.
    """

    table: pd.DataFrame
    slope: float
    intercept: float
    market_premium: float


def _ratio(numerator: float, denominator: float) -> float:
    """Divide with IEEE semantics so a zero denominator gives infinity or NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.float64(numerator) / np.float64(denominator))


def _aligned_pair(
    asset_returns: pd.Series, market_returns: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Validate an asset and the market and return them as float series."""
    asset = as_series(asset_returns, "asset_returns")
    market = as_series(market_returns, "market_returns")
    require_same_index(asset, market, "asset_returns", "market_returns")
    return asset, market


def capm_regression(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    risk_free_rate: float | pd.Series = 0.0,
    periods_per_year: int = 252,
    hac_lags: int | None = None,
) -> CapmResult:
    """Estimate Jensen's alpha and beta of an asset against the market.

    Args:
        asset_returns: Per-period returns of the asset.
        market_returns: Per-period returns of the market on the same index.
        risk_free_rate: Per-period risk-free rate, as a scalar or a series on the
            same index. It is subtracted from both returns before the regression.
        periods_per_year: Periods in a year, used to annualise.
        hac_lags: ``None`` for classical standard errors, or the number of
            Newey-West lags (see ``ols``).

    Returns:
        The alpha, beta, their significance and the annualised risk ratios.

    Raises:
        ConfigurationError: If ``periods_per_year``, ``hac_lags`` or a scalar
            ``risk_free_rate`` is invalid.
        DataError: If an input is not a numeric, finite Series, the indexes differ,
            there are fewer than three observations, or the asset or the market
            is constant.
    """
    require_periods_per_year(periods_per_year)
    asset, market = _aligned_pair(asset_returns, market_returns)
    asset_excess = subtract_risk_free(asset, risk_free_rate, "asset_returns")
    market_excess = subtract_risk_free(market, risk_free_rate, "market_returns")
    fit = ols(asset_excess, market_excess.rename(_MARKET_LABEL), hac_lags=hac_lags)
    beta = float(fit.coefficients.at[_MARKET_LABEL])
    active = asset - market
    tracking_error = float(active.std(ddof=1)) * math.sqrt(periods_per_year)
    alpha = float(fit.coefficients.at[CONSTANT_NAME])
    return CapmResult(
        alpha=alpha,
        annualized_alpha=alpha * periods_per_year,
        alpha_t_statistic=float(fit.t_statistics.at[CONSTANT_NAME]),
        alpha_p_value=float(fit.p_values.at[CONSTANT_NAME]),
        beta=beta,
        beta_std_error=float(fit.std_errors.at[_MARKET_LABEL]),
        beta_t_statistic=float(fit.t_statistics.at[_MARKET_LABEL]),
        r_squared=fit.r_squared,
        treynor_ratio=_ratio(annualized_mean(asset_excess, periods_per_year), beta),
        tracking_error=tracking_error,
        information_ratio=_ratio(
            annualized_mean(active, periods_per_year), tracking_error
        ),
        n_observations=fit.n_observations,
    )


def rolling_beta(
    asset_returns: pd.Series, market_returns: pd.Series, window: int
) -> pd.Series:
    """Return the beta of an asset over a trailing window.

    Each value is the sample covariance of the asset with the market over the last
    ``window`` periods divided by the sample variance of the market over the same
    periods, so it uses no observation after its own date. A constant risk-free
    rate does not change a covariance, so none is taken.

    Args:
        asset_returns: Per-period returns of the asset.
        market_returns: Per-period returns of the market on the same index.
        window: Number of trailing periods per estimate, at least 2.

    Returns:
        A series named ``"beta"`` on the index of the inputs, NaN until ``window``
        observations exist and wherever the market did not move within the window.

    Raises:
        ConfigurationError: If ``window`` is not an integer of at least 2.
        DataError: If an input is not a numeric, finite Series, the indexes differ,
            or ``window`` exceeds the number of observations.
    """
    require_window(window)
    asset, market = _aligned_pair(asset_returns, market_returns)
    if window > len(asset):
        msg = f"window ({window}) exceeds the {len(asset)} observations"
        raise DataError(msg)
    variance = market.rolling(window).var()
    covariance = asset.rolling(window).cov(market)
    return (covariance / variance.where(variance > 0.0)).rename(_BETA_LABEL)


def _betas(asset_excess: pd.DataFrame, market_excess: pd.Series) -> pd.Series:
    """Return the slope of each asset's excess return on the market's."""
    if np.ptp(market_excess) == 0.0:
        msg = "market_returns is constant, so beta is undefined"
        raise DataError(msg)
    centred_market = (market_excess - market_excess.mean()).to_numpy()
    centred_assets = (asset_excess - asset_excess.mean()).to_numpy()
    slopes = centred_assets.T @ centred_market / float(centred_market @ centred_market)
    return pd.Series(slopes, index=asset_excess.columns, name=_BETA_LABEL)


def security_market_line(
    assets: pd.DataFrame,
    market_returns: pd.Series,
    risk_free_rate: float | pd.Series = 0.0,
    periods_per_year: int = 252,
) -> SmlResult:
    """Fit the security market line across a set of assets.

    Each asset's beta is estimated against the market, then the assets' annualised
    mean excess returns are regressed on those betas. Under CAPM the points lie on a
    line through the origin whose slope is the market premium; the distance of an
    asset above the line is its alpha.

    Args:
        assets: One column of per-period returns for each asset, at least three.
        market_returns: Per-period returns of the market on the same index.
        risk_free_rate: Per-period risk-free rate, as a scalar or a series on the
            same index.
        periods_per_year: Periods in a year, used to annualise.

    Returns:
        The per-asset table, the fitted slope and intercept and the market premium.

    Raises:
        ConfigurationError: If ``periods_per_year`` or a scalar ``risk_free_rate``
            is invalid.
        DataError: If an input is not numeric and finite, the indexes differ,
            asset names repeat, fewer than three assets are given, the market is
            constant, or the betas do not differ.
    """
    require_periods_per_year(periods_per_year)
    frame = as_frame(assets, "assets")
    market = as_series(market_returns, "market_returns")
    require_same_index(frame, market, "assets", "market_returns")
    if frame.shape[1] < _MIN_ASSETS_FOR_LINE:
        msg = f"at least three assets are needed to fit the line, got {frame.shape[1]}"
        raise DataError(msg)
    asset_excess = subtract_risk_free(frame, risk_free_rate, "assets")
    market_excess = subtract_risk_free(market, risk_free_rate, "market_returns")
    betas = _betas(asset_excess, market_excess)
    mean_excess = asset_excess.mean() * periods_per_year
    premium = annualized_mean(market_excess, periods_per_year)
    line = ols(mean_excess, betas)
    table = pd.DataFrame(
        {
            _BETA_LABEL: betas,
            "annualized_mean_excess_return": mean_excess,
            "capm_expected_excess_return": betas * premium,
            "alpha_vs_sml": mean_excess - betas * premium,
        }
    )
    return SmlResult(
        table=table,
        slope=float(line.coefficients.at[_BETA_LABEL]),
        intercept=float(line.coefficients.at[CONSTANT_NAME]),
        market_premium=premium,
    )
