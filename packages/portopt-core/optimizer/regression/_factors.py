"""Multifactor and Fama-French-style regressions.

An asset's excess return is regressed on a set of factor returns:

    r_i - r_f = alpha + sum_k beta_k F_k + e.

``alpha`` is the return left unexplained by the factors and ``beta_k`` the loading
on factor ``k``. With the market, size and value factors this is the three-factor
model of Fama and French (1993), "Common Risk Factors in the Returns on Stocks and
Bonds", Journal of Financial Economics 33(1), 3-56; the five-factor extension adds
profitability and investment (Fama and French, 2015) and momentum comes from
Carhart (1997).

The factor returns are used as given. The Fama-French factors are long-short
spreads, or the market return already net of the risk-free rate, so no risk-free
rate is subtracted from them; only the asset's return is put in excess form. The
module reads no data: the caller supplies the factor series.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from optimizer.exceptions import ConfigurationError
from optimizer.regression._config import (
    CONSTANT_NAME,
    FamaFrenchFactor,
    require_periods_per_year,
)
from optimizer.regression._inputs import (
    as_frame,
    as_series,
    require_same_index,
    subtract_risk_free,
)
from optimizer.regression._ols import ols


@dataclass(frozen=True, eq=False)
class FactorRegressionResult:
    """Outcome of a regression of an asset's excess return on factor returns.

    The comparison operators are left at identity because a field holding a pandas
    object has no unambiguous equality.

    Attributes:
        alpha: Intercept per period.
        annualized_alpha: ``alpha`` times the periods per year.
        alpha_t_statistic: t-statistic of ``alpha``.
        loadings: Slope on each factor, labelled by factor name.
        loading_t_statistics: t-statistic of each loading, labelled by factor name.
        r_squared: Share of the excess-return variance the factors explain.
        adjusted_r_squared: ``r_squared`` corrected for the number of factors.
        residual_volatility: Annualised standard error of the regression, the risk
            the factors leave unexplained, using the residual degrees of freedom.
    """

    alpha: float
    annualized_alpha: float
    alpha_t_statistic: float
    loadings: pd.Series
    loading_t_statistics: pd.Series
    r_squared: float
    adjusted_r_squared: float
    residual_volatility: float


def multifactor_regression(
    asset_returns: pd.Series,
    factor_returns: pd.DataFrame,
    risk_free_rate: float | pd.Series = 0.0,
    hac_lags: int | None = None,
    periods_per_year: int = 252,
) -> FactorRegressionResult:
    """Regress an asset's excess return on a set of factor returns.

    Args:
        asset_returns: Per-period returns of the asset.
        factor_returns: One column of per-period returns for each factor, on the
            same index. They are used as given, without subtracting the risk-free
            rate: pass a market factor that is already in excess form.
        risk_free_rate: Per-period risk-free rate, as a scalar or a series on the
            same index, subtracted from the asset's return only.
        hac_lags: ``None`` for classical standard errors, or the number of
            Newey-West lags (see ``ols``).
        periods_per_year: Periods in a year, used to annualise.

    Returns:
        The alpha, the loadings with their t-statistics and the fit statistics.

    Raises:
        ConfigurationError: If ``periods_per_year``, ``hac_lags`` or a scalar
            ``risk_free_rate`` is invalid.
        DataError: If an input is not numeric and finite, the indexes differ,
            factor names repeat or collide with ``"const"``, the factors are
            collinear, or there are not more observations than factors plus one.
    """
    require_periods_per_year(periods_per_year)
    asset = as_series(asset_returns, "asset_returns")
    factors = as_frame(factor_returns, "factor_returns")
    require_same_index(asset, factors, "asset_returns", "factor_returns")
    excess = subtract_risk_free(asset, risk_free_rate, "asset_returns")
    fit = ols(excess, factors, hac_lags=hac_lags)
    alpha = float(fit.coefficients.at[CONSTANT_NAME])
    squared_error = float((fit.residuals**2).sum()) / fit.degrees_of_freedom
    return FactorRegressionResult(
        alpha=alpha,
        annualized_alpha=alpha * periods_per_year,
        alpha_t_statistic=float(fit.t_statistics.at[CONSTANT_NAME]),
        loadings=fit.coefficients.iloc[1:],
        loading_t_statistics=fit.t_statistics.iloc[1:],
        r_squared=fit.r_squared,
        adjusted_r_squared=fit.adjusted_r_squared,
        residual_volatility=math.sqrt(squared_error * periods_per_year),
    )


def _require_fama_french_names(columns: pd.Index) -> None:
    """Raise unless the columns are standard names that include the market."""
    allowed = [factor.value for factor in FamaFrenchFactor]
    names = list(columns)
    unknown = [name for name in names if name not in allowed]
    if unknown or FamaFrenchFactor.MKT_RF.value not in names:
        msg = (
            f"factor columns must be a subset of {allowed} that includes "
            f"{FamaFrenchFactor.MKT_RF.value!r}, got {names}; a risk-free column "
            "belongs in risk_free_rate, not in the factors"
        )
        raise ConfigurationError(msg)


def fama_french_regression(
    asset_returns: pd.Series,
    factors: pd.DataFrame,
    risk_free_rate: float | pd.Series = 0.0,
    hac_lags: int | None = None,
    periods_per_year: int = 252,
) -> FactorRegressionResult:
    """Regress an asset's excess return on Fama-French factors.

    This is ``multifactor_regression`` with the factor names checked. The factors
    are taken as already in excess form (``Mkt-RF`` is the market return net of the
    risk-free rate; the others are long-short spreads), and only the asset's return
    has ``risk_free_rate`` subtracted.

    Args:
        asset_returns: Per-period returns of the asset.
        factors: One column per factor, named from ``"Mkt-RF"``, ``"SMB"``,
            ``"HML"``, ``"RMW"``, ``"CMA"`` and ``"MOM"``, always including
            ``"Mkt-RF"``, on the same index as the asset.
        risk_free_rate: Per-period risk-free rate, as a scalar or a series on the
            same index.
        hac_lags: ``None`` for classical standard errors, or the number of
            Newey-West lags (see ``ols``).
        periods_per_year: Periods in a year, used to annualise.

    Returns:
        The alpha, the loadings with their t-statistics and the fit statistics.

    Raises:
        ConfigurationError: If the factor names are not a subset of the standard
            six that includes ``"Mkt-RF"``, or another setting is invalid.
        DataError: Under the conditions of ``multifactor_regression``.
    """
    if isinstance(factors, pd.DataFrame):
        _require_fama_french_names(factors.columns)
    return multifactor_regression(
        asset_returns,
        factors,
        risk_free_rate=risk_free_rate,
        hac_lags=hac_lags,
        periods_per_year=periods_per_year,
    )
