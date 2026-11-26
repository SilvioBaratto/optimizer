"""PCA residual statistical arbitrage (Avellaneda and Lee, 2010).

Avellaneda and Lee, "Statistical Arbitrage in the U.S. Equities Market",
Quantitative Finance 10(7), take the market's common movements out of every stock
with principal components and trade what is left:

1. standardise each stock's returns and take the leading eigenvectors of their
   correlation matrix;
2. form *eigenportfolios*, whose weights are an eigenvector divided by each
   stock's volatility, and apply them to the raw returns to get factor returns;
3. regress each stock on the factor returns; the residual is its idiosyncratic
   return;
4. cumulate the residual into a path ``X`` and fit the AR(1)
   ``X[t+1] = a + b X[t] + zeta``, which is an Ornstein-Uhlenbeck process sampled
   once per period when ``0 < b < 1``;
5. the *s-score* ``(X[T] - m) / sigma_eq`` says how far the path is from its
   equilibrium level, in equilibrium standard deviations, and the thresholds in
   ``pca_signals`` turn it into positions.

Because the factor regression has a constant, the residuals sum to zero and
``X[T]`` ends at zero; the s-score is then the distance of that endpoint from the
level ``m`` the path wandered around. The same constant pulls the path back to zero
at the end of the window, which biases the AR(1) coefficient upward and the
reversion speed ``kappa`` downward (about 0.83 instead of a planted 0.80 in
simulation) and keeps the coefficient below one even for a random walk.

Research tools only: nothing in ``optimizer.statarb`` models transaction costs,
borrow costs or short-sale limits, and the fund mandate is long-only and
paper-only, so none of it is wired to execution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from optimizer.exceptions import ConfigurationError, DataError
from optimizer.statarb._config import PcaResidualConfig
from optimizer.statarb._inputs import (
    EXACT_FIT_RATIO,
    as_float_array,
    require_finite,
    require_integer,
)

_FloatArray = npt.NDArray[np.float64]
# A correlation matrix of one asset has no common factor to remove.
_MIN_ASSETS: Final = 2


@dataclass(frozen=True, eq=False)
class PcaResidualModel:
    """A fitted PCA residual model.

    The model holds pandas objects, so it compares by identity.

    Attributes:
        eigenvalues: Every eigenvalue of the correlation matrix, largest first,
            indexed ``PC1``, ``PC2``, and so on.
        explained_variance_ratio: ``eigenvalues`` divided by the number of assets,
            the share of total variance each component carries; sums to 1.
        loadings: Eigenvectors of the correlation matrix (assets by components)
            for the standardised returns, unit length. Each column's sign is
            chosen so its entries sum to a non-negative number.
        factor_returns: Returns of the eigenportfolios (dates by components): the
            eigenvector divided by each asset's volatility, applied to the raw
            returns.
        betas: Regression slope of each asset on each factor (assets by
            components).
        residuals: Regression residuals (dates by assets); each column has zero
            mean.
        ou_parameters: One row per asset with ``kappa`` (annualised reversion
            speed), ``m`` (equilibrium level of the cumulative residual),
            ``sigma_eq`` (its equilibrium standard deviation), ``half_life`` (in
            periods) and ``mean_reverting`` (whether the residual qualifies). The
            four numbers are NaN where the AR(1) coefficient is outside ``(0, 1)``.
        s_scores: Latest s-score of each asset; NaN where it is not mean
            reverting.
    """

    eigenvalues: pd.Series
    explained_variance_ratio: pd.Series
    loadings: pd.DataFrame
    factor_returns: pd.DataFrame
    betas: pd.DataFrame
    residuals: pd.DataFrame
    ou_parameters: pd.DataFrame
    s_scores: pd.Series


def _validated_returns(
    returns: pd.DataFrame, n_components: int, config: PcaResidualConfig
) -> _FloatArray:
    """Return the returns as a float array after checking them against the fit."""
    if not isinstance(returns, pd.DataFrame):
        msg = "returns must be a pandas DataFrame (dates by assets)"
        raise DataError(msg)
    n_dates, n_assets = returns.shape
    if n_assets < _MIN_ASSETS:
        msg = f"at least {_MIN_ASSETS} assets are needed, got {n_assets}"
        raise DataError(msg)
    if not 1 <= n_components < n_assets:
        msg = f"n_components must be in [1, {n_assets - 1}], got {n_components}"
        raise ConfigurationError(msg)
    if n_dates < config.min_observations:
        msg = (
            f"at least {config.min_observations} observations are needed, got {n_dates}"
        )
        raise DataError(msg)
    if n_dates <= n_components + 1:
        msg = (
            f"{n_components} factors and a constant would fit {n_dates} observations "
            "exactly: more observations than n_components + 1 are needed"
        )
        raise DataError(msg)
    values = as_float_array(returns, "returns")
    if not np.isfinite(values).all():
        msg = "returns must be finite: drop or fill missing values first"
        raise DataError(msg)
    flat = np.flatnonzero(np.ptp(values, axis=0) == 0.0)
    if flat.size:
        msg = f"asset {returns.columns[flat[0]]!r} has constant returns"
        raise DataError(msg)
    return values


def _principal_components(
    values: _FloatArray, n_components: int
) -> tuple[_FloatArray, _FloatArray, _FloatArray]:
    """Return all eigenvalues, the leading eigenvectors and the volatilities.

    The eigenvalues come largest first. Each eigenvector column is flipped, if
    needed, so that its entries sum to a non-negative number: eigenvectors are
    defined only up to sign, and this makes the leading "market" component
    positive on every asset.
    """
    volatility = values.std(axis=0, ddof=1)
    standardised = (values - values.mean(axis=0)) / volatility
    correlation = standardised.T @ standardised / (len(values) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    leading = eigenvectors[:, ::-1][:, :n_components]
    signs = np.where(leading.sum(axis=0) < 0.0, -1.0, 1.0)
    return eigenvalues[::-1], leading * signs, volatility


def _factor_regression(
    values: _FloatArray, factors: _FloatArray
) -> tuple[_FloatArray, _FloatArray]:
    """Return the slopes (assets by factors) and residuals of the regression."""
    design = np.column_stack([np.ones(len(values)), factors])
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    return coefficients[1:].T, values - design @ coefficients


def _ar1_fit(paths: _FloatArray) -> tuple[_FloatArray, _FloatArray, _FloatArray]:
    """Fit ``X[t+1] = a + b X[t] + zeta`` to every column of ``paths``.

    Returns:
        The intercepts, slopes and residual variances. A column whose lagged
        values are constant gets a NaN slope, and one the line fits to machine
        precision gets a NaN variance.
    """
    lagged, following = paths[:-1], paths[1:]
    lagged_centred = lagged - lagged.mean(axis=0)
    following_centred = following - following.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (lagged_centred * following_centred).sum(axis=0) / (
            lagged_centred**2
        ).sum(axis=0)
    intercept = following.mean(axis=0) - slope * lagged.mean(axis=0)
    errors = following - intercept - slope * lagged
    residual_ss = (errors**2).sum(axis=0)
    total_ss = (following_centred**2).sum(axis=0)
    variance = np.where(
        residual_ss > EXACT_FIT_RATIO * total_ss,
        residual_ss / (len(errors) - 2),
        np.nan,
    )
    return intercept, slope, variance


def _ou_summary(
    paths: _FloatArray, columns: pd.Index, config: PcaResidualConfig
) -> tuple[pd.DataFrame, pd.Series]:
    """Return the OU parameter table and the s-scores of the cumulative residuals."""
    intercept, slope, variance = _ar1_fit(paths)
    reverting = (slope > 0.0) & (slope < 1.0) & (variance > 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = -np.log(slope) * config.periods_per_year
        level = intercept / (1.0 - slope)
        sigma_eq = np.sqrt(variance / (1.0 - slope**2))
        half_life = -math.log(2.0) / np.log(slope)
    qualifies = reverting & (kappa > config.min_kappa)
    table = pd.DataFrame(
        {
            "kappa": np.where(reverting, kappa, np.nan),
            "m": np.where(reverting, level, np.nan),
            "sigma_eq": np.where(reverting, sigma_eq, np.nan),
            "half_life": np.where(reverting, half_life, np.nan),
            "mean_reverting": qualifies,
        },
        index=columns,
    )
    scores = np.where(qualifies, (paths[-1] - level) / sigma_eq, np.nan)
    return table, pd.Series(scores, index=columns, name="s_score")


def fit_pca_residual_model(
    returns: pd.DataFrame,
    n_components: int = 5,
    config: PcaResidualConfig | None = None,
) -> PcaResidualModel:
    """Fit the Avellaneda-Lee PCA residual model to a window of returns.

    Each asset's returns are regressed (with a constant) on the returns of the
    leading eigenportfolios; the residuals are cumulated and an AR(1) is fitted to
    each cumulative path. ``kappa = -ln(b) * periods_per_year``, ``m = a / (1 - b)``
    and ``sigma_eq = sqrt(var(zeta) / (1 - b ** 2))`` with ``var(zeta)`` the
    residual variance of the AR(1) regression on ``n - 2`` degrees of freedom. An
    asset whose coefficient ``b`` is outside ``(0, 1)`` (or whose AR(1) residual
    variance is zero) is flagged not mean reverting: its OU parameters and
    s-score are NaN. Because the cumulative residual returns to zero at the end of
    the window, ``b`` is almost always below one: a random walk or an explosive
    residual shows up as ``b`` just below one (``kappa`` near zero), not above it.
    Set ``PcaResidualConfig.min_kappa`` to drop residuals that revert too slowly.

    This is a research tool. It ignores costs and borrow limits, and nothing here
    is wired to execution.

    Args:
        returns: Simple returns, dates by assets, with no missing values.
        n_components: Number of eigenportfolios, at least 1 and below the number of
            assets.
        config: Settings; defaults to ``PcaResidualConfig()``.

    Returns:
        The eigenstructure, factor returns, betas, residuals, OU parameters and
        latest s-scores.

    Raises:
        DataError: If ``returns`` is not a DataFrame, has fewer than two assets or
            fewer rows than ``config.min_observations`` or than
            ``n_components + 2``, has a value that is not a finite number, or an
            asset is constant.
        ConfigurationError: If ``n_components`` is not an integer or is outside
            ``[1, n_assets - 1]``.
    """
    settings = config or PcaResidualConfig()
    n_components = require_integer("n_components", n_components, 1)
    values = _validated_returns(returns, n_components, settings)
    eigenvalues, eigenvectors, volatility = _principal_components(values, n_components)
    factors = values @ (eigenvectors / volatility[:, None])
    betas, residuals = _factor_regression(values, factors)
    table, scores = _ou_summary(np.cumsum(residuals, axis=0), returns.columns, settings)
    components = [f"PC{i + 1}" for i in range(n_components)]
    ranks = [f"PC{i + 1}" for i in range(len(eigenvalues))]
    return PcaResidualModel(
        eigenvalues=pd.Series(eigenvalues, index=ranks, name="eigenvalue"),
        explained_variance_ratio=pd.Series(
            eigenvalues / len(eigenvalues), index=ranks, name="explained_variance_ratio"
        ),
        loadings=pd.DataFrame(eigenvectors, index=returns.columns, columns=components),
        factor_returns=pd.DataFrame(factors, index=returns.index, columns=components),
        betas=pd.DataFrame(betas, index=returns.columns, columns=components),
        residuals=pd.DataFrame(residuals, index=returns.index, columns=returns.columns),
        ou_parameters=table,
        s_scores=scores,
    )


def _validate_thresholds(
    buy_below: float, sell_above: float, exit_long: float, exit_short: float
) -> None:
    buy_below = require_finite("buy_below", buy_below)
    sell_above = require_finite("sell_above", sell_above)
    exit_long = require_finite("exit_long", exit_long)
    exit_short = require_finite("exit_short", exit_short)
    if not buy_below < exit_long:
        msg = f"buy_below ({buy_below}) must be below exit_long ({exit_long})"
        raise ConfigurationError(msg)
    if not exit_short < sell_above:
        msg = f"exit_short ({exit_short}) must be below sell_above ({sell_above})"
        raise ConfigurationError(msg)
    if not buy_below < sell_above:
        msg = f"buy_below ({buy_below}) must be below sell_above ({sell_above})"
        raise ConfigurationError(msg)


def _held_positions(
    index: pd.Index, previous_positions: pd.Series | None
) -> npt.NDArray[np.int64]:
    """Return the positions already held for each asset in ``index``."""
    if previous_positions is None:
        return np.zeros(len(index), dtype=np.int64)
    if not isinstance(previous_positions, pd.Series):
        msg = "previous_positions must be a pandas Series"
        raise DataError(msg)
    if not previous_positions.isin((-1, 0, 1)).all():
        msg = "previous_positions must contain only -1, 0 and 1"
        raise DataError(msg)
    held = previous_positions.reindex(index, fill_value=0)
    return np.asarray(held.to_numpy(), dtype=np.int64)


def pca_signals(
    s_scores: pd.Series,
    buy_below: float = -1.25,
    sell_above: float = 1.25,
    exit_long: float = -0.5,
    exit_short: float = 0.75,
    previous_positions: pd.Series | None = None,
) -> pd.Series:
    """Turn one date's s-scores into target positions in ``{-1, 0, +1}``.

    The defaults are the thresholds of Avellaneda and Lee (2010): buy to open
    when ``s < -1.25``, sell to open when ``s > +1.25``, close a short when
    ``s < +0.75`` and close a long when ``s > -0.50``. Every comparison is strict.

    A held position stays open until its own exit rule closes it, and a position
    closed on a date is not reopened on that date. An asset without a score (NaN,
    a residual that does not mean revert) is flat: there is no basis to hold it.

    This is a research tool. It ignores costs and borrow limits, and nothing here
    is wired to execution.

    Args:
        s_scores: Latest s-score of each asset, for example
            ``PcaResidualModel.s_scores``.
        buy_below: S-score below which a long position opens.
        sell_above: S-score above which a short position opens.
        exit_long: Held longs close when the score rises above this level.
        exit_short: Held shorts close when the score falls below this level.
        previous_positions: Positions held before this date by asset; assets
            missing from it start flat and extra assets are ignored. ``None``
            means everything starts flat.

    Returns:
        Integer target positions on the index of ``s_scores``.

    Raises:
        DataError: If ``s_scores`` is not a Series of numbers, or
            ``previous_positions`` is not a Series or holds a value other than
            -1, 0 or 1.
        ConfigurationError: If a threshold is not finite or the levels are not
            ordered ``buy_below < exit_long``, ``exit_short < sell_above`` and
            ``buy_below < sell_above``.
    """
    _validate_thresholds(buy_below, sell_above, exit_long, exit_short)
    if not isinstance(s_scores, pd.Series):
        msg = "s_scores must be a pandas Series"
        raise DataError(msg)
    held = _held_positions(s_scores.index, previous_positions)
    scores = as_float_array(s_scores, "s_scores")
    long_position = ((held == 1) & (scores <= exit_long)) | (
        (held == 0) & (scores < buy_below)
    )
    short_position = ((held == -1) & (scores >= exit_short)) | (
        (held == 0) & (scores > sell_above)
    )
    positions = long_position.astype(np.int64) - short_position.astype(np.int64)
    return pd.Series(positions, index=s_scores.index, name="position")
