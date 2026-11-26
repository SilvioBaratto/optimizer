"""Vectorised backtest of target weights, with transaction costs.

The backtest turns a table of target weights into a return series without looking
ahead. Conventions, all of which are fixed by this module and not left to the
caller:

* **Timing.** The return of row ``t`` is the simple return from the price on row
  ``t - 1`` to the price on row ``t``. A weight decided at the close of row ``t``
  is held over the period ending on row ``t + execution_lag`` and earns that
  period's return, so with the default lag of 1 it earns the next period. The
  result covers rows ``1`` onwards; row ``0`` only supplies the first price.
* **Constant mix.** The weights of a row are the weights held *throughout* that
  period, so the book is traded back to its targets every period. A target that
  never changes still has turnover wherever the assets drifted apart.
* **Turnover.** With ``h_t`` the weights held in period ``t``, ``r_t`` the asset
  returns of that period and ``g_t = sum(h_t * r_t)`` the gross portfolio return,
  the weights drift to ``d_t = h_t * (1 + r_t) / (1 + g_t)`` (the value of each
  position over the value of the book, before costs). The turnover of period
  ``t`` is ``sum(|h_t - d_{t-1}|)`` and the first period buys the whole book,
  ``sum(|h_1|)``. For a fully invested long-only book this is the usual
  drifted-weight turnover; buying 1 and selling 1 counts as 2.
* **Costs.** ``cost_t = turnover_t * proportional_bps / 10000`` is deducted from
  the return of the period in which the trade happens, so ``net = gross - cost``.
  Drifting uses the gross value, ignoring the second-order effect of the cost.
  Cash, leverage and short positions earn and pay nothing: there is no financing
  or borrow cost.
* **Sharpe ratio.** It is computed on net returns with a zero risk-free rate, so
  it is the mean return over its standard deviation, annualised.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from optimizer.exceptions import DataError
from optimizer.strategies._config import BacktestConfig
from optimizer.strategies._signals import require_price_frame, to_float_frame

_BPS_PER_UNIT: Final = 10_000.0
# One base row plus two return periods, the fewest from which a standard deviation
# exists.
_MIN_ROWS: Final = 3


@dataclass(frozen=True)
class BacktestMetrics:
    """Summary statistics of a backtest, all on net returns unless stated.

    Attributes:
        annualized_return: Compound annual growth rate of the net equity curve.
        annualized_volatility: Sample standard deviation of the net returns,
            scaled by the square root of the periods per year.
        sharpe_ratio: Mean net return over its standard deviation, annualised,
            with a zero risk-free rate; NaN when the returns do not vary.
        max_drawdown: Largest fall of the equity curve from its running peak, as
            a non-positive fraction (-0.25 is a 25% fall). The initial capital
            counts as the first peak.
        average_turnover: Mean turnover per period, the sum of the absolute
            weight changes, so a full switch from one asset to another is 2.
        total_cost_drag: Annualised gross return less the annualised net return,
            both compounded: the yearly return lost to trading costs.
    """

    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    average_turnover: float
    total_cost_drag: float


# Equality is off because the pandas fields have no scalar truth value to compare.
@dataclass(frozen=True, eq=False)
class BacktestResult:
    """Outcome of a weight backtest, one entry per return period.

    Attributes:
        gross_returns: Portfolio return of each period before costs.
        net_returns: ``gross_returns`` less ``costs``.
        weights: Weights actually held in each period, after the execution lag.
        turnover: Sum of the absolute weight changes needed to reach ``weights``
            from the previous period's drifted weights.
        costs: Trading cost of each period as a fraction of equity.
        equity_curve: Net equity, starting from ``initial_capital``.
        metrics: Summary statistics.
    """

    gross_returns: pd.Series
    net_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    equity_curve: pd.Series
    metrics: BacktestMetrics


def _require_aligned_weights(
    prices: pd.DataFrame, target_weights: pd.DataFrame
) -> pd.DataFrame:
    """Return the weights as float64 after checking they fit the prices."""
    weights = to_float_frame(target_weights, "target_weights")
    if not (
        weights.index.equals(prices.index) and weights.columns.equals(prices.columns)
    ):
        msg = "prices and target_weights must share the same index and columns"
        raise DataError(msg)
    if not np.isfinite(weights.to_numpy()).all():
        msg = "target_weights must be finite: fill missing weights with 0 first"
        raise DataError(msg)
    return weights


def _held_weights(targets: pd.DataFrame, execution_lag: int) -> pd.DataFrame:
    """Return the weights held in each return period, after the execution lag."""
    return targets.shift(execution_lag).fillna(0.0).iloc[1:]


def _asset_returns(prices: pd.DataFrame, held: pd.DataFrame) -> pd.DataFrame:
    """Return simple asset returns, failing where a held asset has none."""
    returns = (prices / prices.shift(1) - 1.0).iloc[1:]
    undefined = returns.isna() & (held != 0.0)
    if undefined.to_numpy().any():
        row, column = np.argwhere(undefined.to_numpy())[0]
        msg = (
            f"asset {undefined.columns[column]!r} is held on {undefined.index[row]} "
            "but a price it needs is missing"
        )
        raise DataError(msg)
    return returns.fillna(0.0)


def _require_positive_value(growth: pd.Series, when: str) -> None:
    """Fail if the book's value would reach zero or turn negative."""
    if (growth <= 0.0).any():
        date = growth.index[int(np.argmax((growth <= 0.0).to_numpy()))]
        msg = f"the portfolio value is not positive {when} on {date}"
        raise DataError(msg)


def _turnover(held: pd.DataFrame, returns: pd.DataFrame, gross: pd.Series) -> pd.Series:
    """Return the turnover of each period against the previous drifted weights."""
    drifted = held.mul(1.0 + returns).div(1.0 + gross, axis=0)
    previous = drifted.shift(1).fillna(0.0)
    return (held - previous).abs().sum(axis=1)


def _annualized_return(returns: pd.Series, periods_per_year: int) -> float:
    """Return the compound annual growth rate of a return series."""
    log_growth = float(np.log1p(returns.to_numpy()).sum())
    return math.expm1(log_growth * periods_per_year / returns.size)


def _sharpe_ratio(returns: pd.Series, periods_per_year: int) -> float:
    """Return the annualised Sharpe ratio, NaN when the returns are constant."""
    values = returns.to_numpy()
    if np.ptp(values) == 0.0:
        return math.nan
    return float(values.mean() / values.std(ddof=1)) * math.sqrt(periods_per_year)


def _max_drawdown(equity: pd.Series, initial_capital: float) -> float:
    """Return the worst fall from a running peak, counting the starting capital."""
    levels = np.concatenate(([initial_capital], equity.to_numpy()))
    peaks = np.maximum.accumulate(levels)
    return float((levels / peaks - 1.0).min())


def _compute_metrics(
    gross: pd.Series,
    net: pd.Series,
    turnover: pd.Series,
    equity: pd.Series,
    config: BacktestConfig,
) -> BacktestMetrics:
    """Summarise a finished backtest."""
    periods = config.periods_per_year
    net_return = _annualized_return(net, periods)
    return BacktestMetrics(
        annualized_return=net_return,
        annualized_volatility=float(np.std(net.to_numpy(), ddof=1))
        * math.sqrt(periods),
        sharpe_ratio=_sharpe_ratio(net, periods),
        max_drawdown=_max_drawdown(equity, config.initial_capital),
        average_turnover=float(turnover.mean()),
        total_cost_drag=_annualized_return(gross, periods) - net_return,
    )


def backtest_weights(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Backtest target weights against prices, without look-ahead.

    The conventions (timing, constant mix, turnover, costs) are described in the
    module documentation. Weights may be long-short and may sum to more or less
    than 1; the remainder is held as cash that earns nothing.

    Args:
        prices: One column of prices per asset, indexed by increasing dates, with
            at least three rows. A price may be missing (NaN) for an asset that
            is not held when the gap affects its return.
        target_weights: Weights decided from each date's close, on the same index
            and columns as ``prices``, finite, as fractions of equity.
        config: Backtest conventions; defaults to ``BacktestConfig()``.

    Returns:
        The gross and net returns, held weights, turnover, costs, equity curve
        and summary metrics for the periods after the first row.

    Raises:
        DataError: If the inputs are malformed or misaligned, a held asset has an
            undefined return, or the book's value would reach zero or below.
    """
    settings = config or BacktestConfig()
    frame = require_price_frame(prices, min_rows=_MIN_ROWS)
    targets = _require_aligned_weights(frame, target_weights)
    held = _held_weights(targets, settings.execution_lag)
    asset_returns = _asset_returns(frame, held)
    gross = (held * asset_returns).sum(axis=1)
    _require_positive_value(1.0 + gross, "before costs")
    turnover = _turnover(held, asset_returns, gross)
    costs = turnover * (settings.costs.proportional_bps / _BPS_PER_UNIT)
    net = gross - costs
    _require_positive_value(1.0 + net, "after costs")
    equity = settings.initial_capital * (1.0 + net).cumprod()
    return BacktestResult(
        gross_returns=gross,
        net_returns=net,
        weights=held,
        turnover=turnover,
        costs=costs,
        equity_curve=equity,
        metrics=_compute_metrics(gross, net, turnover, equity, settings),
    )
