"""Trend and momentum strategies with a cost-aware weight backtest.

Moving-average crossover positions, 12-1 momentum scores and cross-sectional
momentum weights turn a price table into target weights. ``backtest_weights``
then simulates holding those weights with a configurable execution lag and a
proportional trading cost, and reports gross and net returns, turnover, the
equity curve and summary metrics.

The backtest never looks ahead: a weight decided at the close of a date earns
returns only from ``execution_lag`` periods later. The default lag of one
period means "decide at the close, hold over the next period". A lag of zero
credits a weight with the return that ended on the date it was decided, which
is a look-ahead leak; it is allowed so that the leak can be measured, not
because it is a valid way to trade.

A backtest is in-sample evidence about a rule that was fitted to the same data.
Its Sharpe ratio is optimistic when many variants of the rule were tried and the
best one is reported, so correct it for the number of variants tried before
trusting it.
"""

from optimizer.strategies._backtest import (
    BacktestMetrics,
    BacktestResult,
    backtest_weights,
)
from optimizer.strategies._config import (
    BacktestConfig,
    CostConfig,
    MovingAverageKind,
)
from optimizer.strategies._signals import (
    cross_sectional_momentum_weights,
    momentum_scores,
    moving_average,
    moving_average_crossover_positions,
)

__all__ = [
    "BacktestConfig",
    "BacktestMetrics",
    "BacktestResult",
    "CostConfig",
    "MovingAverageKind",
    "backtest_weights",
    "cross_sectional_momentum_weights",
    "momentum_scores",
    "moving_average",
    "moving_average_crossover_positions",
]
