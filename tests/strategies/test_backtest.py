"""Tests for the weight backtest.

The golden numbers were worked out by hand (exact fractions where the arithmetic
allows it) and share no code with the module under test. Row ``0`` of the prices
is the base date: the result covers the rows after it.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from optimizer.exceptions import DataError
from optimizer.strategies import (
    BacktestConfig,
    BacktestMetrics,
    BacktestResult,
    CostConfig,
    backtest_weights,
    cross_sectional_momentum_weights,
    moving_average_crossover_positions,
)

_NAN = float("nan")
_NO_COSTS = BacktestConfig(costs=CostConfig(proportional_bps=0.0))


def _prices(**columns: Sequence[float]) -> pd.DataFrame:
    """Return a price frame with one column per keyword, on business days."""
    length = len(next(iter(columns.values())))
    index = pd.bdate_range("2024-01-01", periods=length)
    return pd.DataFrame(columns, index=index, dtype=float)


def _targets(prices: pd.DataFrame, rows: Sequence[Sequence[float]]) -> pd.DataFrame:
    """Return target weights on the prices' index and columns."""
    return pd.DataFrame(rows, index=prices.index, columns=prices.columns, dtype=float)


def _constant(prices: pd.DataFrame, weights: Sequence[float]) -> pd.DataFrame:
    """Return the same target row on every date."""
    return _targets(prices, [weights] * len(prices))


# One asset that returns +2%, -2% and +5%: 100 * 1.02 = 102, * 0.98 = 99.96,
# * 1.05 = 104.958.
_SINGLE = [100, 102, 99.96, 104.958]
# Two assets: A returns +10%, +10%, 0 and B returns 0, 0, +10%.
_TWO_A = [100, 110, 121, 121]
_TWO_B = [100, 100, 100, 110]


class TestReturns:
    """Gross and net returns, held weights and the equity curve."""

    def test_buy_and_hold_without_costs_reproduces_the_asset_returns(self) -> None:
        """One fully held asset earns exactly its own returns."""
        prices = _prices(x=_SINGLE)
        result = backtest_weights(prices, _constant(prices, [1.0]), _NO_COSTS)
        expected = [0.02, -0.02, 0.05]
        np.testing.assert_allclose(result.gross_returns.to_numpy(), expected)
        np.testing.assert_allclose(result.net_returns.to_numpy(), expected)
        np.testing.assert_allclose(
            result.equity_curve.to_numpy(), [1.02, 0.9996, 1.04958]
        )
        assert result.gross_returns.index.equals(prices.index[1:])

    def test_a_constant_even_split_earns_the_average_return(self) -> None:
        """50/50 earns (0.10 + 0) / 2 = 0.05 on each of the three periods."""
        prices = _prices(a=_TWO_A, b=_TWO_B)
        result = backtest_weights(prices, _constant(prices, [0.5, 0.5]), _NO_COSTS)
        np.testing.assert_allclose(result.gross_returns.to_numpy(), [0.05] * 3)
        # 1.05, 1.05 ** 2 and 1.05 ** 3 by hand.
        np.testing.assert_allclose(
            result.equity_curve.to_numpy(), [1.05, 1.1025, 1.157625]
        )

    def test_the_equity_curve_scales_with_the_initial_capital(self) -> None:
        """Starting from 1000 multiplies every equity value by 1000."""
        prices = _prices(x=_SINGLE)
        config = BacktestConfig(
            costs=CostConfig(proportional_bps=0.0), initial_capital=1000.0
        )
        result = backtest_weights(prices, _constant(prices, [1.0]), config)
        np.testing.assert_allclose(
            result.equity_curve.to_numpy(), [1020.0, 999.6, 1049.58]
        )

    @pytest.mark.parametrize(
        ("lag", "held", "gross"),
        [
            (0, [[0, 1], [1, 0], [0, 1]], [0.0, 0.1, 0.1]),
            (1, [[1, 0], [0, 1], [1, 0]], [0.1, 0.0, 0.0]),
            (2, [[0, 0], [1, 0], [0, 1]], [0.0, 0.1, 0.1]),
        ],
    )
    def test_weights_earn_returns_only_from_the_lagged_period(
        self, lag: int, held: list[list[float]], gross: list[float]
    ) -> None:
        """Weights decided on row t are held on row t + lag, not before."""
        prices = _prices(a=_TWO_A, b=_TWO_B)
        targets = _targets(prices, [[1, 0], [0, 1], [1, 0], [0, 1]])
        config = BacktestConfig(
            execution_lag=lag, costs=CostConfig(proportional_bps=0.0)
        )
        result = backtest_weights(prices, targets, config)
        np.testing.assert_allclose(result.weights.to_numpy(), held)
        np.testing.assert_allclose(result.gross_returns.to_numpy(), gross)

    def test_the_book_is_flat_until_the_first_decision_lands(self) -> None:
        """With a two-row lag the first period holds nothing and costs nothing."""
        prices = _prices(x=_SINGLE)
        config = BacktestConfig(execution_lag=2)
        result = backtest_weights(prices, _constant(prices, [1.0]), config)
        np.testing.assert_allclose(result.weights["x"].to_numpy(), [0.0, 1.0, 1.0])
        np.testing.assert_allclose(result.turnover.to_numpy(), [0.0, 1.0, 0.0])
        np.testing.assert_allclose(result.costs.to_numpy(), [0.0, 0.001, 0.0])

    def test_a_missing_price_of_an_unheld_asset_is_harmless(self) -> None:
        """An asset that is never held may have gaps without affecting the book."""
        prices = _prices(a=[100, 101, 102, 103], b=[_NAN, _NAN, 50, 51])
        result = backtest_weights(prices, _constant(prices, [1.0, 0.0]), _NO_COSTS)
        expected = [0.01, 1.0 / 101.0, 1.0 / 102.0]
        np.testing.assert_allclose(result.gross_returns.to_numpy(), expected)

    def test_gross_and_net_returns_differ_by_the_costs(self) -> None:
        """The net return is the gross return less the cost of the period."""
        prices = _prices(a=_TWO_A, b=_TWO_B)
        targets = _targets(prices, [[1, 0], [0, 1], [1, 0], [0, 1]])
        result = backtest_weights(prices, targets)
        np.testing.assert_allclose(
            (result.gross_returns - result.net_returns).to_numpy(),
            result.costs.to_numpy(),
        )

    def test_the_result_is_frozen(self) -> None:
        """A result cannot be altered after the fact."""
        prices = _prices(x=_SINGLE)
        result = backtest_weights(prices, _constant(prices, [1.0]))
        assert isinstance(result, BacktestResult)
        assert isinstance(result.metrics, BacktestMetrics)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.metrics = result.metrics  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.metrics.sharpe_ratio = 1.0  # type: ignore[misc]


class TestTurnoverAndCosts:
    """The drifted-weight turnover and the proportional cost."""

    def test_the_first_period_buys_the_whole_book(self) -> None:
        """Turnover on the first held period is the sum of the absolute weights."""
        prices = _prices(a=_TWO_A, b=_TWO_B)
        result = backtest_weights(prices, _constant(prices, [0.6, 0.4]))
        assert result.turnover.iloc[0] == pytest.approx(1.0)
        assert result.costs.iloc[0] == pytest.approx(0.001)

    def test_a_constant_book_still_trades_to_undo_the_drift(self) -> None:
        """Holding 50/50 means trading back after A rises 10% while B is flat."""
        # Hand worked with 100 bps. Period 1 buys 1.0 (cost 0.01) and the book
        # earns 0.05. A is then 0.55 / 1.05 of the book, so the next period
        # trades 2 * (0.55 / 1.05 - 0.5) = 1/21 and pays 1/2100.
        prices = _prices(a=_TWO_A, b=_TWO_B)
        config = BacktestConfig(costs=CostConfig(proportional_bps=100.0))
        result = backtest_weights(prices, _constant(prices, [0.5, 0.5]), config)
        np.testing.assert_allclose(result.turnover.to_numpy(), [1.0, 1 / 21, 1 / 21])
        np.testing.assert_allclose(result.costs.to_numpy(), [0.01, 1 / 2100, 1 / 2100])
        np.testing.assert_allclose(
            result.net_returns.to_numpy(),
            [0.05 - 0.01, 0.05 - 1 / 2100, 0.05 - 1 / 2100],
        )

    def test_a_long_short_book_drifts_by_its_own_value(self) -> None:
        """Weights are renormalised by the portfolio value, here 1 - 0.1 = 0.9."""
        # Long A and short B in equal size; A rises 10% and B 20% in period 1.
        # The book earns 0.1 - 0.2 = -0.1, A drifts to 1.1 / 0.9 = 11/9 and B to
        # -1.2 / 0.9 = -4/3, so period 2 trades 2/9 + 1/3 = 5/9.
        prices = _prices(a=[100, 110, 110], b=[100, 120, 120])
        result = backtest_weights(prices, _constant(prices, [1.0, -1.0]), _NO_COSTS)
        np.testing.assert_allclose(result.turnover.to_numpy(), [2.0, 5.0 / 9.0])
        np.testing.assert_allclose(result.gross_returns.to_numpy(), [-0.1, 0.0])

    def test_switching_assets_costs_two_units_of_turnover(self) -> None:
        """Selling all of A and buying all of B trades 2; the purchase trades 1."""
        prices = _prices(
            a=[100, 101, 102.01, 103.0301, 104.060401, 105.10100501],
            b=[100, 102, 104.04, 106.1208, 108.243216, 110.40808032],
        )
        targets = _targets(prices, [[1, 0]] * 3 + [[0, 1]] * 3)
        result = backtest_weights(prices, targets)
        # Held A on periods 1-3 and B on periods 4-5 (one-row lag). A earns 1% a
        # day and B 2%; the purchase costs 10 bps and the switch 2 * 10 bps.
        np.testing.assert_allclose(result.turnover.to_numpy(), [1, 0, 0, 2, 0])
        np.testing.assert_allclose(
            result.costs.to_numpy(), [0.001, 0, 0, 0.002, 0], atol=1e-15
        )
        np.testing.assert_allclose(
            result.net_returns.to_numpy(), [0.009, 0.01, 0.01, 0.018, 0.02]
        )

    def test_costs_are_proportional_to_the_rate(self) -> None:
        """Doubling the basis points doubles the cost of every period."""
        prices = _prices(a=_TWO_A, b=_TWO_B)
        targets = _constant(prices, [0.5, 0.5])
        low = backtest_weights(
            prices, targets, BacktestConfig(costs=CostConfig(proportional_bps=10.0))
        )
        high = backtest_weights(
            prices, targets, BacktestConfig(costs=CostConfig(proportional_bps=20.0))
        )
        np.testing.assert_allclose(high.costs.to_numpy(), 2.0 * low.costs.to_numpy())

    def test_zero_cost_leaves_net_equal_to_gross(self) -> None:
        """A frictionless run charges nothing."""
        prices = _prices(a=_TWO_A, b=_TWO_B)
        result = backtest_weights(prices, _constant(prices, [0.5, 0.5]), _NO_COSTS)
        assert (result.costs == 0.0).all()
        pd.testing.assert_series_equal(
            result.gross_returns, result.net_returns, check_names=False
        )


def _leak_inputs() -> tuple[pd.DataFrame, np.ndarray]:
    """Return seeded random-walk prices and the return on each row after the first."""
    returns = np.random.default_rng(20240607).normal(0.0, 0.01, 500)
    levels = 100.0 * np.cumprod(np.concatenate(([1.0], 1.0 + returns)))
    index = pd.bdate_range("2020-01-01", periods=levels.size)
    return pd.DataFrame({"x": levels}, index=index), returns


class TestNoLookAhead:
    """A position built from a return must not earn that same return."""

    def test_a_position_stamped_with_the_return_it_will_earn_leaks_at_lag_zero(
        self,
    ) -> None:
        """The sign of the return landing on a row earns |r| only when lag is 0."""
        prices, returns = _leak_inputs()
        signs = np.concatenate(([0.0], np.sign(returns)))
        targets = pd.DataFrame({"x": signs}, index=prices.index)
        leaked = backtest_weights(
            prices, targets, BacktestConfig(execution_lag=0, costs=_NO_COSTS.costs)
        )
        honest = backtest_weights(
            prices, targets, BacktestConfig(execution_lag=1, costs=_NO_COSTS.costs)
        )
        # E|r| is 0.01 * sqrt(2 / pi) = 0.008 per period, so the leak is large.
        assert leaked.gross_returns.mean() > 0.005
        # With one row of lag the sign is independent of the return it meets, so
        # the mean is zero up to four standard errors, 4 * 0.01 / sqrt(500).
        assert abs(honest.gross_returns.mean()) < 0.002

    def test_lag_one_credits_a_position_with_the_next_return_exactly(self) -> None:
        """A position equal to the sign of the following return earns |r| at lag 1."""
        prices, returns = _leak_inputs()
        oracle = np.concatenate((np.sign(returns), [0.0]))
        targets = pd.DataFrame({"x": oracle}, index=prices.index)
        one = backtest_weights(
            prices, targets, BacktestConfig(execution_lag=1, costs=_NO_COSTS.costs)
        )
        two = backtest_weights(
            prices, targets, BacktestConfig(execution_lag=2, costs=_NO_COSTS.costs)
        )
        np.testing.assert_allclose(one.gross_returns.to_numpy(), np.abs(returns))
        assert abs(two.gross_returns.mean()) < 0.002

    def test_a_crossover_strategy_misses_the_jump_that_triggers_it(self) -> None:
        """The crossover signal fires after the jump, so the jump is not earned."""
        # Positions on this path are long on rows 3-5 only; the +10% jump arrives
        # on row 3, the signal that sees it is held from row 4, and the equity is
        # (12/11) * (11/12) * (9/11) = 9/11 over the rise to 12, the fall to 11
        # and the fall to 9.
        prices = _prices(x=[10, 10, 10, 11, 12, 11, 9, 8, 9, 11])
        positions = moving_average_crossover_positions(prices, fast=2, slow=3)
        result = backtest_weights(prices, positions, _NO_COSTS)
        np.testing.assert_allclose(
            result.weights["x"].to_numpy(), [0, 0, 0, 1, 1, 1, 0, 0, 0]
        )
        assert result.equity_curve.iloc[-1] == pytest.approx(9.0 / 11.0)


class TestMetrics:
    """``BacktestMetrics`` against hand-computed values."""

    @staticmethod
    def _run(periods: int, capital: float = 1.0) -> BacktestResult:
        """Hold one asset that returns +10%, -20%, +5% and +10%, without costs."""
        prices = _prices(x=[100, 110, 88, 92.4, 101.64])
        config = BacktestConfig(
            periods_per_year=periods,
            initial_capital=capital,
            costs=CostConfig(proportional_bps=0.0),
        )
        return backtest_weights(prices, _constant(prices, [1.0]), config)

    def test_four_periods_a_year(self) -> None:
        """Compounding gives 1.0164 over four periods; mean 0.0125, std 0.14361."""
        # The sample standard deviation of the returns is 0.143614066 and their
        # mean is 0.0125. With four periods a year the annualised return is the
        # total 1.0164 less one, the volatility is twice the standard deviation
        # and the Sharpe ratio is the mean over the standard deviation, times two.
        metrics = self._run(4).metrics
        assert metrics.annualized_return == pytest.approx(0.0164)
        assert metrics.annualized_volatility == pytest.approx(0.2872281323)
        assert metrics.sharpe_ratio == pytest.approx(0.1740776560)
        # The peak of 1.1 falls to 0.88: 0.88 / 1.1 - 1 = -0.2.
        assert metrics.max_drawdown == pytest.approx(-0.2)

    def test_daily_annualisation(self) -> None:
        """With 252 periods the same returns annualise by 63 and by sqrt(252)."""
        metrics = self._run(252).metrics
        # Worked by hand: 1.0164 ** 63 - 1, then 0.143614066 * sqrt(252), then
        # 0.0125 / 0.143614066 * sqrt(252).
        assert metrics.annualized_return == pytest.approx(1.7865917810)
        assert metrics.annualized_volatility == pytest.approx(2.2798026230)
        assert metrics.sharpe_ratio == pytest.approx(1.3816985594)

    def test_metrics_do_not_depend_on_the_initial_capital(self) -> None:
        """Returns, ratios and drawdowns are scale free."""
        scaled = dataclasses.astuple(self._run(4, capital=1000.0).metrics)
        assert scaled == pytest.approx(dataclasses.astuple(self._run(4).metrics))

    def test_the_initial_capital_is_the_first_peak(self) -> None:
        """A loss on the first period is already a drawdown from the start."""
        # Returns of -10%, +5%, +4% and -2%: the equity never regains 1.0, so the
        # worst drawdown is the first -10%.
        prices = _prices(x=[100, 90, 94.5, 98.28, 96.3144])
        result = backtest_weights(prices, _constant(prices, [1.0]), _NO_COSTS)
        assert result.metrics.max_drawdown == pytest.approx(-0.1)

    def test_a_book_that_only_rises_has_no_drawdown(self) -> None:
        """Without a fall the maximum drawdown is exactly zero."""
        prices = _prices(x=[100, 101, 103, 106, 110])
        result = backtest_weights(prices, _constant(prices, [1.0]), _NO_COSTS)
        assert result.metrics.max_drawdown == 0.0

    def test_average_turnover_is_the_mean_per_period(self) -> None:
        """The turnovers 1, 0, 0, 2 and 0 of the switch average 0.6."""
        prices = _prices(
            a=[100, 101, 102.01, 103.0301, 104.060401, 105.10100501],
            b=[100, 102, 104.04, 106.1208, 108.243216, 110.40808032],
        )
        targets = _targets(prices, [[1, 0]] * 3 + [[0, 1]] * 3)
        assert backtest_weights(prices, targets).metrics.average_turnover == (
            pytest.approx(0.6)
        )

    def test_the_cost_drag_is_the_annualised_gross_less_net_return(self) -> None:
        """One 10 bps purchase turns 1.04958 into 1.048551 over three periods."""
        # Three periods a year over three periods: the exponent is one. Gross is
        # 1.02 * 0.98 * 1.05 = 1.04958; net replaces 1.02 by 1.019: 1.048551.
        prices = _prices(x=_SINGLE)
        config = BacktestConfig(periods_per_year=3)
        metrics = backtest_weights(prices, _constant(prices, [1.0]), config).metrics
        assert metrics.total_cost_drag == pytest.approx(1.04958 - 1.048551)
        assert metrics.annualized_return == pytest.approx(0.048551)

    def test_no_costs_means_no_cost_drag(self) -> None:
        """Without trading costs gross and net annualise identically."""
        assert self._run(252).metrics.total_cost_drag == pytest.approx(0.0, abs=1e-12)

    def test_a_flat_book_has_zero_volatility_and_no_sharpe_ratio(self) -> None:
        """A book that never invests earns nothing; the Sharpe ratio is undefined."""
        prices = _prices(x=_SINGLE)
        result = backtest_weights(prices, _constant(prices, [0.0]))
        assert result.metrics.annualized_volatility == 0.0
        assert math.isnan(result.metrics.sharpe_ratio)
        assert result.metrics.annualized_return == 0.0
        assert result.metrics.max_drawdown == 0.0
        assert result.metrics.average_turnover == 0.0

    def test_the_metrics_are_frozen(self) -> None:
        """A metrics record cannot be altered."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            self._run(4).metrics.sharpe_ratio = 0.0  # type: ignore[misc]


class TestStrategyPipelines:
    """The signals feed the backtest without further glue."""

    def test_momentum_weights_run_end_to_end(self) -> None:
        """A seeded universe backtests with costs that follow from its turnover."""
        rng = np.random.default_rng(7)
        steps = 1.0 + rng.normal(0.0004, 0.01, size=(160, 6))
        prices = pd.DataFrame(
            100.0 * np.cumprod(steps, axis=0),
            index=pd.bdate_range("2023-01-02", periods=160),
            columns=list("ABCDEF"),
        )
        targets = cross_sectional_momentum_weights(
            prices, lookback=20, skip=2, top_fraction=1 / 3, rebalance_every=5
        )
        result = backtest_weights(prices, targets)
        assert result.costs.sum() > 0.0
        assert result.metrics.total_cost_drag > 0.0
        np.testing.assert_allclose(
            result.equity_curve.to_numpy(), (1.0 + result.net_returns).cumprod()
        )
        np.testing.assert_allclose(
            result.costs.to_numpy(), result.turnover.to_numpy() * 0.001
        )
        assert math.isfinite(result.metrics.sharpe_ratio)


def _bad_inputs(kind: str) -> tuple[object, object]:
    """Return a ``(prices, weights)`` pair that is invalid in the way ``kind`` says."""
    prices = _prices(a=[100, 101, 102, 103], b=[50, 51, 52, 53])
    good = _constant(prices, [0.5, 0.5])
    shifted = good.set_axis(good.index + pd.Timedelta(days=1), axis=0)
    nan_weights, inf_weights = good.copy(), good.copy()
    nan_weights.iloc[1, 0] = _NAN
    inf_weights.iloc[1, 0] = np.inf
    held_gap = _prices(a=[100, _NAN, 102, 103], b=[50, 51, 52, 53])
    bought_gap = _prices(a=[100, 101, 102, 103], b=[_NAN, _NAN, 52, 53])
    switch = _targets(prices, [[1, 0], [0, 1], [0, 1], [0, 1]])
    cases: dict[str, tuple[object, object]] = {
        "prices_not_a_frame": (prices["a"], good),
        "weights_not_a_frame": (prices, good["a"]),
        "two_rows_only": (prices.iloc[:2], good.iloc[:2]),
        "index_mismatch": (prices, shifted),
        "column_order_mismatch": (prices, good[["b", "a"]]),
        "extra_weight_column": (prices, good.assign(c=0.0)),
        "nan_weight": (prices, nan_weights),
        "infinite_weight": (prices, inf_weights),
        "bad_price": (prices.assign(a=[100, 0.0, 102, 103]), good),
        "held_asset_gap": (held_gap, good),
        "asset_bought_at_a_gap": (bought_gap, switch),
    }
    return cases[kind]


class TestValidation:
    """Every way the inputs can be wrong raises a ``DataError``."""

    @pytest.mark.parametrize(
        "kind",
        [
            "prices_not_a_frame",
            "weights_not_a_frame",
            "two_rows_only",
            "index_mismatch",
            "column_order_mismatch",
            "extra_weight_column",
            "nan_weight",
            "infinite_weight",
            "bad_price",
            "held_asset_gap",
            "asset_bought_at_a_gap",
        ],
    )
    def test_invalid_inputs_raise(self, kind: str) -> None:
        """Misaligned, non-finite, missing or malformed data is rejected."""
        prices, weights = _bad_inputs(kind)
        with pytest.raises(DataError):
            backtest_weights(prices, weights)  # type: ignore[arg-type]

    def test_a_book_that_loses_more_than_its_value_raises(self) -> None:
        """Twice the equity in an asset that falls 60% leaves a negative value."""
        prices = _prices(x=[100, 40, 40])
        with pytest.raises(DataError, match="value"):
            backtest_weights(prices, _constant(prices, [2.0]), _NO_COSTS)

    def test_costs_larger_than_the_book_raise(self) -> None:
        """A cost above 100% of equity cannot be charged."""
        prices = _prices(x=[100, 101, 102])
        config = BacktestConfig(costs=CostConfig(proportional_bps=20000.0))
        with pytest.raises(DataError, match="after costs"):
            backtest_weights(prices, _constant(prices, [1.0]), config)
