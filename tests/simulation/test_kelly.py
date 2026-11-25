"""Tests for the Kelly-criterion functions.

The golden numbers come from closed forms worked out independently of the module:
the even-odds bet f* = 2p - 1, the odds formula f* = (bp - q) / b, the Gaussian
f* = (mu - r) / sigma^2, the stationary point of a two-outcome log utility with a
risk-free rate, and an exact dynamic programme over the binomial lattice of wins
and losses for the ruin and loss probabilities.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from optimizer.exceptions import ConfigurationError, ConvergenceError, DataError
from optimizer.simulation import (
    _kelly,
    kelly_fraction_empirical,
    kelly_fraction_gaussian,
    simulate_kelly_growth,
)

_SEED = 5
_SE_BANDS = 4.0
# Sixty winning and forty losing flips of an even-odds stake.
_EVEN_ODDS = np.array([1.0] * 60 + [-1.0] * 40)
_P_WIN = 0.6


class TestKellyFractionGaussian:
    """``kelly_fraction_gaussian``: (mu - r) / sigma^2."""

    def test_matches_the_closed_form(self) -> None:
        """0.08 / 0.2^2 = 2 and (0.08 - 0.02) / 0.2^2 = 1.5."""
        assert kelly_fraction_gaussian(0.08, 0.2) == pytest.approx(2.0)
        assert kelly_fraction_gaussian(0.08, 0.2, risk_free_rate=0.02) == pytest.approx(
            1.5
        )

    def test_no_excess_return_means_no_bet(self) -> None:
        """An asset that only earns the risk-free rate gets a zero fraction."""
        assert kelly_fraction_gaussian(0.02, 0.2, risk_free_rate=0.02) == 0.0

    def test_negative_excess_return_gives_a_short_position(self) -> None:
        """Below the risk-free rate the unconstrained answer is negative."""
        assert kelly_fraction_gaussian(-0.04, 0.2) == pytest.approx(-1.0)

    @pytest.mark.parametrize("sigma", [0.0, -0.2, math.nan, math.inf])
    def test_invalid_volatility_raises(self, sigma: float) -> None:
        """The volatility is positive and finite."""
        with pytest.raises(ConfigurationError, match="sigma"):
            kelly_fraction_gaussian(0.08, sigma)

    @pytest.mark.parametrize(("mu", "rate"), [(math.nan, 0.0), (0.05, math.inf)])
    def test_non_finite_return_or_rate_raises(self, mu: float, rate: float) -> None:
        """Both the mean return and the risk-free rate are finite."""
        with pytest.raises(ConfigurationError):
            kelly_fraction_gaussian(mu, 0.2, risk_free_rate=rate)


class TestKellyFractionEmpirical:
    """``kelly_fraction_empirical``: the maximiser of the mean log growth."""

    def test_even_odds_bet_recovers_two_p_minus_one(self) -> None:
        """f* = 2p - 1 = 0.2 for +1 with probability 0.6 and -1 otherwise."""
        fraction = kelly_fraction_empirical(_EVEN_ODDS)
        assert fraction == pytest.approx(2.0 * _P_WIN - 1.0, abs=1e-4)

    def test_odds_formula_for_a_three_to_one_payoff(self) -> None:
        """f* = (bp - q) / b = (3 * 0.5 - 0.5) / 3 = 1/3 for +3 or -1 at even chance."""
        fraction = kelly_fraction_empirical(np.array([3.0, -1.0] * 50))
        assert fraction == pytest.approx(1.0 / 3.0, abs=1e-4)

    def test_risk_free_rate_shifts_the_optimum(self) -> None:
        """Stationary point of the two-outcome log utility with a 1% risk-free rate."""
        returns = np.array([0.10] * 60 + [-0.05] * 40)
        fraction = kelly_fraction_empirical(
            returns, risk_free_rate_per_period=0.01, max_leverage=10.0
        )
        # f = -(1 + rf) (p xu + q xd) / (xu xd) with xu = 0.09, xd = -0.06.
        assert fraction == pytest.approx(5.611111111111111, abs=1e-4)

    def test_the_objective_is_maximal_at_the_returned_fraction(self) -> None:
        """Neither neighbour of the optimum beats it."""
        best = kelly_fraction_empirical(_EVEN_ODDS)

        def growth(fraction: float) -> float:
            """Return the exact mean log growth of the 60/40 even-odds bet."""
            return 0.6 * math.log(1.0 + fraction) + 0.4 * math.log(1.0 - fraction)

        assert growth(best) >= growth(best - 0.01)
        assert growth(best) >= growth(best + 0.01)

    def test_no_edge_means_no_bet(self) -> None:
        """A fair coin has a non-positive drift in log wealth, so the answer is zero."""
        assert kelly_fraction_empirical(np.array([1.0, -1.0] * 10)) == 0.0

    def test_negative_edge_means_no_bet(self) -> None:
        """Losing on average, the best fraction is not to play."""
        assert kelly_fraction_empirical(np.array([0.01, -0.03] * 10)) == 0.0

    def test_a_sure_gain_is_capped_by_max_leverage(self) -> None:
        """Log growth rises without limit, so the leverage cap decides."""
        fraction = kelly_fraction_empirical(np.array([0.05] * 5), max_leverage=3.0)
        assert fraction == pytest.approx(3.0, abs=1e-4)

    def test_the_cap_binds_below_the_unconstrained_optimum(self) -> None:
        """A cap of 0.1 beats the optimum 0.2 of the even-odds bet."""
        fraction = kelly_fraction_empirical(_EVEN_ODDS, max_leverage=0.1)
        assert fraction == pytest.approx(0.1, abs=1e-4)

    def test_total_loss_outcome_keeps_the_fraction_below_one(self) -> None:
        """A -100% outcome makes any fraction of one or more ruinous."""
        returns = np.array([0.5] * 90 + [-1.0] * 10)
        fraction = kelly_fraction_empirical(returns, max_leverage=50.0)
        # Odds formula: b = 0.5, p = 0.9, q = 0.1 gives (0.45 - 0.1) / 0.5 = 0.7.
        assert fraction == pytest.approx(0.7, abs=1e-4)

    def test_accepts_a_pandas_series(self) -> None:
        """A return series with a date index is a valid input."""
        series = pd.Series(_EVEN_ODDS, index=pd.RangeIndex(100))
        assert kelly_fraction_empirical(series) == pytest.approx(0.2, abs=1e-4)

    @pytest.mark.parametrize(
        "returns",
        [[0.1], [], [[0.1, 0.2]], [0.1, math.nan], [0.1, -2.0]],
    )
    def test_invalid_returns_raise(self, returns: list[float]) -> None:
        """Returns are at least two finite simple returns of -100% or more."""
        with pytest.raises(DataError):
            kelly_fraction_empirical(returns)

    @pytest.mark.parametrize("max_leverage", [0.0, -1.0, math.nan, math.inf])
    def test_invalid_leverage_cap_raises(self, max_leverage: float) -> None:
        """The cap is positive and finite."""
        with pytest.raises(ConfigurationError, match="max_leverage"):
            kelly_fraction_empirical(_EVEN_ODDS, max_leverage=max_leverage)

    @pytest.mark.parametrize("rate", [-1.0, -2.0, math.nan, math.inf])
    def test_invalid_risk_free_rate_raises(self, rate: float) -> None:
        """The per-period rate is finite and above -100%."""
        with pytest.raises(ConfigurationError, match="risk_free_rate_per_period"):
            kelly_fraction_empirical(_EVEN_ODDS, risk_free_rate_per_period=rate)

    def test_solver_failure_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A scalar minimiser that does not converge raises ``ConvergenceError``."""

        class _Failed:
            """Stand-in for a scipy result of a search that stopped early."""

            success = False
            message = "stalled"
            x = 0.0

        monkeypatch.setattr(
            _kelly.optimize, "minimize_scalar", lambda *a, **k: _Failed()
        )
        with pytest.raises(ConvergenceError, match="stalled"):
            kelly_fraction_empirical(_EVEN_ODDS)


def _ruin_probability(
    up: float, down: float, p_up: float, n_periods: int, threshold: float
) -> float:
    """Return the exact probability that wealth ever drops below ``threshold``.

    Dynamic programme over the number of winning periods: every path with the same
    number of wins at the same time has the same wealth. Paths that cross the
    threshold are absorbed. Written with ``math`` only, so it shares nothing with
    the simulator.
    """
    alive = {0: 1.0}
    hit = 0.0
    for period in range(1, n_periods + 1):
        step: dict[int, float] = {}
        for wins, probability in alive.items():
            for won, chance in ((1, p_up), (0, 1.0 - p_up)):
                new_wins = wins + won
                wealth = up**new_wins * down ** (period - new_wins)
                if wealth < threshold:
                    hit += probability * chance
                else:
                    step[new_wins] = step.get(new_wins, 0.0) + probability * chance
        alive = step
    return hit


def _loss_probability(up: float, down: float, p_up: float, n_periods: int) -> float:
    """Return the exact probability that terminal wealth is below the start."""
    return sum(
        math.comb(n_periods, wins) * p_up**wins * (1.0 - p_up) ** (n_periods - wins)
        for wins in range(n_periods + 1)
        if up**wins * down ** (n_periods - wins) < 1.0
    )


class TestSimulateKellyGrowth:
    """``simulate_kelly_growth`` over a grid of betting fractions."""

    def test_constant_return_has_a_deterministic_outcome(self) -> None:
        """With r = +10% and f = 0.5 wealth is 1.05^n every period, for every path."""
        frame = simulate_kelly_growth([0.1] * 3, [0.0, 0.5], 20, 10, random_state=_SEED)
        row = frame.loc[0.5]
        assert row["growth_rate"] == pytest.approx(math.log(1.05))
        assert row["median_terminal_wealth"] == pytest.approx(1.05**10)
        assert row["probability_below_threshold"] == 0.0
        assert row["probability_of_loss"] == 0.0
        flat = frame.loc[0.0]
        assert flat["growth_rate"] == 0.0
        assert flat["median_terminal_wealth"] == 1.0
        assert flat["probability_of_loss"] == 0.0

    def test_a_steady_loss_crosses_the_ruin_threshold_only_when_deep_enough(
        self,
    ) -> None:
        """For r = -10%, f = 1 ends at 0.9^10 = 0.349 and f = 0.5 at 0.95^10 = 0.599."""
        frame = simulate_kelly_growth(
            [-0.1] * 3, [0.5, 1.0], 20, 10, random_state=_SEED
        )
        deep, shallow = frame.loc[1.0], frame.loc[0.5]
        assert deep["probability_below_threshold"] == 1.0
        assert shallow["probability_below_threshold"] == 0.0
        assert deep["median_terminal_wealth"] == pytest.approx(0.9**10)
        assert shallow["median_terminal_wealth"] == pytest.approx(0.95**10)
        assert deep["probability_of_loss"] == 1.0
        assert shallow["probability_of_loss"] == 1.0
        assert deep["growth_rate"] == pytest.approx(math.log(0.9))

    def test_a_fraction_that_can_lose_more_than_everything_is_limited_to_ruin(
        self,
    ) -> None:
        """Wealth cannot go below zero: f = 2 on a -100% outcome ends at zero."""
        frame = simulate_kelly_growth([-1.0] * 3, [2.0], 5, 4, random_state=_SEED)
        row = frame.loc[2.0]
        assert row["median_terminal_wealth"] == 0.0
        assert row["probability_below_threshold"] == 1.0
        assert row["probability_of_loss"] == 1.0
        assert row["growth_rate"] == -math.inf

    def test_frame_layout(self) -> None:
        """One row per fraction, named index and the four documented columns."""
        frame = simulate_kelly_growth(
            _EVEN_ODDS, [0.1, 0.2, 0.3], 10, 5, random_state=1
        )
        assert list(frame.columns) == [
            "growth_rate",
            "median_terminal_wealth",
            "probability_below_threshold",
            "probability_of_loss",
        ]
        assert frame.index.name == "fraction"
        assert list(frame.index) == [0.1, 0.2, 0.3]

    def test_growth_rate_matches_the_expected_log_growth(self) -> None:
        """g(f) = 0.6 ln(1 + f) + 0.4 ln(1 - f), within four standard errors."""
        fractions = [0.0, 0.1, 0.2, 0.3, 0.4]
        paths, periods = 20_000, 100
        frame = simulate_kelly_growth(
            _EVEN_ODDS, fractions, paths, periods, random_state=_SEED
        )
        for fraction in fractions:
            exact = _P_WIN * math.log1p(fraction) + (1 - _P_WIN) * math.log1p(-fraction)
            spread = abs(math.log1p(fraction) - math.log1p(-fraction))
            standard_error = (
                math.sqrt(_P_WIN * (1 - _P_WIN)) * spread / math.sqrt(paths * periods)
            )
            assert abs(frame.loc[fraction, "growth_rate"] - exact) < max(
                _SE_BANDS * standard_error, 1e-12
            )

    def test_growth_is_maximised_next_to_the_kelly_fraction(self) -> None:
        """The best fraction on the grid is the Kelly fraction or a neighbour."""
        grid = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
        frame = simulate_kelly_growth(_EVEN_ODDS, grid, 20_000, 100, random_state=_SEED)
        assert frame["growth_rate"].idxmax() in (0.15, 0.2, 0.25)
        assert frame.loc[0.4, "growth_rate"] < frame.loc[0.2, "growth_rate"]
        assert frame.loc[0.4, "growth_rate"] < 0.0

    def test_median_terminal_wealth_is_the_median_number_of_wins(self) -> None:
        """Sixty wins out of a hundred flips is the median of Binomial(100, 0.6)."""
        frame = simulate_kelly_growth(
            _EVEN_ODDS, [0.2], 20_000, 100, random_state=_SEED
        )
        expected = 1.2**60 * 0.8**40
        assert frame.loc[0.2, "median_terminal_wealth"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_ruin_and_loss_probabilities_match_the_exact_lattice(self) -> None:
        """Simulated frequencies agree with the exact probabilities within four SE."""
        paths = 20_000
        frame = simulate_kelly_growth(
            _EVEN_ODDS, [0.2, 0.4], paths, 100, random_state=_SEED, ruin_threshold=0.5
        )
        for fraction in (0.2, 0.4):
            up, down = 1.0 + fraction, 1.0 - fraction
            expected = {
                "probability_below_threshold": _ruin_probability(
                    up, down, _P_WIN, 100, 0.5
                ),
                "probability_of_loss": _loss_probability(up, down, _P_WIN, 100),
            }
            for column, exact in expected.items():
                standard_error = math.sqrt(exact * (1.0 - exact) / paths)
                assert (
                    abs(frame.loc[fraction, column] - exact)
                    < _SE_BANDS * standard_error
                )

    def test_over_leveraging_raises_the_probability_of_ruin(self) -> None:
        """Twice the Kelly fraction hits the ruin threshold far more often."""
        frame = simulate_kelly_growth(
            _EVEN_ODDS, [0.2, 0.4], 5_000, 100, random_state=_SEED
        )
        kelly, double = frame["probability_below_threshold"]
        # The exact lattice gives 0.415 at the Kelly fraction and 0.836 at twice it.
        assert double > kelly + 0.3

    def test_same_seed_reproduces_and_another_seed_differs(self) -> None:
        """The seed alone decides the draw."""
        args = (_EVEN_ODDS, [0.1, 0.3], 50, 20)
        first = simulate_kelly_growth(*args, random_state=1)
        again = simulate_kelly_growth(*args, random_state=1)
        other = simulate_kelly_growth(*args, random_state=2)
        pd.testing.assert_frame_equal(first, again)
        assert not first.equals(other)

    def test_every_fraction_sees_the_same_draws(self) -> None:
        """Common random numbers: growth is ordered the same way as the exact curve."""
        frame = simulate_kelly_growth(_EVEN_ODDS, [0.1, 0.2], 100, 10, random_state=3)
        repeated = simulate_kelly_growth(
            _EVEN_ODDS, [0.1, 0.2, 0.3], 100, 10, random_state=3
        )
        pd.testing.assert_frame_equal(frame, repeated.loc[[0.1, 0.2]])

    @pytest.mark.parametrize(
        "returns", [[0.1], [], [[0.1, 0.2]], [0.1, math.nan], [0.1, -2.0]]
    )
    def test_invalid_returns_raise(self, returns: list[float]) -> None:
        """Returns are at least two finite simple returns of -100% or more."""
        with pytest.raises(DataError):
            simulate_kelly_growth(returns, [0.1], 10, 5)

    @pytest.mark.parametrize("fractions", [[], [-0.1], [math.nan], [[0.1, 0.2]]])
    def test_invalid_fractions_raise(self, fractions: list[float]) -> None:
        """Fractions are a non-empty list of finite, non-negative numbers."""
        with pytest.raises(ConfigurationError, match="fractions"):
            simulate_kelly_growth(_EVEN_ODDS, fractions, 10, 5)

    @pytest.mark.parametrize(
        ("n_paths", "n_periods", "name"), [(0, 5, "n_paths"), (5, 0, "n_periods")]
    )
    def test_invalid_sizes_raise(self, n_paths: int, n_periods: int, name: str) -> None:
        """Paths and periods are at least one."""
        with pytest.raises(ConfigurationError, match=name):
            simulate_kelly_growth(_EVEN_ODDS, [0.1], n_paths, n_periods)

    @pytest.mark.parametrize("threshold", [0.0, 1.0, -0.5, 1.5, math.nan])
    def test_invalid_ruin_threshold_raises(self, threshold: float) -> None:
        """The ruin threshold is a fraction of the start strictly between 0 and 1."""
        with pytest.raises(ConfigurationError, match="ruin_threshold"):
            simulate_kelly_growth(_EVEN_ODDS, [0.1], 10, 5, ruin_threshold=threshold)

    def test_negative_seed_raises(self) -> None:
        """A negative seed is rejected up front."""
        with pytest.raises(ConfigurationError, match="random_state"):
            simulate_kelly_growth(_EVEN_ODDS, [0.1], 10, 5, random_state=-1)
