"""Tests for ``OptionType.payoff`` and the unchecked kernel behind it.

The expected payoffs are worked by hand from ``max(S - K, 0)`` and ``max(K - S, 0)``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from optimizer.derivatives import OptionType
from optimizer.derivatives._config import exercise_value
from optimizer.exceptions import ConfigurationError, DataError

_CALL = OptionType.CALL
_PUT = OptionType.PUT


class TestPayoff:
    """``OptionType.payoff``: the exercise value at expiry."""

    def test_a_call_pays_the_excess_of_spot_over_strike(self) -> None:
        """A call struck at 100 with the spot at 112 is worth 12."""
        assert float(_CALL.payoff(112.0, 100.0)) == pytest.approx(12.0)

    def test_a_put_pays_the_excess_of_strike_over_spot(self) -> None:
        """A put struck at 100 with the spot at 85 is worth 15."""
        assert float(_PUT.payoff(85.0, 100.0)) == pytest.approx(15.0)

    @pytest.mark.parametrize(("kind", "spot"), [(_CALL, 90.0), (_PUT, 110.0)])
    def test_an_out_of_the_money_option_is_worth_nothing(
        self, kind: OptionType, spot: float
    ) -> None:
        """The holder simply lets an out-of-the-money option expire."""
        assert float(kind.payoff(spot, 100.0)) == 0.0

    def test_a_grid_of_spots_broadcasts_against_one_strike(self) -> None:
        """Each spot is paid against the same strike, in order."""
        spots = np.array([80.0, 100.0, 120.0])
        np.testing.assert_allclose(_CALL.payoff(spots, 100.0), [0.0, 0.0, 20.0])
        np.testing.assert_allclose(_PUT.payoff(spots, 100.0), [20.0, 0.0, 0.0])

    def test_a_list_is_accepted_and_the_result_is_a_float_array(self) -> None:
        """Plain Python numbers are converted, and the result is float64."""
        payoff = _CALL.payoff([95.0, 105.0], 100.0)
        assert payoff.dtype == np.float64
        np.testing.assert_allclose(payoff, [0.0, 5.0])

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, 0.0, -1.0, "abc"])
    @pytest.mark.parametrize("name", ["spot", "strike"])
    def test_a_price_that_is_not_positive_and_finite_raises(
        self, name: str, bad: object
    ) -> None:
        """A NaN or negative price used to return a payoff without a word."""
        arguments: dict[str, Any] = {"spot": 105.0, "strike": 100.0}
        arguments[name] = bad
        with pytest.raises(ConfigurationError, match=f"{name} must be"):
            _CALL.payoff(**arguments)

    def test_one_bad_element_in_an_array_raises(self) -> None:
        """A single NaN among valid spots is enough to refuse the whole array."""
        with pytest.raises(ConfigurationError, match="spot must be"):
            _PUT.payoff(np.array([90.0, math.nan]), 100.0)

    def test_inputs_that_cannot_be_broadcast_raise(self) -> None:
        """Three spots against four strikes have no common shape."""
        with pytest.raises(DataError, match="broadcast"):
            _CALL.payoff(np.ones(3) * 100.0, np.ones(4) * 100.0)


class TestExerciseValue:
    """``exercise_value``: the unchecked kernel the pricers use internally."""

    @pytest.mark.parametrize("kind", [_CALL, _PUT])
    def test_it_matches_the_public_payoff_on_valid_inputs(
        self, kind: OptionType
    ) -> None:
        """The kernel and the checked method agree wherever the inputs are valid."""
        spots = np.linspace(60.0, 140.0, 9)
        np.testing.assert_allclose(
            exercise_value(kind, spots, 100.0), kind.payoff(spots, 100.0)
        )

    def test_it_does_not_check_its_inputs(self) -> None:
        """Validation happens once at the public entry point, not on every call."""
        assert math.isnan(float(exercise_value(_CALL, math.nan, 100.0)))
