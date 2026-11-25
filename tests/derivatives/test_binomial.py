"""Tests for the Cox-Ross-Rubinstein binomial tree.

Golden numbers come from an independent pure-Python tree (plain lists, no numpy,
built level by level) and from the Black-Scholes closed form evaluated with
``statistics.NormalDist``; neither shares code with the module under test.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pytest

from optimizer.derivatives import OptionType, binomial_price, black_scholes_price
from optimizer.exceptions import ConfigurationError

_NORMAL = NormalDist()
_CALL = OptionType.CALL
_PUT = OptionType.PUT
_BASE: dict[str, float] = {
    "spot": 100.0,
    "strike": 100.0,
    "maturity": 1.0,
    "rate": 0.05,
    "volatility": 0.2,
    "dividend_yield": 0.0,
}
# S=K=100, T=1, r=5%, sigma=20%: Black-Scholes call from NormalDist.
_BS_CALL = 10.4505835722


def _tree(
    kind: OptionType,
    steps: int = 200,
    *,
    american: bool = False,
    **overrides: float,
) -> float:
    """Return the module tree price of the base contract with fields replaced."""
    p = {**_BASE, **overrides}
    return binomial_price(
        p["spot"],
        p["strike"],
        p["maturity"],
        p["rate"],
        p["volatility"],
        kind,
        steps=steps,
        dividend_yield=p["dividend_yield"],
        american=american,
    )


class TestEuropeanTree:
    """The European tree against closed forms and independent trees."""

    def test_one_step_tree_matches_the_hand_computation(self) -> None:
        """With one step the price is e^{-r dt} p (S u - K)^+, worked out by hand."""
        up = math.exp(0.2)
        down = 1.0 / up
        prob = (math.exp(0.05) - down) / (up - down)
        expected = math.exp(-0.05) * prob * (100.0 * up - 100.0)
        assert _tree(_CALL, steps=1) == pytest.approx(expected, abs=1e-12)
        assert expected == pytest.approx(12.162285, abs=1e-6)

    def test_fifty_step_prices_match_an_independent_tree(self) -> None:
        """A pure-Python CRR tree with 50 steps gives the same call and put."""
        # Independent list-based CRR, S=K=100 T=1 r=5% sigma=20%, n=50.
        assert _tree(_CALL, steps=50) == pytest.approx(10.410691540732644, abs=1e-9)
        assert _tree(_PUT, steps=50) == pytest.approx(5.533633990803824, abs=1e-9)

    def test_dividend_yield_put_matches_an_independent_tree(self) -> None:
        """The continuous dividend yield shifts the risk-neutral probability."""
        # Independent list-based CRR, S=100 K=95 T=0.5 r=4% sigma=30% q=3%, n=100.
        price = _tree(
            _PUT,
            steps=100,
            strike=95.0,
            maturity=0.5,
            rate=0.04,
            volatility=0.3,
            dividend_yield=0.03,
        )
        assert price == pytest.approx(5.685124950179273, abs=1e-9)

    def test_thousand_steps_are_close_to_black_scholes(self) -> None:
        """At 1000 steps the call is within one cent of the closed form."""
        assert abs(_tree(_CALL, steps=1000) - _BS_CALL) < 0.01

    def test_error_shrinks_as_the_tree_gets_finer(self) -> None:
        """The error against Black-Scholes falls at every doubling from 50 to 800."""
        errors = [
            abs(_tree(_CALL, steps=n) - _BS_CALL) for n in (50, 100, 200, 400, 800)
        ]
        assert errors == sorted(errors, reverse=True)
        assert errors[-1] < errors[0] / 8.0

    @pytest.mark.parametrize("dividend_yield", [0.0, 0.04])
    def test_put_call_parity_holds_exactly_in_the_tree(
        self, dividend_yield: float
    ) -> None:
        """The CRR probability makes E[S_T] = S e^{(r-q)T}, so parity is exact."""
        call = _tree(_CALL, steps=50, strike=110.0, dividend_yield=dividend_yield)
        put = _tree(_PUT, steps=50, strike=110.0, dividend_yield=dividend_yield)
        forward = 100.0 * math.exp(-dividend_yield) - 110.0 * math.exp(-0.05)
        assert call - put == pytest.approx(forward, abs=1e-9)

    def test_american_flag_defaults_to_european(self) -> None:
        """Without ``american`` a deep in-the-money put keeps its time value."""
        assert _tree(_PUT, steps=50, spot=80.0) == pytest.approx(16.95319941828835)

    @pytest.mark.parametrize("kind", [_CALL, _PUT])
    def test_zero_maturity_returns_the_intrinsic_value(self, kind: OptionType) -> None:
        """At expiry there is nothing to model: the price is the payoff."""
        in_the_money = 110.0 if kind is _CALL else 90.0
        assert _tree(kind, spot=in_the_money, maturity=0.0) == pytest.approx(10.0)
        assert _tree(kind, american=True, spot=in_the_money, maturity=0.0) == (
            pytest.approx(10.0)
        )


class TestAmericanTree:
    """Early exercise against European prices and independent trees."""

    def test_american_put_is_at_least_the_european_put(self) -> None:
        """The right to exercise early can only add value."""
        for strike in (80.0, 100.0, 120.0):
            european = _tree(_PUT, steps=300, strike=strike)
            american = _tree(_PUT, steps=300, strike=strike, american=True)
            assert american >= european - 1e-12

    def test_deep_in_the_money_put_is_strictly_worth_more_early(self) -> None:
        """With r > 0 a deep put is exercised at once and beats the European."""
        # Independent list-based CRR, S=80 K=100, 500 steps: the American value is
        # the intrinsic 20.0 and the European value is 16.983137118259393.
        american = _tree(_PUT, steps=500, american=True, spot=80.0)
        european = _tree(_PUT, steps=500, spot=80.0)
        assert american == pytest.approx(20.0, abs=1e-12)
        assert european == pytest.approx(16.983137118259393, abs=1e-9)
        assert american > european + 3.0

    def test_at_the_money_put_matches_an_independent_tree(self) -> None:
        """An interior early-exercise boundary is reproduced by a list-based tree."""
        # Independent pure-Python CRR American put, S=K=100 T=1 r=5% sigma=20%.
        assert _tree(_PUT, steps=500, american=True) == pytest.approx(
            6.088810110703388, abs=1e-8
        )

    def test_ten_thousand_steps_match_a_high_step_reference(self) -> None:
        """A very fine tree stays numerically stable and equals the reference."""
        # Independent pure-Python CRR American put with 10000 steps (scratch run).
        price = _tree(_PUT, steps=10_000, american=True)
        assert price == pytest.approx(6.090295412881005, abs=1e-8)

    def test_american_put_never_beats_the_strike(self) -> None:
        """A deep in-the-money put is worth its intrinsic value, never more than K."""
        value = _tree(_PUT, steps=200, american=True, spot=40.0)
        assert value == pytest.approx(60.0, abs=1e-12)
        assert _tree(_PUT, steps=200, american=True, spot=1000.0) >= 0.0

    def test_american_call_without_dividends_equals_the_european_call(self) -> None:
        """With q = 0 early exercise of a call is never optimal."""
        # Independent list-based CRR, S=K=100 T=1 r=5% sigma=20%, n=500.
        american = _tree(_CALL, steps=500, american=True)
        european = _tree(_CALL, steps=500)
        assert american == pytest.approx(european, abs=1e-10)
        assert american == pytest.approx(10.44658513644654, abs=1e-9)

    def test_american_call_with_a_high_dividend_yield_is_worth_more_early(
        self,
    ) -> None:
        """A large q makes holding the call costly, so early exercise has value."""
        # Independent list-based CRR, S=100 K=90 T=1 r=5% sigma=20% q=8%, n=500.
        american = _tree(
            _CALL, steps=500, american=True, strike=90.0, dividend_yield=0.08
        )
        european = _tree(_CALL, steps=500, strike=90.0, dividend_yield=0.08)
        assert american == pytest.approx(11.912903210832193, abs=1e-9)
        assert european == pytest.approx(10.9307419690381, abs=1e-9)

    def test_american_put_without_interest_equals_the_european_put(self) -> None:
        """With r = q = 0 the spot is a martingale, so waiting is never worse."""
        american = _tree(_PUT, steps=200, american=True, rate=0.0)
        european = _tree(_PUT, steps=200, rate=0.0)
        assert american == pytest.approx(european, abs=1e-10)


class TestValidation:
    """Inputs the tree cannot price are rejected."""

    def test_probability_above_one_raises(self) -> None:
        """A large drift against a tiny volatility makes p exceed one."""
        with pytest.raises(ConfigurationError, match="probability"):
            _tree(_CALL, steps=10, rate=0.5, volatility=0.01)

    def test_probability_below_zero_raises(self) -> None:
        """A very negative drift makes p negative."""
        with pytest.raises(ConfigurationError, match="probability"):
            _tree(_CALL, steps=10, rate=-1.0, volatility=0.01)

    def test_volatility_too_small_for_the_step_raises(self) -> None:
        """When the up and down moves round to the same number there is no tree."""
        with pytest.raises(ConfigurationError, match="too small"):
            _tree(_CALL, steps=10, volatility=1e-20)

    def test_overflowing_tree_raises(self) -> None:
        """A volatility so large that node prices overflow is reported, not priced."""
        with pytest.raises(ConfigurationError, match="overflow"):
            _tree(_CALL, steps=200, volatility=500.0)

    @pytest.mark.parametrize("steps", [0, -3, 2.5])
    def test_invalid_step_counts_raise(self, steps: float) -> None:
        """The tree needs a whole, positive number of steps."""
        with pytest.raises(ConfigurationError, match="steps"):
            _tree(_CALL, steps=steps)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("spot", 0.0),
            ("spot", -1.0),
            ("strike", 0.0),
            ("maturity", -0.5),
            ("volatility", 0.0),
            ("rate", math.nan),
            ("dividend_yield", math.inf),
        ],
    )
    def test_invalid_contract_raises(self, field: str, value: float) -> None:
        """Non-positive spot, strike or volatility, a negative maturity, or a
        non-finite input is rejected."""
        params = {**_BASE, field: value}
        with pytest.raises(ConfigurationError):
            binomial_price(
                params["spot"],
                params["strike"],
                params["maturity"],
                params["rate"],
                params["volatility"],
                _CALL,
                dividend_yield=params["dividend_yield"],
            )

    def test_array_inputs_raise(self) -> None:
        """The tree prices one contract at a time."""
        with pytest.raises(ConfigurationError, match="scalar"):
            binomial_price(np.array([100.0, 110.0]), 100.0, 1.0, 0.05, 0.2, _CALL)

    def test_unknown_option_type_raises(self) -> None:
        """Anything other than a call or a put is a configuration error."""
        with pytest.raises(ConfigurationError, match="option_type"):
            binomial_price(100.0, 100.0, 1.0, 0.05, 0.2, "straddle")

    def test_matches_black_scholes_for_string_option_type(self) -> None:
        """A string option type works and converges to the closed form."""
        tree = binomial_price(100.0, 100.0, 1.0, 0.05, 0.2, "put", steps=800)
        closed = black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2, "put")
        assert tree == pytest.approx(closed, abs=0.01)
