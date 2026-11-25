"""Tests for Black-Scholes pricing, Greeks and the implied-volatility solver.

Golden numbers come from the closed-form formulas evaluated independently of the
module under test with ``statistics.NormalDist`` and plain ``math``.
"""

from __future__ import annotations

import dataclasses
import math
from statistics import NormalDist

import numpy as np
import pytest

from optimizer.derivatives import (
    Greeks,
    OptionType,
    black_scholes_greeks,
    black_scholes_price,
    implied_volatility,
)
from optimizer.exceptions import ConfigurationError, ConvergenceError, DataError

_NORMAL = NormalDist()
_CALL = OptionType.CALL
_PUT = OptionType.PUT
_BOTH = (_CALL, _PUT)
_BASE: dict[str, float] = {
    "spot": 100.0,
    "strike": 100.0,
    "maturity": 1.0,
    "rate": 0.05,
    "volatility": 0.2,
    "dividend_yield": 0.0,
}
# A second contract with a dividend yield, so q enters every formula.
_YIELD: dict[str, float] = {
    "spot": 100.0,
    "strike": 95.0,
    "maturity": 0.5,
    "rate": 0.04,
    "volatility": 0.3,
    "dividend_yield": 0.03,
}


def _price(kind: OptionType = _CALL, **overrides: float) -> float:
    """Return the module price of the base contract with some fields replaced."""
    p = {**_BASE, **overrides}
    price = black_scholes_price(
        p["spot"],
        p["strike"],
        p["maturity"],
        p["rate"],
        p["volatility"],
        kind,
        p["dividend_yield"],
    )
    return float(price)


def _greeks(kind: OptionType = _CALL, **overrides: float) -> Greeks:
    """Return the module Greeks of the base contract with some fields replaced."""
    p = {**_BASE, **overrides}
    return black_scholes_greeks(
        p["spot"],
        p["strike"],
        p["maturity"],
        p["rate"],
        p["volatility"],
        kind,
        p["dividend_yield"],
    )


def _implied(price: float, kind: OptionType = _CALL, **overrides: float) -> float:
    """Return the module implied volatility of a price on the base contract."""
    p = {**_BASE, **overrides}
    return implied_volatility(
        price,
        p["spot"],
        p["strike"],
        p["maturity"],
        p["rate"],
        kind,
        p["dividend_yield"],
    )


def _reference_price(kind: OptionType, **overrides: float) -> float:
    """Return the textbook price, written per option type with NormalDist."""
    p = {**_BASE, **overrides}
    root_t = math.sqrt(p["maturity"])
    d1 = (
        math.log(p["spot"] / p["strike"])
        + (p["rate"] - p["dividend_yield"] + 0.5 * p["volatility"] ** 2) * p["maturity"]
    ) / (p["volatility"] * root_t)
    d2 = d1 - p["volatility"] * root_t
    asset_leg = p["spot"] * math.exp(-p["dividend_yield"] * p["maturity"])
    cash_leg = p["strike"] * math.exp(-p["rate"] * p["maturity"])
    if kind is _CALL:
        return asset_leg * _NORMAL.cdf(d1) - cash_leg * _NORMAL.cdf(d2)
    return cash_leg * _NORMAL.cdf(-d2) - asset_leg * _NORMAL.cdf(-d1)


def _central_difference(
    kind: OptionType, base: dict[str, float], field: str, step: float
) -> float:
    """Return the central difference of the price in one input of the contract."""
    centre = base.get(field, _BASE[field])
    up = _price(kind, **{**base, field: centre + step})
    down = _price(kind, **{**base, field: centre - step})
    return (up - down) / (2.0 * step)


class TestPrice:
    """``black_scholes_price`` against textbook values and identities."""

    def test_textbook_call(self) -> None:
        """The S=K=100, T=1, r=5%, sigma=20% call is worth 10.4506."""
        # NormalDist closed form 10.4505835722; textbook table 10.450584.
        assert _price(_CALL) == pytest.approx(10.4505835722, abs=1e-9)

    def test_textbook_put(self) -> None:
        """The same put is worth 5.5735."""
        # NormalDist closed form 5.5735260223; textbook table 5.573526.
        assert _price(_PUT) == pytest.approx(5.5735260223, abs=1e-9)

    def test_dividend_yield_prices(self) -> None:
        """A continuous dividend yield lowers the call and raises the put."""
        # NormalDist closed form, S=100 K=95 T=0.5 r=4% sigma=30% q=3%.
        assert _price(_CALL, **_YIELD) == pytest.approx(11.0718191381, abs=1e-9)
        assert _price(_PUT, **_YIELD) == pytest.approx(5.6794991419, abs=1e-9)

    def test_deep_out_of_the_money_golden(self) -> None:
        """A long-dated out-of-the-money put with q > 0 matches the closed form."""
        # NormalDist closed form, S=50 K=60 T=2 r=2% sigma=35% q=1%.
        contract = {
            "spot": 50.0,
            "strike": 60.0,
            "maturity": 2.0,
            "rate": 0.02,
            "volatility": 0.35,
            "dividend_yield": 0.01,
        }
        assert _price(_CALL, **contract) == pytest.approx(6.6481943610, abs=1e-9)
        assert _price(_PUT, **contract) == pytest.approx(15.2856270448, abs=1e-9)

    @pytest.mark.parametrize(
        "contract",
        [
            _BASE,
            _YIELD,
            {**_BASE, "dividend_yield": 0.06, "rate": 0.01},
            {**_BASE, "strike": 140.0, "maturity": 3.0, "volatility": 0.5},
            {**_BASE, "spot": 30.0, "maturity": 0.1, "volatility": 0.1},
        ],
    )
    def test_put_call_parity(self, contract: dict[str, float]) -> None:
        """C - P equals the discounted forward, S e^{-qT} - K e^{-rT}."""
        forward = contract["spot"] * math.exp(
            -contract["dividend_yield"] * contract["maturity"]
        ) - contract["strike"] * math.exp(-contract["rate"] * contract["maturity"])
        gap = _price(_CALL, **contract) - _price(_PUT, **contract)
        assert gap == pytest.approx(forward, abs=1e-10)

    @pytest.mark.parametrize("kind", _BOTH)
    @pytest.mark.parametrize("strike", [70.0, 100.0, 130.0])
    @pytest.mark.parametrize("dividend_yield", [0.0, 0.04])
    def test_matches_the_per_type_textbook_formula(
        self, kind: OptionType, strike: float, dividend_yield: float
    ) -> None:
        """The module agrees with a separately written formula on a grid."""
        overrides = {"strike": strike, "dividend_yield": dividend_yield}
        assert _price(kind, **overrides) == pytest.approx(
            _reference_price(kind, **overrides), abs=1e-10
        )

    def test_scalar_inputs_return_a_python_float(self) -> None:
        """Scalar inputs give a float, not a zero-dimensional array."""
        price = black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2, _CALL)
        assert type(price) is float
        numpy_scalar = black_scholes_price(
            np.float64(100.0), 100.0, 1.0, 0.05, 0.2, _PUT
        )
        assert type(numpy_scalar) is float

    def test_option_type_accepts_its_string_value(self) -> None:
        """``"call"`` and ``"put"`` are accepted in place of the enum."""
        as_text = black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2, "put")
        assert as_text == pytest.approx(_price(_PUT))

    @pytest.mark.parametrize("kind", _BOTH)
    def test_intrinsic_value_at_maturity_zero(self, kind: OptionType) -> None:
        """At expiry the price is the undiscounted payoff, in and out of the money."""
        in_the_money = 110.0 if kind is _CALL else 90.0
        out_of_the_money = 90.0 if kind is _CALL else 110.0
        assert _price(kind, spot=in_the_money, maturity=0.0) == pytest.approx(10.0)
        assert _price(kind, spot=out_of_the_money, maturity=0.0) == 0.0

    def test_arrays_broadcast_to_the_scalar_results(self) -> None:
        """Every element of a broadcast grid equals the matching scalar price."""
        spots = np.array([[80.0], [100.0], [120.0]])
        vols = np.array([[0.1, 0.2, 0.4]])
        grid = black_scholes_price(spots, 100.0, 1.0, 0.05, vols, _CALL, 0.02)
        assert isinstance(grid, np.ndarray)
        assert grid.shape == (3, 3)
        for i, spot in enumerate((80.0, 100.0, 120.0)):
            for j, vol in enumerate((0.1, 0.2, 0.4)):
                expected = _price(_CALL, spot=spot, volatility=vol, dividend_yield=0.02)
                assert grid[i, j] == pytest.approx(expected, abs=1e-12)

    def test_array_with_a_zero_maturity_entry(self) -> None:
        """Expired and live entries of one array are priced side by side."""
        maturities = np.array([0.0, 1.0])
        result = black_scholes_price(110.0, 100.0, maturities, 0.05, 0.2, _CALL)
        assert isinstance(result, np.ndarray)
        assert result[0] == pytest.approx(10.0)
        assert result[1] == pytest.approx(_price(_CALL, spot=110.0), abs=1e-12)

    def test_price_rises_with_spot_and_volatility_for_a_call(self) -> None:
        """A call gains from a higher spot and from a higher volatility."""
        spots = [_price(_CALL, spot=s) for s in (80.0, 90.0, 100.0, 110.0)]
        vols = [_price(_CALL, volatility=v) for v in (0.1, 0.2, 0.3, 0.4)]
        assert spots == sorted(spots)
        assert vols == sorted(vols)

    def test_put_price_falls_with_spot(self) -> None:
        """A put loses value as the spot rises."""
        puts = [_price(_PUT, spot=s) for s in (80.0, 90.0, 100.0, 110.0)]
        assert puts == sorted(puts, reverse=True)

    @pytest.mark.parametrize("kind", _BOTH)
    def test_price_is_at_least_the_discounted_intrinsic_value(
        self, kind: OptionType
    ) -> None:
        """No European option trades below its no-arbitrage lower bound."""
        for spot in (40.0, 80.0, 100.0, 140.0, 300.0):
            asset_leg = spot * math.exp(-0.02)
            cash_leg = 100.0 * math.exp(-0.05)
            sign = 1.0 if kind is _CALL else -1.0
            floor = max(sign * (asset_leg - cash_leg), 0.0)
            assert _price(kind, spot=spot, dividend_yield=0.02) >= floor - 1e-12

    def test_huge_volatility_call_approaches_the_spot(self) -> None:
        """When volatility explodes the call is worth the (dividend-adjusted) spot."""
        assert _price(_CALL, volatility=50.0) == pytest.approx(100.0, abs=1e-6)

    def test_tiny_volatility_call_approaches_the_discounted_forward(self) -> None:
        """With no randomness left a deep in-the-money call is S - K e^{-rT}."""
        value = _price(_CALL, spot=150.0, volatility=1e-4)
        assert value == pytest.approx(150.0 - 100.0 * math.exp(-0.05), abs=1e-9)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("spot", 0.0),
            ("spot", -1.0),
            ("spot", math.nan),
            ("spot", math.inf),
            ("strike", 0.0),
            ("strike", -5.0),
            ("maturity", -0.1),
            ("maturity", math.nan),
            ("volatility", 0.0),
            ("volatility", -0.2),
            ("rate", math.nan),
            ("rate", math.inf),
            ("dividend_yield", math.nan),
        ],
    )
    def test_invalid_scalar_inputs_raise(self, field: str, value: float) -> None:
        """Non-positive, non-finite or negative-maturity inputs are rejected."""
        with pytest.raises(ConfigurationError):
            _price(_CALL, **{field: value})

    def test_one_bad_array_element_raises(self) -> None:
        """A single invalid entry in an array invalidates the whole call."""
        with pytest.raises(ConfigurationError, match="spot"):
            black_scholes_price(np.array([100.0, -1.0]), 100.0, 1.0, 0.05, 0.2, _CALL)

    def test_unknown_option_type_raises(self) -> None:
        """Anything other than a call or a put is a configuration error."""
        with pytest.raises(ConfigurationError, match="option_type"):
            black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2, "straddle")

    def test_non_numeric_input_raises(self) -> None:
        """Text that is not a number is rejected by name."""
        with pytest.raises(ConfigurationError, match="spot must be numeric"):
            black_scholes_price("abc", 100.0, 1.0, 0.05, 0.2, _CALL)  # type: ignore[arg-type]

    def test_non_broadcastable_shapes_raise(self) -> None:
        """Arrays that cannot be broadcast together are a data error."""
        with pytest.raises(DataError, match="broadcast"):
            black_scholes_price(
                np.ones(3) * 100.0, np.ones(2) * 100.0, 1.0, 0.05, 0.2, _CALL
            )


class TestGreeks:
    """``black_scholes_greeks`` against textbook values and finite differences."""

    def test_textbook_call_greeks(self) -> None:
        """The reference call: delta .6368, gamma .01876, vega 37.52, theta -6.414."""
        # NormalDist closed form; the rounded textbook values (0.636831, 0.018762,
        # 37.5240, -6.4140, 53.2325) agree to every quoted digit.
        greeks = _greeks(_CALL)
        assert greeks.delta == pytest.approx(0.6368306512, abs=1e-9)
        assert greeks.gamma == pytest.approx(0.0187620173, abs=1e-9)
        assert greeks.vega == pytest.approx(37.5240346917, abs=1e-7)
        assert greeks.theta == pytest.approx(-6.4140275464, abs=1e-7)
        assert greeks.rho == pytest.approx(53.2324815454, abs=1e-7)

    def test_textbook_put_greeks(self) -> None:
        """The reference put has the call's gamma and vega, and its own delta."""
        # NormalDist closed form for the reference put.
        greeks = _greeks(_PUT)
        assert greeks.delta == pytest.approx(-0.3631693488, abs=1e-9)
        assert greeks.gamma == pytest.approx(0.0187620173, abs=1e-9)
        assert greeks.vega == pytest.approx(37.5240346917, abs=1e-7)
        assert greeks.theta == pytest.approx(-1.6578804239, abs=1e-7)
        assert greeks.rho == pytest.approx(-41.8904609047, abs=1e-7)

    def test_dividend_yield_call_greeks(self) -> None:
        """A continuous dividend yield enters delta, theta and rho."""
        # NormalDist closed form, S=100 K=95 T=0.5 r=4% sigma=30% q=3%.
        greeks = _greeks(_CALL, **_YIELD)
        assert greeks.delta == pytest.approx(0.6352428104, abs=1e-9)
        assert greeks.gamma == pytest.approx(0.0172914285, abs=1e-9)
        assert greeks.vega == pytest.approx(25.9371428044, abs=1e-7)
        assert greeks.theta == pytest.approx(-7.9735128862, abs=1e-7)
        assert greeks.rho == pytest.approx(26.2262309533, abs=1e-7)

    def test_dividend_yield_put_greeks(self) -> None:
        """The mirror formulas hold for a put with a dividend yield."""
        # NormalDist closed form, same contract as the call above.
        greeks = _greeks(_PUT, **_YIELD)
        assert greeks.delta == pytest.approx(-0.3498691292, abs=1e-9)
        assert greeks.theta == pytest.approx(-7.2040937465, abs=1e-7)
        assert greeks.rho == pytest.approx(-20.3332060288, abs=1e-7)

    @pytest.mark.parametrize("spot", [40.0, 80.0, 100.0, 120.0, 250.0])
    @pytest.mark.parametrize("dividend_yield", [0.0, 0.05])
    def test_delta_gamma_vega_ranges(self, spot: float, dividend_yield: float) -> None:
        """Call delta is in [0, 1], put delta in [-1, 0], gamma and vega positive."""
        call = _greeks(_CALL, spot=spot, dividend_yield=dividend_yield)
        put = _greeks(_PUT, spot=spot, dividend_yield=dividend_yield)
        assert 0.0 <= call.delta <= 1.0
        assert -1.0 <= put.delta <= 0.0
        assert call.gamma > 0.0
        assert call.vega > 0.0

    def test_call_and_put_share_gamma_and_vega(self) -> None:
        """Gamma and vega do not depend on the option type."""
        call, put = _greeks(_CALL, **_YIELD), _greeks(_PUT, **_YIELD)
        assert call.gamma == pytest.approx(put.gamma, abs=1e-14)
        assert call.vega == pytest.approx(put.vega, abs=1e-12)

    def test_deltas_differ_by_the_dividend_discount(self) -> None:
        """Differentiating put-call parity gives delta_C - delta_P = e^{-qT}."""
        call, put = _greeks(_CALL, **_YIELD), _greeks(_PUT, **_YIELD)
        expected = math.exp(-_YIELD["dividend_yield"] * _YIELD["maturity"])
        assert call.delta - put.delta == pytest.approx(expected, abs=1e-12)

    @pytest.mark.parametrize("kind", _BOTH)
    @pytest.mark.parametrize("strike", [90.0, 100.0, 110.0])
    @pytest.mark.parametrize("dividend_yield", [0.0, 0.03])
    def test_greeks_match_central_finite_differences(
        self, kind: OptionType, strike: float, dividend_yield: float
    ) -> None:
        """Each Greek equals the central difference of the price it derives from."""
        base = {
            "strike": strike,
            "dividend_yield": dividend_yield,
            "maturity": 0.75,
            "volatility": 0.3,
            "rate": 0.04,
        }
        greeks = _greeks(kind, **base)
        spot_step = 0.05
        centre = _price(kind, **base)
        up = _price(kind, **base, spot=_BASE["spot"] + spot_step)
        down = _price(kind, **base, spot=_BASE["spot"] - spot_step)
        assert greeks.delta == pytest.approx(
            _central_difference(kind, base, "spot", spot_step), rel=1e-6, abs=1e-8
        )
        assert greeks.gamma == pytest.approx(
            (up - 2.0 * centre + down) / spot_step**2, abs=1e-6
        )
        assert greeks.vega == pytest.approx(
            _central_difference(kind, base, "volatility", 1e-5), rel=1e-6
        )
        assert greeks.theta == pytest.approx(
            -_central_difference(kind, base, "maturity", 1e-5), rel=1e-6
        )
        assert greeks.rho == pytest.approx(
            _central_difference(kind, base, "rate", 1e-5), rel=1e-6
        )

    def test_arrays_broadcast_to_the_scalar_results(self) -> None:
        """Every Greek of an array call equals the scalar call element by element."""
        spots = np.array([80.0, 100.0, 120.0])
        greeks = black_scholes_greeks(spots, 100.0, 1.0, 0.05, 0.2, _CALL, 0.01)
        assert isinstance(greeks.delta, np.ndarray)
        assert greeks.theta.shape == (3,)
        for i, spot in enumerate(spots):
            scalar = _greeks(_CALL, spot=float(spot), dividend_yield=0.01)
            assert greeks.delta[i] == pytest.approx(scalar.delta, abs=1e-14)
            assert greeks.gamma[i] == pytest.approx(scalar.gamma, abs=1e-14)
            assert greeks.vega[i] == pytest.approx(scalar.vega, abs=1e-12)
            assert greeks.theta[i] == pytest.approx(scalar.theta, abs=1e-12)
            assert greeks.rho[i] == pytest.approx(scalar.rho, abs=1e-12)

    def test_scalar_inputs_return_python_floats(self) -> None:
        """A scalar call yields five plain floats."""
        greeks = _greeks(_CALL)
        assert all(
            type(getattr(greeks, field.name)) is float
            for field in dataclasses.fields(greeks)
        )

    def test_greeks_are_immutable(self) -> None:
        """The result dataclass is frozen."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            _greeks(_CALL).delta = 0.0  # type: ignore[misc]

    def test_zero_maturity_is_rejected(self) -> None:
        """Greeks are undefined at expiry, so the call refuses it."""
        with pytest.raises(ConfigurationError, match="maturity"):
            _greeks(_CALL, maturity=0.0)

    @pytest.mark.parametrize(
        ("field", "value"),
        [("spot", -1.0), ("strike", 0.0), ("volatility", 0.0), ("rate", math.nan)],
    )
    def test_invalid_inputs_raise(self, field: str, value: float) -> None:
        """The same input checks as the price apply."""
        with pytest.raises(ConfigurationError):
            _greeks(_CALL, **{field: value})


class TestImpliedVolatility:
    """``implied_volatility`` inverts ``black_scholes_price``."""

    def test_recovers_the_textbook_volatility(self) -> None:
        """The textbook call price 10.4505835722 implies exactly 20%."""
        # NormalDist closed form of the reference call.
        assert _implied(10.4505835722, _CALL) == pytest.approx(0.2, abs=1e-8)

    @pytest.mark.parametrize("kind", _BOTH)
    @pytest.mark.parametrize("volatility", [0.1, 0.2, 0.4, 0.8])
    @pytest.mark.parametrize("strike", [90.0, 100.0, 110.0])
    @pytest.mark.parametrize("dividend_yield", [0.0, 0.03])
    def test_round_trip_on_a_grid(
        self,
        kind: OptionType,
        volatility: float,
        strike: float,
        dividend_yield: float,
    ) -> None:
        """Pricing then inverting returns the volatility to 1e-8."""
        price = _price(
            kind, volatility=volatility, strike=strike, dividend_yield=dividend_yield
        )
        recovered = _implied(price, kind, strike=strike, dividend_yield=dividend_yield)
        assert recovered == pytest.approx(volatility, abs=1e-8)

    def test_returns_a_python_float(self) -> None:
        """The solver returns a plain float."""
        assert type(_implied(10.45, _CALL)) is float

    def test_option_type_accepts_its_string_value(self) -> None:
        """``"put"`` is accepted in place of the enum."""
        price = _price(_PUT)
        assert implied_volatility(
            price, 100.0, 100.0, 1.0, 0.05, "put"
        ) == pytest.approx(0.2, abs=1e-8)

    @pytest.mark.parametrize("kind", _BOTH)
    def test_price_below_the_discounted_intrinsic_value_raises(
        self, kind: OptionType
    ) -> None:
        """A price under the no-arbitrage floor has no volatility."""
        # Deep in the money: the floor is S - K e^{-rT} = 19.15 for the call and
        # K e^{-rT} - S = 10.85 for the put.
        spot, strike = (100.0, 85.0) if kind is _CALL else (70.0, 85.0)
        floor = abs(spot - strike * math.exp(-0.05))
        with pytest.raises(DataError, match="below"):
            _implied(floor - 0.5, kind, spot=spot, strike=strike)

    def test_negative_price_raises(self) -> None:
        """A negative price violates the zero floor."""
        with pytest.raises(DataError, match="below"):
            _implied(-0.01, _CALL)

    def test_call_price_above_the_spot_raises(self) -> None:
        """A call can never be worth more than S e^{-qT}."""
        with pytest.raises(DataError, match="above"):
            _implied(100.5, _CALL)
        with pytest.raises(DataError, match="above"):
            _implied(98.0, _CALL, dividend_yield=0.1)

    def test_put_price_above_the_discounted_strike_raises(self) -> None:
        """A put can never be worth more than K e^{-rT}."""
        with pytest.raises(DataError, match="above"):
            _implied(100.0 * math.exp(-0.05) + 0.01, _PUT)

    def test_non_finite_price_raises(self) -> None:
        """NaN and infinite prices are data errors."""
        with pytest.raises(DataError):
            _implied(math.nan, _CALL)
        with pytest.raises(DataError):
            _implied(math.inf, _CALL)

    def test_volatility_above_the_bracket_raises_convergence_error(self) -> None:
        """A price inside the bounds but needing sigma > 500% is not bracketed."""
        # With r = q = 0 the call is bounded by S = 100; the price at sigma = 5 is
        # 98.76, so 99.5 lies between that and the bound.
        with pytest.raises(ConvergenceError, match="bracket"):
            _implied(99.5, _CALL, rate=0.0)

    def test_volatility_below_the_bracket_raises_convergence_error(self) -> None:
        """A price barely above the floor needing sigma < 1e-6 is not bracketed."""
        with pytest.raises(ConvergenceError, match="bracket"):
            _implied(1e-9, _CALL, rate=0.0)

    def test_solver_failure_is_reported_as_a_convergence_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A root finder that gives up surfaces as ``ConvergenceError``."""

        def give_up(*args: object, **kwargs: object) -> float:
            """Stand in for ``brentq`` running out of iterations."""
            raise RuntimeError("no convergence")

        monkeypatch.setattr("optimizer.derivatives._black_scholes.brentq", give_up)
        with pytest.raises(ConvergenceError, match="did not converge"):
            _implied(10.45, _CALL)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("maturity", 0.0),
            ("maturity", -1.0),
            ("spot", 0.0),
            ("strike", -1.0),
            ("rate", math.nan),
            ("dividend_yield", math.inf),
        ],
    )
    def test_invalid_contract_raises(self, field: str, value: float) -> None:
        """The contract must be priceable: a positive maturity, finite inputs."""
        with pytest.raises(ConfigurationError):
            _implied(10.0, _CALL, **{field: value})

    def test_unknown_option_type_raises(self) -> None:
        """Anything other than a call or a put is a configuration error."""
        with pytest.raises(ConfigurationError, match="option_type"):
            implied_volatility(10.0, 100.0, 100.0, 1.0, 0.05, "straddle")
