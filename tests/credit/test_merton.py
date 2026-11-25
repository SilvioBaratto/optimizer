"""Tests for the Merton structural credit model.

Golden numbers come from calculations that do not touch the module under test:
the forward Merton formulas evaluated with ``statistics.NormalDist``, and a
plain-``math`` solver (bisection on the asset value inside a fixed-point iteration
on the asset volatility) run in a scratch script.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from statistics import NormalDist

import pytest

from optimizer.credit import MertonConfig, MertonResult, solve_merton
from optimizer.credit import _merton as merton_module
from optimizer.exceptions import ConfigurationError, ConvergenceError

_CDF = NormalDist().cdf

# Equity value, equity volatility, debt face value, risk-free rate, maturity.
_HULL = (3.0, 0.80, 10.0, 0.05, 1.0)


def _forward(
    asset_value: float,
    asset_volatility: float,
    debt: float,
    rate: float,
    maturity: float,
) -> tuple[float, float]:
    """Return the equity value and equity volatility the Merton formulas imply."""
    root_t = math.sqrt(maturity)
    d1 = (
        math.log(asset_value / debt) + (rate + asset_volatility**2 / 2) * maturity
    ) / (asset_volatility * root_t)
    d2 = d1 - asset_volatility * root_t
    equity = asset_value * _CDF(d1) - debt * math.exp(-rate * maturity) * _CDF(d2)
    return equity, _CDF(d1) * asset_volatility * asset_value / equity


def _hull_result(config: MertonConfig | None = None) -> MertonResult:
    """Solve the textbook example."""
    return solve_merton(*_HULL[:4], config=config)


class TestMertonConfig:
    """``MertonConfig`` defaults and validation."""

    def test_defaults(self) -> None:
        """The defaults are a one-year, risk-neutral, tight-tolerance setup."""
        config = MertonConfig()
        assert config.maturity == 1.0
        assert config.solver_tolerance == 1e-10
        assert config.max_iterations == 200
        assert config.asset_drift is None

    def test_is_frozen(self) -> None:
        """A configuration cannot change after it was built."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            MertonConfig().maturity = 2.0  # type: ignore[misc]

    @pytest.mark.parametrize("maturity", [0.0, -1.0, math.nan, math.inf])
    def test_rejects_a_bad_maturity(self, maturity: float) -> None:
        """The debt matures in a positive, finite number of years."""
        with pytest.raises(ConfigurationError, match="maturity"):
            MertonConfig(maturity=maturity)

    @pytest.mark.parametrize("tolerance", [0.0, -1e-10, math.nan, math.inf])
    def test_rejects_a_bad_tolerance(self, tolerance: float) -> None:
        """The tolerance is a positive, finite residual size."""
        with pytest.raises(ConfigurationError, match="solver_tolerance"):
            MertonConfig(solver_tolerance=tolerance)

    @pytest.mark.parametrize("max_iterations", [0, -5])
    def test_rejects_a_bad_iteration_budget(self, max_iterations: int) -> None:
        """The solver needs at least one evaluation."""
        with pytest.raises(ConfigurationError, match="max_iterations"):
            MertonConfig(max_iterations=max_iterations)

    @pytest.mark.parametrize("drift", [math.nan, math.inf, -math.inf])
    def test_rejects_a_non_finite_drift(self, drift: float) -> None:
        """A drift, when given, is a finite rate."""
        with pytest.raises(ConfigurationError, match="asset_drift"):
            MertonConfig(asset_drift=drift)

    def test_accepts_a_negative_drift(self) -> None:
        """Assets can lose value on average."""
        assert MertonConfig(asset_drift=-0.05).asset_drift == -0.05


class TestRoundTrip:
    """Feeding the model its own forward output recovers the firm."""

    def test_recovers_the_asset_value_and_volatility(self) -> None:
        """A firm worth 12.4 with 21.23% volatility is found again to 1e-6."""
        equity, equity_vol = _forward(12.4, 0.2123, 10.0, 0.05, 1.0)
        result = solve_merton(equity, equity_vol, 10.0, 0.05)
        assert result.asset_value == pytest.approx(12.4, abs=1e-6)
        assert result.asset_volatility == pytest.approx(0.2123, abs=1e-6)

    @pytest.mark.parametrize(
        ("asset_value", "asset_vol", "debt", "rate", "maturity"),
        [
            (25.0, 0.35, 20.0, -0.01, 2.0),
            (64.0, 0.17, 40.0, 0.03, 5.0),
            (100.0, 0.60, 30.0, 0.00, 0.5),
            (5.0, 0.25, 4.9, 0.08, 10.0),
            (8.0, 0.90, 10.0, 0.04, 0.25),
        ],
    )
    def test_recovers_other_firms(
        self,
        asset_value: float,
        asset_vol: float,
        debt: float,
        rate: float,
        maturity: float,
    ) -> None:
        """Different leverage, rates and maturities all round-trip."""
        equity, equity_vol = _forward(asset_value, asset_vol, debt, rate, maturity)
        result = solve_merton(
            equity, equity_vol, debt, rate, MertonConfig(maturity=maturity)
        )
        assert result.asset_value == pytest.approx(asset_value, rel=1e-6)
        assert result.asset_volatility == pytest.approx(asset_vol, rel=1e-6)


class TestTextbookExample:
    """Hull's Merton example: equity 3, volatility 80%, debt 10, rate 5%, one year."""

    def test_matches_the_published_figures(self) -> None:
        """Hull reports an asset value of 12.40 and a volatility of 21.23%."""
        result = _hull_result()
        assert result.asset_value == pytest.approx(12.40, abs=0.005)
        assert result.asset_volatility == pytest.approx(0.2123, abs=5e-5)

    def test_matches_an_independent_solve(self) -> None:
        """Scratch plain-math solve: V and sigma_V to eight digits."""
        result = _hull_result()
        assert result.asset_value == pytest.approx(12.39538718863966, abs=1e-8)
        assert result.asset_volatility == pytest.approx(0.21230471342320778, abs=1e-8)

    def test_risk_neutral_terms(self) -> None:
        """Scratch plain-math solve: d1, d2 and the risk-neutral default probability."""
        result = _hull_result()
        assert result.d1 == pytest.approx(1.3531303687520282, abs=1e-7)
        assert result.d2 == pytest.approx(1.1408256553288205, abs=1e-7)
        assert result.default_probability == pytest.approx(
            0.12697124106279656, abs=1e-8
        )

    def test_debt_spread_and_leverage(self) -> None:
        """Scratch plain-math solve: debt worth V - E, yield spread, leverage."""
        result = _hull_result()
        assert result.debt_value == pytest.approx(9.39538718863966, abs=1e-8)
        assert result.credit_spread == pytest.approx(0.01236624877561747, abs=1e-8)
        assert result.leverage == pytest.approx(0.7674059793569928, abs=1e-8)

    def test_spread_is_the_debt_yield_over_the_risk_free_rate(self) -> None:
        """The spread restates y = -ln(B / D) / T minus r from the debt value."""
        result = _hull_result()
        debt_yield = -math.log(result.debt_value / 10.0) / 1.0
        assert result.credit_spread == pytest.approx(debt_yield - 0.05, abs=1e-12)


class TestOtherInverseSolves:
    """Plain-math scratch solves for other firms, in the inverse direction."""

    def test_negative_rate_and_two_year_maturity(self) -> None:
        """Equity 5, volatility 50%, debt 20, rate -1%, two years."""
        result = solve_merton(5.0, 0.5, 20.0, -0.01, MertonConfig(maturity=2.0))
        assert result.asset_value == pytest.approx(25.288789909696398, abs=1e-8)
        assert result.asset_volatility == pytest.approx(0.10580013111753225, abs=1e-8)
        assert result.d2 == pytest.approx(1.3596434824478338, abs=1e-7)
        assert result.credit_spread == pytest.approx(0.00283188061687632, abs=1e-8)

    def test_five_year_maturity(self) -> None:
        """Equity 30, volatility 35%, debt 40, rate 3%, five years."""
        result = solve_merton(30.0, 0.35, 40.0, 0.03, MertonConfig(maturity=5.0))
        assert result.asset_value == pytest.approx(64.0533843654431, abs=1e-8)
        assert result.asset_volatility == pytest.approx(0.1696824645933359, abs=1e-8)
        assert result.default_probability == pytest.approx(
            0.07400970526659428, abs=1e-8
        )
        assert result.leverage == pytest.approx(0.5374941448929348, abs=1e-8)


class TestDefaultProbability:
    """The real-world and risk-neutral default probabilities."""

    def test_risk_neutral_probability_is_phi_of_minus_d2(self) -> None:
        """Without a drift the distance to default is d2."""
        result = _hull_result()
        assert result.distance_to_default == pytest.approx(result.d2, abs=1e-12)
        assert result.default_probability == pytest.approx(_CDF(-result.d2), abs=1e-12)

    def test_a_drift_equal_to_the_rate_is_the_risk_neutral_case(self) -> None:
        """Passing mu = r explicitly changes nothing."""
        neutral = _hull_result()
        explicit = _hull_result(MertonConfig(asset_drift=0.05))
        assert explicit.default_probability == pytest.approx(
            neutral.default_probability, abs=1e-12
        )

    def test_a_drift_above_the_rate_lowers_the_probability(self) -> None:
        """Scratch plain-math solve: mu = 12% gives a 7.07% default probability."""
        neutral = _hull_result()
        drifted = _hull_result(MertonConfig(asset_drift=0.12))
        assert drifted.default_probability < neutral.default_probability
        assert drifted.default_probability == pytest.approx(
            0.07070772255109259, abs=1e-8
        )

    def test_a_drift_below_the_rate_raises_the_probability(self) -> None:
        """Assets that grow slower than the risk-free rate default more often."""
        neutral = _hull_result()
        drifted = _hull_result(MertonConfig(asset_drift=-0.02))
        assert drifted.default_probability > neutral.default_probability

    def test_the_drift_does_not_move_the_asset_estimates(self) -> None:
        """The drift enters only the real-world default probability."""
        neutral = _hull_result()
        drifted = _hull_result(MertonConfig(asset_drift=0.12))
        assert drifted.asset_value == pytest.approx(neutral.asset_value, abs=1e-9)
        assert drifted.credit_spread == pytest.approx(neutral.credit_spread, abs=1e-9)

    def test_the_probability_is_phi_of_minus_the_distance_to_default(self) -> None:
        """The probability and the distance to default are two views of one number."""
        result = _hull_result(MertonConfig(asset_drift=0.12))
        assert result.default_probability == pytest.approx(
            _CDF(-result.distance_to_default), abs=1e-12
        )


class TestEconomicProperties:
    """Directional properties that hold for any consistent credit model."""

    def test_more_debt_on_the_same_assets_raises_probability_and_spread(self) -> None:
        """Holding the firm's assets fixed, a larger face value is riskier."""
        outcomes = []
        for debt in (5.0, 8.0, 10.0, 12.5):
            equity, equity_vol = _forward(12.4, 0.2123, debt, 0.05, 1.0)
            result = solve_merton(equity, equity_vol, debt, 0.05)
            outcomes.append((result.default_probability, result.credit_spread))
        probabilities, spreads = zip(*outcomes, strict=True)
        assert list(probabilities) == sorted(probabilities)
        assert list(spreads) == sorted(spreads)
        assert len(set(probabilities)) == len(probabilities)

    @pytest.mark.parametrize(
        ("debt", "equity_vol", "probability", "spread"),
        [
            # Scratch plain-math solves, equity fixed at 3, rate 5%, one year.
            (8.0, 0.80, 0.11840267353527445, 0.013058294243902313),
            (12.0, 0.80, 0.13323402952972002, 0.011611846466941286),
            (15.0, 0.80, 0.1400023775060557, 0.01053462929263426),
            (10.0, 0.40, 0.002521976841672602, 7.022660349575993e-05),
            (10.0, 1.20, 0.3788913131862152, 0.0902259546646348),
        ],
    )
    def test_market_inputs_to_default_probability_and_spread(
        self, debt: float, equity_vol: float, probability: float, spread: float
    ) -> None:
        """Equity value fixed at 3: golden outputs for other debt and volatility."""
        result = solve_merton(3.0, equity_vol, debt, 0.05)
        assert result.default_probability == pytest.approx(probability, abs=1e-8)
        assert result.credit_spread == pytest.approx(spread, abs=1e-8)

    def test_more_debt_at_fixed_equity_raises_the_probability(self) -> None:
        """At the same equity value and volatility, more debt raises default risk."""
        probabilities = [
            solve_merton(3.0, 0.8, debt, 0.05).default_probability
            for debt in (8.0, 10.0, 12.0, 15.0)
        ]
        assert probabilities == sorted(probabilities)

    def test_more_debt_at_fixed_equity_lowers_the_spread(self) -> None:
        """Observed equity of 3 against more debt implies a bigger, calmer firm.

        The implied asset volatility falls faster than leverage rises, so the
        spread shrinks even though the default probability grows; the golden
        values above are the independent evidence.
        """
        spreads = [
            solve_merton(3.0, 0.8, debt, 0.05).credit_spread
            for debt in (8.0, 10.0, 12.0, 15.0)
        ]
        assert spreads == sorted(spreads, reverse=True)

    def test_higher_equity_volatility_raises_probability_and_spread(self) -> None:
        """Wilder equity means a riskier firm."""
        results = [solve_merton(3.0, vol, 10.0, 0.05) for vol in (0.2, 0.4, 0.8, 1.2)]
        probabilities = [r.default_probability for r in results]
        spreads = [r.credit_spread for r in results]
        assert probabilities == sorted(probabilities)
        assert spreads == sorted(spreads)
        assert len(set(probabilities)) == len(probabilities)

    @pytest.mark.parametrize("debt", [1e-6, 5.0, 12.0, 30.0])
    @pytest.mark.parametrize("equity_vol", [0.1, 0.5, 1.5])
    def test_spread_is_never_negative(self, debt: float, equity_vol: float) -> None:
        """Risky debt never yields less than the risk-free bond."""
        assert solve_merton(3.0, equity_vol, debt, 0.05).credit_spread >= 0.0

    def test_vanishing_debt_means_vanishing_default_probability(self) -> None:
        """As the face value goes to zero the firm cannot default."""
        probabilities = [
            solve_merton(3.0, 0.8, debt, 0.05).default_probability
            for debt in (1.0, 1e-1, 1e-3, 1e-6)
        ]
        assert probabilities == sorted(probabilities, reverse=True)
        assert len(set(probabilities)) == len(probabilities)
        assert probabilities[-1] < 1e-12

    def test_vanishing_debt_means_vanishing_spread(self) -> None:
        """Almost no debt leaves nothing to price a credit premium on."""
        assert solve_merton(3.0, 0.8, 1e-3, 0.05).credit_spread < 1e-6

    def test_a_longer_maturity_changes_the_estimate(self) -> None:
        """The configured maturity reaches the solver."""
        short = solve_merton(3.0, 0.8, 10.0, 0.05)
        long = solve_merton(3.0, 0.8, 10.0, 0.05, MertonConfig(maturity=3.0))
        assert long.asset_value != pytest.approx(short.asset_value, abs=1e-3)


class TestSolutionConsistency:
    """The returned numbers satisfy the Merton equations."""

    @pytest.mark.parametrize(
        ("inputs", "maturity"),
        [
            (_HULL[:4], 1.0),
            ((5.0, 0.5, 20.0, -0.01), 2.0),
            ((30.0, 0.35, 40.0, 0.03), 5.0),
            ((1.0, 2.5, 10.0, 0.02), 1.0),
        ],
    )
    def test_both_equations_hold_at_the_solution(
        self, inputs: tuple[float, float, float, float], maturity: float
    ) -> None:
        """Recompute equity value and equity volatility from the result."""
        equity, equity_vol, debt, rate = inputs
        config = MertonConfig(maturity=maturity)
        result = solve_merton(*inputs, config=config)
        root_t = math.sqrt(maturity)
        d1 = (
            math.log(result.asset_value / debt)
            + (rate + result.asset_volatility**2 / 2) * maturity
        ) / (result.asset_volatility * root_t)
        d2 = d1 - result.asset_volatility * root_t
        model_equity = result.asset_value * _CDF(d1) - debt * math.exp(
            -rate * maturity
        ) * _CDF(d2)
        model_vol_term = _CDF(d1) * result.asset_volatility * result.asset_value
        assert abs(model_equity / equity - 1.0) < config.solver_tolerance
        assert (
            abs(model_vol_term / (equity_vol * equity) - 1.0) < config.solver_tolerance
        )
        assert result.d1 == pytest.approx(d1, abs=1e-12)
        assert result.d2 == pytest.approx(d2, abs=1e-12)

    def test_reported_residual_is_below_the_tolerance(self) -> None:
        """The residual norm the result carries passes the configured bar."""
        config = MertonConfig(solver_tolerance=1e-9)
        result = _hull_result(config)
        assert 0.0 <= result.residual_norm <= config.solver_tolerance

    def test_debt_value_is_assets_minus_equity(self) -> None:
        """Equity and debt split the firm: V = E + B."""
        result = _hull_result()
        assert result.asset_value == pytest.approx(3.0 + result.debt_value, abs=1e-12)

    def test_debt_value_matches_the_closed_form_debt_price(self) -> None:
        """Merton's debt value D e^{-rT} N(d2) + V N(-d1), from the result's d1, d2."""
        result = _hull_result()
        closed_form = 10.0 * math.exp(-0.05) * _CDF(
            result.d2
        ) + result.asset_value * _CDF(-result.d1)
        assert result.debt_value == pytest.approx(closed_form, abs=1e-9)

    def test_leverage_is_discounted_debt_over_assets(self) -> None:
        """Leverage restates D e^{-rT} / V from the result's own asset value."""
        result = _hull_result()
        expected = 10.0 * math.exp(-0.05) / result.asset_value
        assert result.leverage == pytest.approx(expected, abs=1e-12)

    def test_the_result_is_frozen(self) -> None:
        """A result is a value: it cannot be edited."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            _hull_result().asset_value = 1.0  # type: ignore[misc]

    def test_a_solve_is_reproducible(self) -> None:
        """There is no randomness: the same inputs give the same result."""
        assert _hull_result() == _hull_result()


class TestHardCorners:
    """Inputs far from a typical balance sheet still solve."""

    @pytest.mark.parametrize(
        ("equity", "equity_vol", "maturity"),
        [
            (1e-3, 3.0, 5.0),
            (1e-2, 0.8, 30.0),
            (1e6, 0.05, 10.0),
            (10.0, 0.01, 0.05),
            (50.0, 8.0, 1.0),
        ],
    )
    def test_solves_with_extreme_leverage_and_volatility(
        self, equity: float, equity_vol: float, maturity: float
    ) -> None:
        """Equity from a thousandth to a hundred thousand times the debt."""
        config = MertonConfig(maturity=maturity)
        result = solve_merton(equity, equity_vol, 10.0, 0.03, config)
        assert result.residual_norm <= config.solver_tolerance
        assert 0.0 <= result.default_probability <= 1.0
        assert result.asset_value >= equity

    def test_a_looser_tolerance_still_solves(self) -> None:
        """A coarse tolerance is met and reported."""
        config = MertonConfig(solver_tolerance=1e-6)
        assert _hull_result(config).residual_norm <= 1e-6


class TestInputValidation:
    """``solve_merton`` rejects inputs the model cannot interpret."""

    @pytest.mark.parametrize("value", [0.0, -3.0, math.nan, math.inf])
    def test_rejects_a_bad_equity_value(self, value: float) -> None:
        """Equity is worth something positive."""
        with pytest.raises(ConfigurationError, match="equity_value"):
            solve_merton(value, 0.8, 10.0, 0.05)

    @pytest.mark.parametrize("value", [0.0, -0.8, math.nan, math.inf])
    def test_rejects_a_bad_equity_volatility(self, value: float) -> None:
        """Equity volatility is positive."""
        with pytest.raises(ConfigurationError, match="equity_volatility"):
            solve_merton(3.0, value, 10.0, 0.05)

    @pytest.mark.parametrize("value", [0.0, -10.0, math.nan, math.inf])
    def test_rejects_a_bad_debt_face_value(self, value: float) -> None:
        """The debt to repay is positive."""
        with pytest.raises(ConfigurationError, match="debt_face_value"):
            solve_merton(3.0, 0.8, value, 0.05)

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_rejects_a_non_finite_rate(self, value: float) -> None:
        """The risk-free rate is a number; it may be zero or negative."""
        with pytest.raises(ConfigurationError, match="risk_free_rate"):
            solve_merton(3.0, 0.8, 10.0, value)

    def test_accepts_a_negative_rate(self) -> None:
        """Negative rates are real."""
        assert solve_merton(3.0, 0.8, 10.0, -0.01).asset_value > 3.0

    def test_rejects_a_discount_factor_that_cannot_be_represented(self) -> None:
        """exp(-r T) must stay inside the floating-point range."""
        with pytest.raises(ConfigurationError, match="risk_free_rate"):
            solve_merton(3.0, 0.8, 10.0, 0.5, MertonConfig(maturity=2000.0))


class TestConvergenceFailure:
    """A solve that cannot reach the tolerance is an error, never a guess."""

    @pytest.mark.parametrize("max_iterations", [1, 2])
    def test_a_tiny_budget_raises(self, max_iterations: int) -> None:
        """Too few residual evaluations cannot find the root from the start."""
        with pytest.raises(ConvergenceError, match="residual"):
            _hull_result(MertonConfig(max_iterations=max_iterations))

    def test_the_message_names_the_inputs_and_the_tolerance(self) -> None:
        """The error says which problem failed, to be debuggable from a log."""
        with pytest.raises(ConvergenceError) as raised:
            _hull_result(MertonConfig(max_iterations=1))
        message = str(raised.value)
        assert "equity_value=3" in message
        assert "debt_face_value=10" in message
        assert "solver_tolerance" in message

    def test_a_starting_guess_beyond_double_precision_raises(self) -> None:
        """Equity plus debt overflowing the float range leaves nothing to solve."""
        with pytest.raises(ConvergenceError, match="starting guess"):
            solve_merton(1e308, 0.8, 1e308, 0.05)

    def test_an_absurd_equity_volatility_raises_instead_of_returning_nan(self) -> None:
        """A volatility whose square overflows cannot give a finite solution."""
        with pytest.raises(ConvergenceError, match="not finite"):
            solve_merton(3.0, 1e300, 10.0, 0.05)

    def test_an_unresolvable_debt_value_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A solution with V equal to E has no debt value to take a yield of."""
        monkeypatch.setattr(
            merton_module, "_solve_unknowns", lambda problem, config: (3.0, 0.8, 0.0)
        )
        with pytest.raises(ConvergenceError, match="not positive"):
            _hull_result()

    def test_a_failed_solve_emits_no_numpy_warnings(self) -> None:
        """Overflow in trial steps is part of the search, not something to report."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ConvergenceError):
                solve_merton(3.0, 1e300, 10.0, 0.05)
            with pytest.raises(ConvergenceError):
                solve_merton(1e308, 0.8, 1e308, 0.05)
