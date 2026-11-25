"""Tests for the delta-hedging simulator.

The golden numbers come from closed forms evaluated independently of the module
under test (``statistics.NormalDist`` and plain ``math``): the mean P&L of a
one-rebalance hedge under a drifting GBM, the exact cost shift of one trade, the
zero mean of a hedge under the risk-neutral measure, and the perfect hedge of a
deep in-the-money option with no volatility left.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import pytest

from optimizer.derivatives import (
    DeltaHedgeConfig,
    DeltaHedgeResult,
    OptionType,
    simulate_delta_hedge,
)
from optimizer.exceptions import ConfigurationError

_CALL = OptionType.CALL
_PUT = OptionType.PUT
_SEED = 20240611


def _run(
    kind: OptionType = _CALL,
    *,
    spot: float = 100.0,
    strike: float = 100.0,
    maturity: float = 1.0,
    **config: Any,
) -> DeltaHedgeResult:
    """Simulate a hedge of the at-the-money one-year option with some overrides."""
    return simulate_delta_hedge(
        spot, strike, maturity, kind, DeltaHedgeConfig(**config)
    )


def _standard_error(result: DeltaHedgeResult) -> float:
    """Return the standard error of the mean P&L."""
    return result.std_pnl / math.sqrt(result.pnl.size)


class TestConfig:
    """``DeltaHedgeConfig`` defaults and validation."""

    def test_defaults(self) -> None:
        """The defaults describe a daily-hedged one-year option at 20% volatility."""
        config = DeltaHedgeConfig()
        assert config.n_paths == 5000
        assert config.n_rebalances == 252
        assert config.drift == 0.05
        assert config.true_volatility == 0.2
        assert config.hedge_volatility == 0.2
        assert config.rate == 0.02
        assert config.transaction_cost_bps == 0.0
        assert config.random_state is None

    def test_config_is_immutable(self) -> None:
        """The configuration is a frozen dataclass."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            DeltaHedgeConfig().n_paths = 10  # type: ignore[misc]

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("n_paths", 1),
            ("n_paths", 0),
            ("n_paths", 2.5),
            ("n_rebalances", 0),
            ("n_rebalances", -4),
            ("n_rebalances", 1.5),
            ("true_volatility", 0.0),
            ("true_volatility", -0.2),
            ("true_volatility", math.nan),
            ("hedge_volatility", 0.0),
            ("hedge_volatility", math.inf),
            ("drift", math.nan),
            ("drift", math.inf),
            ("rate", math.nan),
            ("transaction_cost_bps", -1.0),
            ("transaction_cost_bps", math.nan),
            ("random_state", -1),
        ],
    )
    def test_invalid_settings_raise(self, field: str, value: float) -> None:
        """Counts, volatilities, costs, rates and seeds are all checked."""
        with pytest.raises(ConfigurationError, match=field):
            DeltaHedgeConfig(**{field: value})


class TestResult:
    """The shape and bookkeeping of ``DeltaHedgeResult``."""

    def test_result_fields_describe_the_simulation(self) -> None:
        """The summary statistics are computed from the stored P&L vector."""
        result = _run(n_paths=300, n_rebalances=10, random_state=_SEED)
        assert result.pnl.shape == (300,)
        assert result.n_rebalances == 10
        assert result.mean_pnl == pytest.approx(float(result.pnl.mean()), abs=1e-12)
        assert result.std_pnl == pytest.approx(float(result.pnl.std(ddof=1)), abs=1e-12)

    def test_premium_is_the_black_scholes_price_at_the_hedge_volatility(self) -> None:
        """The hedger sells at the Black-Scholes price of the hedge volatility."""
        # NormalDist closed form of an at-the-money call, T=1, r=2%, sigma=20%.
        result = _run(n_paths=10, n_rebalances=2, random_state=_SEED)
        assert result.premium == pytest.approx(8.916037278572539, abs=1e-9)

    def test_result_is_immutable_and_its_pnl_read_only(self) -> None:
        """Neither the dataclass nor its P&L vector can be modified."""
        result = _run(n_paths=10, n_rebalances=2, random_state=_SEED)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.mean_pnl = 0.0  # type: ignore[misc]
        with pytest.raises(ValueError, match="read-only"):
            result.pnl[0] = 0.0

    def test_option_type_accepts_its_string_value(self) -> None:
        """``"put"`` is accepted in place of the enum."""
        config = DeltaHedgeConfig(n_paths=20, n_rebalances=3, random_state=_SEED)
        by_text = simulate_delta_hedge(100.0, 100.0, 1.0, "put", config)
        by_enum = simulate_delta_hedge(100.0, 100.0, 1.0, _PUT, config)
        np.testing.assert_array_equal(by_text.pnl, by_enum.pnl)


class TestReproducibility:
    """Seeds control every random draw."""

    def test_same_seed_gives_the_same_pnl(self) -> None:
        """Two runs with one seed are identical."""
        first = _run(n_paths=200, n_rebalances=20, random_state=_SEED)
        second = _run(n_paths=200, n_rebalances=20, random_state=_SEED)
        np.testing.assert_array_equal(first.pnl, second.pnl)

    def test_different_seeds_give_different_pnl(self) -> None:
        """Changing the seed changes the paths."""
        first = _run(n_paths=200, n_rebalances=20, random_state=1)
        second = _run(n_paths=200, n_rebalances=20, random_state=2)
        assert not np.array_equal(first.pnl, second.pnl)

    def test_unseeded_runs_still_return_a_full_vector(self) -> None:
        """Without a seed the simulation draws fresh entropy and still completes."""
        assert _run(n_paths=50, n_rebalances=5).pnl.shape == (50,)


class TestAnalyticChecks:
    """Closed-form expectations the simulator must reproduce."""

    def test_one_rebalance_mean_matches_the_closed_form(self) -> None:
        """With one hedge the mean P&L is analytic under a drifting GBM."""
        # NormalDist closed form: premium and delta at t=0, then E[S_T] = S e^{mu T}
        # and E[(S_T - K)^+] under the real drift; call S=K=100, T=1, r=2%, mu=5%.
        result = _run(n_paths=400_000, n_rebalances=1, random_state=_SEED)
        assert abs(result.mean_pnl - (-0.09049746685049342)) < 4 * _standard_error(
            result
        )

    def test_one_rebalance_mean_with_a_wrong_hedge_volatility(self) -> None:
        """The closed form also covers a hedge volatility below the true one."""
        # NormalDist closed form, true sigma 30%, hedge sigma 20%.
        result = _run(
            n_paths=400_000, n_rebalances=1, true_volatility=0.3, random_state=_SEED
        )
        assert abs(result.mean_pnl - (-4.065007838819561)) < 4 * _standard_error(result)

    def test_one_rebalance_mean_for_an_out_of_the_money_call(self) -> None:
        """A different strike, maturity, rate and drift stay on the closed form."""
        # NormalDist closed form for a spot of 100, a strike of 105 and half a year,
        # with a 3% rate, 8% drift, 25% true volatility and 22% hedge volatility.
        result = _run(
            n_paths=400_000,
            n_rebalances=1,
            strike=105.0,
            maturity=0.5,
            rate=0.03,
            drift=0.08,
            true_volatility=0.25,
            hedge_volatility=0.22,
            random_state=_SEED,
        )
        assert abs(result.mean_pnl - (-0.9627695041511473)) < 4 * _standard_error(
            result
        )

    def test_one_rebalance_delta_uses_the_hedge_volatility(self) -> None:
        """With a strong drift the mean P&L depends on which volatility sets delta."""
        # NormalDist closed form, true sigma 20%, hedge sigma 40%, mu=15%, r=2%:
        # the mean depends on delta through S (e^{mu T} - e^{rT}) * delta, so a
        # delta taken from the wrong volatility moves it by about 0.3.
        result = _run(
            n_paths=400_000,
            n_rebalances=1,
            drift=0.15,
            hedge_volatility=0.4,
            random_state=_SEED,
        )
        assert abs(result.mean_pnl - 6.5185960457247205) < 4 * _standard_error(result)

    def test_call_and_put_hedges_earn_the_same_without_costs(self) -> None:
        """Put-call parity: delta_put = delta_call - 1 and the forward is financed."""
        call = _run(_CALL, n_paths=500, n_rebalances=30, random_state=_SEED)
        put = _run(_PUT, n_paths=500, n_rebalances=30, random_state=_SEED)
        np.testing.assert_allclose(put.pnl, call.pnl, atol=1e-9)

    @pytest.mark.parametrize("kind", [_CALL, _PUT])
    def test_hedging_under_the_risk_neutral_measure_has_zero_mean(
        self, kind: OptionType
    ) -> None:
        """With drift = rate and one volatility the discounted hedge is a martingale."""
        result = _run(
            kind, n_paths=20_000, n_rebalances=12, drift=0.02, random_state=_SEED
        )
        assert abs(result.mean_pnl) < 4 * _standard_error(result)

    def test_many_rebalances_under_the_real_drift_have_near_zero_mean(self) -> None:
        """Daily hedging at the true volatility earns nothing in expectation."""
        result = _run(n_paths=20_000, n_rebalances=252, random_state=_SEED)
        assert abs(result.mean_pnl) < 4 * _standard_error(result)

    @pytest.mark.parametrize("kind", [_CALL, _PUT])
    def test_deep_in_the_money_option_without_volatility_is_perfectly_hedged(
        self, kind: OptionType
    ) -> None:
        """With no volatility the hedge holds one share and the P&L is exactly zero."""
        # A deep in-the-money call is S - K e^{-rT} and its delta is 1; the put is the
        # mirror image with delta -1. Cash, shares and payoff then cancel on every path.
        spot, strike = (120.0, 100.0) if kind is _CALL else (80.0, 100.0)
        result = _run(
            kind,
            spot=spot,
            strike=strike,
            n_paths=50,
            n_rebalances=10,
            true_volatility=1e-6,
            hedge_volatility=1e-6,
            random_state=_SEED,
        )
        np.testing.assert_allclose(result.pnl, 0.0, atol=1e-8)

    def test_one_trade_costs_exactly_its_financed_proportional_fee(self) -> None:
        """With one hedge, 10 bps costs bps * delta * S * e^{rT} on every path."""
        # NormalDist closed form: delta = N(0.2) = 0.579259709439103, so the fee
        # is 0.001 * 0.579259709439103 * 100 * e^{0.02} = 0.059096153179328.
        free = _run(n_paths=500, n_rebalances=1, random_state=_SEED)
        costly = _run(
            n_paths=500, n_rebalances=1, transaction_cost_bps=10.0, random_state=_SEED
        )
        np.testing.assert_allclose(free.pnl - costly.pnl, 0.059096153179328, atol=1e-9)


class TestHedgingBehaviour:
    """Qualitative properties of discrete delta hedging."""

    def test_pnl_dispersion_shrinks_like_the_inverse_square_root_of_rebalances(
        self,
    ) -> None:
        """Quadrupling the hedge dates halves the standard deviation."""
        coarse = _run(n_paths=20_000, n_rebalances=25, random_state=_SEED)
        fine = _run(n_paths=20_000, n_rebalances=100, random_state=_SEED)
        assert coarse.std_pnl / fine.std_pnl == pytest.approx(2.0, rel=0.2)

    def test_selling_below_the_true_volatility_loses_money(self) -> None:
        """Underpricing volatility is paid for: the mean P&L is clearly negative."""
        # First-order loss is vega * (sigma_hedge - sigma_true) = 37.5 * (-0.05),
        # about -1.9; require at least half of it.
        result = _run(
            n_paths=10_000, hedge_volatility=0.15, n_rebalances=100, random_state=_SEED
        )
        assert result.mean_pnl < -0.9
        assert result.mean_pnl < -4 * _standard_error(result)

    def test_selling_above_the_true_volatility_makes_money(self) -> None:
        """Overpricing volatility is rewarded: the mean P&L is clearly positive."""
        # First-order gain is vega * (0.25 - 0.20) = +1.9; require at least half.
        result = _run(
            n_paths=10_000, hedge_volatility=0.25, n_rebalances=100, random_state=_SEED
        )
        assert result.mean_pnl > 0.9
        assert result.mean_pnl > 4 * _standard_error(result)

    def test_costs_lower_the_pnl_on_every_path_and_grow_with_the_cost(self) -> None:
        """The same paths earn less as the proportional cost rises."""
        runs = [
            _run(
                n_paths=2_000,
                n_rebalances=50,
                transaction_cost_bps=bps,
                random_state=_SEED,
            )
            for bps in (0.0, 5.0, 20.0)
        ]
        assert runs[0].mean_pnl > runs[1].mean_pnl > runs[2].mean_pnl
        assert np.all(runs[0].pnl > runs[1].pnl)
        assert np.all(runs[1].pnl > runs[2].pnl)

    def test_cost_reduction_is_proportional_to_the_cost_rate(self) -> None:
        """Costs enter linearly, so 20 bps removes twice what 10 bps removes."""
        base, low, high = (
            _run(
                n_paths=2_000,
                n_rebalances=50,
                transaction_cost_bps=bps,
                random_state=_SEED,
            ).mean_pnl
            for bps in (0.0, 10.0, 20.0)
        )
        assert base - high == pytest.approx(2.0 * (base - low), rel=1e-9)

    def test_finer_hedging_costs_more(self) -> None:
        """At a fixed cost rate, rebalancing more often burns more in fees."""
        loss = {
            n: _run(n_paths=2_000, n_rebalances=n, random_state=_SEED).mean_pnl
            - _run(
                n_paths=2_000,
                n_rebalances=n,
                transaction_cost_bps=10.0,
                random_state=_SEED,
            ).mean_pnl
            for n in (10, 100)
        }
        assert loss[100] > loss[10] > 0.0


class TestValidation:
    """Contract inputs are checked before any path is drawn."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("spot", 0.0),
            ("spot", -1.0),
            ("spot", math.nan),
            ("strike", 0.0),
            ("strike", math.inf),
            ("maturity", 0.0),
            ("maturity", -1.0),
            ("maturity", math.nan),
        ],
    )
    def test_invalid_contract_raises(self, field: str, value: float) -> None:
        """Spot, strike and maturity must be positive and finite."""
        contract = {"spot": 100.0, "strike": 100.0, "maturity": 1.0, field: value}
        with pytest.raises(ConfigurationError, match=field):
            simulate_delta_hedge(
                contract["spot"],
                contract["strike"],
                contract["maturity"],
                _CALL,
                DeltaHedgeConfig(n_paths=10, n_rebalances=2),
            )

    def test_unknown_option_type_raises(self) -> None:
        """Anything other than a call or a put is a configuration error."""
        with pytest.raises(ConfigurationError, match="option_type"):
            simulate_delta_hedge(
                100.0, 100.0, 1.0, "straddle", DeltaHedgeConfig(n_paths=10)
            )
