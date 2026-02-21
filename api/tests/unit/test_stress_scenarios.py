"""Unit tests for LLM-driven stress scenario service (issue #16)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from app.services.stress_scenarios import (
    _PROB_MAX,
    _PROB_MIN,
    _SHOCK_MAX,
    _SHOCK_MIN,
    _clamp_scenario,
    _ensure_market_drawdown,
    generate_stress_scenarios,
    scenario_to_synthetic_data_args,
)
from baml_client.types import StressScenario

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TICKERS = ["SPY", "QQQ", "GLD", "TLT"]

PORTFOLIO = {t: 0.25 for t in TICKERS}

MACRO = "US CPI at 4.2%, Fed on hold, elevated geopolitical risk."


def _make_scenario(
    name: str = "Test Recession",
    shocks: dict[str, float] | None = None,
    probability: float = 0.15,
    horizon_days: int = 21,
) -> StressScenario:
    if shocks is None:
        shocks = {"SPY": -0.20, "QQQ": -0.25, "GLD": 0.08, "TLT": 0.05}
    return StressScenario(
        name=name,
        description="A synthetic recession scenario for testing.",
        shocks=shocks,
        probability=probability,
        horizon_days=horizon_days,
    )


# ===========================================================================
# TestClampScenario
# ===========================================================================


class TestClampScenario:
    def test_valid_scenario_unchanged(self) -> None:
        s = _make_scenario()
        clamped = _clamp_scenario(s)
        assert clamped.probability == pytest.approx(s.probability)
        assert clamped.shocks == s.shocks

    def test_probability_clamped_below(self) -> None:
        s = _make_scenario(probability=0.0)
        clamped = _clamp_scenario(s)
        assert clamped.probability >= _PROB_MIN

    def test_probability_clamped_above(self) -> None:
        s = _make_scenario(probability=1.0)
        clamped = _clamp_scenario(s)
        assert clamped.probability <= _PROB_MAX

    def test_shock_clamped_below(self) -> None:
        s = _make_scenario(shocks={"SPY": -2.0, "QQQ": 0.0, "GLD": 0.0, "TLT": 0.0})
        clamped = _clamp_scenario(s)
        assert clamped.shocks["SPY"] >= _SHOCK_MIN

    def test_shock_clamped_above(self) -> None:
        s = _make_scenario(shocks={"SPY": 5.0, "QQQ": 0.0, "GLD": 0.0, "TLT": 0.0})
        clamped = _clamp_scenario(s)
        assert clamped.shocks["SPY"] <= _SHOCK_MAX

    def test_horizon_days_minimum_one(self) -> None:
        s = _make_scenario(horizon_days=0)
        clamped = _clamp_scenario(s)
        assert clamped.horizon_days >= 1

    def test_all_shock_values_in_range(self) -> None:
        shocks = {"SPY": -99.0, "QQQ": 99.0, "GLD": -0.5, "TLT": 0.5}
        s = _make_scenario(shocks=shocks)
        clamped = _clamp_scenario(s)
        for v in clamped.shocks.values():
            assert _SHOCK_MIN <= v <= _SHOCK_MAX


# ===========================================================================
# TestEnsureMarketDrawdown
# ===========================================================================


class TestEnsureMarketDrawdown:
    def test_existing_drawdown_scenario_unchanged(self) -> None:
        s = _make_scenario(
            shocks={"SPY": -0.25, "QQQ": -0.30, "GLD": 0.08, "TLT": 0.05}
        )
        result = _ensure_market_drawdown([s])
        assert result[0].shocks["SPY"] == pytest.approx(-0.25)

    def test_no_drawdown_patches_first(self) -> None:
        """If no scenario has avg negative shock ≥10%, first is patched."""
        s = _make_scenario(
            shocks={"SPY": -0.03, "QQQ": -0.02, "GLD": 0.01, "TLT": 0.01}
        )
        result = _ensure_market_drawdown([s])
        # All shocks should be ≤ -0.10 after patch
        for shock in result[0].shocks.values():
            assert shock <= -0.10

    def test_empty_list_returned_unchanged(self) -> None:
        assert _ensure_market_drawdown([]) == []

    def test_second_scenario_not_patched(self) -> None:
        """Only the first scenario is patched; others remain untouched."""
        s1 = _make_scenario(
            name="S1", shocks={"SPY": -0.02, "QQQ": -0.01, "GLD": 0.0, "TLT": 0.0}
        )
        s2 = _make_scenario(
            name="S2", shocks={"SPY": 0.05, "QQQ": 0.05, "GLD": 0.0, "TLT": 0.0}
        )
        result = _ensure_market_drawdown([s1, s2])
        assert result[1].shocks["SPY"] == pytest.approx(0.05)

    def test_narrows_positive_shocks_to_drawdown(self) -> None:
        """Positive shocks in a patched scenario are forced to ≤ -0.10."""
        s = _make_scenario(shocks={"SPY": 0.10, "QQQ": 0.05, "GLD": 0.0, "TLT": 0.0})
        result = _ensure_market_drawdown([s])
        assert result[0].shocks["SPY"] <= -0.10
        assert result[0].shocks["QQQ"] <= -0.10


# ===========================================================================
# TestScenarioToSyntheticDataArgs
# ===========================================================================


class TestScenarioToSyntheticDataArgs:
    def test_returns_conditioning_key(self) -> None:
        s = _make_scenario()
        args = scenario_to_synthetic_data_args(s)
        assert "conditioning" in args

    def test_conditioning_matches_shocks(self) -> None:
        s = _make_scenario()
        args = scenario_to_synthetic_data_args(s)
        assert args["conditioning"] == s.shocks

    def test_conditioning_is_dict(self) -> None:
        s = _make_scenario()
        args = scenario_to_synthetic_data_args(s)
        assert isinstance(args["conditioning"], dict)

    def test_shock_keys_match_tickers(self) -> None:
        shocks = {t: -0.10 for t in TICKERS}
        s = _make_scenario(shocks=shocks)
        args = scenario_to_synthetic_data_args(s)
        assert set(args["conditioning"].keys()) == set(TICKERS)  # type: ignore[arg-type]

    def test_empty_shocks_produces_empty_conditioning(self) -> None:
        s = _make_scenario(shocks={})
        args = scenario_to_synthetic_data_args(s)
        assert args["conditioning"] == {}


# ===========================================================================
# TestGenerateStressScenarios
# ===========================================================================


class TestGenerateStressScenarios:
    def _mock_scenarios(self, n: int = 3) -> list[StressScenario]:
        scenarios = []
        base_shocks: list[dict[str, float]] = [
            {"SPY": -0.20, "QQQ": -0.25, "GLD": 0.08, "TLT": 0.05},
            {"SPY": -0.10, "QQQ": -0.12, "GLD": 0.15, "TLT": -0.15},
            {"SPY": 0.02, "QQQ": -0.30, "GLD": 0.02, "TLT": 0.03},
        ]
        for i in range(n):
            scenarios.append(
                StressScenario(
                    name=f"Scenario {i}",
                    description=f"Description {i}",
                    shocks=base_shocks[i % len(base_shocks)],
                    probability=0.10 + i * 0.05,
                    horizon_days=21,
                )
            )
        return scenarios

    def test_returns_list_of_stress_scenarios(self) -> None:
        with patch("app.services.stress_scenarios.b") as mock_b:
            mock_b.DesignStressScenarios.return_value = self._mock_scenarios(3)
            result = generate_stress_scenarios(3, PORTFOLIO, MACRO)
        assert isinstance(result, list)
        assert all(isinstance(s, StressScenario) for s in result)

    def test_n_scenarios_forwarded_to_baml(self) -> None:
        with patch("app.services.stress_scenarios.b") as mock_b:
            mock_b.DesignStressScenarios.return_value = self._mock_scenarios(2)
            generate_stress_scenarios(2, PORTFOLIO, MACRO)
        mock_b.DesignStressScenarios.assert_called_once()
        call_kwargs = mock_b.DesignStressScenarios.call_args
        assert call_kwargs.kwargs["n_scenarios"] == 2

    def test_tickers_forwarded_to_baml(self) -> None:
        with patch("app.services.stress_scenarios.b") as mock_b:
            mock_b.DesignStressScenarios.return_value = self._mock_scenarios(1)
            generate_stress_scenarios(1, PORTFOLIO, MACRO)
        call_kwargs = mock_b.DesignStressScenarios.call_args
        assert set(call_kwargs.kwargs["tickers"]) == set(TICKERS)

    def test_probabilities_in_range(self) -> None:
        with patch("app.services.stress_scenarios.b") as mock_b:
            mock_b.DesignStressScenarios.return_value = self._mock_scenarios(3)
            result = generate_stress_scenarios(3, PORTFOLIO, MACRO)
        for s in result:
            assert 0.0 < s.probability < 1.0

    def test_shocks_in_valid_range(self) -> None:
        with patch("app.services.stress_scenarios.b") as mock_b:
            mock_b.DesignStressScenarios.return_value = self._mock_scenarios(3)
            result = generate_stress_scenarios(3, PORTFOLIO, MACRO)
        for s in result:
            for v in s.shocks.values():
                assert -1.0 < v < 1.0

    def test_shock_keys_match_portfolio_tickers(self) -> None:
        with patch("app.services.stress_scenarios.b") as mock_b:
            mock_b.DesignStressScenarios.return_value = self._mock_scenarios(3)
            result = generate_stress_scenarios(3, PORTFOLIO, MACRO)
        for s in result:
            assert set(s.shocks.keys()) == set(TICKERS)

    def test_at_least_one_market_drawdown(self) -> None:
        """At least one scenario must have average negative shock ≥ 10%."""
        with patch("app.services.stress_scenarios.b") as mock_b:
            mock_b.DesignStressScenarios.return_value = self._mock_scenarios(3)
            result = generate_stress_scenarios(3, PORTFOLIO, MACRO)
        has_drawdown = False
        for s in result:
            neg_shocks = [v for v in s.shocks.values() if v < 0]
            if neg_shocks and abs(sum(neg_shocks) / len(neg_shocks)) >= 0.10:
                has_drawdown = True
                break
        assert has_drawdown

    def test_invalid_n_scenarios_raises(self) -> None:
        with pytest.raises(ValueError, match="n_scenarios"):
            generate_stress_scenarios(0, PORTFOLIO, MACRO)

    def test_empty_portfolio_raises(self) -> None:
        with pytest.raises(ValueError, match="current_portfolio"):
            generate_stress_scenarios(1, {}, MACRO)

    def test_baml_failure_raises_runtime_error(self) -> None:
        with patch("app.services.stress_scenarios.b") as mock_b:
            mock_b.DesignStressScenarios.side_effect = Exception("LLM unavailable")
            with pytest.raises(RuntimeError, match="LLM stress scenario"):
                generate_stress_scenarios(1, PORTFOLIO, MACRO)

    def test_out_of_range_probability_clamped(self) -> None:
        bad_scenario = StressScenario(
            name="Bad",
            description="desc",
            shocks={"SPY": -0.20, "QQQ": -0.25, "GLD": 0.08, "TLT": 0.05},
            probability=1.5,  # invalid — should be clamped
            horizon_days=21,
        )
        with patch("app.services.stress_scenarios.b") as mock_b:
            mock_b.DesignStressScenarios.return_value = [bad_scenario]
            result = generate_stress_scenarios(1, PORTFOLIO, MACRO)
        assert result[0].probability <= _PROB_MAX

    def test_out_of_range_shock_clamped(self) -> None:
        bad_scenario = StressScenario(
            name="Bad",
            description="desc",
            shocks={"SPY": -5.0, "QQQ": -0.20, "GLD": 3.0, "TLT": 0.0},
            probability=0.10,
            horizon_days=21,
        )
        with patch("app.services.stress_scenarios.b") as mock_b:
            mock_b.DesignStressScenarios.return_value = [bad_scenario]
            result = generate_stress_scenarios(1, PORTFOLIO, MACRO)
        assert result[0].shocks["SPY"] >= _SHOCK_MIN
        assert result[0].shocks["GLD"] <= _SHOCK_MAX
