"""Tests for regime-conditional subperiod Sharpe validation."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

from optimizer.exceptions import DataError
from optimizer.validation import (
    RegimeValidationConfig,
    RegimeValidationResult,
    run_regime_validation,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def expansion_macro_data() -> pd.DataFrame:
    """60 days of macro data that classifies as EXPANSION."""
    dates = pd.bdate_range("2024-01-02", periods=60)
    return pd.DataFrame(
        {"pmi": 55.0, "spread_2s10s": 1.5, "hy_oas": 300.0},
        index=dates,
    )


@pytest.fixture()
def three_regime_macro_data() -> pd.DataFrame:
    """180 days: 60 expansion + 60 recession + 60 expansion."""
    dates = pd.bdate_range("2024-01-02", periods=180)
    pmi = [55.0] * 60 + [46.0] * 60 + [55.0] * 60
    spread = [1.5] * 60 + [-0.5] * 60 + [1.5] * 60
    hy_oas = [300.0] * 60 + [600.0] * 60 + [300.0] * 60
    return pd.DataFrame(
        {"pmi": pmi, "spread_2s10s": spread, "hy_oas": hy_oas},
        index=dates,
    )


@pytest.fixture()
def synthetic_oos_returns(three_regime_macro_data: pd.DataFrame) -> pd.Series:
    """Returns aligned to three_regime_macro_data.

    Expansion blocks have positive mean; recession block has negative mean.
    This ensures >80% of alpha comes from expansion.
    """
    rng = np.random.default_rng(42)
    dates = three_regime_macro_data.index
    n = len(dates)
    returns = np.empty(n)
    # Expansion 1: strong positive
    returns[:60] = rng.normal(loc=0.002, scale=0.01, size=60)
    # Recession: negative
    returns[60:120] = rng.normal(loc=-0.001, scale=0.015, size=60)
    # Expansion 2: strong positive
    returns[120:] = rng.normal(loc=0.002, scale=0.01, size=60)
    return pd.Series(returns, index=dates, name="portfolio_returns")


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestRegimeValidationConfig:
    def test_defaults(self) -> None:
        cfg = RegimeValidationConfig()
        assert cfg.min_regime_obs == 21
        assert cfg.single_regime_alpha_threshold == 0.80
        assert cfg.trading_days_per_year == 252
        assert cfg.risk_free_rate == 0.0
        assert cfg.include_unknown_regime is False
        assert cfg.max_macro_staleness_days == 120

    def test_frozen(self) -> None:
        cfg = RegimeValidationConfig()
        with pytest.raises(AttributeError):
            cfg.min_regime_obs = 42  # type: ignore[misc]

    def test_for_standard(self) -> None:
        cfg = RegimeValidationConfig.for_standard()
        assert cfg.min_regime_obs == 21

    def test_for_strict(self) -> None:
        cfg = RegimeValidationConfig.for_strict()
        assert cfg.min_regime_obs == 63
        assert cfg.single_regime_alpha_threshold == 0.70

    def test_for_research(self) -> None:
        cfg = RegimeValidationConfig.for_research()
        assert cfg.include_unknown_regime is True

    def test_invalid_min_regime_obs(self) -> None:
        with pytest.raises(ValueError, match="min_regime_obs"):
            RegimeValidationConfig(min_regime_obs=0)

    def test_invalid_threshold_low(self) -> None:
        with pytest.raises(ValueError, match="single_regime_alpha_threshold"):
            RegimeValidationConfig(single_regime_alpha_threshold=0.0)

    def test_invalid_threshold_high(self) -> None:
        with pytest.raises(ValueError, match="single_regime_alpha_threshold"):
            RegimeValidationConfig(single_regime_alpha_threshold=1.5)

    def test_invalid_trading_days(self) -> None:
        with pytest.raises(ValueError, match="trading_days_per_year"):
            RegimeValidationConfig(trading_days_per_year=0)

    def test_invalid_risk_free_rate(self) -> None:
        with pytest.raises(ValueError, match="risk_free_rate"):
            RegimeValidationConfig(risk_free_rate=-0.01)

    def test_invalid_max_macro_staleness_days(self) -> None:
        with pytest.raises(ValueError, match="max_macro_staleness_days"):
            RegimeValidationConfig(max_macro_staleness_days=-1)

    def test_unbounded_macro_staleness_is_allowed(self) -> None:
        assert (
            RegimeValidationConfig(
                max_macro_staleness_days=None
            ).max_macro_staleness_days
            is None
        )


# ---------------------------------------------------------------------------
# Core function tests
# ---------------------------------------------------------------------------


class TestRunRegimeValidation:
    def test_returns_result_type(
        self,
        synthetic_oos_returns: pd.Series,
        three_regime_macro_data: pd.DataFrame,
    ) -> None:
        result = run_regime_validation(synthetic_oos_returns, three_regime_macro_data)
        assert isinstance(result, RegimeValidationResult)

    def test_per_regime_metrics_columns(
        self,
        synthetic_oos_returns: pd.Series,
        three_regime_macro_data: pd.DataFrame,
    ) -> None:
        result = run_regime_validation(synthetic_oos_returns, three_regime_macro_data)
        expected_cols = {
            "obs",
            "coverage_pct",
            "ann_return",
            "ann_vol",
            "sharpe",
            "max_drawdown",
            "obs_sufficient",
        }
        assert set(result.per_regime_metrics.columns) == expected_cols

    def test_per_regime_metrics_index(
        self,
        synthetic_oos_returns: pd.Series,
        three_regime_macro_data: pd.DataFrame,
    ) -> None:
        result = run_regime_validation(synthetic_oos_returns, three_regime_macro_data)
        # Should have expansion and recession
        assert "expansion" in result.per_regime_metrics.index
        assert "recession" in result.per_regime_metrics.index

    def test_per_subperiod_metrics_columns(
        self,
        synthetic_oos_returns: pd.Series,
        three_regime_macro_data: pd.DataFrame,
    ) -> None:
        result = run_regime_validation(synthetic_oos_returns, three_regime_macro_data)
        expected_cols = {
            "start",
            "end",
            "regime",
            "obs",
            "ann_return",
            "ann_vol",
            "sharpe",
            "max_drawdown",
        }
        assert set(result.per_subperiod_metrics.columns) == expected_cols

    def test_three_subperiods_detected(
        self,
        synthetic_oos_returns: pd.Series,
        three_regime_macro_data: pd.DataFrame,
    ) -> None:
        result = run_regime_validation(synthetic_oos_returns, three_regime_macro_data)
        # 60 expansion + 60 recession + 60 expansion = 3 subperiods
        assert len(result.per_subperiod_metrics) == 3

    def test_sharpe_finite_for_sufficient_obs(
        self,
        synthetic_oos_returns: pd.Series,
        three_regime_macro_data: pd.DataFrame,
    ) -> None:
        result = run_regime_validation(synthetic_oos_returns, three_regime_macro_data)
        sufficient = result.per_subperiod_metrics[
            result.per_subperiod_metrics["obs"] >= 21
        ]
        assert sufficient["sharpe"].notna().all()

    def test_sharpe_nan_for_insufficient_obs(
        self,
        synthetic_oos_returns: pd.Series,
        three_regime_macro_data: pd.DataFrame,
    ) -> None:
        # Set min_regime_obs very high so all subperiods are insufficient
        config = RegimeValidationConfig(min_regime_obs=1000)
        result = run_regime_validation(
            synthetic_oos_returns, three_regime_macro_data, config=config
        )
        assert result.per_subperiod_metrics["sharpe"].isna().all()

    def test_alpha_concentration_sums_to_one_for_positive_regimes(
        self,
        synthetic_oos_returns: pd.Series,
        three_regime_macro_data: pd.DataFrame,
    ) -> None:
        result = run_regime_validation(synthetic_oos_returns, three_regime_macro_data)
        positive_sum = result.regime_alpha_concentration[
            result.regime_alpha_concentration > 0
        ].sum()
        assert abs(positive_sum - 1.0) < 1e-10

    def test_concentrated_regime_flagged(
        self,
        synthetic_oos_returns: pd.Series,
        three_regime_macro_data: pd.DataFrame,
    ) -> None:
        result = run_regime_validation(synthetic_oos_returns, three_regime_macro_data)
        # Expansion contributes majority of alpha (positive returns in both
        # expansion blocks, negative in recession)
        assert len(result.concentrated_regimes) >= 1
        assert "expansion" in result.concentrated_regimes

    def test_empty_returns_raises(
        self,
        three_regime_macro_data: pd.DataFrame,
    ) -> None:
        empty = pd.Series(dtype=float)
        with pytest.raises(DataError, match="oos_returns is empty"):
            run_regime_validation(empty, three_regime_macro_data)

    def test_no_macro_overlap_returns_nan_metrics(
        self,
        synthetic_oos_returns: pd.Series,
    ) -> None:
        # Macro data dates don't overlap with OOS returns
        non_overlapping = pd.DataFrame(
            {"pmi": [55.0], "spread_2s10s": [1.5], "hy_oas": [300.0]},
            index=pd.DatetimeIndex(["2000-01-01"]),
        )
        result = run_regime_validation(synthetic_oos_returns, non_overlapping)
        assert result.n_regimes_observed == 0
        assert len(result.per_regime_metrics) == 0

    def test_all_unknown_overlap_returns_empty_result(self) -> None:
        """Overlapping macro data that all classifies as UNKNOWN and is then
        filtered out must return a well-formed empty result, not crash."""
        dates = pd.bdate_range("2024-01-02", periods=60)
        # Columns unrecognised by classify_regime → UNKNOWN for every date.
        macro = pd.DataFrame({"irrelevant": 1.0}, index=dates)
        oos = pd.Series(
            np.random.default_rng(0).normal(0.001, 0.01, size=60), index=dates
        )
        result = run_regime_validation(oos, macro)  # include_unknown_regime=False
        assert result.n_regimes_observed == 0
        assert len(result.per_regime_metrics) == 0
        assert len(result.per_subperiod_metrics) == 0
        assert result.total_obs == 60
        # Result frames still carry their columns and serialize cleanly.
        expected_cols = {
            "obs",
            "coverage_pct",
            "ann_return",
            "ann_vol",
            "sharpe",
            "max_drawdown",
            "obs_sufficient",
        }
        assert set(result.per_regime_metrics.columns) == expected_cols
        import json

        json.dumps(result.to_attribution_dict())

    def test_regime_timeline_length(
        self,
        synthetic_oos_returns: pd.Series,
        three_regime_macro_data: pd.DataFrame,
    ) -> None:
        result = run_regime_validation(synthetic_oos_returns, three_regime_macro_data)
        assert len(result.regime_timeline) == len(synthetic_oos_returns)

    def test_total_obs(
        self,
        synthetic_oos_returns: pd.Series,
        three_regime_macro_data: pd.DataFrame,
    ) -> None:
        result = run_regime_validation(synthetic_oos_returns, three_regime_macro_data)
        assert result.total_obs == len(synthetic_oos_returns)

    def test_config_respected_min_obs(
        self,
        synthetic_oos_returns: pd.Series,
        three_regime_macro_data: pd.DataFrame,
    ) -> None:
        """With very high min_regime_obs, all per-regime metrics should be NaN."""
        config = RegimeValidationConfig(min_regime_obs=1000)
        result = run_regime_validation(
            synthetic_oos_returns, three_regime_macro_data, config=config
        )
        assert result.per_regime_metrics["sharpe"].isna().all()
        assert result.per_regime_metrics["ann_return"].isna().all()

    def test_expansion_sharpe_higher_than_recession(
        self,
        synthetic_oos_returns: pd.Series,
        three_regime_macro_data: pd.DataFrame,
    ) -> None:
        """Sanity check: expansion regime should have higher Sharpe than recession."""
        result = run_regime_validation(synthetic_oos_returns, three_regime_macro_data)
        exp_sharpe = result.per_regime_metrics.loc["expansion", "sharpe"]
        rec_sharpe = result.per_regime_metrics.loc["recession", "sharpe"]
        assert exp_sharpe > rec_sharpe

    def test_single_regime_data(
        self,
        expansion_macro_data: pd.DataFrame,
    ) -> None:
        """All data in one regime should give 100% concentration."""
        rng = np.random.default_rng(99)
        returns = pd.Series(
            rng.normal(loc=0.001, scale=0.01, size=60),
            index=expansion_macro_data.index,
        )
        result = run_regime_validation(returns, expansion_macro_data)
        assert result.n_regimes_observed == 1
        assert "expansion" in result.per_regime_metrics.index
        assert result.regime_alpha_concentration["expansion"] == pytest.approx(1.0)

    def test_include_unknown_regime(
        self,
        synthetic_oos_returns: pd.Series,
        three_regime_macro_data: pd.DataFrame,
    ) -> None:
        """When include_unknown_regime=True, timeline matches OOS length."""
        config = RegimeValidationConfig.for_research()
        result = run_regime_validation(
            synthetic_oos_returns, three_regime_macro_data, config=config
        )
        assert len(result.regime_timeline) == len(synthetic_oos_returns)

    def test_all_negative_alpha_no_spurious_concentration(
        self,
        three_regime_macro_data: pd.DataFrame,
    ) -> None:
        """When all regimes have negative returns, no regime is flagged."""
        rng = np.random.default_rng(123)
        dates = three_regime_macro_data.index
        # All negative returns
        returns = pd.Series(
            rng.normal(loc=-0.002, scale=0.01, size=len(dates)),
            index=dates,
        )
        result = run_regime_validation(returns, three_regime_macro_data)
        # No regime should be flagged as concentrated
        assert result.concentrated_regimes == []
        # All concentration values should be zero
        assert (result.regime_alpha_concentration == 0.0).all()

    def test_geometric_annualization(
        self,
        expansion_macro_data: pd.DataFrame,
    ) -> None:
        """Verify annualized return uses geometric compounding."""
        # Constant daily return of 0.001 → geometric ann = (1.001)^252 - 1
        dates = expansion_macro_data.index
        returns = pd.Series(0.001, index=dates)
        result = run_regime_validation(returns, expansion_macro_data)
        ann_ret = result.per_regime_metrics.loc["expansion", "ann_return"]
        expected = (1.001) ** 252 - 1.0
        assert ann_ret == pytest.approx(expected, rel=1e-6)

    def test_ann_return_is_realised_cagr_not_compounded_mean(
        self,
        expansion_macro_data: pd.DataFrame,
    ) -> None:
        """Volatility drag must reduce ann_return: +10% / -10% days lose money."""
        dates = expansion_macro_data.index
        returns = pd.Series(np.tile([0.10, -0.10], len(dates) // 2), index=dates)
        result = run_regime_validation(returns, expansion_macro_data)
        ann_ret = result.per_regime_metrics.loc["expansion", "ann_return"]
        expected = float(np.prod(1.0 + returns.to_numpy())) ** (252 / len(dates)) - 1.0
        assert expected < 0
        assert ann_ret == pytest.approx(expected, rel=1e-9)

    def test_ann_return_equals_total_return_over_one_year(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=252)
        macro = pd.DataFrame(
            {"pmi": 55.0, "spread_2s10s": 1.5, "hy_oas": 300.0}, index=dates
        )
        rng = np.random.default_rng(7)
        returns = pd.Series(rng.normal(0.0005, 0.02, size=252), index=dates)
        result = run_regime_validation(returns, macro)
        ann_ret = result.per_regime_metrics.loc["expansion", "ann_return"]
        assert ann_ret == pytest.approx(float((1.0 + returns).prod() - 1.0), rel=1e-9)

    def test_single_observation_block_reports_undefined_dispersion_as_nan(
        self,
    ) -> None:
        """One day has no volatility, and annualising it only extrapolates."""
        dates = pd.bdate_range("2024-01-02", periods=61)
        macro = pd.DataFrame(
            {
                "pmi": [55.0] * 60 + [46.0],
                "spread_2s10s": [1.5] * 60 + [-0.5],
                "hy_oas": [300.0] * 60 + [600.0],
            },
            index=dates,
        )
        returns = pd.Series([0.001] * 60 + [-0.02], index=dates)
        config = RegimeValidationConfig(min_regime_obs=1)

        result = run_regime_validation(returns, macro, config=config)

        one_day = result.per_regime_metrics.loc["recession"]
        assert one_day["obs"] == 1
        assert np.isnan(one_day["ann_vol"])
        assert np.isnan(one_day["ann_return"])
        assert np.isnan(one_day["sharpe"])
        assert one_day["max_drawdown"] == pytest.approx(-0.02)
        last_block = result.per_subperiod_metrics.iloc[-1]
        assert last_block["obs"] == 1
        assert np.isnan(last_block["ann_vol"])
        attribution = result.to_attribution_dict()
        regimes = cast(list[dict[str, object]], attribution["regimes"])
        recession = next(r for r in regimes if r["regime"] == "recession")
        assert recession["ann_vol"] is None

    def test_max_drawdown_counts_loss_from_starting_equity(self) -> None:
        """A block that opens with a loss is a drawdown from the initial 1.0."""
        dates = pd.bdate_range("2024-01-02", periods=4)
        macro = pd.DataFrame(
            {"pmi": 55.0, "spread_2s10s": 1.5, "hy_oas": 300.0}, index=dates
        )
        returns = pd.Series([-0.20, 0.01, 0.01, 0.01], index=dates)
        config = RegimeValidationConfig(min_regime_obs=1)

        result = run_regime_validation(returns, macro, config=config)

        assert result.per_regime_metrics.loc["expansion", "max_drawdown"] == (
            pytest.approx(-0.20)
        )
        assert result.per_subperiod_metrics["max_drawdown"].iloc[0] == (
            pytest.approx(-0.20)
        )

    def test_max_drawdown_zero_for_monotonic_gains(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=4)
        macro = pd.DataFrame(
            {"pmi": 55.0, "spread_2s10s": 1.5, "hy_oas": 300.0}, index=dates
        )
        returns = pd.Series([0.01, 0.02, 0.0, 0.01], index=dates)
        config = RegimeValidationConfig(min_regime_obs=1)

        result = run_regime_validation(returns, macro, config=config)

        assert result.per_regime_metrics.loc["expansion", "max_drawdown"] == 0.0

    def test_to_attribution_dict_structure(
        self,
        synthetic_oos_returns: pd.Series,
        three_regime_macro_data: pd.DataFrame,
    ) -> None:
        """to_attribution_dict returns expected keys and types."""
        result = run_regime_validation(synthetic_oos_returns, three_regime_macro_data)
        d = result.to_attribution_dict()

        assert "regimes" in d
        assert "subperiods" in d
        assert "summary" in d

        # Regime entries
        assert len(d["regimes"]) >= 1
        regime_entry = d["regimes"][0]
        expected_keys = (
            "regime",
            "obs",
            "coverage_pct",
            "ann_return",
            "ann_vol",
            "sharpe",
            "max_drawdown",
            "alpha_concentration",
            "is_concentrated",
        )
        for key in expected_keys:
            assert key in regime_entry

        # Subperiod entries
        assert len(d["subperiods"]) >= 1
        sp_entry = d["subperiods"][0]
        for key in (
            "start",
            "end",
            "regime",
            "obs",
            "ann_return",
            "ann_vol",
            "sharpe",
            "max_drawdown",
        ):
            assert key in sp_entry

        # Summary
        assert d["summary"]["total_obs"] == len(synthetic_oos_returns)
        assert isinstance(d["summary"]["concentrated_regimes"], list)
        assert isinstance(d["summary"]["has_concentration_warning"], bool)

    def test_to_attribution_dict_serializable(
        self,
        synthetic_oos_returns: pd.Series,
        three_regime_macro_data: pd.DataFrame,
    ) -> None:
        """to_attribution_dict output is JSON-serializable (no NaN, no Timestamps)."""
        import json

        result = run_regime_validation(synthetic_oos_returns, three_regime_macro_data)
        d = result.to_attribution_dict()
        # Should not raise
        serialized = json.dumps(d)
        assert isinstance(serialized, str)

    def test_to_attribution_dict_with_insufficient_obs(
        self,
        synthetic_oos_returns: pd.Series,
        three_regime_macro_data: pd.DataFrame,
    ) -> None:
        """NaN metrics become None in attribution dict (JSON-safe)."""
        import json

        config = RegimeValidationConfig(min_regime_obs=1000)
        result = run_regime_validation(
            synthetic_oos_returns, three_regime_macro_data, config=config
        )
        d = result.to_attribution_dict()
        # Should be JSON-serializable (NaN → None)
        serialized = json.dumps(d)
        assert isinstance(serialized, str)
        # Sharpe should be None for insufficient data
        for regime in d["regimes"]:
            assert regime["sharpe"] is None


class TestMixedFrequencyMacroData:
    def test_sparse_pmi_is_carried_forward_between_releases(self) -> None:
        """A monthly PMI on a daily frame keeps its last published value."""
        dates = pd.bdate_range("2024-01-02", periods=120)
        pmi = pd.Series(np.nan, index=dates)
        pmi.iloc[0] = 55.0
        macro = pd.DataFrame({"pmi": pmi, "spread_2s10s": 0.2}, index=dates)
        returns = pd.Series(0.001, index=dates)

        result = run_regime_validation(returns, macro)

        assert list(result.per_regime_metrics.index) == ["expansion"]
        assert result.per_regime_metrics.loc["expansion", "obs"] == len(dates)
        assert len(result.per_subperiod_metrics) == 1

    def test_carry_forward_does_not_look_ahead(self) -> None:
        """Days before the first PMI release stay neutral, not back-filled."""
        dates = pd.bdate_range("2024-01-02", periods=60)
        pmi = pd.Series(np.nan, index=dates)
        pmi.iloc[30] = 44.0
        macro = pd.DataFrame({"pmi": pmi, "spread_2s10s": -0.5}, index=dates)
        returns = pd.Series(0.001, index=dates)

        result = run_regime_validation(returns, macro)

        timeline = result.regime_timeline
        assert (timeline.iloc[:30] == "recovery").all()
        assert (timeline.iloc[30:] == "recession").all()

    def test_sparse_gdp_growth_keeps_native_release_cadence(self) -> None:
        """Forward-filling must not flatten the GDP series' period-over-period step."""
        dates = pd.bdate_range("2024-01-02", periods=8)
        gdp = pd.Series(np.nan, index=dates)
        gdp.iloc[[0, 2, 4, 6]] = [1.0, 1.5, 2.0, 3.0]
        macro = pd.DataFrame({"gdp_growth": gdp}, index=dates)
        returns = pd.Series(0.001, index=dates)

        result = run_regime_validation(
            returns, macro, config=RegimeValidationConfig(min_regime_obs=1)
        )

        assert result.regime_timeline.iloc[-1] == "expansion"


_EXPANSION = {"pmi": 55.0, "spread_2s10s": 1.5, "hy_oas": 300.0}
_RECOVERY = {"pmi": 46.0, "spread_2s10s": 0.5, "hy_oas": 400.0}
_SLOWDOWN = {"pmi": 50.0, "spread_2s10s": 0.5, "hy_oas": 400.0}


def _blocks_macro(*blocks: tuple[dict[str, float], int]) -> pd.DataFrame:
    """Concatenate constant-indicator blocks on a business-day index."""
    n = sum(length for _, length in blocks)
    dates = pd.bdate_range("2024-01-02", periods=n)
    rows = [ind for ind, length in blocks for _ in range(length)]
    return pd.DataFrame(rows, index=dates)


class TestAlphaConcentration:
    def test_dominant_regime_with_uniform_daily_edge_is_not_flagged(self) -> None:
        """Identical per-day performance must not read as concentrated alpha."""
        macro = _blocks_macro((_EXPANSION, 255), (_RECOVERY, 45))
        returns = pd.Series(0.0005, index=macro.index)

        result = run_regime_validation(returns, macro)

        assert result.n_regimes_observed == 2
        assert result.concentrated_regimes == []
        conc = result.regime_alpha_concentration
        assert conc["expansion"] == pytest.approx(0.5)
        assert conc["recovery"] == pytest.approx(0.5)

    def test_regimes_below_min_obs_take_no_share(self) -> None:
        """An under-sampled regime has NaN metrics and must not absorb alpha."""
        macro = _blocks_macro((_EXPANSION, 60), (_RECOVERY, 60), (_SLOWDOWN, 3))
        returns = pd.Series(
            [0.001] * 120 + [0.05] * 3,
            index=macro.index,
        )

        result = run_regime_validation(returns, macro)

        assert not result.per_regime_metrics.loc["slowdown", "obs_sufficient"]
        conc = result.regime_alpha_concentration
        assert conc["slowdown"] == 0.0
        assert conc["expansion"] == pytest.approx(0.5)
        assert conc["recovery"] == pytest.approx(0.5)
        assert result.concentrated_regimes == []

    def test_concentration_uses_excess_over_risk_free_rate(self) -> None:
        """Alpha is return above the risk-free rate, as in the Sharpe ratio."""
        macro = _blocks_macro((_EXPANSION, 60), (_RECOVERY, 60))
        returns = pd.Series([0.0006] * 60 + [0.0004] * 60, index=macro.index)
        config = RegimeValidationConfig(risk_free_rate=0.0004 * 252)

        result = run_regime_validation(returns, macro, config=config)

        conc = result.regime_alpha_concentration
        assert conc["expansion"] == pytest.approx(1.0)
        assert conc["recovery"] == pytest.approx(0.0)
        assert result.concentrated_regimes == ["expansion"]

    def test_concentration_flags_regime_with_all_the_edge(self) -> None:
        macro = _blocks_macro((_EXPANSION, 60), (_RECOVERY, 60))
        returns = pd.Series([0.002] * 60 + [-0.001] * 60, index=macro.index)

        result = run_regime_validation(returns, macro)

        assert result.concentrated_regimes == ["expansion"]


class TestMacroDataOrdering:
    @staticmethod
    def _sparse_pmi_macro() -> tuple[pd.DataFrame, pd.Series]:
        dates = pd.bdate_range("2024-01-02", periods=180)
        pmi = pd.Series(np.nan, index=dates)
        pmi.iloc[[0, 60, 120]] = [55.0, 46.0, 55.0]
        macro = pd.DataFrame({"pmi": pmi, "spread_2s10s": 0.2}, index=dates)
        rng = np.random.default_rng(1)
        returns = pd.Series(rng.normal(0.001, 0.01, size=180), index=dates)
        return macro, returns

    def test_newest_first_macro_data_gives_the_same_regimes(self) -> None:
        """Row order of the macro frame must not decide what counts as past."""
        macro, returns = self._sparse_pmi_macro()

        expected = run_regime_validation(returns, macro)
        result = run_regime_validation(returns, macro.iloc[::-1])

        pd.testing.assert_series_equal(result.regime_timeline, expected.regime_timeline)
        pd.testing.assert_frame_equal(
            result.per_regime_metrics, expected.per_regime_metrics
        )

    def test_shuffled_macro_data_gives_the_same_regimes(self) -> None:
        macro, returns = self._sparse_pmi_macro()

        expected = run_regime_validation(returns, macro)
        result = run_regime_validation(returns, macro.sample(frac=1.0, random_state=0))

        pd.testing.assert_series_equal(result.regime_timeline, expected.regime_timeline)

    def test_duplicate_macro_dates_keep_the_latest_row(self) -> None:
        """A superseded row must not leak its values into the carry-forward.

        The surviving row has no PMI release; if the superseded duplicate were
        kept, forward-filling would hand it the superseded PMI of 40.0.
        """
        dates = pd.bdate_range("2024-01-02", periods=60)
        pmi = pd.Series(np.nan, index=dates)
        pmi.iloc[30] = 55.0
        macro = pd.DataFrame({"pmi": pmi, "spread_2s10s": 0.2}, index=dates)
        superseded = macro.iloc[[0]].copy()
        superseded["pmi"] = 40.0
        with_duplicate = pd.concat([superseded, macro])
        returns = pd.Series(0.001, index=dates)

        expected = run_regime_validation(returns, macro)
        result = run_regime_validation(returns, with_duplicate)

        pd.testing.assert_series_equal(result.regime_timeline, expected.regime_timeline)


class TestMacroDataStaleness:
    @staticmethod
    def _expansion_macro_ending_early() -> tuple[pd.DataFrame, pd.Series]:
        macro_dates = pd.bdate_range("2024-01-02", periods=60)
        macro = pd.DataFrame(
            {"pmi": 55.0, "spread_2s10s": 1.5, "hy_oas": 300.0}, index=macro_dates
        )
        oos_dates = pd.bdate_range("2024-01-02", periods=250)
        rng = np.random.default_rng(3)
        returns = pd.Series(rng.normal(0.0005, 0.01, size=250), index=oos_dates)
        return macro, returns

    def test_regime_is_not_carried_past_the_staleness_limit(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Macro data that stops early must not label the rest of the sample."""
        macro, returns = self._expansion_macro_ending_early()
        last_macro = macro.index[-1]
        limit = pd.Timedelta(days=120)
        expected_obs = int((returns.index <= last_macro + limit).sum())
        assert expected_obs < len(returns)

        with caplog.at_level("WARNING"):
            result = run_regime_validation(returns, macro)

        assert result.per_regime_metrics.loc["expansion", "obs"] == expected_obs
        assert result.per_regime_metrics.loc["expansion", "coverage_pct"] < 1.0
        assert len(result.regime_timeline) == expected_obs
        assert result.total_obs == len(returns)
        assert "stale" in caplog.text

    def test_include_unknown_marks_stale_days_unknown(self) -> None:
        macro, returns = self._expansion_macro_ending_early()
        config = RegimeValidationConfig(include_unknown_regime=True)

        result = run_regime_validation(returns, macro, config=config)

        assert result.regime_timeline.iloc[-1] == "unknown"
        assert result.regime_timeline.iloc[0] == "expansion"

    def test_unbounded_staleness_restores_open_ended_carry_forward(self) -> None:
        macro, returns = self._expansion_macro_ending_early()
        config = RegimeValidationConfig(max_macro_staleness_days=None)

        result = run_regime_validation(returns, macro, config=config)

        assert result.per_regime_metrics.loc["expansion", "coverage_pct"] == 1.0

    def test_quarterly_cadence_stays_within_the_default_limit(self) -> None:
        """Quarter-start stamps leave gaps of about 90 days between releases."""
        oos_dates = pd.bdate_range("2024-01-02", periods=250)
        macro = pd.DataFrame(
            {"pmi": 55.0, "spread_2s10s": 1.5, "hy_oas": 300.0},
            index=pd.DatetimeIndex(
                ["2024-01-01", "2024-04-01", "2024-07-01", "2024-10-01"]
            ),
        )
        returns = pd.Series(0.001, index=oos_dates)

        result = run_regime_validation(returns, macro)

        assert result.per_regime_metrics.loc["expansion", "coverage_pct"] == 1.0
