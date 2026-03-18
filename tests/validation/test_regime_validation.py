"""Tests for regime-conditional subperiod Sharpe validation."""

from __future__ import annotations

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
            "regime", "obs", "coverage_pct", "ann_return",
            "ann_vol", "sharpe", "max_drawdown",
            "alpha_concentration", "is_concentrated",
        )
        for key in expected_keys:
            assert key in regime_entry

        # Subperiod entries
        assert len(d["subperiods"]) >= 1
        sp_entry = d["subperiods"][0]
        for key in ("start", "end", "regime", "obs", "ann_return",
                     "ann_vol", "sharpe", "max_drawdown"):
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
