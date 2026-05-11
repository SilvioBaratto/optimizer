"""Tests for research.pipeline._metrics — pure metric computation functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# _annualized_return
# ---------------------------------------------------------------------------


class TestAnnualizedReturn:
    def test_when_empty_series_then_returns_nan(self) -> None:
        from research.pipeline._metrics import _annualized_return

        result = _annualized_return(pd.Series(dtype=float))
        assert np.isnan(result)

    def test_when_flat_zero_returns_then_returns_zero(self) -> None:
        from research.pipeline._metrics import _annualized_return

        r = pd.Series([0.0] * 252)
        result = _annualized_return(r)
        assert result == pytest.approx(0.0)

    def test_when_positive_returns_then_compounds_correctly(self) -> None:
        from research.pipeline._metrics import _annualized_return

        r = pd.Series([0.001] * 252)
        result = _annualized_return(r)
        expected = (1.001) ** 252 - 1.0
        assert result == pytest.approx(expected, rel=1e-9)

    def test_when_below_one_year_then_still_annualizes(self) -> None:
        from research.pipeline._metrics import _annualized_return

        r = pd.Series([0.001] * 126)
        result = _annualized_return(r)
        expected = (1.001) ** 252 - 1.0
        assert result == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# _daily_rf
# ---------------------------------------------------------------------------


class TestDailyRf:
    def test_when_rf_none_then_returns_zero_series(self) -> None:
        from research.pipeline._metrics import _daily_rf

        idx = pd.date_range("2024-01-02", periods=3, freq="B")
        returns = pd.Series([0.01, -0.02, 0.005], index=idx)
        result = _daily_rf(returns, None)
        assert (result == 0.0).all()
        assert result.index.equals(returns.index)

    def test_when_rf_empty_then_returns_zero_series(self) -> None:
        from research.pipeline._metrics import _daily_rf

        idx = pd.date_range("2024-01-02", periods=2, freq="B")
        returns = pd.Series([0.01, -0.02], index=idx)
        result = _daily_rf(returns, pd.Series(dtype=float))
        assert (result == 0.0).all()

    def test_when_rf_provided_then_forward_fills_and_divides_by_252(self) -> None:
        from research.pipeline._metrics import _daily_rf

        idx = pd.date_range("2024-01-02", periods=5, freq="B")
        returns = pd.Series([0.01] * 5, index=idx)
        rf = pd.Series([0.05, 0.06], index=[idx[0], idx[3]])
        result = _daily_rf(returns, rf)
        expected = pd.Series(
            [0.05 / 252, 0.05 / 252, 0.05 / 252, 0.06 / 252, 0.06 / 252],
            index=idx,
        )
        pd.testing.assert_series_equal(result, expected)


# ---------------------------------------------------------------------------
# _sharpe
# ---------------------------------------------------------------------------


class TestSharpe:
    def test_when_empty_returns_then_returns_nan(self) -> None:
        from research.pipeline._metrics import _sharpe

        result = _sharpe(pd.Series(dtype=float))
        assert np.isnan(result)

    def test_when_zero_vol_then_returns_nan(self) -> None:
        from research.pipeline._metrics import _sharpe

        # Single-element series → std(ddof=1) = 0.0 → vol = 0 → NaN
        idx = pd.date_range("2024-01-01", periods=1, freq="B")
        r = pd.Series([0.001], index=idx)
        result = _sharpe(r)
        assert np.isnan(result)

    def test_when_no_rf_then_uses_zero(self) -> None:
        from research.pipeline._metrics import _sharpe

        idx = pd.date_range("2024-01-01", periods=252, freq="B")
        rng = np.random.default_rng(42)
        r = pd.Series(rng.normal(0.001, 0.015, 252), index=idx)
        result = _sharpe(r)
        assert result > 0.0
        assert not np.isnan(result)

    def test_when_rf_series_provided_then_adjusts_excess(self) -> None:
        from research.pipeline._metrics import _sharpe

        idx = pd.date_range("2024-01-01", periods=252, freq="B")
        r = pd.Series(np.linspace(0.0005, 0.002, 252), index=idx)
        rf = pd.Series([0.05] * 252, index=idx)
        result = _sharpe(r, rf)
        assert result > 0.0


# ---------------------------------------------------------------------------
# _sortino
# ---------------------------------------------------------------------------


class TestSortino:
    def test_when_empty_returns_then_returns_zero(self) -> None:
        from research.pipeline._metrics import _sortino

        result = _sortino(pd.Series(dtype=float))
        assert result == 0.0

    def test_when_always_positive_then_returns_zero(self) -> None:
        from research.pipeline._metrics import _sortino

        idx = pd.date_range("2024-01-01", periods=252, freq="B")
        r = pd.Series([0.001] * 252, index=idx)
        result = _sortino(r)
        assert result == 0.0

    def test_when_mixed_returns_then_positive(self) -> None:
        from research.pipeline._metrics import _sortino

        idx = pd.date_range("2024-01-01", periods=252, freq="B")
        rng = np.random.default_rng(42)
        r = pd.Series(rng.normal(0.001, 0.015, 252), index=idx)
        result = _sortino(r)
        assert result > 0.0

    def test_when_rf_provided_then_adjusts_downside(self) -> None:
        from research.pipeline._metrics import _sortino

        idx = pd.date_range("2024-01-01", periods=252, freq="B")
        r = pd.Series(np.linspace(0.0001, 0.002, 252), index=idx)
        rf = pd.Series([0.05] * 252, index=idx)
        result = _sortino(r, rf)
        assert not np.isnan(result)


# ---------------------------------------------------------------------------
# _downside_vol
# ---------------------------------------------------------------------------


class TestDownsideVol:
    def test_when_empty_returns_then_returns_zero(self) -> None:
        from research.pipeline._metrics import _downside_vol

        result = _downside_vol(pd.Series(dtype=float))
        assert result == 0.0

    def test_when_always_positive_then_returns_zero(self) -> None:
        from research.pipeline._metrics import _downside_vol

        idx = pd.date_range("2024-01-01", periods=252, freq="B")
        r = pd.Series([0.001] * 252, index=idx)
        result = _downside_vol(r)
        assert result == 0.0

    def test_when_mixed_returns_then_positive(self) -> None:
        from research.pipeline._metrics import _downside_vol

        idx = pd.date_range("2024-01-01", periods=252, freq="B")
        rng = np.random.default_rng(42)
        r = pd.Series(rng.normal(0.001, 0.015, 252), index=idx)
        result = _downside_vol(r)
        assert result > 0.0


# ---------------------------------------------------------------------------
# _information_ratio
# ---------------------------------------------------------------------------


class TestInformationRatio:
    def test_when_portfolio_empty_then_returns_zero(self) -> None:
        from research.pipeline._metrics import _information_ratio

        idx = pd.date_range("2024-01-02", periods=2, freq="B")
        bm = pd.Series([0.01, -0.02], index=idx)
        result = _information_ratio(pd.Series(dtype=float), bm)
        assert result == 0.0

    def test_when_benchmark_empty_then_returns_zero(self) -> None:
        from research.pipeline._metrics import _information_ratio

        idx = pd.date_range("2024-01-02", periods=2, freq="B")
        pf = pd.Series([0.01, -0.02], index=idx)
        result = _information_ratio(pf, pd.Series(dtype=float))
        assert result == 0.0

    def test_when_no_common_dates_then_returns_nan(self) -> None:
        from research.pipeline._metrics import _information_ratio

        pf = pd.Series([0.01], index=pd.date_range("2024-01-02", periods=1, freq="B"))
        bm = pd.Series([0.02], index=pd.date_range("2024-01-03", periods=1, freq="B"))
        result = _information_ratio(pf, bm)
        assert np.isnan(result)

    def test_when_zero_tracking_error_then_returns_zero(self) -> None:
        from research.pipeline._metrics import _information_ratio

        idx = pd.date_range("2024-01-02", periods=252, freq="B")
        r = pd.Series([0.001] * 252, index=idx)
        result = _information_ratio(r, r.copy())
        assert result == 0.0

    def test_when_portfolio_outperforms_then_positive(self) -> None:
        from research.pipeline._metrics import _information_ratio

        idx = pd.date_range("2024-01-02", periods=252, freq="B")
        pf = pd.Series(np.linspace(0.001, 0.002, 252), index=idx)
        bm = pd.Series(np.linspace(0.0005, 0.001, 252), index=idx)
        result = _information_ratio(pf, bm)
        assert result > 0.0


# ---------------------------------------------------------------------------
# _METRICS_KEY_MAP
# ---------------------------------------------------------------------------


class TestMetricsKeyMap:
    def test_contains_expected_keys(self) -> None:
        from research.pipeline._metrics import _METRICS_KEY_MAP

        assert "Ann. Return" in _METRICS_KEY_MAP
        assert "Sharpe (rf)" in _METRICS_KEY_MAP
        assert "Sortino" in _METRICS_KEY_MAP
        assert "Info Ratio" in _METRICS_KEY_MAP

    def test_all_values_are_snake_case(self) -> None:
        from research.pipeline._metrics import _METRICS_KEY_MAP

        for v in _METRICS_KEY_MAP.values():
            assert v == v.lower()
            assert " " not in v


# ---------------------------------------------------------------------------
# _to_json_safe
# ---------------------------------------------------------------------------


class TestToJsonSafe:
    def test_when_none_input_then_returns_none(self) -> None:
        from research.pipeline._metrics import _to_json_safe

        assert _to_json_safe(None) is None

    def test_when_nan_then_returns_none(self) -> None:
        from research.pipeline._metrics import _to_json_safe

        assert _to_json_safe(float("nan")) is None

    def test_when_finite_float_then_returns_same(self) -> None:
        from research.pipeline._metrics import _to_json_safe

        assert _to_json_safe(3.14) == 3.14

    def test_when_numpy_float_then_returns_python_float(self) -> None:
        from research.pipeline._metrics import _to_json_safe

        result = _to_json_safe(np.float64(2.718))
        assert result == pytest.approx(2.718)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# _project_metrics
# ---------------------------------------------------------------------------


class TestProjectMetrics:
    def test_when_metrics_provided_then_maps_keys(self) -> None:
        from research.pipeline._metrics import _project_metrics

        metrics = {
            "Ann. Return": 0.15,
            "Ann. Vol": 0.12,
            "Sharpe (rf)": 1.25,
            "Sortino": 1.8,
            "Info Ratio": 0.9,
            "Downside Vol": 0.08,
            "Max Drawdown": -0.15,
        }
        result = _project_metrics(metrics)
        assert result["ann_return"] == 0.15
        assert result["sharpe"] == 1.25
        assert result["sortino"] == 1.8

    def test_when_key_missing_then_returns_none(self) -> None:
        from research.pipeline._metrics import _project_metrics

        result = _project_metrics({})
        assert result["ann_return"] is None
        assert result["sharpe"] is None


# ---------------------------------------------------------------------------
# TOP_N_DISPLAY
# ---------------------------------------------------------------------------


class TestTopNDisplay:
    def test_is_positive_integer(self) -> None:
        from research.pipeline._metrics import TOP_N_DISPLAY

        assert isinstance(TOP_N_DISPLAY, int)
        assert TOP_N_DISPLAY > 0
