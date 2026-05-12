"""Tests for research/_preprocessing.py — FX + sklearn return pipeline (issue #521)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("typer")

from skfolio.preprocessing import prices_to_returns
from sklearn.pipeline import Pipeline

from optimizer.preprocessing import (
    DataValidator,
    OutlierTreater,
    RegressionImputer,
    SectorImputer,
)
from research.returns import (
    apply_fx_to_prices,
    build_return_preprocessing_pipeline,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_synthetic_prices(
    n_days: int = 50,
    tickers: tuple[str, ...] = ("AAA", "BBB", "CCC", "DDD", "EEE"),
) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    increments = rng.normal(0, 0.01, size=(n_days, len(tickers)))
    data = 100.0 * np.exp(np.cumsum(increments, axis=0))
    return pd.DataFrame(data, index=dates, columns=list(tickers))


def _build_fx_rates(index: pd.Index, ccys: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        {ccy: np.full(len(index), 0.9) for ccy in ccys},
        index=index,
    )


# ---------------------------------------------------------------------------
# apply_fx_to_prices
# ---------------------------------------------------------------------------


class TestApplyFxToPrices:
    def test_when_all_tickers_already_eur_then_output_equals_input(self) -> None:
        prices = _build_synthetic_prices()
        ccy_map = dict.fromkeys(prices.columns, "EUR")
        fx_rates = pd.DataFrame(index=prices.index)
        out = apply_fx_to_prices(prices, ccy_map, fx_rates, base_currency="EUR")
        pd.testing.assert_frame_equal(out, prices)

    def test_when_one_ticker_in_usd_then_converted_by_rate(self) -> None:
        prices = _build_synthetic_prices()
        ccy_map = {"AAA": "USD"}
        for c in prices.columns[1:]:
            ccy_map[c] = "EUR"
        fx_rates = _build_fx_rates(prices.index, ("USD",))
        out = apply_fx_to_prices(prices, ccy_map, fx_rates, base_currency="EUR")
        np.testing.assert_allclose(out["AAA"].values, prices["AAA"].values * 0.9)
        for col in prices.columns[1:]:
            np.testing.assert_allclose(out[col].values, prices[col].values)

    def test_when_called_then_shape_preserved(self) -> None:
        prices = _build_synthetic_prices()
        ccy_map = {"AAA": "USD", "BBB": "GBP", "CCC": "EUR", "DDD": "EUR", "EEE": "EUR"}
        fx_rates = _build_fx_rates(prices.index, ("USD", "GBP"))
        out = apply_fx_to_prices(prices, ccy_map, fx_rates)
        assert out.shape == prices.shape
        assert list(out.columns) == list(prices.columns)
        assert list(out.index) == list(prices.index)

    def test_when_called_then_no_inf_introduced(self) -> None:
        prices = _build_synthetic_prices()
        ccy_map = {"AAA": "USD", "BBB": "EUR", "CCC": "EUR", "DDD": "EUR", "EEE": "EUR"}
        fx_rates = _build_fx_rates(prices.index, ("USD",))
        out = apply_fx_to_prices(prices, ccy_map, fx_rates)
        assert not np.isinf(out.to_numpy()).any()

    def test_when_called_then_no_new_nan_introduced(self) -> None:
        prices = _build_synthetic_prices()
        ccy_map = {"AAA": "USD", "BBB": "EUR", "CCC": "EUR", "DDD": "EUR", "EEE": "EUR"}
        fx_rates = _build_fx_rates(prices.index, ("USD",))
        out = apply_fx_to_prices(prices, ccy_map, fx_rates)
        # Cells finite in input must remain finite in output (with full FX coverage)
        assert out.notna().all().all()


# ---------------------------------------------------------------------------
# build_return_preprocessing_pipeline
# ---------------------------------------------------------------------------


class TestBuildReturnPreprocessingPipeline:
    def test_when_called_then_returns_sklearn_pipeline(self) -> None:
        pipe = build_return_preprocessing_pipeline(sector_mapping={})
        assert isinstance(pipe, Pipeline)

    def test_when_called_then_steps_in_correct_order(self) -> None:
        pipe = build_return_preprocessing_pipeline(sector_mapping={})
        names = [name for name, _ in pipe.steps]
        assert names == ["validate", "outlier", "sector_impute", "regression_impute"]

    def test_when_called_then_step_types_correct(self) -> None:
        pipe = build_return_preprocessing_pipeline(sector_mapping={})
        steps = dict(pipe.steps)
        assert isinstance(steps["validate"], DataValidator)
        assert isinstance(steps["outlier"], OutlierTreater)
        assert isinstance(steps["sector_impute"], SectorImputer)
        assert isinstance(steps["regression_impute"], RegressionImputer)

    def test_when_called_then_outlier_thresholds_match_spec(self) -> None:
        pipe = build_return_preprocessing_pipeline(sector_mapping={})
        outlier: OutlierTreater = pipe.named_steps["outlier"]
        assert outlier.winsorize_threshold == 4.0
        assert outlier.remove_threshold == 8.0

    def test_when_called_then_regression_imputer_settings_match_spec(self) -> None:
        pipe = build_return_preprocessing_pipeline(sector_mapping={})
        ri: RegressionImputer = pipe.named_steps["regression_impute"]
        assert ri.n_neighbors == 5
        assert ri.min_train_periods == 60

    def test_when_mapping_provided_then_propagated_to_sector_imputer(self) -> None:
        mapping = {"AAA": "Tech", "BBB": "Health"}
        pipe = build_return_preprocessing_pipeline(sector_mapping=mapping)
        si: SectorImputer = pipe.named_steps["sector_impute"]
        assert si.sector_mapping == mapping


# ---------------------------------------------------------------------------
# End-to-end: FX → returns → pipeline
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_when_chain_run_then_output_has_no_nan_or_inf(self) -> None:
        n_days = 80
        tickers = ("AAA", "BBB", "CCC", "DDD", "EEE")
        prices = _build_synthetic_prices(n_days=n_days, tickers=tickers)
        ccy_map = {"AAA": "USD", "BBB": "EUR", "CCC": "EUR", "DDD": "EUR", "EEE": "GBP"}
        fx_rates = _build_fx_rates(prices.index, ("USD", "GBP"))

        eur_prices = apply_fx_to_prices(prices, ccy_map, fx_rates)
        returns = prices_to_returns(eur_prices)

        sector_mapping = dict.fromkeys(tickers, "Tech")
        pipe = build_return_preprocessing_pipeline(sector_mapping=sector_mapping)
        out: pd.DataFrame = pipe.fit_transform(returns)

        assert isinstance(out, pd.DataFrame)
        arr = out.to_numpy()
        assert not np.isnan(arr).any()
        assert not np.isinf(arr).any()

    def test_when_input_has_nan_then_pipeline_imputes_it(self) -> None:
        n_days = 80
        tickers = ("AAA", "BBB", "CCC", "DDD", "EEE")
        prices = _build_synthetic_prices(n_days=n_days, tickers=tickers)
        ccy_map = dict.fromkeys(tickers, "EUR")
        fx_rates = pd.DataFrame(index=prices.index)

        eur_prices = apply_fx_to_prices(prices, ccy_map, fx_rates)
        returns = prices_to_returns(eur_prices)
        # Inject NaN cells in the middle to simulate missing data
        returns.iloc[20, 0] = np.nan
        returns.iloc[30, 2] = np.nan

        sector_mapping = dict.fromkeys(tickers, "Tech")
        pipe = build_return_preprocessing_pipeline(sector_mapping=sector_mapping)
        out: pd.DataFrame = pipe.fit_transform(returns)

        assert not np.isnan(out.to_numpy()).any()
