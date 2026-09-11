"""Tests for the factor-mimicking-returns to skfolio factor-prior bridge."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from skfolio.prior import LoadingMatrixRegression, TimeSeriesFactorModel

from optimizer.exceptions import ConfigurationError, DataError
from optimizer.factors import (
    FactorPriorConfig,
    build_all_factor_mimicking_portfolios,
    build_time_series_factor_model,
    fit_factor_prior,
)

N_DATES = 80
N_ASSETS = 12
DATES = pd.date_range("2022-01-01", periods=N_DATES, freq="B")
ASSETS = [f"A{i:02d}" for i in range(N_ASSETS)]
FACTORS = ["value", "momentum", "quality"]


@pytest.fixture()
def asset_returns() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        rng.normal(0.0004, 0.02, (N_DATES, N_ASSETS)),
        index=DATES,
        columns=ASSETS,
    )


@pytest.fixture()
def factor_returns() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    return pd.DataFrame(
        rng.normal(0.0003, 0.01, (N_DATES, len(FACTORS))),
        index=DATES,
        columns=FACTORS,
    )


class TestFactorPriorConfig:
    def test_defaults(self) -> None:
        cfg = FactorPriorConfig()
        assert cfg.higham is False
        assert cfg.max_iteration == 100
        assert cfg.min_observations == 12

    def test_for_higham_preset(self) -> None:
        assert FactorPriorConfig.for_higham().higham is True

    def test_frozen(self) -> None:
        cfg = FactorPriorConfig()
        with pytest.raises(Exception):  # noqa: B017 - FrozenInstanceError
            cfg.higham = True  # type: ignore[misc]

    def test_non_positive_max_iteration_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="max_iteration"):
            FactorPriorConfig(max_iteration=0)

    def test_too_small_min_observations_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="min_observations"):
            FactorPriorConfig(min_observations=1)


class TestBuildTimeSeriesFactorModel:
    def test_returns_unfitted_estimator(self) -> None:
        model = build_time_series_factor_model()
        assert isinstance(model, TimeSeriesFactorModel)
        assert not hasattr(model, "return_distribution_")

    def test_forwards_higham_flag(self) -> None:
        model = build_time_series_factor_model(FactorPriorConfig.for_higham())
        assert model.higham is True

    def test_accepts_custom_loading_matrix_estimator(self) -> None:
        lmr = LoadingMatrixRegression()
        model = build_time_series_factor_model(loading_matrix_estimator=lmr)
        assert model.loading_matrix_estimator is lmr


class TestFitFactorPrior:
    def test_fits_and_exposes_return_distribution(
        self, asset_returns: pd.DataFrame, factor_returns: pd.DataFrame
    ) -> None:
        model = fit_factor_prior(asset_returns, factor_returns)
        rd = model.return_distribution_
        assert np.asarray(rd.mu).shape == (N_ASSETS,)
        assert np.asarray(rd.covariance).shape == (N_ASSETS, N_ASSETS)

    def test_end_to_end_from_mimicking_portfolios(self) -> None:
        rng = np.random.default_rng(21)
        scores = {
            name: pd.DataFrame(
                rng.standard_normal((N_DATES, N_ASSETS)),
                index=DATES,
                columns=ASSETS,
            )
            for name in FACTORS
        }
        rets = pd.DataFrame(
            rng.normal(0.0004, 0.02, (N_DATES, N_ASSETS)),
            index=DATES,
            columns=ASSETS,
        )
        fac = build_all_factor_mimicking_portfolios(scores, rets)
        model = fit_factor_prior(rets, fac)
        assert np.asarray(model.return_distribution_.mu).shape == (N_ASSETS,)

    def test_drops_non_finite_periods(
        self, asset_returns: pd.DataFrame, factor_returns: pd.DataFrame
    ) -> None:
        # Inject NaN into a handful of factor rows; must not raise (skfolio
        # itself rejects NaN, so the bridge must drop those periods).
        f = factor_returns.copy()
        f.iloc[:5, 0] = np.nan
        model = fit_factor_prior(asset_returns, f)
        assert hasattr(model, "return_distribution_")

    def test_no_common_dates_raises(
        self, asset_returns: pd.DataFrame, factor_returns: pd.DataFrame
    ) -> None:
        shifted = factor_returns.copy()
        shifted.index = shifted.index + pd.Timedelta(days=5000)
        with pytest.raises(DataError, match="common dates"):
            fit_factor_prior(asset_returns, shifted)

    def test_insufficient_finite_periods_raises(
        self, asset_returns: pd.DataFrame, factor_returns: pd.DataFrame
    ) -> None:
        few = asset_returns.iloc[:8]
        few_f = factor_returns.iloc[:8]
        with pytest.raises(DataError, match="fully-finite"):
            fit_factor_prior(few, few_f)

    def test_empty_frame_raises(self, asset_returns: pd.DataFrame) -> None:
        with pytest.raises(DataError, match="non-empty"):
            fit_factor_prior(asset_returns.iloc[:0], asset_returns.iloc[:0])
