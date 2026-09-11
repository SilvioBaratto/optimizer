"""Tests for the cross-sectional linear-regression factory."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from skfolio.linear_model import CSLinearRegression, CSLinearRegressorWrapper
from sklearn.linear_model import Ridge

from optimizer.exceptions import ConfigurationError
from optimizer.linear_model import (
    CSLinearRegressionConfig,
    CSLinearRegressorWrapperConfig,
    build_cs_linear_regression,
    build_cs_linear_regressor_wrapper,
)


class TestCSLinearRegressionConfig:
    def test_when_default_then_fit_intercept_true(self) -> None:
        cfg = CSLinearRegressionConfig()
        assert cfg.fit_intercept is True

    def test_when_default_then_weighted_false(self) -> None:
        cfg = CSLinearRegressionConfig()
        assert cfg.weighted is False

    def test_when_default_then_min_observations_10(self) -> None:
        cfg = CSLinearRegressionConfig()
        assert cfg.min_observations == 10

    def test_when_constructed_then_frozen(self) -> None:
        cfg = CSLinearRegressionConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.fit_intercept = False  # type: ignore[misc]

    def test_when_min_observations_negative_then_raises(self) -> None:
        from optimizer.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="min_observations"):
            CSLinearRegressionConfig(min_observations=-1)


class TestPresets:
    def test_when_for_default_then_default_fields(self) -> None:
        cfg = CSLinearRegressionConfig.for_default()
        assert cfg.fit_intercept is True
        assert cfg.weighted is False

    def test_when_for_weighted_then_weighted_true(self) -> None:
        cfg = CSLinearRegressionConfig.for_weighted()
        assert cfg.weighted is True


class TestBuildCSLinearRegression:
    def test_when_built_then_returns_skfolio_class(self) -> None:
        est = build_cs_linear_regression(CSLinearRegressionConfig())
        assert isinstance(est, CSLinearRegression)

    def test_when_fit_intercept_true_then_forwarded(self) -> None:
        cfg = CSLinearRegressionConfig(fit_intercept=True)
        est = build_cs_linear_regression(cfg)
        assert est.fit_intercept is True

    def test_when_fit_intercept_false_then_forwarded(self) -> None:
        cfg = CSLinearRegressionConfig(fit_intercept=False)
        est = build_cs_linear_regression(cfg)
        assert est.fit_intercept is False


class TestSklearnRoundTrip:
    @pytest.fixture(scope="class")
    def cs_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        T, N, K = 60, 8, 3
        X = rng.normal(size=(T, N, K))
        beta_true = rng.normal(size=K)
        y = (X @ beta_true).reshape(T, N) + 0.1 * rng.normal(size=(T, N))
        return X, y, beta_true

    def test_when_fit_then_coef_matches_panel_shape(
        self,
        cs_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        X, y, _ = cs_data
        est = build_cs_linear_regression(CSLinearRegressionConfig.for_default())
        est.fit(X, y)
        T, _, K = X.shape
        assert est.coef_.shape == (T, K)

    def test_when_predict_then_output_matches_y_shape(
        self,
        cs_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        X, y, _ = cs_data
        est = build_cs_linear_regression(CSLinearRegressionConfig.for_default())
        est.fit(X, y)
        pred = est.predict(X)
        assert pred.shape == y.shape

    def test_when_fit_then_recovers_true_beta(
        self,
        cs_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        X, y, beta_true = cs_data
        est = build_cs_linear_regression(CSLinearRegressionConfig(fit_intercept=False))
        est.fit(X, y)
        # Average of per-period coefs should be close to true beta.
        assert np.allclose(est.coef_.mean(axis=0), beta_true, atol=0.05)

    def test_when_cs_weights_zero_then_nan_pairs_excluded(
        self,
        cs_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        X, y, _ = cs_data
        T, N, _ = X.shape
        w = np.ones((T, N))
        w[0, 0] = 0.0
        X_nan = X.copy()
        X_nan[0, 0, :] = np.nan  # excluded pair may be NaN
        est = build_cs_linear_regression(CSLinearRegressionConfig.for_weighted())
        est.fit(X_nan, y, cs_weights=w)
        assert np.isfinite(est.coef_).all()


class TestCSLinearRegressorWrapperConfig:
    def test_when_default_then_n_jobs_one(self) -> None:
        cfg = CSLinearRegressorWrapperConfig()
        assert cfg.n_jobs == 1
        assert cfg.weighted is False
        assert cfg.min_observations == 10

    def test_when_constructed_then_frozen(self) -> None:
        cfg = CSLinearRegressorWrapperConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.n_jobs = 2  # type: ignore[misc]

    def test_when_n_jobs_zero_then_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="n_jobs"):
            CSLinearRegressorWrapperConfig(n_jobs=0)

    def test_when_n_jobs_below_minus_one_then_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="n_jobs"):
            CSLinearRegressorWrapperConfig(n_jobs=-2)

    def test_when_n_jobs_minus_one_then_allowed(self) -> None:
        cfg = CSLinearRegressorWrapperConfig(n_jobs=-1)
        assert cfg.n_jobs == -1

    def test_when_min_observations_negative_then_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="min_observations"):
            CSLinearRegressorWrapperConfig(min_observations=-1)

    def test_when_for_default_then_default_fields(self) -> None:
        cfg = CSLinearRegressorWrapperConfig.for_default()
        assert cfg.n_jobs == 1
        assert cfg.weighted is False

    def test_when_for_weighted_then_weighted_true(self) -> None:
        cfg = CSLinearRegressorWrapperConfig.for_weighted()
        assert cfg.weighted is True


class TestBuildCSLinearRegressorWrapper:
    def test_when_built_then_returns_skfolio_class(self) -> None:
        est = build_cs_linear_regressor_wrapper(
            CSLinearRegressorWrapperConfig(), regressor=Ridge(alpha=1.0)
        )
        assert isinstance(est, CSLinearRegressorWrapper)

    def test_when_n_jobs_set_then_forwarded(self) -> None:
        est = build_cs_linear_regressor_wrapper(
            CSLinearRegressorWrapperConfig(n_jobs=2), regressor=Ridge()
        )
        assert est.n_jobs == 2

    def test_when_regressor_set_then_forwarded(self) -> None:
        reg = Ridge(alpha=0.5)
        est = build_cs_linear_regressor_wrapper(
            CSLinearRegressorWrapperConfig(), regressor=reg
        )
        assert est.regressor is reg

    def test_when_regressor_none_then_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="regressor"):
            build_cs_linear_regressor_wrapper(
                CSLinearRegressorWrapperConfig(), regressor=None
            )

    def test_when_fit_then_coef_and_predict_shapes(self) -> None:
        rng = np.random.default_rng(1)
        T, N, K = 40, 7, 3
        X = rng.normal(size=(T, N, K))
        beta_true = rng.normal(size=K)
        y = (X @ beta_true).reshape(T, N) + 0.05 * rng.normal(size=(T, N))
        est = build_cs_linear_regressor_wrapper(
            CSLinearRegressorWrapperConfig.for_default(),
            regressor=Ridge(alpha=1e-6, fit_intercept=False),
        )
        est.fit(X, y)
        assert est.coef_.shape == (T, K)
        assert est.predict(X).shape == y.shape
        assert np.allclose(est.coef_.mean(axis=0), beta_true, atol=0.1)
