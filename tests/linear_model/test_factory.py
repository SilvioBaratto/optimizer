"""Tests for the cross-sectional linear-regression factory."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from skfolio.linear_model import CSLinearRegression

from optimizer.linear_model import (
    CSLinearRegressionConfig,
    build_cs_linear_regression,
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
