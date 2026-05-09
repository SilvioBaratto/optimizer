"""Tests for distance estimator factory functions."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from skfolio.distance import (
    CovarianceDistance,
    DistanceCorrelation,
    KendallDistance,
    MutualInformation,
    PearsonDistance,
    SpearmanDistance,
)
from skfolio.distance._base import BaseDistance

from optimizer.distance import (
    DistanceConfig,
    DistanceEstimatorType,
    build_distance,
)
from optimizer.exceptions import ConfigurationError


class TestDistanceEstimatorType:
    def test_when_listed_then_six_members_present(self) -> None:
        members = {m.name for m in DistanceEstimatorType}
        assert members == {
            "PEARSON",
            "KENDALL",
            "SPEARMAN",
            "COVARIANCE",
            "DISTANCE_CORRELATION",
            "MUTUAL_INFORMATION",
        }

    def test_when_str_enum_then_value_is_lowercase_string(self) -> None:
        assert DistanceEstimatorType.PEARSON.value == "pearson"
        assert DistanceEstimatorType.MUTUAL_INFORMATION.value == "mutual_information"


class TestDistanceConfig:
    def test_when_default_then_pearson_estimator(self) -> None:
        cfg = DistanceConfig()
        assert cfg.estimator == DistanceEstimatorType.PEARSON

    def test_when_constructed_then_frozen_dataclass(self) -> None:
        cfg = DistanceConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.estimator = DistanceEstimatorType.KENDALL  # type: ignore[misc]

    def test_when_default_then_n_bins_and_bandwidth_are_none(self) -> None:
        cfg = DistanceConfig()
        assert cfg.n_bins is None
        assert cfg.bandwidth is None

    def test_when_n_bins_set_on_non_mi_then_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="MUTUAL_INFORMATION"):
            DistanceConfig(estimator=DistanceEstimatorType.PEARSON, n_bins=10)

    def test_when_bandwidth_set_on_non_mi_then_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="MUTUAL_INFORMATION"):
            DistanceConfig(estimator=DistanceEstimatorType.KENDALL, bandwidth=1.0)

    def test_when_bandwidth_set_then_raises_unsupported(self) -> None:
        # skfolio 0.20.1 MutualInformation has no bandwidth parameter.
        with pytest.raises(ConfigurationError, match="bandwidth"):
            DistanceConfig(
                estimator=DistanceEstimatorType.MUTUAL_INFORMATION,
                bandwidth=0.5,
            )


class TestPresets:
    def test_when_for_pearson_then_pearson_estimator(self) -> None:
        cfg = DistanceConfig.for_pearson()
        assert cfg.estimator == DistanceEstimatorType.PEARSON

    def test_when_for_kendall_then_kendall_estimator(self) -> None:
        cfg = DistanceConfig.for_kendall()
        assert cfg.estimator == DistanceEstimatorType.KENDALL

    def test_when_for_spearman_then_spearman_estimator(self) -> None:
        cfg = DistanceConfig.for_spearman()
        assert cfg.estimator == DistanceEstimatorType.SPEARMAN

    def test_when_for_covariance_then_covariance_estimator(self) -> None:
        cfg = DistanceConfig.for_covariance()
        assert cfg.estimator == DistanceEstimatorType.COVARIANCE

    def test_when_for_distance_correlation_then_distance_correlation(self) -> None:
        cfg = DistanceConfig.for_distance_correlation()
        assert cfg.estimator == DistanceEstimatorType.DISTANCE_CORRELATION

    def test_when_for_mutual_information_then_mi_with_n_bins(self) -> None:
        cfg = DistanceConfig.for_mutual_information(n_bins=10)
        assert cfg.estimator == DistanceEstimatorType.MUTUAL_INFORMATION
        assert cfg.n_bins == 10


class TestBuildDistance:
    @pytest.mark.parametrize(
        ("estimator_type", "expected_class"),
        [
            (DistanceEstimatorType.PEARSON, PearsonDistance),
            (DistanceEstimatorType.KENDALL, KendallDistance),
            (DistanceEstimatorType.SPEARMAN, SpearmanDistance),
            (DistanceEstimatorType.COVARIANCE, CovarianceDistance),
            (DistanceEstimatorType.DISTANCE_CORRELATION, DistanceCorrelation),
            (DistanceEstimatorType.MUTUAL_INFORMATION, MutualInformation),
        ],
    )
    def test_when_dispatched_then_correct_class_returned(
        self,
        estimator_type: DistanceEstimatorType,
        expected_class: type[BaseDistance],
    ) -> None:
        cfg = DistanceConfig(estimator=estimator_type)
        est = build_distance(cfg)
        assert isinstance(est, expected_class)

    def test_when_built_then_subclass_of_base_distance(self) -> None:
        est = build_distance(DistanceConfig())
        assert isinstance(est, BaseDistance)

    def test_when_mi_with_n_bins_then_n_bins_forwarded(self) -> None:
        cfg = DistanceConfig.for_mutual_information(n_bins=12)
        est = build_distance(cfg)
        assert isinstance(est, MutualInformation)
        assert est.n_bins == 12


class TestIntegration:
    @pytest.fixture(scope="class")
    def returns(self) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        n_obs, n_assets = 250, 8
        data = rng.normal(loc=0.0005, scale=0.012, size=(n_obs, n_assets))
        cols = [f"A{i:02d}" for i in range(n_assets)]
        return pd.DataFrame(data, columns=cols)

    @pytest.mark.parametrize(
        "estimator_type",
        list(DistanceEstimatorType),
    )
    def test_when_fitted_then_distance_matrix_is_square(
        self,
        estimator_type: DistanceEstimatorType,
        returns: pd.DataFrame,
    ) -> None:
        if estimator_type == DistanceEstimatorType.MUTUAL_INFORMATION:
            cfg = DistanceConfig.for_mutual_information(n_bins=8)
        else:
            cfg = DistanceConfig(estimator=estimator_type)
        est = build_distance(cfg)
        est.fit(returns)
        n = returns.shape[1]
        assert est.distance_.shape == (n, n)

    def test_when_fitted_on_sp500_then_distance_matrix_square(self) -> None:
        from skfolio.datasets import load_sp500_dataset
        from skfolio.preprocessing import prices_to_returns

        prices = load_sp500_dataset()
        returns = prices_to_returns(prices)
        est = build_distance(DistanceConfig.for_pearson())
        est.fit(returns)
        n = returns.shape[1]
        assert est.distance_.shape == (n, n)
        assert np.allclose(est.distance_, est.distance_.T)
