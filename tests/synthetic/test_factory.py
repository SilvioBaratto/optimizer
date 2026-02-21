"""Tests for synthetic data factory functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from skfolio.distribution import DependenceMethod, SelectionCriterion, VineCopula
from skfolio.prior import SyntheticData

from optimizer.synthetic import (
    DependenceMethodType,
    SelectionCriterionType,
    SyntheticDataConfig,
    VineCopulaConfig,
    build_synthetic_data,
    build_vine_copula,
)


@pytest.fixture()
def returns_df() -> pd.DataFrame:
    """Synthetic return DataFrame with 5 assets and 200 observations."""
    rng = np.random.default_rng(42)
    n_obs, n_assets = 200, 5
    data = rng.normal(loc=0.001, scale=0.02, size=(n_obs, n_assets))
    tickers = [f"TICK_{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(
        data,
        columns=tickers,
        index=pd.date_range("2023-01-01", periods=n_obs, freq="B"),
    )


# ---------------------------------------------------------------------------
# VineCopula
# ---------------------------------------------------------------------------


class TestBuildVineCopula:
    def test_default_produces_vine_copula(self) -> None:
        model = build_vine_copula()
        assert isinstance(model, VineCopula)

    def test_max_depth_forwarded(self) -> None:
        cfg = VineCopulaConfig(max_depth=6)
        model = build_vine_copula(cfg)
        assert model.max_depth == 6

    def test_fit_marginals_forwarded(self) -> None:
        cfg = VineCopulaConfig(fit_marginals=False)
        model = build_vine_copula(cfg)
        assert model.fit_marginals is False

    def test_log_transform_forwarded(self) -> None:
        cfg = VineCopulaConfig(log_transform=True)
        model = build_vine_copula(cfg)
        assert model.log_transform is True

    def test_dependence_method_forwarded(self) -> None:
        cfg = VineCopulaConfig(
            dependence_method=DependenceMethodType.MUTUAL_INFORMATION,
        )
        model = build_vine_copula(cfg)
        assert model.dependence_method == DependenceMethod.MUTUAL_INFORMATION

    def test_selection_criterion_forwarded(self) -> None:
        cfg = VineCopulaConfig(
            selection_criterion=SelectionCriterionType.BIC,
        )
        model = build_vine_copula(cfg)
        assert model.selection_criterion == SelectionCriterion.BIC

    def test_independence_level_forwarded(self) -> None:
        cfg = VineCopulaConfig(independence_level=0.01)
        model = build_vine_copula(cfg)
        assert model.independence_level == 0.01

    def test_n_jobs_forwarded(self) -> None:
        cfg = VineCopulaConfig(n_jobs=4)
        model = build_vine_copula(cfg)
        assert model.n_jobs == 4

    def test_random_state_forwarded(self) -> None:
        cfg = VineCopulaConfig(random_state=42)
        model = build_vine_copula(cfg)
        assert model.random_state == 42


# ---------------------------------------------------------------------------
# SyntheticData
# ---------------------------------------------------------------------------


class TestBuildSyntheticData:
    def test_default_produces_synthetic_data(self) -> None:
        model = build_synthetic_data()
        assert isinstance(model, SyntheticData)

    def test_n_samples_forwarded(self) -> None:
        cfg = SyntheticDataConfig(n_samples=5_000)
        model = build_synthetic_data(cfg)
        assert model.n_samples == 5_000

    def test_vine_copula_config_builds_distribution(self) -> None:
        cfg = SyntheticDataConfig(
            vine_copula_config=VineCopulaConfig(max_depth=3),
        )
        model = build_synthetic_data(cfg)
        assert isinstance(model.distribution_estimator, VineCopula)
        assert model.distribution_estimator.max_depth == 3

    def test_explicit_distribution_overrides_config(self) -> None:
        explicit = VineCopula(max_depth=8)
        cfg = SyntheticDataConfig(
            vine_copula_config=VineCopulaConfig(max_depth=3),
        )
        model = build_synthetic_data(cfg, distribution_estimator=explicit)
        assert model.distribution_estimator is explicit
        assert model.distribution_estimator.max_depth == 8

    def test_sample_args_forwarded(self) -> None:
        conditioning = {"TICK_00": -0.10}
        model = build_synthetic_data(
            sample_args={"conditioning": conditioning},
        )
        assert model.sample_args == {"conditioning": conditioning}

    def test_for_scenario_generation(self) -> None:
        cfg = SyntheticDataConfig.for_scenario_generation(n_samples=10_000)
        model = build_synthetic_data(cfg)
        assert isinstance(model, SyntheticData)
        assert model.n_samples == 10_000
        assert isinstance(model.distribution_estimator, VineCopula)

    def test_for_stress_test_with_conditioning(self) -> None:
        cfg = SyntheticDataConfig.for_stress_test(n_samples=5_000)
        model = build_synthetic_data(
            cfg,
            sample_args={"conditioning": {"TICK_00": -0.30}},
        )
        assert isinstance(model, SyntheticData)
        assert model.n_samples == 5_000
        assert model.sample_args == {"conditioning": {"TICK_00": -0.30}}


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests using real skfolio fit/predict."""

    def test_vine_copula_fit(self, returns_df: pd.DataFrame) -> None:
        model = build_vine_copula(VineCopulaConfig(random_state=42))
        model.fit(returns_df.values)
        samples = model.sample(n_samples=100)
        assert samples.shape == (100, returns_df.shape[1])

    def test_synthetic_data_fit_predict(self, returns_df: pd.DataFrame) -> None:
        cfg = SyntheticDataConfig(
            n_samples=500,
            vine_copula_config=VineCopulaConfig(random_state=42),
        )
        model = build_synthetic_data(cfg)
        model.fit(returns_df)
        prior = model.return_distribution_
        assert prior.mu is not None
        assert prior.covariance is not None
        assert len(prior.mu) == returns_df.shape[1]
