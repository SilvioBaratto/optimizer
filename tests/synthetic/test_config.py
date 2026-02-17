"""Tests for synthetic data configs and enums."""

from __future__ import annotations

import pytest

from optimizer.synthetic import (
    DependenceMethodType,
    SelectionCriterionType,
    SyntheticDataConfig,
    VineCopulaConfig,
)


class TestDependenceMethodType:
    def test_members(self) -> None:
        assert set(DependenceMethodType) == {
            DependenceMethodType.KENDALL_TAU,
            DependenceMethodType.MUTUAL_INFORMATION,
            DependenceMethodType.WASSERSTEIN_DISTANCE,
        }

    def test_str_serialization(self) -> None:
        assert DependenceMethodType.KENDALL_TAU.value == "kendall_tau"
        assert DependenceMethodType.WASSERSTEIN_DISTANCE.value == "wasserstein_distance"


class TestSelectionCriterionType:
    def test_members(self) -> None:
        assert set(SelectionCriterionType) == {
            SelectionCriterionType.AIC,
            SelectionCriterionType.BIC,
        }

    def test_str_serialization(self) -> None:
        assert SelectionCriterionType.AIC.value == "aic"
        assert SelectionCriterionType.BIC.value == "bic"


class TestVineCopulaConfig:
    def test_default_values(self) -> None:
        cfg = VineCopulaConfig()
        assert cfg.fit_marginals is True
        assert cfg.max_depth == 4
        assert cfg.log_transform is False
        assert cfg.dependence_method == DependenceMethodType.KENDALL_TAU
        assert cfg.selection_criterion == SelectionCriterionType.AIC
        assert cfg.independence_level == 0.05
        assert cfg.n_jobs is None
        assert cfg.random_state is None

    def test_frozen(self) -> None:
        cfg = VineCopulaConfig()
        with pytest.raises(AttributeError):
            cfg.max_depth = 8  # type: ignore[misc]

    def test_custom_values(self) -> None:
        cfg = VineCopulaConfig(
            fit_marginals=False,
            max_depth=6,
            log_transform=True,
            dependence_method=DependenceMethodType.MUTUAL_INFORMATION,
            selection_criterion=SelectionCriterionType.BIC,
            independence_level=0.01,
            n_jobs=4,
            random_state=42,
        )
        assert cfg.fit_marginals is False
        assert cfg.max_depth == 6
        assert cfg.log_transform is True
        assert cfg.dependence_method == DependenceMethodType.MUTUAL_INFORMATION
        assert cfg.selection_criterion == SelectionCriterionType.BIC
        assert cfg.independence_level == 0.01
        assert cfg.n_jobs == 4
        assert cfg.random_state == 42


class TestSyntheticDataConfig:
    def test_default_values(self) -> None:
        cfg = SyntheticDataConfig()
        assert cfg.n_samples == 1_000
        assert cfg.vine_copula_config is None

    def test_frozen(self) -> None:
        cfg = SyntheticDataConfig()
        with pytest.raises(AttributeError):
            cfg.n_samples = 5000  # type: ignore[misc]

    def test_custom_values(self) -> None:
        vine_cfg = VineCopulaConfig(max_depth=6)
        cfg = SyntheticDataConfig(
            n_samples=10_000,
            vine_copula_config=vine_cfg,
        )
        assert cfg.n_samples == 10_000
        assert cfg.vine_copula_config is vine_cfg
        assert cfg.vine_copula_config.max_depth == 6

    def test_for_scenario_generation(self) -> None:
        cfg = SyntheticDataConfig.for_scenario_generation(n_samples=5_000)
        assert cfg.n_samples == 5_000
        assert cfg.vine_copula_config is not None
        assert cfg.vine_copula_config.fit_marginals is True

    def test_for_stress_test(self) -> None:
        cfg = SyntheticDataConfig.for_stress_test(n_samples=20_000)
        assert cfg.n_samples == 20_000
        assert cfg.vine_copula_config is not None
