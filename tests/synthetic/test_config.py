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

    def test_when_default_then_marginal_candidates_none(self) -> None:
        assert VineCopulaConfig().marginal_candidates is None

    def test_when_default_then_copula_candidates_none(self) -> None:
        assert VineCopulaConfig().copula_candidates is None

    def test_when_default_then_central_assets_none(self) -> None:
        assert VineCopulaConfig().central_assets is None

    def test_when_marginal_candidates_set_then_stored_as_tuple(self) -> None:
        cfg = VineCopulaConfig(marginal_candidates=("Gaussian", "StudentT"))
        assert cfg.marginal_candidates == ("Gaussian", "StudentT")

    def test_when_copula_candidates_set_then_stored_as_tuple(self) -> None:
        cfg = VineCopulaConfig(copula_candidates=("ClaytonCopula",))
        assert cfg.copula_candidates == ("ClaytonCopula",)

    def test_when_central_assets_set_then_stored_as_tuple(self) -> None:
        cfg = VineCopulaConfig(central_assets=("AAPL", "MSFT"))
        assert cfg.central_assets == ("AAPL", "MSFT")


class TestVineCopulaPresets:
    def test_when_for_with_t_marginals_then_gaussian_and_t(self) -> None:
        cfg = VineCopulaConfig.for_with_t_marginals()
        assert cfg.marginal_candidates == ("Gaussian", "StudentT")

    def test_when_for_clayton_only_then_clayton_copula(self) -> None:
        cfg = VineCopulaConfig.for_clayton_only()
        assert cfg.copula_candidates == ("ClaytonCopula",)


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

    def test_stress_test_uses_bic(self) -> None:
        cfg = SyntheticDataConfig.for_stress_test()
        assert cfg.vine_copula_config is not None
        assert cfg.vine_copula_config.selection_criterion == SelectionCriterionType.BIC

    def test_stress_test_max_depth_6(self) -> None:
        cfg = SyntheticDataConfig.for_stress_test()
        assert cfg.vine_copula_config is not None
        assert cfg.vine_copula_config.max_depth == 6

    def test_scenario_vs_stress_differ(self) -> None:
        scenario = SyntheticDataConfig.for_scenario_generation()
        stress = SyntheticDataConfig.for_stress_test()
        assert scenario.vine_copula_config != stress.vine_copula_config
