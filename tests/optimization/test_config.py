"""Tests for optimization configs and enums."""

from __future__ import annotations

import pytest

from optimizer.optimization import (
    BenchmarkTrackerConfig,
    ClusteringConfig,
    DistanceConfig,
    DistanceType,
    EqualWeightedConfig,
    ExtraRiskMeasureType,
    HERCConfig,
    HRPConfig,
    InverseVolatilityConfig,
    LinkageMethodType,
    MaxDiversificationConfig,
    MeanRiskConfig,
    NCOConfig,
    ObjectiveFunctionType,
    RatioMeasureType,
    RiskBudgetingConfig,
    RiskMeasureType,
    StackingConfig,
)


class TestObjectiveFunctionType:
    def test_members(self) -> None:
        assert set(ObjectiveFunctionType) == {
            ObjectiveFunctionType.MINIMIZE_RISK,
            ObjectiveFunctionType.MAXIMIZE_RETURN,
            ObjectiveFunctionType.MAXIMIZE_UTILITY,
            ObjectiveFunctionType.MAXIMIZE_RATIO,
        }

    def test_str_serialization(self) -> None:
        assert ObjectiveFunctionType.MINIMIZE_RISK.value == "minimize_risk"
        assert ObjectiveFunctionType.MAXIMIZE_RATIO.value == "maximize_ratio"


class TestRiskMeasureType:
    def test_members(self) -> None:
        assert len(RiskMeasureType) == 15

    def test_str_serialization(self) -> None:
        assert RiskMeasureType.VARIANCE.value == "variance"
        assert RiskMeasureType.CVAR.value == "cvar"
        assert RiskMeasureType.MAX_DRAWDOWN.value == "max_drawdown"
        assert RiskMeasureType.GINI_MEAN_DIFFERENCE.value == "gini_mean_difference"


class TestExtraRiskMeasureType:
    def test_members(self) -> None:
        assert set(ExtraRiskMeasureType) == {
            ExtraRiskMeasureType.VALUE_AT_RISK,
            ExtraRiskMeasureType.DRAWDOWN_AT_RISK,
            ExtraRiskMeasureType.ENTROPIC_RISK_MEASURE,
            ExtraRiskMeasureType.FOURTH_CENTRAL_MOMENT,
            ExtraRiskMeasureType.FOURTH_LOWER_PARTIAL_MOMENT,
            ExtraRiskMeasureType.SKEW,
            ExtraRiskMeasureType.KURTOSIS,
        }

    def test_str_serialization(self) -> None:
        assert ExtraRiskMeasureType.VALUE_AT_RISK.value == "value_at_risk"


class TestDistanceType:
    def test_members(self) -> None:
        assert set(DistanceType) == {
            DistanceType.PEARSON,
            DistanceType.KENDALL,
            DistanceType.SPEARMAN,
            DistanceType.COVARIANCE,
            DistanceType.DISTANCE_CORRELATION,
            DistanceType.MUTUAL_INFORMATION,
        }

    def test_str_serialization(self) -> None:
        assert DistanceType.PEARSON.value == "pearson"
        assert DistanceType.MUTUAL_INFORMATION.value == "mutual_information"


class TestLinkageMethodType:
    def test_members(self) -> None:
        assert set(LinkageMethodType) == {
            LinkageMethodType.SINGLE,
            LinkageMethodType.COMPLETE,
            LinkageMethodType.AVERAGE,
            LinkageMethodType.WEIGHTED,
            LinkageMethodType.CENTROID,
            LinkageMethodType.MEDIAN,
            LinkageMethodType.WARD,
        }

    def test_str_serialization(self) -> None:
        assert LinkageMethodType.WARD.value == "ward"
        assert LinkageMethodType.SINGLE.value == "single"


class TestRatioMeasureType:
    def test_members(self) -> None:
        assert len(RatioMeasureType) == 19

    def test_str_serialization(self) -> None:
        assert RatioMeasureType.SHARPE_RATIO.value == "sharpe_ratio"
        assert RatioMeasureType.SORTINO_RATIO.value == "sortino_ratio"
        assert RatioMeasureType.CALMAR_RATIO.value == "calmar_ratio"
        assert RatioMeasureType.CVAR_RATIO.value == "cvar_ratio"


class TestDistanceConfig:
    def test_default_values(self) -> None:
        cfg = DistanceConfig()
        assert cfg.distance_type == DistanceType.PEARSON
        assert cfg.absolute is False
        assert cfg.power == 1.0
        assert cfg.threshold == 0.5

    def test_frozen(self) -> None:
        cfg = DistanceConfig()
        with pytest.raises(AttributeError):
            cfg.absolute = True  # type: ignore[misc]

    def test_custom_values(self) -> None:
        cfg = DistanceConfig(
            distance_type=DistanceType.SPEARMAN,
            absolute=True,
            power=2.0,
        )
        assert cfg.distance_type == DistanceType.SPEARMAN
        assert cfg.absolute is True
        assert cfg.power == 2.0


class TestClusteringConfig:
    def test_default_values(self) -> None:
        cfg = ClusteringConfig()
        assert cfg.max_clusters is None
        assert cfg.linkage_method == LinkageMethodType.WARD

    def test_frozen(self) -> None:
        cfg = ClusteringConfig()
        with pytest.raises(AttributeError):
            cfg.max_clusters = 5  # type: ignore[misc]

    def test_custom_values(self) -> None:
        cfg = ClusteringConfig(
            max_clusters=10,
            linkage_method=LinkageMethodType.COMPLETE,
        )
        assert cfg.max_clusters == 10
        assert cfg.linkage_method == LinkageMethodType.COMPLETE


class TestMeanRiskConfig:
    def test_default_values(self) -> None:
        cfg = MeanRiskConfig()
        assert cfg.objective == ObjectiveFunctionType.MINIMIZE_RISK
        assert cfg.risk_measure == RiskMeasureType.VARIANCE
        assert cfg.risk_aversion == 1.0
        assert cfg.efficient_frontier_size is None
        assert cfg.min_weights == 0.0
        assert cfg.max_weights == 1.0
        assert cfg.budget == 1.0
        assert cfg.max_short is None
        assert cfg.max_long is None
        assert cfg.cardinality is None
        assert cfg.transaction_costs == 0.0
        assert cfg.management_fees == 0.0
        assert cfg.max_tracking_error is None
        assert cfg.l1_coef == 0.0
        assert cfg.l2_coef == 0.0
        assert cfg.risk_free_rate == 0.0
        assert cfg.cvar_beta == 0.95
        assert cfg.solver == "CLARABEL"
        assert cfg.solver_params is None
        assert cfg.prior_config is None

    def test_frozen(self) -> None:
        cfg = MeanRiskConfig()
        with pytest.raises(AttributeError):
            cfg.risk_aversion = 2.0  # type: ignore[misc]

    def test_custom_values(self) -> None:
        cfg = MeanRiskConfig(
            objective=ObjectiveFunctionType.MAXIMIZE_RATIO,
            risk_measure=RiskMeasureType.CVAR,
            risk_aversion=2.5,
            cvar_beta=0.99,
            l1_coef=0.01,
        )
        assert cfg.objective == ObjectiveFunctionType.MAXIMIZE_RATIO
        assert cfg.risk_measure == RiskMeasureType.CVAR
        assert cfg.risk_aversion == 2.5
        assert cfg.cvar_beta == 0.99
        assert cfg.l1_coef == 0.01

    def test_transaction_costs_and_fees(self) -> None:
        cfg = MeanRiskConfig(
            transaction_costs=0.001,
            management_fees=0.002,
        )
        assert cfg.transaction_costs == 0.001
        assert cfg.management_fees == 0.002

    def test_max_tracking_error(self) -> None:
        cfg = MeanRiskConfig(max_tracking_error=0.02)
        assert cfg.max_tracking_error == 0.02

    def test_for_min_variance(self) -> None:
        cfg = MeanRiskConfig.for_min_variance()
        assert cfg.objective == ObjectiveFunctionType.MINIMIZE_RISK
        assert cfg.risk_measure == RiskMeasureType.VARIANCE

    def test_for_max_sharpe(self) -> None:
        cfg = MeanRiskConfig.for_max_sharpe()
        assert cfg.objective == ObjectiveFunctionType.MAXIMIZE_RATIO
        assert cfg.risk_measure == RiskMeasureType.VARIANCE

    def test_for_max_utility(self) -> None:
        cfg = MeanRiskConfig.for_max_utility(risk_aversion=3.0)
        assert cfg.objective == ObjectiveFunctionType.MAXIMIZE_UTILITY
        assert cfg.risk_aversion == 3.0

    def test_for_min_cvar(self) -> None:
        cfg = MeanRiskConfig.for_min_cvar(beta=0.99)
        assert cfg.risk_measure == RiskMeasureType.CVAR
        assert cfg.cvar_beta == 0.99

    def test_for_efficient_frontier(self) -> None:
        cfg = MeanRiskConfig.for_efficient_frontier(size=30)
        assert cfg.efficient_frontier_size == 30
        assert cfg.objective == ObjectiveFunctionType.MINIMIZE_RISK


class TestRiskBudgetingConfig:
    def test_default_values(self) -> None:
        cfg = RiskBudgetingConfig()
        assert cfg.risk_measure == RiskMeasureType.VARIANCE
        assert cfg.min_weights == 0.0
        assert cfg.max_weights == 1.0
        assert cfg.risk_free_rate == 0.0
        assert cfg.solver == "CLARABEL"
        assert cfg.prior_config is None

    def test_frozen(self) -> None:
        cfg = RiskBudgetingConfig()
        with pytest.raises(AttributeError):
            cfg.solver = "SCS"  # type: ignore[misc]

    def test_for_risk_parity(self) -> None:
        cfg = RiskBudgetingConfig.for_risk_parity()
        assert cfg.risk_measure == RiskMeasureType.VARIANCE

    def test_for_cvar_parity(self) -> None:
        cfg = RiskBudgetingConfig.for_cvar_parity(beta=0.99)
        assert cfg.risk_measure == RiskMeasureType.CVAR
        assert cfg.cvar_beta == 0.99

    def test_for_cdar_parity(self) -> None:
        cfg = RiskBudgetingConfig.for_cdar_parity(beta=0.99)
        assert cfg.risk_measure == RiskMeasureType.CDAR
        assert cfg.cdar_beta == 0.99


class TestMaxDiversificationConfig:
    def test_default_values(self) -> None:
        cfg = MaxDiversificationConfig()
        assert cfg.min_weights == 0.0
        assert cfg.max_weights == 1.0
        assert cfg.budget == 1.0
        assert cfg.l1_coef == 0.0
        assert cfg.l2_coef == 0.0
        assert cfg.solver == "CLARABEL"
        assert cfg.prior_config is None

    def test_frozen(self) -> None:
        cfg = MaxDiversificationConfig()
        with pytest.raises(AttributeError):
            cfg.budget = 0.5  # type: ignore[misc]


class TestHRPConfig:
    def test_default_values(self) -> None:
        cfg = HRPConfig()
        assert cfg.risk_measure == RiskMeasureType.VARIANCE
        assert cfg.extra_risk_measure is None
        assert cfg.min_weights == 0.0
        assert cfg.max_weights == 1.0
        assert cfg.distance_config is None
        assert cfg.clustering_config is None
        assert cfg.prior_config is None

    def test_frozen(self) -> None:
        cfg = HRPConfig()
        with pytest.raises(AttributeError):
            cfg.risk_measure = RiskMeasureType.CVAR  # type: ignore[misc]

    def test_for_variance(self) -> None:
        cfg = HRPConfig.for_variance()
        assert cfg.risk_measure == RiskMeasureType.VARIANCE

    def test_for_cvar(self) -> None:
        cfg = HRPConfig.for_cvar()
        assert cfg.risk_measure == RiskMeasureType.CVAR

    def test_with_distance_config(self) -> None:
        cfg = HRPConfig(
            distance_config=DistanceConfig(distance_type=DistanceType.SPEARMAN),
        )
        assert cfg.distance_config is not None
        assert cfg.distance_config.distance_type == DistanceType.SPEARMAN

    def test_with_clustering_config(self) -> None:
        cfg = HRPConfig(
            clustering_config=ClusteringConfig(max_clusters=5),
        )
        assert cfg.clustering_config is not None
        assert cfg.clustering_config.max_clusters == 5

    def test_with_extra_risk_measure(self) -> None:
        cfg = HRPConfig(
            extra_risk_measure=ExtraRiskMeasureType.VALUE_AT_RISK,
        )
        assert cfg.extra_risk_measure == ExtraRiskMeasureType.VALUE_AT_RISK


class TestHERCConfig:
    def test_default_values(self) -> None:
        cfg = HERCConfig()
        assert cfg.risk_measure == RiskMeasureType.VARIANCE
        assert cfg.extra_risk_measure is None
        assert cfg.solver == "CLARABEL"
        assert cfg.distance_config is None
        assert cfg.clustering_config is None
        assert cfg.prior_config is None

    def test_frozen(self) -> None:
        cfg = HERCConfig()
        with pytest.raises(AttributeError):
            cfg.solver = "SCS"  # type: ignore[misc]

    def test_for_variance(self) -> None:
        cfg = HERCConfig.for_variance()
        assert cfg.risk_measure == RiskMeasureType.VARIANCE

    def test_for_cvar(self) -> None:
        cfg = HERCConfig.for_cvar()
        assert cfg.risk_measure == RiskMeasureType.CVAR


class TestNCOConfig:
    def test_default_values(self) -> None:
        cfg = NCOConfig()
        assert cfg.quantile == 0.5
        assert cfg.n_jobs is None
        assert cfg.distance_config is None
        assert cfg.clustering_config is None

    def test_frozen(self) -> None:
        cfg = NCOConfig()
        with pytest.raises(AttributeError):
            cfg.quantile = 0.8  # type: ignore[misc]

    def test_custom_values(self) -> None:
        cfg = NCOConfig(
            quantile=0.75,
            n_jobs=4,
            distance_config=DistanceConfig(distance_type=DistanceType.KENDALL),
            clustering_config=ClusteringConfig(
                max_clusters=8,
                linkage_method=LinkageMethodType.AVERAGE,
            ),
        )
        assert cfg.quantile == 0.75
        assert cfg.n_jobs == 4
        assert cfg.distance_config is not None
        assert cfg.distance_config.distance_type == DistanceType.KENDALL
        assert cfg.clustering_config is not None
        assert cfg.clustering_config.max_clusters == 8
        assert cfg.clustering_config.linkage_method == LinkageMethodType.AVERAGE


class TestBenchmarkTrackerConfig:
    def test_default_values(self) -> None:
        cfg = BenchmarkTrackerConfig()
        assert cfg.risk_measure == RiskMeasureType.STANDARD_DEVIATION
        assert cfg.min_weights == 0.0
        assert cfg.max_weights == 1.0
        assert cfg.max_short is None
        assert cfg.max_long is None
        assert cfg.cardinality is None
        assert cfg.transaction_costs == 0.0
        assert cfg.management_fees == 0.0
        assert cfg.l1_coef == 0.0
        assert cfg.l2_coef == 0.0
        assert cfg.risk_free_rate == 0.0
        assert cfg.solver == "CLARABEL"
        assert cfg.solver_params is None
        assert cfg.prior_config is None

    def test_frozen(self) -> None:
        cfg = BenchmarkTrackerConfig()
        with pytest.raises(AttributeError):
            cfg.risk_measure = RiskMeasureType.VARIANCE  # type: ignore[misc]

    def test_custom_values(self) -> None:
        cfg = BenchmarkTrackerConfig(
            risk_measure=RiskMeasureType.VARIANCE,
            transaction_costs=0.001,
            management_fees=0.002,
            l1_coef=0.01,
        )
        assert cfg.risk_measure == RiskMeasureType.VARIANCE
        assert cfg.transaction_costs == 0.001
        assert cfg.management_fees == 0.002
        assert cfg.l1_coef == 0.01


class TestEqualWeightedConfig:
    def test_instantiation(self) -> None:
        cfg = EqualWeightedConfig()
        assert cfg is not None

    def test_frozen(self) -> None:
        cfg = EqualWeightedConfig()
        with pytest.raises(AttributeError):
            cfg.x = 1  # type: ignore[attr-defined]


class TestInverseVolatilityConfig:
    def test_default_values(self) -> None:
        cfg = InverseVolatilityConfig()
        assert cfg.prior_config is None

    def test_frozen(self) -> None:
        cfg = InverseVolatilityConfig()
        with pytest.raises(AttributeError):
            cfg.prior_config = None  # type: ignore[misc]

    def test_with_prior_config(self) -> None:
        from optimizer.moments import MomentEstimationConfig

        prior_cfg = MomentEstimationConfig()
        cfg = InverseVolatilityConfig(prior_config=prior_cfg)
        assert cfg.prior_config is prior_cfg


class TestStackingConfig:
    def test_default_values(self) -> None:
        cfg = StackingConfig()
        assert cfg.quantile == 0.5
        assert cfg.quantile_measure == RatioMeasureType.SHARPE_RATIO
        assert cfg.n_jobs is None
        assert cfg.cv is None

    def test_frozen(self) -> None:
        cfg = StackingConfig()
        with pytest.raises(AttributeError):
            cfg.quantile = 0.8  # type: ignore[misc]

    def test_custom_values(self) -> None:
        cfg = StackingConfig(
            quantile=0.75,
            quantile_measure=RatioMeasureType.SORTINO_RATIO,
            n_jobs=4,
            cv=5,
        )
        assert cfg.quantile == 0.75
        assert cfg.quantile_measure == RatioMeasureType.SORTINO_RATIO
        assert cfg.n_jobs == 4
        assert cfg.cv == 5
