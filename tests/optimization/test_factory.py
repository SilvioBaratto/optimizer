"""Tests for optimization factory functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from skfolio.cluster import HierarchicalClustering, LinkageMethod
from skfolio.distance import (
    CovarianceDistance,
    DistanceCorrelation,
    KendallDistance,
    MutualInformation,
    PearsonDistance,
    SpearmanDistance,
)
from skfolio.measures import ExtraRiskMeasure, RatioMeasure, RiskMeasure
from skfolio.optimization import (
    BenchmarkTracker,
    EqualWeighted,
    HierarchicalEqualRiskContribution,
    HierarchicalRiskParity,
    InverseVolatility,
    MaximumDiversification,
    MeanRisk,
    NestedClustersOptimization,
    RiskBudgeting,
    StackingOptimization,
)
from skfolio.optimization.convex._base import ObjectiveFunction

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
    build_benchmark_tracker,
    build_clustering_estimator,
    build_distance_estimator,
    build_equal_weighted,
    build_herc,
    build_hrp,
    build_inverse_volatility,
    build_max_diversification,
    build_mean_risk,
    build_nco,
    build_risk_budgeting,
    build_stacking,
)

# ---------------------------------------------------------------------------
# Distance estimator
# ---------------------------------------------------------------------------


class TestBuildDistanceEstimator:
    @pytest.mark.parametrize(
        ("distance_type", "expected_class"),
        [
            (DistanceType.PEARSON, PearsonDistance),
            (DistanceType.KENDALL, KendallDistance),
            (DistanceType.SPEARMAN, SpearmanDistance),
            (DistanceType.COVARIANCE, CovarianceDistance),
            (DistanceType.DISTANCE_CORRELATION, DistanceCorrelation),
            (DistanceType.MUTUAL_INFORMATION, MutualInformation),
        ],
    )
    def test_each_type_produces_correct_class(
        self,
        distance_type: DistanceType,
        expected_class: type,
    ) -> None:
        cfg = DistanceConfig(distance_type=distance_type)
        estimator = build_distance_estimator(cfg)
        assert isinstance(estimator, expected_class)

    def test_default_config(self) -> None:
        estimator = build_distance_estimator()
        assert isinstance(estimator, PearsonDistance)

    def test_absolute_forwarded(self) -> None:
        cfg = DistanceConfig(absolute=True, power=2.0)
        estimator = build_distance_estimator(cfg)
        assert isinstance(estimator, PearsonDistance)
        assert estimator.absolute is True
        assert estimator.power == 2.0

    def test_threshold_forwarded(self) -> None:
        cfg = DistanceConfig(
            distance_type=DistanceType.DISTANCE_CORRELATION,
            threshold=0.7,
        )
        estimator = build_distance_estimator(cfg)
        assert isinstance(estimator, DistanceCorrelation)
        assert estimator.threshold == 0.7


# ---------------------------------------------------------------------------
# Clustering estimator
# ---------------------------------------------------------------------------


class TestBuildClusteringEstimator:
    def test_default_config(self) -> None:
        estimator = build_clustering_estimator()
        assert isinstance(estimator, HierarchicalClustering)
        assert estimator.linkage_method == LinkageMethod.WARD
        assert estimator.max_clusters is None

    def test_custom_config(self) -> None:
        cfg = ClusteringConfig(
            max_clusters=5,
            linkage_method=LinkageMethodType.COMPLETE,
        )
        estimator = build_clustering_estimator(cfg)
        assert isinstance(estimator, HierarchicalClustering)
        assert estimator.linkage_method == LinkageMethod.COMPLETE
        assert estimator.max_clusters == 5

    @pytest.mark.parametrize(
        ("linkage_type", "expected"),
        [
            (LinkageMethodType.SINGLE, LinkageMethod.SINGLE),
            (LinkageMethodType.COMPLETE, LinkageMethod.COMPLETE),
            (LinkageMethodType.AVERAGE, LinkageMethod.AVERAGE),
            (LinkageMethodType.WEIGHTED, LinkageMethod.WEIGHTED),
            (LinkageMethodType.CENTROID, LinkageMethod.CENTROID),
            (LinkageMethodType.MEDIAN, LinkageMethod.MEDIAN),
            (LinkageMethodType.WARD, LinkageMethod.WARD),
        ],
    )
    def test_all_linkage_methods(
        self,
        linkage_type: LinkageMethodType,
        expected: LinkageMethod,
    ) -> None:
        cfg = ClusteringConfig(linkage_method=linkage_type)
        estimator = build_clustering_estimator(cfg)
        assert estimator.linkage_method == expected


# ---------------------------------------------------------------------------
# MeanRisk
# ---------------------------------------------------------------------------


class TestBuildMeanRisk:
    def test_default_produces_mean_risk(self) -> None:
        model = build_mean_risk()
        assert isinstance(model, MeanRisk)

    def test_objective_forwarded(self) -> None:
        cfg = MeanRiskConfig(objective=ObjectiveFunctionType.MAXIMIZE_RATIO)
        model = build_mean_risk(cfg)
        assert model.objective_function == ObjectiveFunction.MAXIMIZE_RATIO

    def test_risk_measure_forwarded(self) -> None:
        cfg = MeanRiskConfig(risk_measure=RiskMeasureType.CVAR)
        model = build_mean_risk(cfg)
        assert model.risk_measure == RiskMeasure.CVAR

    def test_risk_aversion_forwarded(self) -> None:
        cfg = MeanRiskConfig(
            objective=ObjectiveFunctionType.MAXIMIZE_UTILITY,
            risk_aversion=2.5,
        )
        model = build_mean_risk(cfg)
        assert model.risk_aversion == 2.5

    def test_efficient_frontier_size_forwarded(self) -> None:
        cfg = MeanRiskConfig(efficient_frontier_size=30)
        model = build_mean_risk(cfg)
        assert model.efficient_frontier_size == 30

    def test_weight_bounds_forwarded(self) -> None:
        cfg = MeanRiskConfig(min_weights=-0.1, max_weights=0.5)
        model = build_mean_risk(cfg)
        assert model.min_weights == -0.1
        assert model.max_weights == 0.5

    def test_budget_forwarded(self) -> None:
        cfg = MeanRiskConfig(budget=0.5)
        model = build_mean_risk(cfg)
        assert model.budget == 0.5

    def test_cardinality_forwarded(self) -> None:
        cfg = MeanRiskConfig(cardinality=10)
        model = build_mean_risk(cfg)
        assert model.cardinality == 10

    def test_transaction_costs_forwarded(self) -> None:
        cfg = MeanRiskConfig(transaction_costs=0.001)
        model = build_mean_risk(cfg)
        assert model.transaction_costs == 0.001

    def test_management_fees_forwarded(self) -> None:
        cfg = MeanRiskConfig(management_fees=0.002)
        model = build_mean_risk(cfg)
        assert model.management_fees == 0.002

    def test_max_tracking_error_forwarded(self) -> None:
        cfg = MeanRiskConfig(max_tracking_error=0.02)
        model = build_mean_risk(cfg)
        assert model.max_tracking_error == 0.02

    def test_regularisation_forwarded(self) -> None:
        cfg = MeanRiskConfig(l1_coef=0.01, l2_coef=0.02)
        model = build_mean_risk(cfg)
        assert model.l1_coef == 0.01
        assert model.l2_coef == 0.02

    def test_solver_forwarded(self) -> None:
        cfg = MeanRiskConfig(solver="SCS")
        model = build_mean_risk(cfg)
        assert model.solver == "SCS"

    def test_beta_params_forwarded(self) -> None:
        cfg = MeanRiskConfig(
            cvar_beta=0.99,
            evar_beta=0.90,
            cdar_beta=0.85,
            edar_beta=0.80,
        )
        model = build_mean_risk(cfg)
        assert model.cvar_beta == 0.99
        assert model.evar_beta == 0.90
        assert model.cdar_beta == 0.85
        assert model.edar_beta == 0.80

    def test_prior_config_builds_prior(self) -> None:
        from skfolio.moments import ShrunkMu
        from skfolio.prior import EmpiricalPrior

        from optimizer.moments import MomentEstimationConfig, MuEstimatorType

        prior_cfg = MomentEstimationConfig(mu_estimator=MuEstimatorType.SHRUNK)
        cfg = MeanRiskConfig(prior_config=prior_cfg)
        model = build_mean_risk(cfg)
        assert isinstance(model.prior_estimator, EmpiricalPrior)
        assert isinstance(model.prior_estimator.mu_estimator, ShrunkMu)

    def test_explicit_prior_estimator_overrides_config(self) -> None:
        from skfolio.prior import EmpiricalPrior

        from optimizer.moments import MomentEstimationConfig, MuEstimatorType

        explicit_prior = EmpiricalPrior()
        cfg = MeanRiskConfig(
            prior_config=MomentEstimationConfig(
                mu_estimator=MuEstimatorType.SHRUNK,
            ),
        )
        model = build_mean_risk(cfg, prior_estimator=explicit_prior)
        assert model.prior_estimator is explicit_prior

    def test_kwargs_forwarded(self) -> None:
        groups = {"tech": ["TICK_00", "TICK_01"]}
        model = build_mean_risk(groups=groups)
        assert model.groups == groups


# ---------------------------------------------------------------------------
# RiskBudgeting
# ---------------------------------------------------------------------------


class TestBuildRiskBudgeting:
    def test_default_produces_risk_budgeting(self) -> None:
        model = build_risk_budgeting()
        assert isinstance(model, RiskBudgeting)

    def test_risk_measure_forwarded(self) -> None:
        cfg = RiskBudgetingConfig(risk_measure=RiskMeasureType.CVAR)
        model = build_risk_budgeting(cfg)
        assert model.risk_measure == RiskMeasure.CVAR

    def test_risk_budget_forwarded(self) -> None:
        budget = np.array([0.3, 0.7])
        model = build_risk_budgeting(risk_budget=budget)
        assert np.array_equal(model.risk_budget, budget)

    def test_solver_forwarded(self) -> None:
        cfg = RiskBudgetingConfig(solver="SCS")
        model = build_risk_budgeting(cfg)
        assert model.solver == "SCS"


# ---------------------------------------------------------------------------
# MaximumDiversification
# ---------------------------------------------------------------------------


class TestBuildMaxDiversification:
    def test_default_produces_max_div(self) -> None:
        model = build_max_diversification()
        assert isinstance(model, MaximumDiversification)

    def test_budget_forwarded(self) -> None:
        cfg = MaxDiversificationConfig(budget=0.5)
        model = build_max_diversification(cfg)
        assert model.budget == 0.5

    def test_regularisation_forwarded(self) -> None:
        cfg = MaxDiversificationConfig(l1_coef=0.01, l2_coef=0.02)
        model = build_max_diversification(cfg)
        assert model.l1_coef == 0.01
        assert model.l2_coef == 0.02

    def test_solver_forwarded(self) -> None:
        cfg = MaxDiversificationConfig(solver="SCS")
        model = build_max_diversification(cfg)
        assert model.solver == "SCS"


# ---------------------------------------------------------------------------
# HRP
# ---------------------------------------------------------------------------


class TestBuildHRP:
    def test_default_produces_hrp(self) -> None:
        model = build_hrp()
        assert isinstance(model, HierarchicalRiskParity)

    def test_risk_measure_forwarded(self) -> None:
        cfg = HRPConfig(risk_measure=RiskMeasureType.CVAR)
        model = build_hrp(cfg)
        assert model.risk_measure == RiskMeasure.CVAR

    def test_extra_risk_measure_forwarded(self) -> None:
        cfg = HRPConfig(
            extra_risk_measure=ExtraRiskMeasureType.VALUE_AT_RISK,
        )
        model = build_hrp(cfg)
        assert model.risk_measure == ExtraRiskMeasure.VALUE_AT_RISK

    def test_weight_bounds_forwarded(self) -> None:
        cfg = HRPConfig(min_weights=0.01, max_weights=0.5)
        model = build_hrp(cfg)
        assert model.min_weights == 0.01
        assert model.max_weights == 0.5

    def test_distance_config_builds_estimator(self) -> None:
        cfg = HRPConfig(
            distance_config=DistanceConfig(distance_type=DistanceType.SPEARMAN),
        )
        model = build_hrp(cfg)
        assert isinstance(model.distance_estimator, SpearmanDistance)

    def test_clustering_config_builds_estimator(self) -> None:
        cfg = HRPConfig(
            clustering_config=ClusteringConfig(
                max_clusters=5,
                linkage_method=LinkageMethodType.COMPLETE,
            ),
        )
        model = build_hrp(cfg)
        assert isinstance(
            model.hierarchical_clustering_estimator,
            HierarchicalClustering,
        )
        assert model.hierarchical_clustering_estimator.max_clusters == 5
        assert (
            model.hierarchical_clustering_estimator.linkage_method
            == LinkageMethod.COMPLETE
        )

    def test_explicit_distance_overrides_config(self) -> None:
        explicit = KendallDistance()
        cfg = HRPConfig(
            distance_config=DistanceConfig(distance_type=DistanceType.SPEARMAN),
        )
        model = build_hrp(cfg, distance_estimator=explicit)
        assert model.distance_estimator is explicit


# ---------------------------------------------------------------------------
# HERC
# ---------------------------------------------------------------------------


class TestBuildHERC:
    def test_default_produces_herc(self) -> None:
        model = build_herc()
        assert isinstance(model, HierarchicalEqualRiskContribution)

    def test_risk_measure_forwarded(self) -> None:
        cfg = HERCConfig(risk_measure=RiskMeasureType.CVAR)
        model = build_herc(cfg)
        assert model.risk_measure == RiskMeasure.CVAR

    def test_extra_risk_measure_forwarded(self) -> None:
        cfg = HERCConfig(
            extra_risk_measure=ExtraRiskMeasureType.VALUE_AT_RISK,
        )
        model = build_herc(cfg)
        assert model.risk_measure == ExtraRiskMeasure.VALUE_AT_RISK

    def test_solver_forwarded(self) -> None:
        cfg = HERCConfig(solver="SCS")
        model = build_herc(cfg)
        assert model.solver == "SCS"

    def test_distance_config_builds_estimator(self) -> None:
        cfg = HERCConfig(
            distance_config=DistanceConfig(distance_type=DistanceType.KENDALL),
        )
        model = build_herc(cfg)
        assert isinstance(model.distance_estimator, KendallDistance)

    def test_clustering_config_builds_estimator(self) -> None:
        cfg = HERCConfig(
            clustering_config=ClusteringConfig(max_clusters=3),
        )
        model = build_herc(cfg)
        assert isinstance(
            model.hierarchical_clustering_estimator,
            HierarchicalClustering,
        )
        assert model.hierarchical_clustering_estimator.max_clusters == 3


# ---------------------------------------------------------------------------
# NCO
# ---------------------------------------------------------------------------


class TestBuildNCO:
    def test_default_produces_nco(self) -> None:
        model = build_nco()
        assert isinstance(model, NestedClustersOptimization)

    def test_quantile_forwarded(self) -> None:
        cfg = NCOConfig(quantile=0.75)
        model = build_nco(cfg)
        assert model.quantile == 0.75

    def test_n_jobs_forwarded(self) -> None:
        cfg = NCOConfig(n_jobs=4)
        model = build_nco(cfg)
        assert model.n_jobs == 4

    def test_inner_outer_estimators_forwarded(self) -> None:
        inner = MeanRisk()
        outer = MeanRisk()
        model = build_nco(inner_estimator=inner, outer_estimator=outer)
        assert model.inner_estimator is inner
        assert model.outer_estimator is outer

    def test_distance_config_builds_estimator(self) -> None:
        cfg = NCOConfig(
            distance_config=DistanceConfig(distance_type=DistanceType.PEARSON),
        )
        model = build_nco(cfg)
        assert isinstance(model.distance_estimator, PearsonDistance)

    def test_clustering_config_builds_estimator(self) -> None:
        cfg = NCOConfig(
            clustering_config=ClusteringConfig(
                max_clusters=4,
                linkage_method=LinkageMethodType.WARD,
            ),
        )
        model = build_nco(cfg)
        assert isinstance(
            model.clustering_estimator,
            HierarchicalClustering,
        )
        assert model.clustering_estimator.max_clusters == 4


# ---------------------------------------------------------------------------
# BenchmarkTracker
# ---------------------------------------------------------------------------


class TestBuildBenchmarkTracker:
    def test_default_produces_benchmark_tracker(self) -> None:
        model = build_benchmark_tracker()
        assert isinstance(model, BenchmarkTracker)

    def test_risk_measure_forwarded(self) -> None:
        cfg = BenchmarkTrackerConfig(risk_measure=RiskMeasureType.VARIANCE)
        model = build_benchmark_tracker(cfg)
        assert model.risk_measure == RiskMeasure.VARIANCE

    def test_weight_bounds_forwarded(self) -> None:
        cfg = BenchmarkTrackerConfig(min_weights=-0.1, max_weights=0.5)
        model = build_benchmark_tracker(cfg)
        assert model.min_weights == -0.1
        assert model.max_weights == 0.5

    def test_transaction_costs_forwarded(self) -> None:
        cfg = BenchmarkTrackerConfig(transaction_costs=0.001)
        model = build_benchmark_tracker(cfg)
        assert model.transaction_costs == 0.001

    def test_management_fees_forwarded(self) -> None:
        cfg = BenchmarkTrackerConfig(management_fees=0.002)
        model = build_benchmark_tracker(cfg)
        assert model.management_fees == 0.002

    def test_regularisation_forwarded(self) -> None:
        cfg = BenchmarkTrackerConfig(l1_coef=0.01, l2_coef=0.02)
        model = build_benchmark_tracker(cfg)
        assert model.l1_coef == 0.01
        assert model.l2_coef == 0.02

    def test_solver_forwarded(self) -> None:
        cfg = BenchmarkTrackerConfig(solver="SCS")
        model = build_benchmark_tracker(cfg)
        assert model.solver == "SCS"

    def test_kwargs_forwarded(self) -> None:
        groups = {"tech": ["TICK_00", "TICK_01"]}
        model = build_benchmark_tracker(groups=groups)
        assert model.groups == groups


# ---------------------------------------------------------------------------
# EqualWeighted
# ---------------------------------------------------------------------------


class TestBuildEqualWeighted:
    def test_default_produces_equal_weighted(self) -> None:
        model = build_equal_weighted()
        assert isinstance(model, EqualWeighted)

    def test_with_config(self) -> None:
        cfg = EqualWeightedConfig()
        model = build_equal_weighted(cfg)
        assert isinstance(model, EqualWeighted)


# ---------------------------------------------------------------------------
# InverseVolatility
# ---------------------------------------------------------------------------


class TestBuildInverseVolatility:
    def test_default_produces_inverse_volatility(self) -> None:
        model = build_inverse_volatility()
        assert isinstance(model, InverseVolatility)

    def test_prior_config_builds_prior(self) -> None:
        from skfolio.moments import ShrunkMu
        from skfolio.prior import EmpiricalPrior

        from optimizer.moments import MomentEstimationConfig, MuEstimatorType

        prior_cfg = MomentEstimationConfig(mu_estimator=MuEstimatorType.SHRUNK)
        cfg = InverseVolatilityConfig(prior_config=prior_cfg)
        model = build_inverse_volatility(cfg)
        assert isinstance(model.prior_estimator, EmpiricalPrior)
        assert isinstance(model.prior_estimator.mu_estimator, ShrunkMu)

    def test_explicit_prior_overrides_config(self) -> None:
        from skfolio.prior import EmpiricalPrior

        from optimizer.moments import MomentEstimationConfig

        explicit = EmpiricalPrior()
        cfg = InverseVolatilityConfig(prior_config=MomentEstimationConfig())
        model = build_inverse_volatility(cfg, prior_estimator=explicit)
        assert model.prior_estimator is explicit


# ---------------------------------------------------------------------------
# StackingOptimization
# ---------------------------------------------------------------------------


class TestBuildStacking:
    def test_default_produces_stacking(self) -> None:
        model = build_stacking()
        assert isinstance(model, StackingOptimization)

    def test_default_estimators(self) -> None:
        model = build_stacking()
        assert len(model.estimators) == 2
        assert model.estimators[0][0] == "mean_risk"
        assert model.estimators[1][0] == "hrp"

    def test_custom_estimators(self) -> None:
        estimators = [
            ("mr1", MeanRisk()),
            ("mr2", MeanRisk()),
        ]
        model = build_stacking(estimators=estimators)
        assert model.estimators is estimators

    def test_final_estimator_forwarded(self) -> None:
        final = MeanRisk()
        model = build_stacking(final_estimator=final)
        assert model.final_estimator is final

    def test_quantile_forwarded(self) -> None:
        cfg = StackingConfig(quantile=0.75)
        model = build_stacking(cfg)
        assert model.quantile == 0.75

    def test_quantile_measure_forwarded(self) -> None:
        cfg = StackingConfig(
            quantile_measure=RatioMeasureType.SORTINO_RATIO,
        )
        model = build_stacking(cfg)
        assert model.quantile_measure == RatioMeasure.SORTINO_RATIO

    def test_n_jobs_forwarded(self) -> None:
        cfg = StackingConfig(n_jobs=4)
        model = build_stacking(cfg)
        assert model.n_jobs == 4

    def test_cv_forwarded(self) -> None:
        cfg = StackingConfig(cv=5)
        model = build_stacking(cfg)
        assert model.cv == 5


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests using real skfolio fit/predict."""

    def test_mean_risk_min_variance_fit(self, returns_df: pd.DataFrame) -> None:
        cfg = MeanRiskConfig.for_min_variance()
        model = build_mean_risk(cfg)
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert portfolio.weights is not None
        assert len(portfolio.weights) == returns_df.shape[1]
        assert abs(sum(portfolio.weights) - 1.0) < 1e-6

    def test_mean_risk_max_sharpe_fit(self, returns_df: pd.DataFrame) -> None:
        cfg = MeanRiskConfig.for_max_sharpe()
        model = build_mean_risk(cfg)
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert portfolio.weights is not None

    def test_mean_risk_max_utility_fit(self, returns_df: pd.DataFrame) -> None:
        cfg = MeanRiskConfig.for_max_utility(risk_aversion=2.0)
        model = build_mean_risk(cfg)
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert portfolio.weights is not None

    def test_mean_risk_min_cvar_fit(self, returns_df: pd.DataFrame) -> None:
        cfg = MeanRiskConfig.for_min_cvar()
        model = build_mean_risk(cfg)
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert portfolio.weights is not None

    def test_mean_risk_with_transaction_costs(self, returns_df: pd.DataFrame) -> None:
        cfg = MeanRiskConfig(
            objective=ObjectiveFunctionType.MINIMIZE_RISK,
            transaction_costs=0.001,
            management_fees=0.0005,
        )
        model = build_mean_risk(cfg)
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert portfolio.weights is not None

    def test_risk_budgeting_fit(self, returns_df: pd.DataFrame) -> None:
        cfg = RiskBudgetingConfig.for_risk_parity()
        model = build_risk_budgeting(cfg)
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert portfolio.weights is not None
        assert len(portfolio.weights) == returns_df.shape[1]

    def test_risk_budgeting_custom_budget(self, returns_df: pd.DataFrame) -> None:
        n = returns_df.shape[1]
        budget = np.ones(n) / n
        model = build_risk_budgeting(risk_budget=budget)
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert portfolio.weights is not None

    def test_max_diversification_fit(self, returns_df: pd.DataFrame) -> None:
        model = build_max_diversification()
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert portfolio.weights is not None
        assert len(portfolio.weights) == returns_df.shape[1]

    def test_hrp_fit(self, returns_df: pd.DataFrame) -> None:
        cfg = HRPConfig.for_variance()
        model = build_hrp(cfg)
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert portfolio.weights is not None
        assert len(portfolio.weights) == returns_df.shape[1]

    def test_hrp_cvar_fit(self, returns_df: pd.DataFrame) -> None:
        cfg = HRPConfig.for_cvar()
        model = build_hrp(cfg)
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert portfolio.weights is not None

    def test_herc_fit(self, returns_df: pd.DataFrame) -> None:
        cfg = HERCConfig.for_variance()
        model = build_herc(cfg)
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert portfolio.weights is not None
        assert len(portfolio.weights) == returns_df.shape[1]

    def test_herc_cvar_fit(self, returns_df: pd.DataFrame) -> None:
        cfg = HERCConfig.for_cvar()
        model = build_herc(cfg)
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert portfolio.weights is not None

    def test_nco_fit(self, returns_df: pd.DataFrame) -> None:
        model = build_nco()
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert portfolio.weights is not None
        assert len(portfolio.weights) == returns_df.shape[1]

    def test_nco_with_custom_estimators(self, returns_df: pd.DataFrame) -> None:
        inner = MeanRisk()
        outer = MeanRisk()
        model = build_nco(inner_estimator=inner, outer_estimator=outer)
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert portfolio.weights is not None

    def test_benchmark_tracker_fit(self, returns_df: pd.DataFrame) -> None:
        model = build_benchmark_tracker()
        benchmark = returns_df.mean(axis=1)
        model.fit(returns_df, y=benchmark)
        portfolio = model.predict(returns_df)
        assert portfolio.weights is not None
        assert len(portfolio.weights) == returns_df.shape[1]

    def test_equal_weighted_fit(self, returns_df: pd.DataFrame) -> None:
        model = build_equal_weighted()
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert portfolio.weights is not None
        assert len(portfolio.weights) == returns_df.shape[1]
        expected_weight = 1.0 / returns_df.shape[1]
        assert all(abs(w - expected_weight) < 1e-10 for w in portfolio.weights)

    def test_inverse_volatility_fit(self, returns_df: pd.DataFrame) -> None:
        model = build_inverse_volatility()
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert portfolio.weights is not None
        assert len(portfolio.weights) == returns_df.shape[1]
        assert abs(sum(portfolio.weights) - 1.0) < 1e-6

    def test_stacking_fit(self, returns_df: pd.DataFrame) -> None:
        model = build_stacking()
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert portfolio.weights is not None
        assert len(portfolio.weights) == returns_df.shape[1]

    def test_stacking_with_custom_estimators(self, returns_df: pd.DataFrame) -> None:
        estimators = [
            ("min_var", build_mean_risk(MeanRiskConfig.for_min_variance())),
            ("hrp", build_hrp(HRPConfig.for_variance())),
        ]
        final = build_mean_risk(MeanRiskConfig.for_min_variance())
        model = build_stacking(
            estimators=estimators,
            final_estimator=final,
        )
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert portfolio.weights is not None

    def test_sp500_mean_risk(self) -> None:
        from skfolio.datasets import load_sp500_dataset
        from skfolio.preprocessing import prices_to_returns

        prices = load_sp500_dataset()
        returns = prices_to_returns(prices)

        cfg = MeanRiskConfig.for_min_variance()
        model = build_mean_risk(cfg)
        model.fit(returns)
        portfolio = model.predict(returns)
        assert portfolio.weights is not None
        assert len(portfolio.weights) == returns.shape[1]

    def test_sp500_hrp(self) -> None:
        from skfolio.datasets import load_sp500_dataset
        from skfolio.preprocessing import prices_to_returns

        prices = load_sp500_dataset()
        returns = prices_to_returns(prices)

        cfg = HRPConfig.for_variance()
        model = build_hrp(cfg)
        model.fit(returns)
        portfolio = model.predict(returns)
        assert portfolio.weights is not None
        assert len(portfolio.weights) == returns.shape[1]

    def test_hrp_cdar_fit(self, returns_df: pd.DataFrame) -> None:
        cfg = HRPConfig(risk_measure=RiskMeasureType.CDAR)
        model = build_hrp(cfg)
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert portfolio.weights is not None
        assert len(portfolio.weights) == returns_df.shape[1]
        assert abs(sum(portfolio.weights) - 1.0) < 1e-6

    def test_hrp_mutual_information_distance_fit(
        self, returns_df: pd.DataFrame
    ) -> None:
        cfg = HRPConfig(
            distance_config=DistanceConfig(
                distance_type=DistanceType.MUTUAL_INFORMATION
            ),
        )
        model = build_hrp(cfg)
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert portfolio.weights is not None
        assert len(portfolio.weights) == returns_df.shape[1]
        assert abs(sum(portfolio.weights) - 1.0) < 1e-6

    def test_hrp_kendall_distance_fit(self, returns_df: pd.DataFrame) -> None:
        cfg = HRPConfig(
            distance_config=DistanceConfig(distance_type=DistanceType.KENDALL),
        )
        model = build_hrp(cfg)
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert portfolio.weights is not None
        assert len(portfolio.weights) == returns_df.shape[1]
        assert abs(sum(portfolio.weights) - 1.0) < 1e-6

    def test_herc_cdar_fit(self, returns_df: pd.DataFrame) -> None:
        cfg = HERCConfig(risk_measure=RiskMeasureType.CDAR)
        model = build_herc(cfg)
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert portfolio.weights is not None
        assert len(portfolio.weights) == returns_df.shape[1]
        assert abs(sum(portfolio.weights) - 1.0) < 1e-6

    def test_nco_hrp_inner_mean_risk_outer(self, returns_df: pd.DataFrame) -> None:
        inner_cfg = HRPConfig(
            risk_measure=RiskMeasureType.VARIANCE,
            clustering_config=ClusteringConfig(max_clusters=3),
        )
        inner = build_hrp(inner_cfg)
        outer = build_mean_risk(MeanRiskConfig.for_min_variance())
        model = build_nco(inner_estimator=inner, outer_estimator=outer)
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert portfolio.weights is not None
        assert len(portfolio.weights) == returns_df.shape[1]
        assert abs(sum(portfolio.weights) - 1.0) < 1e-6

    def test_inverse_volatility_formula(self, returns_df: pd.DataFrame) -> None:
        """Verify inverse-volatility weights match 1/sigma_i / sum(1/sigma_j)."""
        model = build_inverse_volatility()
        model.fit(returns_df)
        portfolio = model.predict(returns_df)

        # Manual computation
        sigmas = returns_df.std().to_numpy()
        inv_vol = 1.0 / sigmas
        expected_weights = inv_vol / inv_vol.sum()

        np.testing.assert_allclose(portfolio.weights, expected_weights, rtol=1e-6)

    def test_efficient_frontier(self, returns_df: pd.DataFrame) -> None:
        cfg = MeanRiskConfig.for_efficient_frontier(size=5)
        model = build_mean_risk(cfg)
        model.fit(returns_df)
        population = model.predict(returns_df)
        assert len(population) == 5


# ---------------------------------------------------------------------------
# Partial-investment budget (issue #107)
# ---------------------------------------------------------------------------


class TestPartialInvestmentBudget:
    def test_budget_0_8_weights_sum(self, returns_df: pd.DataFrame) -> None:
        """Partial budget=0.8 → weights sum to 0.8."""
        cfg = MeanRiskConfig(
            objective=ObjectiveFunctionType.MINIMIZE_RISK,
            budget=0.8,
        )
        model = build_mean_risk(cfg)
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert float(np.sum(portfolio.weights)) == pytest.approx(0.8, abs=1e-6)

    def test_budget_0_weights_sum(self, returns_df: pd.DataFrame) -> None:
        """Zero budget=0.0 → weights sum to 0.0."""
        cfg = MeanRiskConfig(
            objective=ObjectiveFunctionType.MINIMIZE_RISK,
            budget=0.0,
        )
        model = build_mean_risk(cfg)
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert float(np.sum(portfolio.weights)) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Diversification ratio >= 1 (issue #106)
# ---------------------------------------------------------------------------


class TestDiversificationRatioGeOne:
    def test_min_variance_dr_ge_one(self, returns_df: pd.DataFrame) -> None:
        model = build_mean_risk(MeanRiskConfig.for_min_variance())
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert portfolio.diversification >= 1.0 - 1e-6

    def test_hrp_dr_ge_one(self, returns_df: pd.DataFrame) -> None:
        model = build_hrp(HRPConfig.for_variance())
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert portfolio.diversification >= 1.0 - 1e-6

    def test_max_diversification_dr_ge_one(self, returns_df: pd.DataFrame) -> None:
        model = build_max_diversification()
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert portfolio.diversification >= 1.0 - 1e-6


# ---------------------------------------------------------------------------
# CVaR >= VaR (issue #106)
# ---------------------------------------------------------------------------


class TestCVaRGeVaR:
    def test_min_cvar_portfolio(self, returns_df: pd.DataFrame) -> None:
        model = build_mean_risk(MeanRiskConfig.for_min_cvar())
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert portfolio.cvar >= portfolio.value_at_risk - 1e-8

    def test_min_variance_portfolio(self, returns_df: pd.DataFrame) -> None:
        model = build_mean_risk(MeanRiskConfig.for_min_variance())
        model.fit(returns_df)
        portfolio = model.predict(returns_df)
        assert portfolio.cvar >= portfolio.value_at_risk - 1e-8
