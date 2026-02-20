"""Factory functions for building skfolio optimization estimators."""

from __future__ import annotations

from typing import Any

import numpy as np
from skfolio.cluster import HierarchicalClustering, LinkageMethod
from skfolio.distance import (
    BaseDistance,
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
from skfolio.prior._base import BasePrior

from optimizer.factors._integration import FactorExposureConstraints
from optimizer.moments._factory import build_prior
from optimizer.optimization._config import (
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

# ---------------------------------------------------------------------------
# Mapping dicts
# ---------------------------------------------------------------------------

_OBJECTIVE_MAP: dict[ObjectiveFunctionType, ObjectiveFunction] = {
    ObjectiveFunctionType.MINIMIZE_RISK: ObjectiveFunction.MINIMIZE_RISK,
    ObjectiveFunctionType.MAXIMIZE_RETURN: ObjectiveFunction.MAXIMIZE_RETURN,
    ObjectiveFunctionType.MAXIMIZE_UTILITY: ObjectiveFunction.MAXIMIZE_UTILITY,
    ObjectiveFunctionType.MAXIMIZE_RATIO: ObjectiveFunction.MAXIMIZE_RATIO,
}

_RISK_MEASURE_MAP: dict[RiskMeasureType, RiskMeasure] = {
    RiskMeasureType.VARIANCE: RiskMeasure.VARIANCE,
    RiskMeasureType.SEMI_VARIANCE: RiskMeasure.SEMI_VARIANCE,
    RiskMeasureType.STANDARD_DEVIATION: RiskMeasure.STANDARD_DEVIATION,
    RiskMeasureType.SEMI_DEVIATION: RiskMeasure.SEMI_DEVIATION,
    RiskMeasureType.MEAN_ABSOLUTE_DEVIATION: RiskMeasure.MEAN_ABSOLUTE_DEVIATION,
    RiskMeasureType.FIRST_LOWER_PARTIAL_MOMENT: RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT,
    RiskMeasureType.CVAR: RiskMeasure.CVAR,
    RiskMeasureType.EVAR: RiskMeasure.EVAR,
    RiskMeasureType.WORST_REALIZATION: RiskMeasure.WORST_REALIZATION,
    RiskMeasureType.CDAR: RiskMeasure.CDAR,
    RiskMeasureType.MAX_DRAWDOWN: RiskMeasure.MAX_DRAWDOWN,
    RiskMeasureType.AVERAGE_DRAWDOWN: RiskMeasure.AVERAGE_DRAWDOWN,
    RiskMeasureType.EDAR: RiskMeasure.EDAR,
    RiskMeasureType.ULCER_INDEX: RiskMeasure.ULCER_INDEX,
    RiskMeasureType.GINI_MEAN_DIFFERENCE: RiskMeasure.GINI_MEAN_DIFFERENCE,
}

_EXTRA_RISK_MEASURE_MAP: dict[ExtraRiskMeasureType, ExtraRiskMeasure] = {
    ExtraRiskMeasureType.VALUE_AT_RISK: ExtraRiskMeasure.VALUE_AT_RISK,
    ExtraRiskMeasureType.DRAWDOWN_AT_RISK: ExtraRiskMeasure.DRAWDOWN_AT_RISK,
    ExtraRiskMeasureType.ENTROPIC_RISK_MEASURE: ExtraRiskMeasure.ENTROPIC_RISK_MEASURE,
    ExtraRiskMeasureType.FOURTH_CENTRAL_MOMENT: ExtraRiskMeasure.FOURTH_CENTRAL_MOMENT,
    ExtraRiskMeasureType.FOURTH_LOWER_PARTIAL_MOMENT: (
        ExtraRiskMeasure.FOURTH_LOWER_PARTIAL_MOMENT
    ),
    ExtraRiskMeasureType.SKEW: ExtraRiskMeasure.SKEW,
    ExtraRiskMeasureType.KURTOSIS: ExtraRiskMeasure.KURTOSIS,
}

_LINKAGE_MAP: dict[LinkageMethodType, LinkageMethod] = {
    LinkageMethodType.SINGLE: LinkageMethod.SINGLE,
    LinkageMethodType.COMPLETE: LinkageMethod.COMPLETE,
    LinkageMethodType.AVERAGE: LinkageMethod.AVERAGE,
    LinkageMethodType.WEIGHTED: LinkageMethod.WEIGHTED,
    LinkageMethodType.CENTROID: LinkageMethod.CENTROID,
    LinkageMethodType.MEDIAN: LinkageMethod.MEDIAN,
    LinkageMethodType.WARD: LinkageMethod.WARD,
}

_RATIO_MEASURE_MAP: dict[RatioMeasureType, RatioMeasure] = {
    RatioMeasureType.SHARPE_RATIO: RatioMeasure.SHARPE_RATIO,
    RatioMeasureType.ANNUALIZED_SHARPE_RATIO: RatioMeasure.ANNUALIZED_SHARPE_RATIO,
    RatioMeasureType.SORTINO_RATIO: RatioMeasure.SORTINO_RATIO,
    RatioMeasureType.ANNUALIZED_SORTINO_RATIO: RatioMeasure.ANNUALIZED_SORTINO_RATIO,
    RatioMeasureType.MEAN_ABSOLUTE_DEVIATION_RATIO: (
        RatioMeasure.MEAN_ABSOLUTE_DEVIATION_RATIO
    ),
    RatioMeasureType.FIRST_LOWER_PARTIAL_MOMENT_RATIO: (
        RatioMeasure.FIRST_LOWER_PARTIAL_MOMENT_RATIO
    ),
    RatioMeasureType.VALUE_AT_RISK_RATIO: RatioMeasure.VALUE_AT_RISK_RATIO,
    RatioMeasureType.CVAR_RATIO: RatioMeasure.CVAR_RATIO,
    RatioMeasureType.ENTROPIC_RISK_MEASURE_RATIO: (
        RatioMeasure.ENTROPIC_RISK_MEASURE_RATIO
    ),
    RatioMeasureType.EVAR_RATIO: RatioMeasure.EVAR_RATIO,
    RatioMeasureType.WORST_REALIZATION_RATIO: RatioMeasure.WORST_REALIZATION_RATIO,
    RatioMeasureType.DRAWDOWN_AT_RISK_RATIO: RatioMeasure.DRAWDOWN_AT_RISK_RATIO,
    RatioMeasureType.CDAR_RATIO: RatioMeasure.CDAR_RATIO,
    RatioMeasureType.CALMAR_RATIO: RatioMeasure.CALMAR_RATIO,
    RatioMeasureType.AVERAGE_DRAWDOWN_RATIO: RatioMeasure.AVERAGE_DRAWDOWN_RATIO,
    RatioMeasureType.EDAR_RATIO: RatioMeasure.EDAR_RATIO,
    RatioMeasureType.ULCER_INDEX_RATIO: RatioMeasure.ULCER_INDEX_RATIO,
    RatioMeasureType.GINI_MEAN_DIFFERENCE_RATIO: (
        RatioMeasure.GINI_MEAN_DIFFERENCE_RATIO
    ),
}


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def build_distance_estimator(
    config: DistanceConfig | None = None,
    *,
    covariance_estimator: Any | None = None,
) -> BaseDistance:
    """Build a skfolio distance estimator from *config*.

    Parameters
    ----------
    config : DistanceConfig or None
        Distance configuration.  Defaults to ``DistanceConfig()``
        (Pearson distance).
    covariance_estimator : BaseCovariance or None
        Covariance estimator for ``CovarianceDistance``.  Ignored
        for other distance types.

    Returns
    -------
    BaseDistance
        A fitted-ready skfolio distance estimator.
    """
    if config is None:
        config = DistanceConfig()

    match config.distance_type:
        case DistanceType.PEARSON:
            return PearsonDistance(
                absolute=config.absolute,
                power=config.power,
            )
        case DistanceType.KENDALL:
            return KendallDistance(
                absolute=config.absolute,
                power=config.power,
            )
        case DistanceType.SPEARMAN:
            return SpearmanDistance(
                absolute=config.absolute,
                power=config.power,
            )
        case DistanceType.COVARIANCE:
            return CovarianceDistance(
                covariance_estimator=covariance_estimator,
                absolute=config.absolute,
                power=config.power,
            )
        case DistanceType.DISTANCE_CORRELATION:
            return DistanceCorrelation(threshold=config.threshold)
        case DistanceType.MUTUAL_INFORMATION:
            return MutualInformation()


def build_clustering_estimator(
    config: ClusteringConfig | None = None,
) -> HierarchicalClustering:
    """Build a skfolio hierarchical clustering estimator from *config*.

    Parameters
    ----------
    config : ClusteringConfig or None
        Clustering configuration.  Defaults to ``ClusteringConfig()``
        (Ward linkage, auto max_clusters).

    Returns
    -------
    HierarchicalClustering
        A fitted-ready skfolio clustering estimator.
    """
    if config is None:
        config = ClusteringConfig()

    return HierarchicalClustering(
        max_clusters=config.max_clusters,
        linkage_method=_LINKAGE_MAP[config.linkage_method],
    )


# ---------------------------------------------------------------------------
# Main optimiser factories
# ---------------------------------------------------------------------------


def _resolve_risk_measure(
    risk_measure: RiskMeasureType,
    extra_risk_measure: ExtraRiskMeasureType | None,
) -> RiskMeasure | ExtraRiskMeasure:
    """Resolve a risk measure from config enums."""
    if extra_risk_measure is not None:
        return _EXTRA_RISK_MEASURE_MAP[extra_risk_measure]
    return _RISK_MEASURE_MAP[risk_measure]


def build_mean_risk(
    config: MeanRiskConfig | None = None,
    *,
    prior_estimator: BasePrior | None = None,
    factor_exposure_constraints: FactorExposureConstraints | None = None,
    **kwargs: Any,
) -> MeanRisk:
    """Build a skfolio :class:`MeanRisk` optimiser from *config*.

    Parameters
    ----------
    config : MeanRiskConfig or None
        Mean-risk configuration.  Defaults to ``MeanRiskConfig()``
        (minimum-variance).
    prior_estimator : BasePrior or None
        Prior estimator.  When ``None``, one is built from
        ``config.prior_config`` (or skfolio default).
    factor_exposure_constraints : FactorExposureConstraints or None
        Enforceable factor exposure constraints produced by
        :func:`~optimizer.factors.build_factor_exposure_constraints`.
        When provided, ``left_inequality`` and ``right_inequality`` are
        injected into the :class:`MeanRisk` constructor.  Any explicit
        ``left_inequality`` / ``right_inequality`` entries in ``kwargs``
        take precedence.
    **kwargs
        Additional keyword arguments forwarded to the
        :class:`MeanRisk` constructor (for non-serialisable
        parameters such as ``previous_weights``, ``groups``,
        ``linear_constraints``, etc.).

    Returns
    -------
    MeanRisk
        A fitted-ready skfolio optimiser.
    """
    if config is None:
        config = MeanRiskConfig()

    if prior_estimator is None and config.prior_config is not None:
        prior_estimator = build_prior(config.prior_config)

    if factor_exposure_constraints is not None:
        kwargs.setdefault(
            "left_inequality", factor_exposure_constraints.left_inequality
        )
        kwargs.setdefault(
            "right_inequality", factor_exposure_constraints.right_inequality
        )

    return MeanRisk(
        objective_function=_OBJECTIVE_MAP[config.objective],
        risk_measure=_RISK_MEASURE_MAP[config.risk_measure],
        risk_aversion=config.risk_aversion,
        efficient_frontier_size=config.efficient_frontier_size,
        prior_estimator=prior_estimator,
        min_weights=config.min_weights,
        max_weights=config.max_weights,
        budget=config.budget,
        max_short=config.max_short,
        max_long=config.max_long,
        cardinality=config.cardinality,
        transaction_costs=config.transaction_costs,
        management_fees=config.management_fees,
        max_tracking_error=config.max_tracking_error,
        l1_coef=config.l1_coef,
        l2_coef=config.l2_coef,
        risk_free_rate=config.risk_free_rate,
        cvar_beta=config.cvar_beta,
        evar_beta=config.evar_beta,
        cdar_beta=config.cdar_beta,
        edar_beta=config.edar_beta,
        solver=config.solver,
        solver_params=config.solver_params,
        **kwargs,
    )


def build_risk_budgeting(
    config: RiskBudgetingConfig | None = None,
    *,
    risk_budget: np.ndarray | None = None,
    prior_estimator: BasePrior | None = None,
    **kwargs: Any,
) -> RiskBudgeting:
    """Build a skfolio :class:`RiskBudgeting` optimiser from *config*.

    Parameters
    ----------
    config : RiskBudgetingConfig or None
        Risk-budgeting configuration.  Defaults to
        ``RiskBudgetingConfig()`` (variance risk parity).
    risk_budget : ndarray or None
        Per-asset risk budget.  ``None`` gives equal budgets.
    prior_estimator : BasePrior or None
        Prior estimator.  When ``None``, one is built from
        ``config.prior_config`` (or skfolio default).
    **kwargs
        Additional keyword arguments forwarded to the
        :class:`RiskBudgeting` constructor.

    Returns
    -------
    RiskBudgeting
        A fitted-ready skfolio optimiser.
    """
    if config is None:
        config = RiskBudgetingConfig()

    if prior_estimator is None and config.prior_config is not None:
        prior_estimator = build_prior(config.prior_config)

    return RiskBudgeting(
        risk_measure=_RISK_MEASURE_MAP[config.risk_measure],
        risk_budget=risk_budget,
        prior_estimator=prior_estimator,
        min_weights=config.min_weights,
        max_weights=config.max_weights,
        risk_free_rate=config.risk_free_rate,
        cvar_beta=config.cvar_beta,
        evar_beta=config.evar_beta,
        cdar_beta=config.cdar_beta,
        edar_beta=config.edar_beta,
        solver=config.solver,
        solver_params=config.solver_params,
        **kwargs,
    )


def build_max_diversification(
    config: MaxDiversificationConfig | None = None,
    *,
    prior_estimator: BasePrior | None = None,
    **kwargs: Any,
) -> MaximumDiversification:
    """Build a skfolio :class:`MaximumDiversification` optimiser from *config*.

    Parameters
    ----------
    config : MaxDiversificationConfig or None
        Configuration.  Defaults to ``MaxDiversificationConfig()``.
    prior_estimator : BasePrior or None
        Prior estimator.  When ``None``, one is built from
        ``config.prior_config`` (or skfolio default).
    **kwargs
        Additional keyword arguments forwarded to the
        :class:`MaximumDiversification` constructor.

    Returns
    -------
    MaximumDiversification
        A fitted-ready skfolio optimiser.
    """
    if config is None:
        config = MaxDiversificationConfig()

    if prior_estimator is None and config.prior_config is not None:
        prior_estimator = build_prior(config.prior_config)

    return MaximumDiversification(
        prior_estimator=prior_estimator,
        min_weights=config.min_weights,
        max_weights=config.max_weights,
        budget=config.budget,
        max_short=config.max_short,
        max_long=config.max_long,
        cardinality=config.cardinality,
        l1_coef=config.l1_coef,
        l2_coef=config.l2_coef,
        risk_free_rate=config.risk_free_rate,
        solver=config.solver,
        solver_params=config.solver_params,
        **kwargs,
    )


def build_hrp(
    config: HRPConfig | None = None,
    *,
    prior_estimator: BasePrior | None = None,
    distance_estimator: BaseDistance | None = None,
    **kwargs: Any,
) -> HierarchicalRiskParity:
    """Build a skfolio :class:`HierarchicalRiskParity` optimiser from *config*.

    Parameters
    ----------
    config : HRPConfig or None
        HRP configuration.  Defaults to ``HRPConfig()`` (variance).
    prior_estimator : BasePrior or None
        Prior estimator.
    distance_estimator : BaseDistance or None
        Distance estimator.  When ``None``, one is built from
        ``config.distance_config`` (or skfolio default).
    **kwargs
        Additional keyword arguments forwarded to the
        :class:`HierarchicalRiskParity` constructor.

    Returns
    -------
    HierarchicalRiskParity
        A fitted-ready skfolio optimiser.
    """
    if config is None:
        config = HRPConfig()

    if prior_estimator is None and config.prior_config is not None:
        prior_estimator = build_prior(config.prior_config)

    if distance_estimator is None and config.distance_config is not None:
        distance_estimator = build_distance_estimator(config.distance_config)

    clustering = (
        build_clustering_estimator(config.clustering_config)
        if config.clustering_config is not None
        else None
    )

    risk_measure = _resolve_risk_measure(
        config.risk_measure,
        config.extra_risk_measure,
    )

    return HierarchicalRiskParity(
        risk_measure=risk_measure,
        prior_estimator=prior_estimator,
        distance_estimator=distance_estimator,
        hierarchical_clustering_estimator=clustering,
        min_weights=config.min_weights,
        max_weights=config.max_weights,
        **kwargs,
    )


def build_herc(
    config: HERCConfig | None = None,
    *,
    prior_estimator: BasePrior | None = None,
    distance_estimator: BaseDistance | None = None,
    **kwargs: Any,
) -> HierarchicalEqualRiskContribution:
    """Build a skfolio HERC optimiser from *config*.

    Parameters
    ----------
    config : HERCConfig or None
        HERC configuration.  Defaults to ``HERCConfig()`` (variance).
    prior_estimator : BasePrior or None
        Prior estimator.
    distance_estimator : BaseDistance or None
        Distance estimator.  When ``None``, one is built from
        ``config.distance_config`` (or skfolio default).
    **kwargs
        Additional keyword arguments forwarded to the
        :class:`HierarchicalEqualRiskContribution` constructor.

    Returns
    -------
    HierarchicalEqualRiskContribution
        A fitted-ready skfolio optimiser.
    """
    if config is None:
        config = HERCConfig()

    if prior_estimator is None and config.prior_config is not None:
        prior_estimator = build_prior(config.prior_config)

    if distance_estimator is None and config.distance_config is not None:
        distance_estimator = build_distance_estimator(config.distance_config)

    clustering = (
        build_clustering_estimator(config.clustering_config)
        if config.clustering_config is not None
        else None
    )

    risk_measure = _resolve_risk_measure(
        config.risk_measure,
        config.extra_risk_measure,
    )

    return HierarchicalEqualRiskContribution(
        risk_measure=risk_measure,
        prior_estimator=prior_estimator,
        distance_estimator=distance_estimator,
        hierarchical_clustering_estimator=clustering,
        min_weights=config.min_weights,
        max_weights=config.max_weights,
        solver=config.solver,
        solver_params=config.solver_params,
        **kwargs,
    )


def build_nco(
    config: NCOConfig | None = None,
    *,
    inner_estimator: Any | None = None,
    outer_estimator: Any | None = None,
    distance_estimator: BaseDistance | None = None,
    **kwargs: Any,
) -> NestedClustersOptimization:
    """Build a skfolio :class:`NestedClustersOptimization` optimiser from *config*.

    Parameters
    ----------
    config : NCOConfig or None
        NCO configuration.  Defaults to ``NCOConfig()``.
    inner_estimator : BaseOptimization or None
        Inner cluster optimiser.
    outer_estimator : BaseOptimization or None
        Outer (inter-cluster) optimiser.
    distance_estimator : BaseDistance or None
        Distance estimator.  When ``None``, one is built from
        ``config.distance_config`` (or skfolio default).
    **kwargs
        Additional keyword arguments forwarded to the
        :class:`NestedClustersOptimization` constructor.

    Returns
    -------
    NestedClustersOptimization
        A fitted-ready skfolio optimiser.
    """
    if config is None:
        config = NCOConfig()

    if distance_estimator is None and config.distance_config is not None:
        distance_estimator = build_distance_estimator(config.distance_config)

    clustering = (
        build_clustering_estimator(config.clustering_config)
        if config.clustering_config is not None
        else None
    )

    return NestedClustersOptimization(
        inner_estimator=inner_estimator,
        outer_estimator=outer_estimator,
        distance_estimator=distance_estimator,
        clustering_estimator=clustering,
        quantile=config.quantile,
        n_jobs=config.n_jobs,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Benchmark tracking
# ---------------------------------------------------------------------------


def build_benchmark_tracker(
    config: BenchmarkTrackerConfig | None = None,
    *,
    prior_estimator: BasePrior | None = None,
    **kwargs: Any,
) -> BenchmarkTracker:
    """Build a skfolio :class:`BenchmarkTracker` optimiser from *config*.

    The resulting model minimises tracking error against benchmark
    returns.  Pass benchmark returns as ``y`` in ``model.fit(X, y)``.

    Parameters
    ----------
    config : BenchmarkTrackerConfig or None
        Benchmark tracker configuration.  Defaults to
        ``BenchmarkTrackerConfig()``.
    prior_estimator : BasePrior or None
        Prior estimator.  When ``None``, one is built from
        ``config.prior_config`` (or skfolio default).
    **kwargs
        Additional keyword arguments forwarded to the
        :class:`BenchmarkTracker` constructor (for non-serialisable
        parameters such as ``previous_weights``, ``groups``,
        ``linear_constraints``).

    Returns
    -------
    BenchmarkTracker
        A fitted-ready skfolio optimiser.
    """
    if config is None:
        config = BenchmarkTrackerConfig()

    if prior_estimator is None and config.prior_config is not None:
        prior_estimator = build_prior(config.prior_config)

    return BenchmarkTracker(
        risk_measure=_RISK_MEASURE_MAP[config.risk_measure],
        prior_estimator=prior_estimator,
        min_weights=config.min_weights,
        max_weights=config.max_weights,
        max_short=config.max_short,
        max_long=config.max_long,
        cardinality=config.cardinality,
        transaction_costs=config.transaction_costs,
        management_fees=config.management_fees,
        l1_coef=config.l1_coef,
        l2_coef=config.l2_coef,
        risk_free_rate=config.risk_free_rate,
        solver=config.solver,
        solver_params=config.solver_params,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Naive allocation methods
# ---------------------------------------------------------------------------


def build_equal_weighted(
    config: EqualWeightedConfig | None = None,  # noqa: ARG001
    **kwargs: Any,
) -> EqualWeighted:
    """Build a skfolio :class:`EqualWeighted` (1/N) optimiser.

    Parameters
    ----------
    config : EqualWeightedConfig or None
        Configuration (currently has no parameters).
    **kwargs
        Additional keyword arguments forwarded to the
        :class:`EqualWeighted` constructor.

    Returns
    -------
    EqualWeighted
        A fitted-ready skfolio optimiser.
    """
    return EqualWeighted(**kwargs)


def build_inverse_volatility(
    config: InverseVolatilityConfig | None = None,
    *,
    prior_estimator: BasePrior | None = None,
    **kwargs: Any,
) -> InverseVolatility:
    """Build a skfolio :class:`InverseVolatility` optimiser from *config*.

    Parameters
    ----------
    config : InverseVolatilityConfig or None
        Configuration.  Defaults to ``InverseVolatilityConfig()``.
    prior_estimator : BasePrior or None
        Prior estimator (covariance estimator determines volatility).
        When ``None``, one is built from ``config.prior_config``
        (or skfolio default).
    **kwargs
        Additional keyword arguments forwarded to the
        :class:`InverseVolatility` constructor.

    Returns
    -------
    InverseVolatility
        A fitted-ready skfolio optimiser.
    """
    if config is None:
        config = InverseVolatilityConfig()

    if prior_estimator is None and config.prior_config is not None:
        prior_estimator = build_prior(config.prior_config)

    return InverseVolatility(
        prior_estimator=prior_estimator,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Ensemble optimisation
# ---------------------------------------------------------------------------


def build_stacking(
    config: StackingConfig | None = None,
    *,
    estimators: list[tuple[str, Any]] | None = None,
    final_estimator: Any | None = None,
    **kwargs: Any,
) -> StackingOptimization:
    """Build a skfolio :class:`StackingOptimization` ensemble from *config*.

    Parameters
    ----------
    config : StackingConfig or None
        Stacking configuration.  Defaults to ``StackingConfig()``.
    estimators : list of (name, estimator) tuples or None
        Sub-optimisers whose outputs are combined.  Required by
        skfolio but passed here (not in config) because estimator
        objects are not serialisable.
    final_estimator : BaseOptimization or None
        Meta-optimiser that allocates across sub-portfolios.
    **kwargs
        Additional keyword arguments forwarded to the
        :class:`StackingOptimization` constructor.

    Returns
    -------
    StackingOptimization
        A fitted-ready skfolio ensemble optimiser.
    """
    if config is None:
        config = StackingConfig()

    if estimators is None:
        estimators = [
            ("mean_risk", MeanRisk()),
            ("hrp", HierarchicalRiskParity()),
        ]

    return StackingOptimization(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=config.cv,
        quantile=config.quantile,
        quantile_measure=_RATIO_MEASURE_MAP[config.quantile_measure],
        n_jobs=config.n_jobs,
        **kwargs,
    )
