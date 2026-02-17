"""Configuration for portfolio optimization models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from optimizer.moments._config import MomentEstimationConfig

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ObjectiveFunctionType(str, Enum):
    """Objective function selection.

    Maps to :class:`skfolio.optimization.convex._base.ObjectiveFunction`.
    """

    MINIMIZE_RISK = "minimize_risk"
    MAXIMIZE_RETURN = "maximize_return"
    MAXIMIZE_UTILITY = "maximize_utility"
    MAXIMIZE_RATIO = "maximize_ratio"


class RiskMeasureType(str, Enum):
    """Convex risk measure selection.

    Maps to :class:`skfolio.measures.RiskMeasure`.
    """

    VARIANCE = "variance"
    SEMI_VARIANCE = "semi_variance"
    STANDARD_DEVIATION = "standard_deviation"
    SEMI_DEVIATION = "semi_deviation"
    MEAN_ABSOLUTE_DEVIATION = "mean_absolute_deviation"
    FIRST_LOWER_PARTIAL_MOMENT = "first_lower_partial_moment"
    CVAR = "cvar"
    EVAR = "evar"
    WORST_REALIZATION = "worst_realization"
    CDAR = "cdar"
    MAX_DRAWDOWN = "max_drawdown"
    AVERAGE_DRAWDOWN = "average_drawdown"
    EDAR = "edar"
    ULCER_INDEX = "ulcer_index"
    GINI_MEAN_DIFFERENCE = "gini_mean_difference"


class ExtraRiskMeasureType(str, Enum):
    """Non-convex risk measure selection (for HRP/HERC).

    Maps to :class:`skfolio.measures.ExtraRiskMeasure`.
    """

    VALUE_AT_RISK = "value_at_risk"
    DRAWDOWN_AT_RISK = "drawdown_at_risk"
    ENTROPIC_RISK_MEASURE = "entropic_risk_measure"
    FOURTH_CENTRAL_MOMENT = "fourth_central_moment"
    FOURTH_LOWER_PARTIAL_MOMENT = "fourth_lower_partial_moment"
    SKEW = "skew"
    KURTOSIS = "kurtosis"


class DistanceType(str, Enum):
    """Distance estimator selection.

    Maps to the distance classes in :mod:`skfolio.distance`.
    """

    PEARSON = "pearson"
    KENDALL = "kendall"
    SPEARMAN = "spearman"
    COVARIANCE = "covariance"
    DISTANCE_CORRELATION = "distance_correlation"
    MUTUAL_INFORMATION = "mutual_information"


class LinkageMethodType(str, Enum):
    """Linkage method selection for hierarchical clustering.

    Maps to :class:`skfolio.cluster.LinkageMethod`.
    """

    SINGLE = "single"
    COMPLETE = "complete"
    AVERAGE = "average"
    WEIGHTED = "weighted"
    CENTROID = "centroid"
    MEDIAN = "median"
    WARD = "ward"


class RatioMeasureType(str, Enum):
    """Ratio measure selection for scoring (e.g. stacking meta-optimiser).

    Maps to :class:`skfolio.measures.RatioMeasure`.
    """

    SHARPE_RATIO = "sharpe_ratio"
    ANNUALIZED_SHARPE_RATIO = "annualized_sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    ANNUALIZED_SORTINO_RATIO = "annualized_sortino_ratio"
    MEAN_ABSOLUTE_DEVIATION_RATIO = "mean_absolute_deviation_ratio"
    FIRST_LOWER_PARTIAL_MOMENT_RATIO = "first_lower_partial_moment_ratio"
    VALUE_AT_RISK_RATIO = "value_at_risk_ratio"
    CVAR_RATIO = "cvar_ratio"
    ENTROPIC_RISK_MEASURE_RATIO = "entropic_risk_measure_ratio"
    EVAR_RATIO = "evar_ratio"
    WORST_REALIZATION_RATIO = "worst_realization_ratio"
    DRAWDOWN_AT_RISK_RATIO = "drawdown_at_risk_ratio"
    CDAR_RATIO = "cdar_ratio"
    CALMAR_RATIO = "calmar_ratio"
    AVERAGE_DRAWDOWN_RATIO = "average_drawdown_ratio"
    EDAR_RATIO = "edar_ratio"
    ULCER_INDEX_RATIO = "ulcer_index_ratio"
    GINI_MEAN_DIFFERENCE_RATIO = "gini_mean_difference_ratio"


# ---------------------------------------------------------------------------
# Sub-configs (reusable building blocks)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DistanceConfig:
    """Immutable configuration for a distance estimator.

    Parameters
    ----------
    distance_type : DistanceType
        Which distance estimator to build.
    absolute : bool
        Whether to apply absolute transformation to the correlation
        matrix (Pearson, Kendall, Spearman, Covariance distances).
    power : float
        Exponent of the power transformation applied to the
        correlation matrix.
    threshold : float
        Distance correlation threshold (only for ``DISTANCE_CORRELATION``).
    """

    distance_type: DistanceType = DistanceType.PEARSON
    absolute: bool = False
    power: float = 1.0
    threshold: float = 0.5


@dataclass(frozen=True)
class ClusteringConfig:
    """Immutable configuration for hierarchical clustering.

    Parameters
    ----------
    max_clusters : int or None
        Maximum number of flat clusters.  ``None`` uses the
        Two-Order Difference to Gap Statistic heuristic.
    linkage_method : LinkageMethodType
        Linkage method for building the dendrogram.
    """

    max_clusters: int | None = None
    linkage_method: LinkageMethodType = LinkageMethodType.WARD


# ---------------------------------------------------------------------------
# Main optimiser configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MeanRiskConfig:
    """Immutable configuration for :class:`skfolio.optimization.MeanRisk`.

    Serialisable parameters only.  Non-serialisable objects
    (``prior_estimator``, ``previous_weights``, ``groups``,
    ``linear_constraints``, etc.) are passed as keyword arguments
    to the factory function.

    Parameters
    ----------
    objective : ObjectiveFunctionType
        Objective function.
    risk_measure : RiskMeasureType
        Convex risk measure.
    risk_aversion : float
        Risk-aversion coefficient (``MAXIMIZE_UTILITY``).
    efficient_frontier_size : int or None
        Number of points on the efficient frontier (``None`` = single
        portfolio).
    min_weights : float or None
        Lower bound on asset weights.
    max_weights : float or None
        Upper bound on asset weights.
    budget : float or None
        Portfolio budget (sum of weights).
    max_short : float or None
        Maximum short position.
    max_long : float or None
        Maximum long position.
    cardinality : int or None
        Maximum number of assets.
    transaction_costs : float
        Linear transaction costs penalising turnover relative to
        ``previous_weights``.
    management_fees : float
        Linear management fees proportional to position size.
    max_tracking_error : float or None
        Maximum tracking error relative to benchmark returns
        (passed as ``y`` in ``fit(X, y)``).
    l1_coef : float
        L1 regularisation coefficient.
    l2_coef : float
        L2 regularisation coefficient.
    risk_free_rate : float
        Risk-free rate for ratio objectives.
    cvar_beta : float
        CVaR confidence level.
    evar_beta : float
        EVaR confidence level.
    cdar_beta : float
        CDaR confidence level.
    edar_beta : float
        EDaR confidence level.
    solver : str
        CVXPY solver name.
    solver_params : dict or None
        Additional solver parameters.
    prior_config : MomentEstimationConfig or None
        Inner prior configuration.
    """

    objective: ObjectiveFunctionType = ObjectiveFunctionType.MINIMIZE_RISK
    risk_measure: RiskMeasureType = RiskMeasureType.VARIANCE
    risk_aversion: float = 1.0
    efficient_frontier_size: int | None = None
    min_weights: float | None = 0.0
    max_weights: float | None = 1.0
    budget: float | None = 1.0
    max_short: float | None = None
    max_long: float | None = None
    cardinality: int | None = None
    transaction_costs: float = 0.0
    management_fees: float = 0.0
    max_tracking_error: float | None = None
    l1_coef: float = 0.0
    l2_coef: float = 0.0
    risk_free_rate: float = 0.0
    cvar_beta: float = 0.95
    evar_beta: float = 0.95
    cdar_beta: float = 0.95
    edar_beta: float = 0.95
    solver: str = "CLARABEL"
    solver_params: dict[str, object] | None = None
    prior_config: MomentEstimationConfig | None = None

    # -- factory methods -----------------------------------------------------

    @classmethod
    def for_min_variance(cls) -> MeanRiskConfig:
        """Minimum-variance portfolio."""
        return cls(
            objective=ObjectiveFunctionType.MINIMIZE_RISK,
            risk_measure=RiskMeasureType.VARIANCE,
        )

    @classmethod
    def for_max_sharpe(cls) -> MeanRiskConfig:
        """Maximum Sharpe-ratio portfolio."""
        return cls(
            objective=ObjectiveFunctionType.MAXIMIZE_RATIO,
            risk_measure=RiskMeasureType.VARIANCE,
        )

    @classmethod
    def for_max_utility(cls, risk_aversion: float = 1.0) -> MeanRiskConfig:
        """Maximum utility portfolio."""
        return cls(
            objective=ObjectiveFunctionType.MAXIMIZE_UTILITY,
            risk_measure=RiskMeasureType.VARIANCE,
            risk_aversion=risk_aversion,
        )

    @classmethod
    def for_min_cvar(cls, beta: float = 0.95) -> MeanRiskConfig:
        """Minimum-CVaR portfolio."""
        return cls(
            objective=ObjectiveFunctionType.MINIMIZE_RISK,
            risk_measure=RiskMeasureType.CVAR,
            cvar_beta=beta,
        )

    @classmethod
    def for_efficient_frontier(
        cls,
        size: int = 20,
        risk_measure: RiskMeasureType = RiskMeasureType.VARIANCE,
    ) -> MeanRiskConfig:
        """Efficient frontier with *size* portfolios."""
        return cls(
            objective=ObjectiveFunctionType.MINIMIZE_RISK,
            risk_measure=risk_measure,
            efficient_frontier_size=size,
        )


@dataclass(frozen=True)
class RiskBudgetingConfig:
    """Immutable configuration for :class:`skfolio.optimization.RiskBudgeting`.

    The ``risk_budget`` array is passed as a factory argument (not stored
    here) because ``numpy`` arrays are not hashable in frozen dataclasses.

    Parameters
    ----------
    risk_measure : RiskMeasureType
        Convex risk measure.
    min_weights : float or None
        Lower bound on asset weights.
    max_weights : float or None
        Upper bound on asset weights.
    risk_free_rate : float
        Risk-free rate.
    cvar_beta : float
        CVaR confidence level.
    evar_beta : float
        EVaR confidence level.
    cdar_beta : float
        CDaR confidence level.
    edar_beta : float
        EDaR confidence level.
    solver : str
        CVXPY solver name.
    solver_params : dict or None
        Additional solver parameters.
    prior_config : MomentEstimationConfig or None
        Inner prior configuration.
    """

    risk_measure: RiskMeasureType = RiskMeasureType.VARIANCE
    min_weights: float | None = 0.0
    max_weights: float | None = 1.0
    risk_free_rate: float = 0.0
    cvar_beta: float = 0.95
    evar_beta: float = 0.95
    cdar_beta: float = 0.95
    edar_beta: float = 0.95
    solver: str = "CLARABEL"
    solver_params: dict[str, object] | None = None
    prior_config: MomentEstimationConfig | None = None

    # -- factory methods -----------------------------------------------------

    @classmethod
    def for_risk_parity(cls) -> RiskBudgetingConfig:
        """Equal risk-budget (risk parity) with variance."""
        return cls(risk_measure=RiskMeasureType.VARIANCE)

    @classmethod
    def for_cvar_parity(cls, beta: float = 0.95) -> RiskBudgetingConfig:
        """Equal risk-budget with CVaR."""
        return cls(
            risk_measure=RiskMeasureType.CVAR,
            cvar_beta=beta,
        )


@dataclass(frozen=True)
class MaxDiversificationConfig:
    """Immutable configuration for :class:`skfolio.optimization.MaximumDiversification`.

    Parameters
    ----------
    min_weights : float or None
        Lower bound on asset weights.
    max_weights : float or None
        Upper bound on asset weights.
    budget : float or None
        Portfolio budget (sum of weights).
    max_short : float or None
        Maximum short position.
    max_long : float or None
        Maximum long position.
    cardinality : int or None
        Maximum number of assets.
    l1_coef : float
        L1 regularisation coefficient.
    l2_coef : float
        L2 regularisation coefficient.
    risk_free_rate : float
        Risk-free rate.
    solver : str
        CVXPY solver name.
    solver_params : dict or None
        Additional solver parameters.
    prior_config : MomentEstimationConfig or None
        Inner prior configuration.
    """

    min_weights: float | None = 0.0
    max_weights: float | None = 1.0
    budget: float | None = 1.0
    max_short: float | None = None
    max_long: float | None = None
    cardinality: int | None = None
    l1_coef: float = 0.0
    l2_coef: float = 0.0
    risk_free_rate: float = 0.0
    solver: str = "CLARABEL"
    solver_params: dict[str, object] | None = None
    prior_config: MomentEstimationConfig | None = None


@dataclass(frozen=True)
class HRPConfig:
    """Immutable configuration for :class:`skfolio.optimization.HierarchicalRiskParity`.

    Parameters
    ----------
    risk_measure : RiskMeasureType
        Convex risk measure.
    extra_risk_measure : ExtraRiskMeasureType or None
        Non-convex risk measure (overrides ``risk_measure`` when set).
    min_weights : float or None
        Lower bound on asset weights.
    max_weights : float or None
        Upper bound on asset weights.
    distance_config : DistanceConfig or None
        Distance estimator configuration.
    clustering_config : ClusteringConfig or None
        Hierarchical clustering configuration.
    prior_config : MomentEstimationConfig or None
        Inner prior configuration.
    """

    risk_measure: RiskMeasureType = RiskMeasureType.VARIANCE
    extra_risk_measure: ExtraRiskMeasureType | None = None
    min_weights: float | None = 0.0
    max_weights: float | None = 1.0
    distance_config: DistanceConfig | None = None
    clustering_config: ClusteringConfig | None = None
    prior_config: MomentEstimationConfig | None = None

    # -- factory methods -----------------------------------------------------

    @classmethod
    def for_variance(cls) -> HRPConfig:
        """HRP with variance risk measure."""
        return cls(risk_measure=RiskMeasureType.VARIANCE)

    @classmethod
    def for_cvar(cls) -> HRPConfig:
        """HRP with CVaR risk measure."""
        return cls(risk_measure=RiskMeasureType.CVAR)


@dataclass(frozen=True)
class HERCConfig:
    """Immutable config for HERC optimisation.

    Wraps :class:`skfolio.optimization.HierarchicalEqualRiskContribution`.

    Parameters
    ----------
    risk_measure : RiskMeasureType
        Convex risk measure.
    extra_risk_measure : ExtraRiskMeasureType or None
        Non-convex risk measure (overrides ``risk_measure`` when set).
    min_weights : float or None
        Lower bound on asset weights.
    max_weights : float or None
        Upper bound on asset weights.
    solver : str
        CVXPY solver name.
    solver_params : dict or None
        Additional solver parameters.
    distance_config : DistanceConfig or None
        Distance estimator configuration.
    clustering_config : ClusteringConfig or None
        Hierarchical clustering configuration.
    prior_config : MomentEstimationConfig or None
        Inner prior configuration.
    """

    risk_measure: RiskMeasureType = RiskMeasureType.VARIANCE
    extra_risk_measure: ExtraRiskMeasureType | None = None
    min_weights: float | None = 0.0
    max_weights: float | None = 1.0
    solver: str = "CLARABEL"
    solver_params: dict[str, object] | None = None
    distance_config: DistanceConfig | None = None
    clustering_config: ClusteringConfig | None = None
    prior_config: MomentEstimationConfig | None = None

    # -- factory methods -----------------------------------------------------

    @classmethod
    def for_variance(cls) -> HERCConfig:
        """HERC with variance risk measure."""
        return cls(risk_measure=RiskMeasureType.VARIANCE)

    @classmethod
    def for_cvar(cls) -> HERCConfig:
        """HERC with CVaR risk measure."""
        return cls(risk_measure=RiskMeasureType.CVAR)


@dataclass(frozen=True)
class NCOConfig:
    """Immutable config for NCO optimisation.

    Wraps :class:`skfolio.optimization.NestedClustersOptimization`.

    The ``inner_estimator`` and ``outer_estimator`` are passed as factory
    arguments (not stored here) because they are not serialisable.

    Parameters
    ----------
    quantile : float
        Quantile for portfolio selection across cross-validation folds.
    n_jobs : int or None
        Number of parallel jobs for cross-validation.
    distance_config : DistanceConfig or None
        Distance estimator configuration.
    clustering_config : ClusteringConfig or None
        Hierarchical clustering configuration.
    """

    quantile: float = 0.5
    n_jobs: int | None = None
    distance_config: DistanceConfig | None = None
    clustering_config: ClusteringConfig | None = None


# ---------------------------------------------------------------------------
# Benchmark tracking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkTrackerConfig:
    """Immutable configuration for :class:`skfolio.optimization.BenchmarkTracker`.

    Minimises tracking error against benchmark returns (passed as ``y`` in
    ``fit(X, y)``).  Non-serialisable objects (``prior_estimator``,
    ``previous_weights``, ``groups``, ``linear_constraints``) are passed as
    keyword arguments to the factory function.

    Parameters
    ----------
    risk_measure : RiskMeasureType
        Risk measure for tracking error (default: standard deviation).
    min_weights : float or None
        Lower bound on asset weights.
    max_weights : float or None
        Upper bound on asset weights.
    max_short : float or None
        Maximum short position.
    max_long : float or None
        Maximum long position.
    cardinality : int or None
        Maximum number of assets.
    transaction_costs : float
        Linear transaction costs.
    management_fees : float
        Linear management fees.
    l1_coef : float
        L1 regularisation coefficient.
    l2_coef : float
        L2 regularisation coefficient.
    risk_free_rate : float
        Risk-free rate.
    solver : str
        CVXPY solver name.
    solver_params : dict or None
        Additional solver parameters.
    prior_config : MomentEstimationConfig or None
        Inner prior configuration.
    """

    risk_measure: RiskMeasureType = RiskMeasureType.STANDARD_DEVIATION
    min_weights: float | None = 0.0
    max_weights: float | None = 1.0
    max_short: float | None = None
    max_long: float | None = None
    cardinality: int | None = None
    transaction_costs: float = 0.0
    management_fees: float = 0.0
    l1_coef: float = 0.0
    l2_coef: float = 0.0
    risk_free_rate: float = 0.0
    solver: str = "CLARABEL"
    solver_params: dict[str, object] | None = None
    prior_config: MomentEstimationConfig | None = None


# ---------------------------------------------------------------------------
# Naive allocation methods
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EqualWeightedConfig:
    """Immutable configuration for :class:`skfolio.optimization.EqualWeighted`.

    The equal-weighted portfolio assigns identical weight to each asset
    (1/N).  No estimation is required.
    """


@dataclass(frozen=True)
class InverseVolatilityConfig:
    """Immutable config for :class:`skfolio.optimization.InverseVolatility`.

    Scales each position inversely to its estimated volatility.

    Parameters
    ----------
    prior_config : MomentEstimationConfig or None
        Inner prior configuration (covariance estimator determines
        volatility estimates).
    """

    prior_config: MomentEstimationConfig | None = None


# ---------------------------------------------------------------------------
# Ensemble optimisation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StackingConfig:
    """Immutable config for :class:`skfolio.optimization.StackingOptimization`.

    Combines multiple optimisers into an ensemble via a meta-optimiser.
    The ``estimators`` list and ``final_estimator`` are passed as factory
    arguments (not stored here) because they are not serialisable.

    Parameters
    ----------
    quantile : float
        Quantile for portfolio selection across cross-validation folds.
    quantile_measure : RatioMeasureType
        Ratio measure used when selecting the quantile portfolio.
    n_jobs : int or None
        Number of parallel jobs for cross-validation.
    cv : int or None
        Number of cross-validation folds (``None`` = no cross-validation).
    """

    quantile: float = 0.5
    quantile_measure: RatioMeasureType = RatioMeasureType.SHARPE_RATIO
    n_jobs: int | None = None
    cv: int | None = None
