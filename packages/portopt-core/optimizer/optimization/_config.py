"""Configuration for portfolio optimization models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from optimizer.moments._config import MomentEstimationConfig


class ObjectiveFunctionType(str, Enum):
    """Objective function selection.

    Maps to `ObjectiveFunction`.
    """

    MINIMIZE_RISK = "minimize_risk"
    MAXIMIZE_RETURN = "maximize_return"
    MAXIMIZE_UTILITY = "maximize_utility"
    MAXIMIZE_RATIO = "maximize_ratio"


class RiskMeasureType(str, Enum):
    """Convex risk measure selection.

    Maps to `RiskMeasure`.
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


class RatioMeasureType(str, Enum):
    """Ratio measure selection for scoring.

    Most members map directly to `RatioMeasure`.
    ``INFORMATION_RATIO`` is implemented as a custom scorer (active return
    divided by tracking error) because skfolio does not expose it natively;
    use `build_scorer` with a ``benchmark_returns``
    argument to build the corresponding callable.
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
    INFORMATION_RATIO = "information_ratio"


class FallbackPolicy(str, Enum):
    """Serialisable fallback policy for the optimizer resilience layer.

    Maps to the ``fallback`` argument accepted by every skfolio 1.0
    optimizer.  Only the *serialisable* subset is exposed here — an
    estimator (or list of estimators) is non-serialisable and must be
    supplied as the ``fallback`` factory keyword argument instead.

    * ``NONE`` -> ``fallback=None`` (no fallback; default).
    * ``PREVIOUS_WEIGHTS`` -> ``fallback="previous_weights"`` (reuse the
      last good allocation when a rebalance is infeasible).
    """

    NONE = "none"
    PREVIOUS_WEIGHTS = "previous_weights"


@dataclass(frozen=True)
class MeanRiskConfig:
    """Immutable configuration for `MeanRisk`.

    Serialisable parameters only.  Non-serialisable objects
    (``prior_estimator``, ``previous_weights``, ``groups``,
    ``linear_constraints``, etc.) are passed as keyword arguments
    to the factory function.

    Attributes:
        objective: Objective function.
        risk_measure: Convex risk measure.
        risk_aversion: Risk-aversion coefficient (``MAXIMIZE_UTILITY``).
        efficient_frontier_size: Number of points on the efficient frontier;
            ``None`` yields a single portfolio.
        min_weights: Lower bound on asset weights.
        max_weights: Upper bound on asset weights.
        budget: Portfolio budget (sum of weights).
        max_short: Maximum short position.
        max_long: Maximum long position.
        cardinality: Maximum number of assets.
        transaction_costs: Linear transaction costs penalising turnover relative
            to ``previous_weights``.
        management_fees: Linear management fees proportional to position size.
        max_tracking_error: Maximum tracking error relative to benchmark returns
            (passed as ``y`` in ``fit(X, y)``).
        l1_coef: L1 regularisation coefficient.
        l2_coef: L2 regularisation coefficient.
        risk_free_rate: Risk-free rate for ratio objectives.
        cvar_beta: CVaR confidence level.
        evar_beta: EVaR confidence level.
        cdar_beta: CDaR confidence level.
        edar_beta: EDaR confidence level.
        solver: CVXPY solver name.
        solver_params: Additional solver parameters.
        prior_config: Inner prior configuration.
        max_sector_weight: Maximum total weight allocated to any single sector.
            When set, ``build_mean_risk()`` requires a ``sector_mapping`` kwarg
            to resolve sector membership.  ``None`` disables sector constraints.
        raise_on_failure: Resilience switch forwarded to
            ``MeanRisk.raise_on_failure`` (skfolio 1.0).  When ``True``
            (default) an infeasible / unsolved optimisation raises; when
            ``False`` the estimator yields a ``FailedPortfolio`` sentinel on
            ``predict`` and records the reason in ``error_`` instead of raising
            — useful inside walk-forward backtests that must survive an
            infeasible fold.  An explicit ``raise_on_failure`` factory kwarg
            overrides this field.
        fallback_policy: Serialisable fallback strategy forwarded to
            ``MeanRisk.fallback`` (skfolio 1.0).  ``NONE`` (default) attaches
            no fallback; ``PREVIOUS_WEIGHTS`` reuses the last good allocation
            when a rebalance is infeasible (supply ``previous_weights`` at
            factory time for it to reuse).  For an estimator/list fallback
            (non-serialisable) pass the ``fallback`` factory kwarg directly,
            which overrides this field.
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
    max_sector_weight: float | None = None
    raise_on_failure: bool = True
    fallback_policy: FallbackPolicy = FallbackPolicy.NONE

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
    def for_max_sharpe_diversified(cls) -> MeanRiskConfig:
        """Maximum Sharpe with diversification constraints.

        Targets the Sharpe ratio while using L2 regularisation and
        a per-asset weight cap to enforce diversification and keep
        volatility stable.  Uses ShrunkMu + DenoiseCovariance for
        robust moment estimates.
        """
        return cls(
            objective=ObjectiveFunctionType.MAXIMIZE_RATIO,
            risk_measure=RiskMeasureType.VARIANCE,
            max_weights=0.10,
            l2_coef=0.05,
            prior_config=MomentEstimationConfig.for_shrunk_denoised(),
        )

    @classmethod
    def for_concentrated_sharpe(cls) -> MeanRiskConfig:
        """Max Sharpe for concentrated portfolios (15-30 stocks).

        Uses ``min_weights=0.01`` to ensure all selected stocks get a
        meaningful allocation and ``max_weights=0.10`` to cap individual
        positions.  L2 regularisation pushes toward equal-weight.
        """
        return cls(
            objective=ObjectiveFunctionType.MAXIMIZE_RATIO,
            risk_measure=RiskMeasureType.VARIANCE,
            min_weights=0.01,
            max_weights=0.10,
            l2_coef=0.05,
            prior_config=MomentEstimationConfig.for_shrunk_denoised(),
        )

    @classmethod
    def for_max_sharpe_sector_constrained(
        cls,
        max_sector_weight: float = 0.25,
    ) -> MeanRiskConfig:
        """Max Sharpe with per-asset and per-sector diversification constraints.

        Combines L2 regularisation, a 10% per-asset cap, and a uniform
        per-sector cap to prevent sector concentration.

        Uses ShrunkMu + DenoiseCovariance for robust moment estimates.

        Args:
            max_sector_weight: Maximum total weight for any single sector.
        """
        return cls(
            objective=ObjectiveFunctionType.MAXIMIZE_RATIO,
            risk_measure=RiskMeasureType.VARIANCE,
            max_weights=0.10,
            l2_coef=0.05,
            max_sector_weight=max_sector_weight,
            prior_config=MomentEstimationConfig.for_shrunk_denoised(),
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
        """Efficient frontier with `size` portfolios."""
        return cls(
            objective=ObjectiveFunctionType.MINIMIZE_RISK,
            risk_measure=risk_measure,
            efficient_frontier_size=size,
        )

    @classmethod
    def for_tracking_error(
        cls,
        max_tracking_error: float = 0.05,
    ) -> MeanRiskConfig:
        """Max-Sharpe portfolio with a hard tracking-error cap.

        The benchmark return series MUST be supplied as ``y`` in
        ``fit(X, y)`` — skfolio raises when ``max_tracking_error`` is set
        but ``y`` is omitted.

        Args:
            max_tracking_error: Upper bound on annualised tracking error vs.
                the benchmark.
        """
        return cls(
            objective=ObjectiveFunctionType.MAXIMIZE_RATIO,
            risk_measure=RiskMeasureType.VARIANCE,
            max_tracking_error=max_tracking_error,
        )

    @classmethod
    def for_resilient_walk_forward(cls) -> MeanRiskConfig:
        """Max-Sharpe preset that survives an infeasible rebalance.

        Sets ``fallback_policy=PREVIOUS_WEIGHTS`` and
        ``raise_on_failure=False`` so a walk-forward / online backtest
        continues past an infeasible fold — reusing the previous
        allocation when ``previous_weights`` is supplied, or emitting a
        ``FailedPortfolio`` sentinel otherwise — instead of raising.
        """
        return cls(
            objective=ObjectiveFunctionType.MAXIMIZE_RATIO,
            risk_measure=RiskMeasureType.VARIANCE,
            raise_on_failure=False,
            fallback_policy=FallbackPolicy.PREVIOUS_WEIGHTS,
        )
