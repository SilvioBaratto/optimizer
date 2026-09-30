"""BenchmarkTracker configuration and factory.

Benchmark returns are passed as ``y`` in ``fit(X, y)`` — not as a
Config field. The benchmark is a non-serialisable Series and therefore
must be supplied at fit time, not at config construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from skfolio.optimization import BenchmarkTracker
from skfolio.prior._base import BasePrior

from optimizer.moments._config import MomentEstimationConfig
from optimizer.moments._factory import build_prior
from optimizer.optimization._config import RiskMeasureType
from optimizer.optimization._factory import _RISK_MEASURE_MAP


@dataclass(frozen=True)
class BenchmarkTrackerConfig:
    """Immutable configuration for BenchmarkTracker.

    Attributes:
        risk_measure: Tracking-error risk measure. Defaults to STANDARD_DEVIATION.
        prior_config: Inner prior configuration. None defers to the skfolio default.
        min_weights: Lower bound on asset weights.
        max_weights: Upper bound on asset weights.
        transaction_costs: Linear transaction costs penalising turnover.
        management_fees: Linear management fees proportional to position size.
        l1_coef: L1 regularisation coefficient.
        l2_coef: L2 regularisation coefficient.
        risk_free_rate: Risk-free rate.
        solver: CVXPY solver name.
        solver_params: Additional solver keyword arguments.

    Note:
        The benchmark return series is NOT a config field. Pass it as ``y``
        to ``fit(X, y)`` after building the optimizer; it is a non-serialisable
        runtime object and cannot be held in a frozen dataclass.
    """

    risk_measure: RiskMeasureType = RiskMeasureType.STANDARD_DEVIATION
    prior_config: MomentEstimationConfig | None = None
    min_weights: float = 0.0
    max_weights: float = 1.0
    transaction_costs: float = 0.0
    management_fees: float = 0.0
    l1_coef: float = 0.0
    l2_coef: float = 0.0
    risk_free_rate: float = 0.0
    solver: str = "CLARABEL"
    solver_params: dict[str, object] | None = None

    @classmethod
    def for_te_target(cls, target: float = 0.01) -> BenchmarkTrackerConfig:
        """Return a config with a soft L2 tracking-error penalty.

        Uses a soft L2 penalty proportional to ``target`` to discourage
        large deviations from the benchmark.  ``BenchmarkTracker``
        minimises tracking error as its objective and exposes NO hard
        tracking-error bound parameter — for an explicit cap use
        ``MeanRiskConfig.for_tracking_error`` (which sets
        ``max_tracking_error`` and takes the benchmark as ``y`` at fit
        time), or pass custom ``linear_constraints`` via factory kwargs.

        Args:
            target: Desired tracking-error level; mapped to ``l2_coef``.

        Returns:
            A BenchmarkTrackerConfig with l2_coef set to target.
        """
        return cls(l2_coef=target)

    @classmethod
    def for_information_ratio(cls) -> BenchmarkTrackerConfig:
        """Return a config preset for information-ratio maximisation.

        Returns:
            A BenchmarkTrackerConfig with STANDARD_DEVIATION as the risk measure.
        """
        return cls(risk_measure=RiskMeasureType.STANDARD_DEVIATION)


def build_benchmark_tracker(
    config: BenchmarkTrackerConfig | None = None,
    *,
    prior_estimator: BasePrior | None = None,
    **kwargs: Any,
) -> BenchmarkTracker:
    """Build a skfolio BenchmarkTracker from config.

    The benchmark return series must be supplied as ``y`` in
    ``fit(X, y)`` after construction — the Config does NOT carry it.

    Args:
        config: Benchmark-tracker configuration. Defaults to
            BenchmarkTrackerConfig() when None.
        prior_estimator: Prior estimator. When None, one is built from
            config.prior_config (or the skfolio default).
        **kwargs: Additional keyword arguments forwarded to BenchmarkTracker.

    Returns:
        A fitted-ready skfolio optimiser. Call
        ``estimator.fit(X, y=benchmark_returns)`` to fit it.
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
        transaction_costs=config.transaction_costs,
        management_fees=config.management_fees,
        l1_coef=config.l1_coef,
        l2_coef=config.l2_coef,
        risk_free_rate=config.risk_free_rate,
        solver=config.solver,
        solver_params=config.solver_params,
        **kwargs,
    )


__all__ = ["BenchmarkTrackerConfig", "build_benchmark_tracker"]
