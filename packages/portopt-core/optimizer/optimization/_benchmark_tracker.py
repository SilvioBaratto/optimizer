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
        transaction_costs: Linear transaction costs relative to
            ``previous_weights``.  Applied only to the cost accounting of the
            resulting Portfolio (net returns); it does not change the
            optimised weights, so it cannot damp rebalancing.  Use
            ``MeanRiskConfig`` for turnover-aware optimisation.
        management_fees: Linear management fees proportional to position size.
            Applied only to the cost accounting of the resulting Portfolio; it
            does not change the optimised weights.
        l1_coef: L1 regularisation coefficient.
        l2_coef: L2 regularisation coefficient.  Applied to the absolute
            weights, so it shrinks toward equal weight rather than toward the
            benchmark and typically widens tracking error; it is not a
            tracking-error target.
        risk_free_rate: Risk-free rate.
        solver: CVXPY solver name.
        solver_params: Additional solver keyword arguments.

    Note:
        The benchmark return series is NOT a config field. Pass it as ``y``
        to ``fit(X, y)`` after building the optimizer; it is a non-serialisable
        runtime object and cannot be held in a frozen dataclass.

        ``BenchmarkTracker`` minimises tracking error and exposes no hard
        tracking-error bound.  For an explicit cap use
        ``MeanRiskConfig.for_tracking_error``, which takes the benchmark as
        ``y`` at fit time.

        It has no expected-return term, so it cannot maximise the information
        ratio.  For an active, alpha-seeking portfolio use ``MeanRisk`` with
        ``MAXIMIZE_RATIO`` on excess returns.
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
