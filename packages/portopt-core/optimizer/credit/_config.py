"""Configuration for the Merton structural credit model."""

from __future__ import annotations

from dataclasses import dataclass

from optimizer.credit._validation import (
    require_finite,
    require_integer_at_least,
    require_positive_finite,
)


@dataclass(frozen=True)
class MertonConfig:
    """Immutable configuration for the Merton structural credit model.

    Attributes:
        maturity: Years until the zero-coupon debt falls due; the horizon of the
            option on the firm's assets.
        solver_tolerance: Largest acceptable Euclidean norm of the two relative
            residuals (equity value and equity volatility) at the solution. A
            solve that cannot reach it raises ``ConvergenceError``. Values below
            about 1e-14 are beyond double precision for most inputs.
        max_iterations: Budget of residual evaluations the solver may spend. The
            finite-difference evaluations of its Jacobian are not counted, so the
            figure is close to the number of iterations.
        asset_drift: Annual expected growth rate of the firm's assets, used only
            for the real-world default probability. ``None`` uses the risk-free
            rate, which gives the risk-neutral probability.
    """

    maturity: float = 1.0
    solver_tolerance: float = 1e-10
    max_iterations: int = 200
    asset_drift: float | None = None

    def __post_init__(self) -> None:
        """Reject settings the model cannot use."""
        require_positive_finite("maturity", self.maturity)
        require_positive_finite("solver_tolerance", self.solver_tolerance)
        require_integer_at_least("max_iterations", self.max_iterations, 1)
        if self.asset_drift is not None:
            require_finite("asset_drift", self.asset_drift)
