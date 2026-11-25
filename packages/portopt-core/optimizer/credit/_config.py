"""Configuration for the Merton structural credit model."""

from __future__ import annotations

import math
from dataclasses import dataclass

from optimizer.exceptions import ConfigurationError


def require_positive_finite(name: str, value: float, error: type[Exception]) -> None:
    """Raise ``error`` unless ``value`` is a finite number above zero.

    Args:
        name: Parameter name quoted in the message.
        value: Number to check.
        error: Exception class to raise, so a bad setting and a bad input can use
            different classes.

    Raises:
        Exception: ``error``, when ``value`` is zero, negative, infinite or NaN.
    """
    if not (math.isfinite(value) and value > 0.0):
        msg = f"{name} must be a finite number above zero, got {value}"
        raise error(msg)


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
        require_positive_finite("maturity", self.maturity, ConfigurationError)
        require_positive_finite(
            "solver_tolerance", self.solver_tolerance, ConfigurationError
        )
        if self.max_iterations < 1:
            msg = f"max_iterations must be at least 1, got {self.max_iterations}"
            raise ConfigurationError(msg)
        if self.asset_drift is not None and not math.isfinite(self.asset_drift):
            msg = f"asset_drift must be finite or None, got {self.asset_drift}"
            raise ConfigurationError(msg)
