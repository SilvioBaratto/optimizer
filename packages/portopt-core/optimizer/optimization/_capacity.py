"""Fit-time check that a per-asset weight cap can reach the budget.

``max_weights=0.10`` with a fully invested budget needs at least ten assets. With fewer
the weights cannot sum to the budget, the problem is infeasible, and the solver reports
a generic convex-optimisation failure only after the fit has run. The estimators here
extend the skfolio classes with one warning that names the cap, the asset count and the
largest reachable total, and change nothing else about how they fit, clone or fail.
"""

from __future__ import annotations

import math
import numbers
import warnings
from typing import Any

from skfolio.optimization import MaximumDiversification, MeanRisk

# Weights are floats, so a total a hair under the budget is rounding, not a shortfall.
_TOLERANCE = 1e-9


def warn_if_cap_cannot_fill_budget(estimator: Any, X: Any) -> None:
    """Warn when ``n_assets * max_weights`` falls short of the budget.

    Only a scalar cap and a numeric budget are comparable; a per-asset cap, no cap, or
    no budget leaves nothing to check. The budget is an equality constraint, so no
    choice of floors or shorts can lift the total above ``n_assets * max_weights``.

    Args:
        estimator: An optimiser exposing ``max_weights`` and ``budget``.
        X: The returns being fitted; its column count is the asset count.
    """
    cap = getattr(estimator, "max_weights", None)
    budget = getattr(estimator, "budget", None)
    shape: tuple[int, ...] = getattr(X, "shape", ())
    if len(shape) != 2:
        return
    if isinstance(cap, bool) or not isinstance(cap, numbers.Real) or cap <= 0:
        return
    if isinstance(budget, bool) or not isinstance(budget, numbers.Real):
        return
    n_assets = int(shape[1])
    reachable = n_assets * float(cap)
    if reachable >= float(budget) - _TOLERANCE:
        return
    needed = math.ceil(float(budget) / float(cap) - _TOLERANCE)
    warnings.warn(
        f"max_weights={float(cap):g} across {n_assets} assets cannot reach the budget "
        f"of {float(budget):g}: the weights sum to at most {reachable:g}, so the fit "
        f"is infeasible. Use at least {needed} assets or raise max_weights.",
        UserWarning,
        stacklevel=3,
    )


class CapacityCheckedMeanRisk(MeanRisk):  # type: ignore[misc]
    """`MeanRisk` that warns when its weight cap cannot reach the budget."""

    def fit(self, X: Any, y: Any = None, **fit_params: Any) -> Any:
        """Warn about an unreachable budget, then fit exactly as `MeanRisk` does."""
        warn_if_cap_cannot_fill_budget(self, X)
        return super().fit(X, y, **fit_params)


class CapacityCheckedMaximumDiversification(MaximumDiversification):  # type: ignore[misc]
    """`MaximumDiversification` that warns when its cap cannot reach the budget."""

    def fit(self, X: Any, y: Any = None, **fit_params: Any) -> Any:
        """Warn about an unreachable budget, then fit exactly as the base class does."""
        warn_if_cap_cannot_fill_budget(self, X)
        return super().fit(X, y, **fit_params)
