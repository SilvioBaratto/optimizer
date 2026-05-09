"""RiskBudgeting configuration and factory.

Wraps :class:`skfolio.optimization.RiskBudgeting` behind a frozen
:class:`RiskBudgetingConfig`. ERC (Equal Risk Contribution) is the
default; custom risk budgets are supplied as a dict mapping ticker to
target risk share (must sum to 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from skfolio.optimization import RiskBudgeting
from skfolio.prior._base import BasePrior

from optimizer.exceptions import ConfigurationError
from optimizer.moments._config import MomentEstimationConfig
from optimizer.moments._factory import build_prior

_BUDGET_SUM_ATOL = 1e-6

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskBudgetingConfig:
    """Immutable configuration for :class:`skfolio.optimization.RiskBudgeting`.

    Serialisable parameters only. Non-serialisable objects
    (``prior_estimator``, ``previous_weights``, ``groups``, etc.) are
    passed as keyword arguments to :func:`build_risk_budgeting`.

    Parameters
    ----------
    budgets : tuple[tuple[str, float], ...] or None
        Custom risk budgets ``((ticker, share), ...)``. ``None`` selects
        the equal-risk-contribution (ERC) special case (skfolio default).
        Must sum to 1 within ``1e-6``.
    prior_config : MomentEstimationConfig or None
        Inner prior configuration. ``None`` defers to skfolio default
        (``EmpiricalPrior``).
    min_weights : float
        Lower bound on asset weights.
    max_weights : float
        Upper bound on asset weights.
    transaction_costs : float
        Linear transaction costs penalising turnover.
    management_fees : float
        Linear management fees proportional to position size.
    solver : str
        CVXPY solver name.
    solver_params : dict or None
        Additional solver parameters.
    """

    budgets: tuple[tuple[str, float], ...] | None = None
    prior_config: MomentEstimationConfig | None = None
    min_weights: float = 0.0
    max_weights: float = 1.0
    transaction_costs: float = 0.0
    management_fees: float = 0.0
    solver: str = "CLARABEL"
    solver_params: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if self.budgets is not None:
            total = sum(value for _, value in self.budgets)
            if not np.isclose(total, 1.0, atol=_BUDGET_SUM_ATOL):
                raise ConfigurationError(
                    f"budgets must sum to 1 (got {total:.6f})"
                )

    @classmethod
    def for_equal_risk_contribution(cls) -> RiskBudgetingConfig:
        """Equal risk contribution (ERC) preset — skfolio default."""
        return cls()

    @classmethod
    def for_custom_budgets(
        cls,
        budgets: dict[str, float],
    ) -> RiskBudgetingConfig:
        """Custom risk-budgets preset from a dict mapping ticker to share."""
        return cls(budgets=tuple(budgets.items()))


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_risk_budgeting(
    config: RiskBudgetingConfig | None = None,
    *,
    prior_estimator: BasePrior | None = None,
    **kwargs: Any,
) -> RiskBudgeting:
    """Build a skfolio :class:`RiskBudgeting` optimiser from *config*.

    Parameters
    ----------
    config : RiskBudgetingConfig or None
        Risk-budgeting configuration. ``None`` triggers the ERC default.
    prior_estimator : BasePrior or None
        Prior estimator. When ``None``, one is built from
        ``config.prior_config`` (or skfolio default).
    **kwargs
        Additional keyword arguments forwarded to the
        :class:`RiskBudgeting` constructor (for non-serialisable
        parameters such as ``previous_weights``, ``groups``,
        ``linear_constraints``, etc.).

    Returns
    -------
    RiskBudgeting
        A fitted-ready skfolio optimiser.
    """
    if config is None:
        config = RiskBudgetingConfig()

    if prior_estimator is None and config.prior_config is not None:
        prior_estimator = build_prior(config.prior_config)

    risk_budget = (
        np.array([value for _, value in config.budgets])
        if config.budgets is not None
        else None
    )

    return RiskBudgeting(
        risk_budget=risk_budget,
        prior_estimator=prior_estimator,
        min_weights=config.min_weights,
        max_weights=config.max_weights,
        transaction_costs=config.transaction_costs,
        management_fees=config.management_fees,
        solver=config.solver,
        solver_params=config.solver_params,
        **kwargs,
    )
