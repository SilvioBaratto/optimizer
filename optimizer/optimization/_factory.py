"""Factory functions for building skfolio optimization estimators."""

from __future__ import annotations

import logging
from typing import Any

from skfolio.measures import RiskMeasure
from skfolio.optimization import MeanRisk
from skfolio.optimization.convex._base import ObjectiveFunction
from skfolio.prior._base import BasePrior

from optimizer.factors._integration import FactorExposureConstraints
from optimizer.moments._factory import build_prior
from optimizer.optimization._config import (
    MeanRiskConfig,
    ObjectiveFunctionType,
    RiskMeasureType,
)

logger = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# Helper factory
# ---------------------------------------------------------------------------


def build_sector_constraints(
    sector_mapping: dict[str, str],
    max_sector_weight: float,
) -> tuple[dict[str, str], list[str]]:
    """Build skfolio ``groups`` and ``linear_constraints`` from a sector mapping.

    Parameters
    ----------
    sector_mapping : dict[str, str]
        Mapping from asset ticker to sector name
        (e.g. ``{"AAPL": "Technology", "JPM": "Financials"}``).
    max_sector_weight : float
        Maximum total weight for any single sector (e.g. 0.25 = 25%).

    Returns
    -------
    groups : dict[str, str]
        Ticker -> sector label, as required by
        :class:`skfolio.optimization.MeanRisk` (converted to a 2-D
        group array at fit-time via ``input_to_array``).
    linear_constraints : list[str]
        One constraint string per sector (``"SectorName <= cap"``).
    """
    sectors = sorted(set(sector_mapping.values()))
    cap = round(max_sector_weight, 6)
    linear_constraints = [f"{sector} <= {cap}" for sector in sectors]

    return sector_mapping, linear_constraints


def _build_sector_floor_constraints(
    min_sector_weights: dict[str, float],
) -> list[str]:
    """Emit ``"<sector> >= <floor>"`` rows from a floor mapping (sorted)."""
    return [
        f"{sector} >= {round(floor, 6)}"
        for sector, floor in sorted(min_sector_weights.items())
    ]


def _validate_sector_floors(
    min_sector_weights: dict[str, float],
    sector_mapping: dict[str, str] | None,
    max_sector_weight: float | None,
) -> None:
    """Reject infeasible floors at build time.

    Raises ``ValueError`` when ``sector_mapping`` is missing, when floors sum
    above 1.0, or when any floor exceeds ``max_sector_weight``.
    """
    if sector_mapping is None:
        raise ValueError(
            "min_sector_weights requires sector_mapping to enforce sector "
            "floor constraints (no group mapping available)."
        )
    floor_sum = sum(min_sector_weights.values())
    if floor_sum > 1.0 + 1e-9:
        raise ValueError(
            f"min_sector_weights sum to {floor_sum:.4f} > 1.0; floors are infeasible."
        )
    if max_sector_weight is not None:
        for sector, floor in min_sector_weights.items():
            if floor > max_sector_weight + 1e-9:
                raise ValueError(
                    f"min_sector_weights[{sector!r}]={floor:.4f} exceeds "
                    f"max_sector_weight={max_sector_weight:.4f}."
                )


# ---------------------------------------------------------------------------
# Main optimiser factory
# ---------------------------------------------------------------------------


def build_mean_risk(
    config: MeanRiskConfig | None = None,
    *,
    prior_estimator: BasePrior | None = None,
    factor_exposure_constraints: FactorExposureConstraints | None = None,
    sector_mapping: dict[str, str] | None = None,
    min_sector_weights: dict[str, float] | None = None,
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
    sector_mapping : dict[str, str] or None
        Ticker -> sector name mapping.  Used to build ``groups`` and
        ``linear_constraints`` when ``config.max_sector_weight`` is set
        or when ``min_sector_weights`` is supplied.  Any explicit
        ``groups`` or ``linear_constraints`` in ``kwargs`` take
        precedence.
    min_sector_weights : dict[str, float] or None
        Per-sector floor weights (e.g. ``{"Healthcare": 0.08}``).  Each
        entry emits a ``"<sector> >= <floor>"`` row appended to the
        sector-cap rows.  Requires ``sector_mapping``; raises
        ``ValueError`` if not supplied, if floors sum to > 1.0, or if
        any floor exceeds ``config.max_sector_weight``.  Empty dict is
        a no-op.  The input dict is not mutated.
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

    sector_constraints: list[str] = []
    if config.max_sector_weight is not None:
        if sector_mapping is not None:
            built_groups, built_caps = build_sector_constraints(
                sector_mapping, config.max_sector_weight
            )
            kwargs.setdefault("groups", built_groups)
            sector_constraints.extend(built_caps)
        else:
            logger.warning(
                "MeanRiskConfig.max_sector_weight=%.4f is set but no "
                "sector_mapping was provided; sector constraints are skipped.",
                config.max_sector_weight,
            )

    if min_sector_weights:
        _validate_sector_floors(
            min_sector_weights, sector_mapping, config.max_sector_weight
        )
        kwargs.setdefault("groups", sector_mapping)
        sector_constraints.extend(_build_sector_floor_constraints(min_sector_weights))

    if sector_constraints:
        kwargs.setdefault("linear_constraints", sector_constraints)

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
