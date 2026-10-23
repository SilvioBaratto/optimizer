"""Factory functions for building skfolio optimization estimators."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from skfolio.measures import RiskMeasure
from skfolio.optimization import MeanRisk
from skfolio.optimization.convex._base import ObjectiveFunction
from skfolio.prior._base import BasePrior

from optimizer.factors._integration import FactorExposureConstraints
from optimizer.moments._factory import build_prior
from optimizer.optimization._config import (
    FallbackPolicy,
    MeanRiskConfig,
    ObjectiveFunctionType,
    RiskMeasureType,
)

logger = logging.getLogger(__name__)

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


def build_sector_constraints(
    sector_mapping: dict[str, str],
    max_sector_weight: float,
) -> tuple[dict[str, str], list[str]]:
    """Build skfolio ``groups`` and ``linear_constraints`` from a sector mapping.

    Args:
        sector_mapping: Mapping from asset ticker to sector name
            (e.g. ``{"AAPL": "Technology", "JPM": "Financials"}``).
        max_sector_weight: Maximum total weight for any single sector
            (e.g. 0.25 = 25%).

    Returns:
        A tuple of (groups, linear_constraints) where groups is the
        ticker->sector label dict required by `MeanRisk` (converted to a
        2-D group array at fit-time via ``input_to_array``), and
        linear_constraints
        is one constraint string per sector (``"SectorName <= cap"``).

    Note:
        Tickers whose sector is missing (``None``, ``NaN``, ``pd.NA`` or empty
        string) are excluded from constraint emission.
        ``TickerProfile.sector`` is nullable in the source DB (ETFs and some
        equities carry no GICS sector), so a DB-sourced mapping can
        legitimately contain such values.  skfolio maps an asset that is
        ``None`` / absent in ``groups`` to its own singleton group, leaving it
        unconstrained by sector — the correct behaviour for an unknown-sector
        asset.  Emitting a ``"None <= cap"`` row (or sorting a set that mixes
        ``str`` and ``None``/``NaN``) would instead raise ``TypeError`` or
        attach a bogus constraint, so those values are skipped here.  pandas
        turns a NULL into ``NaN`` in string columns, which is truthy, so only
        a non-empty ``str`` counts as a sector label.
    """
    sectors = sorted(
        {
            sector
            for sector in sector_mapping.values()
            if isinstance(sector, str) and sector
        }
    )
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


def _build_sector_cap_constraints(
    sector_bands: dict[str, tuple[float, float]],
) -> list[str]:
    """Emit ``"<sector> >= <floor>"`` and ``"<sector> <= <cap>"`` rows.

    Used by `build_mean_risk` when ``sector_bands`` is provided
    directly (factory-level injection without going through the orchestrator).

    Args:
        sector_bands: Mapping from sector name to ``(floor, cap)``. Sectors
            with ``floor == 0.0`` emit no floor row.

    Returns:
        Constraint rows sorted by sector name then constraint type
        (floor rows precede cap rows).
    """
    rows: list[str] = []
    for sector in sorted(sector_bands):
        floor, cap = sector_bands[sector]
        if floor > 0.0:
            rows.append(f"{sector} >= {round(floor, 6)}")
    for sector in sorted(sector_bands):
        _, cap = sector_bands[sector]
        rows.append(f"{sector} <= {round(cap, 6)}")
    return rows


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


def _is_sector_label(value: object) -> bool:
    """Return whether ``value`` names a sector (non-empty ``str``)."""
    return isinstance(value, str) and bool(value)


def _resolve_enforceable_bands(
    sector_bands: dict[str, tuple[float, float]],
    groups: object,
    budget: float | None,
) -> dict[str, tuple[float, float]]:
    """Validate regime bands against the sectors named by ``groups``.

    skfolio drops, with only a ``UserWarning``, every constraint row naming a
    group absent from ``groups``.  Rows that survive are checked here for the
    errors that hold whatever the asset universe is.  Cap/budget feasibility
    is only warned about: the universe is not known at build time, and an
    asset missing from the mapping is a singleton group in skfolio, so it is
    unconstrained and may absorb the weight the caps leave over.

    Args:
        sector_bands: Mapping from sector name to ``(floor, cap)``.
        groups: Asset -> sector mapping that skfolio will receive.  Presence
            and cap-sum checks only run for a mapping; any other form
            (e.g. an array) is passed through after the floor/cap checks.
        budget: Portfolio budget, or ``None`` when unconstrained.

    Returns:
        The subset of ``sector_bands`` whose sector occurs in ``groups``.

    Raises:
        ValueError: If a floor exceeds its cap, if no band sector occurs in
            ``groups`` (typically a naming mismatch), or if the floors of the
            present sectors sum above 1.0.
    """
    for sector, (floor, cap) in sector_bands.items():
        if floor > cap + 1e-9:
            raise ValueError(
                f"sector_bands[{sector!r}] floor {floor:.4f} exceeds cap {cap:.4f}."
            )
    if not isinstance(groups, Mapping):
        return dict(sector_bands)

    present = {label for label in groups.values() if _is_sector_label(label)}
    absent = sorted(set(sector_bands) - present)
    if len(absent) == len(sector_bands):
        raise ValueError(
            f"No sector_bands sector {absent} occurs in the sector mapping "
            f"(sectors present: {sorted(present)}); check for a naming mismatch."
        )
    if absent:
        logger.warning(
            "sector_bands for sectors absent from the universe are skipped: %s.",
            absent,
        )
    active = {s: b for s, b in sector_bands.items() if s in present}

    floor_sum = sum(floor for floor, _ in active.values())
    if floor_sum > 1.0 + 1e-9:
        raise ValueError(
            f"sector_bands floors of the present sectors sum to {floor_sum:.4f} "
            "> 1.0; floors are infeasible."
        )

    # An asset without a sector label, or in a sector without a band, is
    # unconstrained and can absorb the remaining weight.
    fully_capped = present <= active.keys() and all(
        _is_sector_label(label) for label in groups.values()
    )
    if budget is not None and fully_capped:
        cap_sum = sum(cap for _, cap in active.values())
        if cap_sum < budget - 1e-9:
            logger.warning(
                "sector_bands caps of the sectors in the mapping sum to %.4f < "
                "budget %.4f; the fit is infeasible unless assets missing from "
                "the mapping absorb the remainder.",
                cap_sum,
                budget,
            )
    return active


def build_mean_risk(
    config: MeanRiskConfig | None = None,
    *,
    prior_estimator: BasePrior | None = None,
    factor_exposure_constraints: FactorExposureConstraints | None = None,
    sector_mapping: dict[str, str] | None = None,
    min_sector_weights: dict[str, float] | None = None,
    sector_bands: dict[str, tuple[float, float]] | None = None,
    **kwargs: Any,
) -> MeanRisk:
    """Build a skfolio `MeanRisk` optimiser from `config`.

    Args:
        config: Mean-risk configuration. Defaults to ``MeanRiskConfig()``
            (minimum-variance).
        prior_estimator: Prior estimator. When ``None``, one is built from
            ``config.prior_config`` (or skfolio default).
        factor_exposure_constraints: Enforceable factor exposure constraints
            produced by `build_factor_exposure_constraints`. When provided,
            ``left_inequality`` and ``right_inequality`` are injected into
            the `MeanRisk` constructor. Any explicit
            ``left_inequality`` / ``right_inequality`` entries in ``kwargs``
            take precedence.
        sector_mapping: Ticker -> sector name mapping. Used to build
            ``groups`` and ``linear_constraints`` when
            ``config.max_sector_weight`` is set or when
            ``min_sector_weights`` is supplied. Any explicit ``groups`` or
            ``linear_constraints`` in ``kwargs`` take precedence.
        min_sector_weights: Per-sector floor weights
            (e.g. ``{"Healthcare": 0.08}``). Each entry emits a
            ``"<sector> >= <floor>"`` row appended to the sector-cap rows.
            Requires ``sector_mapping``; raises ``ValueError`` if not
            supplied, if floors sum to > 1.0, or if any floor exceeds
            ``config.max_sector_weight``. Empty dict is a no-op. The input
            dict is not mutated.
        sector_bands: Regime-conditional per-sector ``(floor, cap)`` bands
            from `resolve_sector_bands`. When provided, both floor rows
            (``"<sector> >= <floor>"``) and cap rows
            (``"<sector> <= <cap>"``) are appended to
            ``linear_constraints`` after any ``min_sector_weights`` rows.
            Sectors with ``floor == 0.0`` emit no floor row. This is the
            low-level factory path for injecting regime-conditional sector
            bands into a `MeanRisk` optimizer. Requires ``sector_mapping``
            (or an explicit ``groups`` in ``kwargs``); raises ``ValueError``
            otherwise. Bands for sectors absent from the mapping are skipped
            with a warning. ``ValueError`` is raised when no band sector
            occurs in the mapping, when a floor exceeds its cap, or when the
            floors sum to > 1.0. Caps summing below ``config.budget`` over a
            fully capped mapping only log a warning, since an asset missing
            from the mapping is unconstrained and may absorb the remainder.
            ``None`` is a no-op.
        **kwargs: Additional keyword arguments forwarded to the
            `MeanRisk` constructor (for non-serialisable parameters
            such as ``previous_weights``, ``groups``,
            ``linear_constraints``, etc.). Notable skfolio 1.0
            passthroughs: ``target_weights`` — L1/L2 regularisation anchor
            applied to ``w - target_weights``, distinct from
            ``max_tracking_error``; ``fallback`` — overrides
            ``config.fallback_policy``; ``raise_on_failure`` — overrides
            ``config.raise_on_failure``.

    Returns:
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

    if sector_bands:
        # skfolio drops (with only a UserWarning) every constraint row naming a
        # group it cannot resolve, so bands without groups would be silently
        # unenforced.
        if sector_mapping is None and kwargs.get("groups") is None:
            raise ValueError(
                "sector_bands requires sector_mapping or an explicit groups "
                "kwarg to enforce sector constraints (no group mapping "
                "available)."
            )
        kwargs.setdefault("groups", sector_mapping)
        enforceable_bands = _resolve_enforceable_bands(
            sector_bands, kwargs["groups"], config.budget
        )
        sector_constraints.extend(_build_sector_cap_constraints(enforceable_bands))

    if sector_constraints:
        kwargs.setdefault("linear_constraints", sector_constraints)

    # Resilience layer (skfolio 1.0). Explicit kwargs win over config fields.
    kwargs.setdefault("raise_on_failure", config.raise_on_failure)
    if config.fallback_policy is FallbackPolicy.PREVIOUS_WEIGHTS:
        kwargs.setdefault("fallback", "previous_weights")

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
