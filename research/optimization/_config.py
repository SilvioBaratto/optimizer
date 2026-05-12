"""Cycle-3 §7 optimizer config construction and builder.

Encapsulates:

- ``_REGION_MAP`` — single source of truth for country → region.
- ``_SECTOR_FLOORS`` — Cycle-3 §7.1 sector minimums.
- ``_make_opt_config`` — builds the hard-constrained ``MeanRiskConfig``
  (with walk-forward feasibility fallback for thin universes).
- ``build_research_optimizer`` — composes config + sector floors + region
  caps + ``previous_weights`` into a fitted-ready ``MeanRisk``.
- ``_select_optimizer`` — dispatch to plain or robust ``MeanRisk``.
- ``_make_builder`` — closure factory for retighten loop compatibility.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from optimizer.optimization import (
    MeanRiskConfig,
    ObjectiveFunctionType,
    RiskMeasureType,
    RobustMeanRiskConfig,
    build_mean_risk,
    build_region_linear_constraints,
    build_robust_mean_risk,
)

if TYPE_CHECKING:
    from skfolio.optimization import MeanRisk

logger = logging.getLogger(__name__)

_REGION_MAP: dict[str, str] = {
    "United States": "Americas",
    "Canada": "Americas",
    "Colombia": "Americas",
    "Argentina": "Americas",
    "Brazil": "Americas",
    "Mexico": "Americas",
    "Chile": "Americas",
    "United Kingdom": "Europe",
    "France": "Europe",
    "Germany": "Europe",
    "Netherlands": "Europe",
    "Switzerland": "Europe",
    "Ireland": "Europe",
    "Italy": "Europe",
    "Spain": "Europe",
    "Monaco": "Europe",
    "Luxembourg": "Europe",
    "Belgium": "Europe",
    "Norway": "Europe",
    "Sweden": "Europe",
    "Denmark": "Europe",
    "Finland": "Europe",
    "Austria": "Europe",
    "Portugal": "Europe",
    "China": "Asia-Pacific",
    "Taiwan": "Asia-Pacific",
    "Japan": "Asia-Pacific",
    "South Korea": "Asia-Pacific",
    "Indonesia": "Asia-Pacific",
    "India": "Asia-Pacific",
    "Singapore": "Asia-Pacific",
    "Australia": "Asia-Pacific",
    "Hong Kong": "Asia-Pacific",
    "Israel": "Middle East & Africa",
    "Turkey": "Middle East & Africa",
    "South Africa": "Middle East & Africa",
}

_SECTOR_FLOORS: dict[str, float] = {"Healthcare": 0.08, "Technology": 0.10}

_MAX_REGION_WEIGHT = 0.60
_SOLVER_PARAMS: dict[str, Any] = {
    "max_iter": 200_000,
    "eps_abs": 1e-8,
    "eps_rel": 1e-8,
}


def _resolve_min_weights(n_survivors: int, target_count: int) -> float:
    """Walk-forward feasibility fallback for thin universes.

    When survivors are below the target, the spec ``min_weights=0.02``
    becomes infeasible.  Fall back to ``1/(2N)`` and emit a warning.
    """
    if n_survivors >= target_count:
        return 0.02
    fallback = 1.0 / (2 * n_survivors) if n_survivors > 0 else 0.02
    logger.warning(
        "Feasibility fallback: %d survivors < target %d; min_weights=%.4f "
        "(=1/(2*N))",
        n_survivors,
        target_count,
        fallback,
    )
    return fallback


def _make_opt_config(
    n_survivors: int,
    target_count: int,
    cost_bps: float,
) -> MeanRiskConfig:
    """Build the Cycle-3 §7.1 hard-constrained ``MeanRiskConfig``."""
    return MeanRiskConfig(
        objective=ObjectiveFunctionType.MAXIMIZE_RATIO,
        risk_measure=RiskMeasureType.VARIANCE,
        min_weights=_resolve_min_weights(n_survivors, target_count),
        max_weights=0.10,
        max_sector_weight=0.15,
        budget=1.0,
        l1_coef=0.0,
        l2_coef=0.05,
        transaction_costs=cost_bps / 1e4,
        solver="CLARABEL",
        solver_params=_SOLVER_PARAMS,
    )


def _select_optimizer(
    mean_risk_config: MeanRiskConfig,
    *,
    robust: bool,
    uncertainty_level: float,
    sector_mapping: dict[str, str],
    min_sector_weights: dict[str, float] | None = None,
    linear_constraints: list[str] | None = None,
    previous_weights: np.ndarray | None = None,
) -> MeanRisk:
    """Cycle-3 §7.4 selector: dispatch to plain or robust ``MeanRisk``.

    When ``robust`` is ``False``, returns ``build_mean_risk(...)`` output
    (plain Cycle-3 §7.1 hard-constrained estimator).  When ``robust`` is
    ``True``, returns ``build_robust_mean_risk(RobustMeanRiskConfig(
    mean_risk_config=..., mu_uncertainty_set_config=for_moderate(...)))``
    so the underlying hard-constrained config travels through unchanged
    while the moderate-confidence ellipsoidal mu uncertainty set is
    attached.

    Both branches forward ``min_sector_weights``, ``linear_constraints``,
    and ``previous_weights`` identically; the caller is responsible for
    pre-merging sector caps + region rows when supplying
    ``linear_constraints``.
    """
    extra: dict[str, Any] = {"sector_mapping": sector_mapping}
    if min_sector_weights is not None:
        extra["min_sector_weights"] = min_sector_weights
    if linear_constraints is not None:
        extra["linear_constraints"] = linear_constraints
    if previous_weights is not None:
        extra["previous_weights"] = previous_weights

    if not robust:
        return build_mean_risk(mean_risk_config, **extra)

    robust_cfg = dataclasses.replace(
        RobustMeanRiskConfig.for_moderate(uncertainty_level=uncertainty_level),
        mean_risk_config=mean_risk_config,
    )
    return build_robust_mean_risk(robust_cfg, **extra)


def _make_builder(
    *,
    sector_mapping: dict[str, str],
    country_map: dict[str, str] | None,
    previous_weights: np.ndarray | None,
    robust: bool = False,
    uncertainty_level: float = 0.95,
) -> Callable[[MeanRiskConfig], MeanRisk]:
    """Return a builder closure that materialises a ``MeanRisk`` per config.

    Snapshots all mutable side-inputs at closure creation time so that
    subsequent caller mutations are invisible to the retighten loop.
    """
    _prev_w = previous_weights.copy() if previous_weights is not None else None
    _sector_mapping = dict(sector_mapping)
    _country_map = dict(country_map) if country_map is not None else None

    def builder(config: MeanRiskConfig) -> MeanRisk:
        opt = _select_optimizer(
            config,
            robust=robust,
            uncertainty_level=uncertainty_level,
            sector_mapping=_sector_mapping,
            min_sector_weights=_SECTOR_FLOORS,
            previous_weights=_prev_w,
        )
        if _country_map:
            _, region_rows = build_region_linear_constraints(
                _country_map, _REGION_MAP, max_region_weight=_MAX_REGION_WEIGHT
            )
            existing = list(opt.linear_constraints or [])
            opt.linear_constraints = existing + region_rows
        return opt

    return builder


def build_research_optimizer(
    *,
    sector_mapping: dict[str, str],
    country_map: dict[str, str] | None,
    n_survivors: int,
    target_count: int,
    cost_bps: float,
    previous_weights: np.ndarray | None = None,
) -> MeanRisk:
    """Compose §7.1 config + sector floors + region caps into a ``MeanRisk``."""
    config = _make_opt_config(n_survivors, target_count, cost_bps)
    builder = _make_builder(
        sector_mapping=sector_mapping,
        country_map=country_map,
        previous_weights=previous_weights,
    )
    return builder(config)
