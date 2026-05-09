"""Cycle-3 §7 helpers for the research pipeline.

Encapsulates:

- ``_REGION_MAP`` — single source of truth for country → region.
- ``_SECTOR_FLOORS`` — Cycle-3 §7.1 sector minimums.
- ``_make_opt_config`` — builds the hard-constrained ``MeanRiskConfig``
  (with walk-forward feasibility fallback for thin universes).
- ``build_research_optimizer`` — composes config + sector floors + region
  caps + ``previous_weights`` into a fitted-ready ``MeanRisk``.
- ``_solve_with_retighten`` — Cycle-3 §7.3 Top-4 retighten loop.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from optimizer.optimization import (
    MeanRiskConfig,
    ObjectiveFunctionType,
    RiskMeasureType,
    RobustMeanRiskConfig,
    build_mean_risk,
    build_region_linear_constraints,
    build_robust_mean_risk,
)
from optimizer.rebalancing import HybridRebalancingConfig, should_rebalance_hybrid

if TYPE_CHECKING:
    from skfolio.optimization import MeanRisk

logger = logging.getLogger(__name__)

_HOCKEY_STICK_NEG_THRESHOLD = 0.0
_HOCKEY_STICK_POS_THRESHOLD = 1.5
_TRADING_DAYS_PER_YEAR = 252


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

# Cycle-3 §7.3 Top-4 retighten constants
_TOP_N = 4
_TOP4_THRESHOLD = 0.30
_SHRINK_FACTOR = 0.95


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

    Captures the immutable side-inputs (sector mapping, country mapping,
    previous weights, robust toggle) so the retighten loop can rebuild
    the estimator from a shrunken config without re-threading them.
    """

    def builder(config: MeanRiskConfig) -> MeanRisk:
        opt = _select_optimizer(
            config,
            robust=robust,
            uncertainty_level=uncertainty_level,
            sector_mapping=sector_mapping,
            min_sector_weights=_SECTOR_FLOORS,
            previous_weights=previous_weights,
        )
        if country_map:
            _, region_rows = build_region_linear_constraints(
                country_map, _REGION_MAP, max_region_weight=_MAX_REGION_WEIGHT
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


def _top4_weight(weights: np.ndarray) -> float:
    """Sum of the top-4 weights (or all if fewer assets)."""
    return float(np.sum(np.sort(weights)[-_TOP_N:]))


def _retighten_error_message(final_entry: dict[str, Any], max_retries: int) -> str:
    return (
        f"Top-4 retighten loop failed: top4={final_entry['top4']:.4f} "
        f">= {_TOP4_THRESHOLD} after {max_retries} retries "
        f"(final max_weights={final_entry['max_weights']:.6f})"
    )


def _solve_with_retighten(
    optimizer: MeanRisk,
    returns: pd.DataFrame,
    *,
    config: MeanRiskConfig,
    builder: Callable[[MeanRiskConfig], MeanRisk],
    max_retries: int = 5,
) -> tuple[MeanRisk, list[dict[str, Any]]]:
    """Cycle-3 §7.3 Top-4 retighten loop.

    Fit the optimiser; if ``sum(sorted(weights)[-4:]) >= 0.30``, shrink
    ``max_weights`` by ``0.95`` and rebuild + refit.  Cap at
    ``max_retries`` retries (initial fit + ``max_retries`` rebuilds).
    Raise ``RuntimeError`` when the loop fails to converge.
    """
    trace: list[dict[str, Any]] = []
    current = optimizer
    current_config = config
    attempt = 1
    while True:
        current.fit(returns)
        top4 = _top4_weight(np.asarray(current.weights_))
        trace.append(
            {
                "attempt": attempt,
                "top4": top4,
                "max_weights": cast(float, current_config.max_weights),
            }
        )
        logger.debug(
            "retighten attempt=%d top4=%.4f max_weights=%.4f",
            attempt,
            top4,
            cast(float, current_config.max_weights),
        )
        if top4 < _TOP4_THRESHOLD:
            return current, trace
        if attempt > max_retries:
            raise RuntimeError(_retighten_error_message(trace[-1], max_retries))
        new_max = cast(float, current_config.max_weights) * _SHRINK_FACTOR
        current_config = dataclasses.replace(current_config, max_weights=new_max)
        current = builder(current_config)
        attempt += 1


def _annualized_sharpe(returns: np.ndarray) -> float:
    """Annualized Sharpe of a daily-return slice; 0 when std == 0."""
    if returns.size == 0:
        return 0.0
    std = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    if std == 0.0:
        return 0.0
    return float(np.mean(returns) / std * np.sqrt(_TRADING_DAYS_PER_YEAR))


def _split_into_subperiods(
    returns: pd.Series, n_subperiods: int
) -> list[pd.Series]:
    """Split *returns* into ``n_subperiods`` equal contiguous slices."""
    n = len(returns)
    edges = np.linspace(0, n, n_subperiods + 1, dtype=int)
    return [returns.iloc[edges[i] : edges[i + 1]] for i in range(n_subperiods)]


def _hockey_stick_warn(
    oos_returns: pd.Series | None,
    n_subperiods: int = 3,
) -> None:
    """Cycle-3 §8 hockey-stick concentration check.

    Splits *oos_returns* into ``n_subperiods`` equal slices, computes the
    annualized Sharpe of each, and emits a ``logger.warning`` when the
    minimum Sharpe is below 0 AND the maximum exceeds 1.5 — indicating
    out-performance concentrated in a single sub-period.  When the
    series is shorter than ``n_subperiods`` (or empty / ``None``), the
    check is skipped with a ``logger.debug`` line and no warning fires.
    """
    if oos_returns is None or len(oos_returns) < n_subperiods:
        logger.debug(
            "Concentration check skipped: returns length=%s < %d sub-periods",
            None if oos_returns is None else len(oos_returns),
            n_subperiods,
        )
        return

    sub_returns = _split_into_subperiods(oos_returns, n_subperiods)
    sharpes = [_annualized_sharpe(s.to_numpy()) for s in sub_returns]
    if not (
        min(sharpes) < _HOCKEY_STICK_NEG_THRESHOLD
        and max(sharpes) > _HOCKEY_STICK_POS_THRESHOLD
    ):
        return

    peak_idx = int(np.argmax(sharpes)) + 1  # 1-indexed for human-readable log
    logger.warning(
        "[WARN] hockey-stick — outperformance concentrated in period %d "
        "(sharpes=%s)",
        peak_idx,
        [round(s, 3) for s in sharpes],
    )


def _decide_rebalance(
    *,
    prev_weights: np.ndarray | None,
    target_weights: np.ndarray,
    current_date: pd.Timestamp,
    last_review_date: pd.Timestamp,
) -> tuple[bool, str]:
    """Cycle-3 §11 hybrid rebalance decision.

    Returns ``(False, "cold_start")`` when ``prev_weights`` is ``None``.
    Otherwise delegates to
    :func:`optimizer.rebalancing.should_rebalance_hybrid` with
    ``HybridRebalancingConfig.for_quarterly_with_10pct_threshold()``.
    """
    if prev_weights is None:
        logger.info(
            "Hybrid rebalance: cold_start (no previous_weights at %s)",
            current_date.date().isoformat(),
        )
        return False, "cold_start"

    config = HybridRebalancingConfig.for_quarterly_with_10pct_threshold()
    return should_rebalance_hybrid(
        prev_weights, target_weights, config, current_date, last_review_date
    )
