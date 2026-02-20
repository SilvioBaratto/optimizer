"""LLM-driven risk budget calibration from qualitative sector outlook.

Architecture:
  1. Call the BAML ``CalibrateRiskBudget`` function with the sector outlook,
     sector universe, and asset-to-sector mapping.
  2. Expand the LLM-returned sector budgets to an asset-level budget vector
     by distributing each sector's budget equally among its assets.
  3. Normalise to sum to 1.0 and enforce non-negativity.

The returned numpy array can be passed directly as ``risk_budget`` to
``build_risk_budgeting()``.

Usage::

    from app.services.risk_budget_service import calibrate_risk_budget

    asset_sector_map = {
        "AAPL": "Technology", "MSFT": "Technology",
        "JNJ": "Healthcare",  "XOM": "Energy",
    }
    budget = calibrate_risk_budget(
        sector_outlook="Overweight Technology; underweight Energy.",
        sector_universe=["Technology", "Healthcare", "Energy"],
        asset_sector_map=asset_sector_map,
    )
    # budget.shape == (4,), budget.sum() ≈ 1.0
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt

from baml_client import b
from baml_client.types import RiskBudgetOutput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _expand_sector_to_asset_budgets(
    sector_budgets: dict[str, float],
    asset_sector_map: dict[str, str],
) -> dict[str, float]:
    """Distribute sector budgets equally among assets within each sector.

    For each sector s with budget b_s and n_s assets, each asset in s
    receives b_s / n_s.  Assets whose sector is absent from
    ``sector_budgets`` receive an equal share of the residual (if any);
    if there is no residual, they receive 0.

    Returns a dict mapping every asset in ``asset_sector_map`` to a
    non-negative float.  Values are NOT normalised here — call
    ``_normalise`` afterwards.
    """
    # Count assets per sector
    sector_counts: dict[str, int] = {}
    for sector in asset_sector_map.values():
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    asset_budgets: dict[str, float] = {}
    for asset, sector in asset_sector_map.items():
        sb = max(0.0, sector_budgets.get(sector, 0.0))
        n = sector_counts.get(sector, 1)
        asset_budgets[asset] = sb / n

    return asset_budgets


def _normalise(values: dict[str, float]) -> dict[str, float]:
    """Normalise a dict of non-negative floats to sum to 1.

    Falls back to equal weights if the total is zero.
    """
    total = sum(max(0.0, v) for v in values.values())
    n = len(values)
    if total <= 0.0 or n == 0:
        equal = 1.0 / max(n, 1)
        return {k: equal for k in values}
    return {k: max(0.0, v) / total for k, v in values.items()}


def _to_budget_array(
    asset_budgets: dict[str, float],
    ordered_assets: list[str],
) -> npt.NDArray[np.float64]:
    """Convert an asset→budget dict to a numpy array in *ordered_assets* order."""
    return np.array(
        [asset_budgets.get(a, 0.0) for a in ordered_assets], dtype=np.float64
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calibrate_risk_budget(
    sector_outlook: str,
    sector_universe: list[str],
    asset_sector_map: dict[str, str],
) -> npt.NDArray[np.float64]:
    """Translate a qualitative sector outlook into an asset-level risk budget vector.

    Calls the BAML ``CalibrateRiskBudget`` function, then expands sector
    budgets to the asset level (equal-weight within sector) and normalises
    the result to sum to 1.

    Args:
        sector_outlook: Free-text qualitative sector view (e.g. ``"Overweight
            Technology and Healthcare; underweight Energy."``).
        sector_universe: Exhaustive list of sector names used in the portfolio.
        asset_sector_map: Dict mapping asset ticker to its sector name.  Must
            be non-empty.  Every sector in *asset_sector_map* should appear in
            *sector_universe* (unmapped sectors receive zero budget).

    Returns:
        1-D numpy array of shape ``(n_assets,)`` in the iteration order of
        ``asset_sector_map``.  Values are non-negative and sum to 1.0 (±1e-8).

    Raises:
        ValueError: If *sector_universe* or *asset_sector_map* is empty.
        RuntimeError: If the BAML call fails.
    """
    if not sector_universe:
        raise ValueError("sector_universe must not be empty")
    if not asset_sector_map:
        raise ValueError("asset_sector_map must not be empty")

    ordered_assets = list(asset_sector_map.keys())

    try:
        result: RiskBudgetOutput = b.CalibrateRiskBudget(
            sector_outlook=sector_outlook,
            sector_universe=sector_universe,
            asset_sector_map=asset_sector_map,
        )
    except Exception as exc:
        logger.error("BAML CalibrateRiskBudget failed: %s", exc)
        raise RuntimeError(f"LLM risk budget calibration failed: {exc}") from exc

    # Use LLM-provided asset_budgets if they cover all assets; otherwise
    # fall back to expanding from sector_budgets ourselves (more robust).
    llm_asset_budgets: dict[str, float] = dict(result.asset_budgets)
    missing = set(ordered_assets) - set(llm_asset_budgets.keys())
    if missing:
        logger.warning(
            "LLM asset_budgets missing %d asset(s) — expanding from sector_budgets",
            len(missing),
        )
        expanded = _expand_sector_to_asset_budgets(
            dict(result.sector_budgets), asset_sector_map
        )
        # Merge: use LLM values where available, expansion for the rest
        for asset in missing:
            llm_asset_budgets[asset] = expanded.get(asset, 0.0)

    normalised = _normalise(llm_asset_budgets)
    budget = _to_budget_array(normalised, ordered_assets)

    logger.info(
        "Calibrated risk budget for %d assets across %d sectors",
        len(ordered_assets),
        len(sector_universe),
    )
    return budget
