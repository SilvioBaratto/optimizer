"""Macro-regime sector weight bands for dynamic portfolio constraints.

Each regime maps the 11 GICS sectors to a ``(floor, cap)`` tuple that
constrains total sector weight in the optimizer.  Bands are applied once
per run using the lagged classified regime (single-regime-per-run
injection — backtest folds and final fit share one regime).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from optimizer.factors._config import MacroRegime

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

# Caps that sum to exactly the budget pin the optimiser to a single point, and
# the 6-decimal rounding of the emitted constraint rows can tip that point
# infeasible, so widened caps keep a little room above full investment.
_MIN_CAP_COVERAGE = 1.05

ALL_11_SECTORS: tuple[str, ...] = (
    "Energy",
    "Basic Materials",
    "Industrials",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Healthcare",
    "Financial Services",
    "Technology",
    "Communication Services",
    "Utilities",
    "Real Estate",
)

SECTOR_REGIME_BANDS: dict[MacroRegime, dict[str, tuple[float, float]]] = {
    MacroRegime.RECOVERY: {
        "Energy": (0.00, 0.10),
        "Basic Materials": (0.06, 0.22),
        "Industrials": (0.10, 0.28),
        "Consumer Cyclical": (0.10, 0.28),
        "Consumer Defensive": (0.00, 0.06),
        "Healthcare": (0.00, 0.10),
        "Financial Services": (0.10, 0.28),
        "Technology": (0.10, 0.28),
        "Communication Services": (0.06, 0.22),
        "Utilities": (0.00, 0.06),
        "Real Estate": (0.06, 0.22),
    },
    MacroRegime.EXPANSION: {
        "Energy": (0.03, 0.16),
        "Basic Materials": (0.00, 0.10),
        "Industrials": (0.06, 0.22),
        "Consumer Cyclical": (0.03, 0.16),
        "Consumer Defensive": (0.03, 0.16),
        "Healthcare": (0.06, 0.22),
        "Financial Services": (0.10, 0.28),
        "Technology": (0.10, 0.28),
        "Communication Services": (0.06, 0.22),
        "Utilities": (0.00, 0.10),
        "Real Estate": (0.06, 0.22),
    },
    MacroRegime.SLOWDOWN: {
        "Energy": (0.10, 0.28),
        "Basic Materials": (0.06, 0.22),
        "Industrials": (0.00, 0.10),
        "Consumer Cyclical": (0.00, 0.06),
        "Consumer Defensive": (0.10, 0.28),
        "Healthcare": (0.10, 0.28),
        "Financial Services": (0.03, 0.16),
        "Technology": (0.00, 0.06),
        "Communication Services": (0.00, 0.10),
        "Utilities": (0.06, 0.22),
        "Real Estate": (0.03, 0.16),
    },
    MacroRegime.RECESSION: {
        "Energy": (0.06, 0.22),
        "Basic Materials": (0.03, 0.16),
        "Industrials": (0.00, 0.06),
        "Consumer Cyclical": (0.03, 0.16),
        "Consumer Defensive": (0.10, 0.28),
        "Healthcare": (0.10, 0.28),
        "Financial Services": (0.00, 0.06),
        "Technology": (0.00, 0.06),
        "Communication Services": (0.00, 0.06),
        "Utilities": (0.10, 0.28),
        "Real Estate": (0.00, 0.06),
    },
    MacroRegime.UNKNOWN: dict.fromkeys(ALL_11_SECTORS, (0.03, 0.16)),
}


def _matrix_to_bands_tuple() -> tuple[tuple[str, str, float, float], ...]:
    """Flatten SECTOR_REGIME_BANDS into a serialisable tuple.

    Returns entries ordered deterministically by regime then sector, used as
    the default value for SectorRegimeBandsConfig.

    Returns:
        Tuple of (regime_value, sector_name, floor, cap) entries.
    """
    rows: list[tuple[str, str, float, float]] = []
    for regime in MacroRegime:
        sector_map = SECTOR_REGIME_BANDS[regime]
        for sector in ALL_11_SECTORS:
            floor, cap = sector_map[sector]
            rows.append((regime.value, sector, floor, cap))
    return tuple(rows)


def _restrict_to_universe(
    bands: dict[str, tuple[float, float]],
    universe: frozenset[str],
    regime: MacroRegime,
) -> dict[str, tuple[float, float]]:
    """Keep the bands of sectors in ``universe`` and keep them satisfiable.

    Args:
        bands: Full (floor, cap) mapping for the regime.
        universe: Sector labels the optimiser's assets actually carry.
        regime: Regime the bands belong to, used for log messages only.

    Returns:
        Bands for the sectors present in the universe. When every universe
        sector is banded and their caps cannot cover a fully invested
        portfolio, the caps are widened by one common factor.
    """
    present = {sector: band for sector, band in bands.items() if sector in universe}

    dropped = {
        sector: floor
        for sector, (floor, _) in bands.items()
        if sector not in universe and floor > 0.0
    }
    if dropped:
        logger.warning(
            "Regime %s sector floors ignored for sectors absent from the universe: %s",
            regime.value,
            ", ".join(f"{sector} >= {floor:g}" for sector, floor in dropped.items()),
        )

    # A universe sector without a band is unconstrained and can absorb the
    # residual weight, so the banded caps do not have to cover the budget.
    if not present or universe - present.keys():
        return present

    cap_sum = sum(cap for _, cap in present.values())
    if cap_sum >= _MIN_CAP_COVERAGE:
        return present

    logger.warning(
        "Regime %s sector caps of the universe sum to %.4f, below the %.2f "
        "needed to keep a fully invested portfolio feasible; widening them "
        "proportionally.",
        regime.value,
        cap_sum,
        _MIN_CAP_COVERAGE,
    )
    if cap_sum <= 0.0:
        return dict.fromkeys(present, (0.0, 1.0))
    scale = _MIN_CAP_COVERAGE / cap_sum
    return {
        sector: (floor, min(1.0, cap * scale))
        for sector, (floor, cap) in present.items()
    }


def resolve_sector_bands(
    regime: MacroRegime,
    config: object | None = None,
    *,
    sectors: Iterable[str] | None = None,
) -> dict[str, tuple[float, float]]:
    """Return sector (floor, cap) bands for the given macro regime.

    Performs a hard switch on regime: when the regime value matches an entry
    in the config bands, that entry's floor/cap is used. Falls back to UNKNOWN
    when the regime does not appear in the config. Returns a fresh copy so
    callers cannot mutate the source data.

    The bands cover all 11 sectors, but a selected universe often spans only
    some of them. Pass ``sectors`` so the result matches the universe: caps of
    absent sectors would otherwise be emitted for nothing while their floors
    are silently dropped by the optimiser, and the caps of the sectors that
    remain can sum to less than 100%, which makes a fully invested portfolio
    infeasible.

    Args:
        regime: The classified macro-economic regime.
        config: Optional config holding a custom bands tuple. When None,
            the module-level SECTOR_REGIME_BANDS is used directly.
        sectors: Sector labels carried by the assets being optimised. When
            given, bands of other sectors are dropped (a warning names any
            dropped floors) and, if every given sector has a band and
            their caps sum to less than a fully invested portfolio needs, the
            caps are widened by one common factor. Floors are never changed.
            Assets with no sector label are unconstrained by sector, so the
            widening may relax the bands more than strictly needed when such
            assets exist. ``None`` returns the full band matrix unchanged.

    Returns:
        Mapping from sector name to (floor, cap) for the resolved regime.
    """
    from optimizer.factors._config import SectorRegimeBandsConfig  # avoid circular

    resolved: dict[str, tuple[float, float]]
    if config is not None and isinstance(config, SectorRegimeBandsConfig):
        custom: dict[str, dict[str, tuple[float, float]]] = {}
        for regime_val, sector, floor, cap in config.bands:
            custom.setdefault(regime_val, {})[sector] = (floor, cap)

        regime_key = regime.value
        unknown_key = MacroRegime.UNKNOWN.value
        if regime_key in custom:
            resolved = dict(custom[regime_key])
        elif unknown_key in custom:
            resolved = dict(custom[unknown_key])
        else:
            resolved = dict.fromkeys(ALL_11_SECTORS, (0.03, 0.16))
    else:
        default = SECTOR_REGIME_BANDS.get(regime)
        if default is None:
            default = SECTOR_REGIME_BANDS[MacroRegime.UNKNOWN]
        resolved = dict(default)

    if sectors is None:
        return resolved
    return _restrict_to_universe(resolved, frozenset(sectors), regime)
