"""Macro-regime sector weight bands for dynamic portfolio constraints.

Each regime maps the 11 GICS sectors to a ``(floor, cap)`` tuple that
constrains total sector weight in the optimizer.  Bands are applied once
per pipeline run using the lagged classified regime (single-regime-per-run
injection — backtest folds and final fit share one regime).
"""

from __future__ import annotations

from optimizer.factors._config import MacroRegime

# ---------------------------------------------------------------------------
# Sector constants
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Regime → sector → (floor, cap) matrix
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _matrix_to_bands_tuple() -> tuple[tuple[str, str, float, float], ...]:
    """Flatten ``SECTOR_REGIME_BANDS`` into a serialisable tuple.

    Returns a tuple of ``(regime_value, sector, floor, cap)`` entries
    ordered deterministically by regime then sector.  Used as the default
    value for :class:`~optimizer.factors._config.SectorRegimeBandsConfig`.

    Returns
    -------
    tuple[tuple[str, str, float, float], ...]
        Flattened bands where each entry is
        ``(regime.value, sector_name, floor, cap)``.
    """
    rows: list[tuple[str, str, float, float]] = []
    for regime in MacroRegime:
        sector_map = SECTOR_REGIME_BANDS[regime]
        for sector in ALL_11_SECTORS:
            floor, cap = sector_map[sector]
            rows.append((regime.value, sector, floor, cap))
    return tuple(rows)


# ---------------------------------------------------------------------------
# Public resolver
# ---------------------------------------------------------------------------


def resolve_sector_bands(
    regime: MacroRegime,
    config: object | None = None,
) -> dict[str, tuple[float, float]]:
    """Return sector ``(floor, cap)`` bands for the given macro regime.

    Performs a hard switch on ``regime``: when the regime value matches an
    entry in the config bands, that entry's floor/cap is used.  Falls back
    to ``UNKNOWN`` when the regime does not appear in the config.  Returns
    a fresh copy so callers cannot mutate the source data.

    Parameters
    ----------
    regime : MacroRegime
        The classified macro-economic regime.
    config : SectorRegimeBandsConfig or None
        Optional config holding a custom bands tuple.  When ``None``,
        the module-level ``SECTOR_REGIME_BANDS`` is used directly.

    Returns
    -------
    dict[str, tuple[float, float]]
        Mapping from sector name to ``(floor, cap)`` for the resolved regime.
    """
    from optimizer.factors._config import SectorRegimeBandsConfig  # avoid circular

    if config is not None and isinstance(config, SectorRegimeBandsConfig):
        # Build a regime → sector → (floor, cap) map from the config tuple.
        custom: dict[str, dict[str, tuple[float, float]]] = {}
        for regime_val, sector, floor, cap in config.bands:
            custom.setdefault(regime_val, {})[sector] = (floor, cap)

        regime_key = regime.value
        if regime_key in custom:
            return dict(custom[regime_key])
        # Fallback to UNKNOWN
        unknown_key = MacroRegime.UNKNOWN.value
        if unknown_key in custom:
            return dict(custom[unknown_key])
        # Last resort: uniform bands from the full matrix
        return dict.fromkeys(ALL_11_SECTORS, (0.03, 0.16))

    # Default: use module-level matrix with UNKNOWN fallback.
    resolved = SECTOR_REGIME_BANDS.get(regime)
    if resolved is None:
        resolved = SECTOR_REGIME_BANDS[MacroRegime.UNKNOWN]
    return dict(resolved)
