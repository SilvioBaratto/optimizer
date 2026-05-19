"""Unit tests for optimizer.factors._sector_bands and SectorRegimeBandsConfig."""

from __future__ import annotations

import pytest

from optimizer.exceptions import ConfigurationError
from optimizer.factors._config import MacroRegime, SectorRegimeBandsConfig
from optimizer.factors._sector_bands import (
    ALL_11_SECTORS,
    SECTOR_REGIME_BANDS,
    _matrix_to_bands_tuple,
    resolve_sector_bands,
)

# ---------------------------------------------------------------------------
# ALL_11_SECTORS completeness
# ---------------------------------------------------------------------------


def test_all_11_sectors_length() -> None:
    assert len(ALL_11_SECTORS) == 11


def test_all_11_sectors_are_strings() -> None:
    for s in ALL_11_SECTORS:
        assert isinstance(s, str)


# ---------------------------------------------------------------------------
# SECTOR_REGIME_BANDS matrix correctness
# ---------------------------------------------------------------------------


def test_matrix_covers_all_regimes() -> None:
    for regime in MacroRegime:
        assert regime in SECTOR_REGIME_BANDS, f"Missing regime: {regime}"


def test_matrix_covers_all_sectors_for_each_regime() -> None:
    for regime in MacroRegime:
        for sector in ALL_11_SECTORS:
            assert sector in SECTOR_REGIME_BANDS[regime], (
                f"Missing sector {sector!r} for regime {regime.value!r}"
            )


def test_matrix_bounds_valid() -> None:
    for regime, sector_map in SECTOR_REGIME_BANDS.items():
        for sector, (floor, cap) in sector_map.items():
            assert 0.0 <= floor <= cap <= 1.0, (
                f"Invalid band ({floor}, {cap}) for "
                f"regime={regime.value!r}, sector={sector!r}"
            )


def test_per_regime_floor_sum_le_one() -> None:
    for regime, sector_map in SECTOR_REGIME_BANDS.items():
        total = sum(floor for floor, _ in sector_map.values())
        assert total <= 1.0 + 1e-9, (
            f"Regime {regime.value!r}: floor sum {total:.4f} > 1.0"
        )


# ---------------------------------------------------------------------------
# Specific value spot-checks (blueprint-mandated)
# ---------------------------------------------------------------------------


def test_recovery_technology_band() -> None:
    bands = SECTOR_REGIME_BANDS[MacroRegime.RECOVERY]
    assert bands["Technology"] == (0.10, 0.28)


def test_recession_technology_band() -> None:
    bands = SECTOR_REGIME_BANDS[MacroRegime.RECESSION]
    assert bands["Technology"] == (0.00, 0.06)


def test_unknown_all_sectors_uniform() -> None:
    bands = SECTOR_REGIME_BANDS[MacroRegime.UNKNOWN]
    for sector in ALL_11_SECTORS:
        assert bands[sector] == (0.03, 0.16), (
            f"UNKNOWN sector {sector!r} expected (0.03, 0.16), got {bands[sector]}"
        )


# ---------------------------------------------------------------------------
# resolve_sector_bands
# ---------------------------------------------------------------------------


def test_resolve_recovery_technology() -> None:
    result = resolve_sector_bands(MacroRegime.RECOVERY)
    assert result["Technology"] == (0.10, 0.28)


def test_resolve_unknown_all_sectors() -> None:
    result = resolve_sector_bands(MacroRegime.UNKNOWN)
    for sector in ALL_11_SECTORS:
        assert result[sector] == (0.03, 0.16), (
            f"UNKNOWN resolver: sector {sector!r} expected (0.03, 0.16), "
            f"got {result[sector]}"
        )


def test_resolve_returns_copy() -> None:
    """Mutating the returned dict must not affect the module-level matrix."""
    result = resolve_sector_bands(MacroRegime.RECESSION)
    original = SECTOR_REGIME_BANDS[MacroRegime.RECESSION]["Technology"]
    result["Technology"] = (0.99, 0.99)
    assert SECTOR_REGIME_BANDS[MacroRegime.RECESSION]["Technology"] == original


def test_resolve_with_config_recovery_technology() -> None:
    cfg = SectorRegimeBandsConfig.for_default()
    result = resolve_sector_bands(MacroRegime.RECOVERY, cfg)
    assert result["Technology"] == (0.10, 0.28)


def test_resolve_with_config_unknown() -> None:
    cfg = SectorRegimeBandsConfig.for_default()
    result = resolve_sector_bands(MacroRegime.UNKNOWN, cfg)
    for sector in ALL_11_SECTORS:
        assert result[sector] == (0.03, 0.16)


def test_resolve_config_returns_copy() -> None:
    cfg = SectorRegimeBandsConfig.for_default()
    result = resolve_sector_bands(MacroRegime.RECESSION, cfg)
    original_tech = SECTOR_REGIME_BANDS[MacroRegime.RECESSION]["Technology"]
    result["Technology"] = (0.77, 0.99)
    assert SECTOR_REGIME_BANDS[MacroRegime.RECESSION]["Technology"] == original_tech


# ---------------------------------------------------------------------------
# SectorRegimeBandsConfig validation
# ---------------------------------------------------------------------------


def test_for_default_construction() -> None:
    cfg = SectorRegimeBandsConfig.for_default()
    assert cfg.enable is True
    assert len(cfg.bands) == len(MacroRegime) * len(ALL_11_SECTORS)


def test_disabled_construction() -> None:
    cfg = SectorRegimeBandsConfig.disabled()
    assert cfg.enable is False
    assert cfg.bands == ()


def test_default_config_no_validation_error() -> None:
    # for_default() must not raise.
    cfg = SectorRegimeBandsConfig.for_default()
    assert cfg is not None


def test_invalid_floor_greater_than_cap_raises() -> None:
    pat = r"floor.*cap|cap.*floor|0\.0 <= floor"
    with pytest.raises(ConfigurationError, match=pat):
        SectorRegimeBandsConfig(
            enable=True,
            bands=(("recovery", "Technology", 0.50, 0.10),),
        )


def test_floor_sum_greater_than_one_raises() -> None:
    """Construct a config where per-regime floor sum exceeds 1.0."""
    # Build one regime with floors summing to > 1.0.
    bands: list[tuple[str, str, float, float]] = []
    for sector in ALL_11_SECTORS:
        # 11 × 0.10 = 1.10 > 1.0
        bands.append(("recovery", sector, 0.10, 0.50))
    with pytest.raises(ConfigurationError, match=r"sum.*1\.0|floors.*infeasible"):
        SectorRegimeBandsConfig(enable=True, bands=tuple(bands))


def test_invalid_regime_value_raises() -> None:
    with pytest.raises(ConfigurationError, match="regime"):
        SectorRegimeBandsConfig(
            enable=True,
            bands=(("not_a_regime", "Technology", 0.05, 0.20),),
        )


def test_invalid_sector_raises() -> None:
    with pytest.raises(ConfigurationError, match="sector"):
        SectorRegimeBandsConfig(
            enable=True,
            bands=(("recovery", "Alien Technology", 0.05, 0.20),),
        )


def test_floor_below_zero_raises() -> None:
    with pytest.raises(ConfigurationError):
        SectorRegimeBandsConfig(
            enable=True,
            bands=(("recovery", "Technology", -0.01, 0.20),),
        )


# ---------------------------------------------------------------------------
# _matrix_to_bands_tuple
# ---------------------------------------------------------------------------


def test_matrix_to_bands_tuple_count() -> None:
    tup = _matrix_to_bands_tuple()
    assert len(tup) == len(MacroRegime) * len(ALL_11_SECTORS)


def test_matrix_to_bands_tuple_types() -> None:
    for entry in _matrix_to_bands_tuple():
        regime_val, sector, floor, cap = entry
        assert isinstance(regime_val, str)
        assert isinstance(sector, str)
        assert isinstance(floor, float)
        assert isinstance(cap, float)
