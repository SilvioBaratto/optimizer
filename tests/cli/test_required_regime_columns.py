"""TDD coverage for required-column raise + GDP fallback in _FRED_REGIME_MAP (#570)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("typer")

from cli.data_assembly import (
    _FRED_REGIME_MAP,
    _REQUIRED_REGIME_COLUMNS,
    assemble_regime_data,
)


def _empty_te() -> pd.DataFrame:
    return pd.DataFrame()


def _empty_fred() -> pd.DataFrame:
    return pd.DataFrame()


def _macro_with_nan_gdp(start: str, end: str) -> pd.DataFrame:
    """Macro DataFrame with gdp_growth all-NaN but yield_spread populated."""
    idx = pd.date_range(start, end, freq="D")
    return pd.DataFrame(
        {"gdp_growth": np.nan, "yield_spread": 0.5},
        index=idx,
    )


def _macro_with_old_gdp(start: str, mid: str, end: str) -> pd.DataFrame:
    """gdp_growth populated up to mid, NaN after; yield_spread populated throughout."""
    idx = pd.date_range(start, end, freq="D")
    gdp = pd.Series(np.nan, index=idx, dtype="float64")
    gdp.loc[start:mid] = 1.0
    return pd.DataFrame({"gdp_growth": gdp, "yield_spread": 0.5}, index=idx)


# ---------------------------------------------------------------------------
# Map + constant surface
# ---------------------------------------------------------------------------


class TestFredRegimeMapGdpEntry:
    def test_when_map_inspected_gdp_series_maps_to_gdp_growth(self) -> None:
        assert _FRED_REGIME_MAP.get("A191RL1Q225SBEA") == "gdp_growth"


class TestRequiredRegimeColumnsConstant:
    def test_when_constant_inspected_only_gdp_growth_is_required(self) -> None:
        assert "gdp_growth" in _REQUIRED_REGIME_COLUMNS
        assert len(_REQUIRED_REGIME_COLUMNS) == 1

    def test_when_constant_inspected_it_is_a_frozenset(self) -> None:
        assert isinstance(_REQUIRED_REGIME_COLUMNS, frozenset)


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------


class TestRequiredColumnAllNanRaises:
    def test_when_gdp_growth_all_nan_value_error_names_column(self) -> None:
        macro = _macro_with_nan_gdp("2024-01-01", "2024-02-10")
        with pytest.raises(ValueError, match="gdp_growth"):
            assemble_regime_data(macro, _empty_fred(), _empty_te())


class TestRequiredColumnStaleEmitsWarningNotRaise:
    def test_when_gdp_growth_stale_beyond_fill_limit_warning_only(self) -> None:
        # gdp populated 2024-01-01 → 2024-01-10; series extends to 2024-04-01.
        # 80+ days of trailing NaN > fill_limit=45 — last valid date is 2024-01-10.
        macro = _macro_with_old_gdp("2024-01-01", "2024-01-10", "2024-04-01")
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always", UserWarning)
            assemble_regime_data(macro, _empty_fred(), _empty_te(), fill_limit=45)

        gdp_warnings = [w for w in record if "gdp_growth" in str(w.message)]
        assert gdp_warnings, "expected UserWarning for stale gdp_growth"
