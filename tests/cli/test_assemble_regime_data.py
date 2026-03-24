"""Regression tests for assemble_regime_data() ffill limit and staleness warning.

Issue #313: assemble_regime_data() was calling ffill() with no limit, allowing
stale monthly PMI observations to propagate indefinitely into the future.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("typer")

from cli.data_assembly import assemble_regime_data

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_macro(index: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Return a macro DataFrame with no gdp_growth/yield_spread columns."""
    if index is None:
        index = pd.DatetimeIndex([])
    return pd.DataFrame(index=index)


def _fred_df(start: str, end: str, value: float = 1.0) -> pd.DataFrame:
    """Daily FRED DataFrame with T10Y2Y (maps to spread_2s10s)."""
    idx = pd.date_range(start, end, freq="D")
    return pd.DataFrame({"T10Y2Y": value}, index=idx)


def _te_df(date: str, pmi_value: float = 52.0) -> pd.DataFrame:
    """TE observations DataFrame with a single PMI observation on *date*."""
    idx = pd.DatetimeIndex([date])
    # _TE_REGIME_MAP maps "manufacturing_pmi" → "pmi"
    return pd.DataFrame({"manufacturing_pmi": [pmi_value]}, index=idx)


# ---------------------------------------------------------------------------
# TestFfillLimitRespected
# ---------------------------------------------------------------------------


class TestFfillLimitRespected:
    """The fill_limit parameter caps forward-fill at N consecutive rows."""

    def test_value_beyond_limit_becomes_nan(self) -> None:
        """PMI observation older than fill_limit days → NaN at tail."""
        # PMI on 2024-01-01; FRED runs to 2024-04-01 (~91 days later)
        te_obs = _te_df("2024-01-01")
        fred = _fred_df("2024-01-01", "2024-04-01")

        result = assemble_regime_data(_empty_macro(), fred, te_obs, fill_limit=45)

        assert "pmi" in result.columns
        assert pd.isna(result["pmi"].iloc[-1]), (
            "PMI 91 days old should be NaN after fill_limit=45"
        )

    def test_value_within_limit_is_filled(self) -> None:
        """PMI observation within fill_limit days → filled at tail."""
        # PMI on 2024-01-01; FRED runs only 30 days (within limit)
        te_obs = _te_df("2024-01-01")
        fred = _fred_df("2024-01-01", "2024-01-31")

        result = assemble_regime_data(_empty_macro(), fred, te_obs, fill_limit=45)

        assert "pmi" in result.columns
        assert not pd.isna(result["pmi"].iloc[-1]), (
            "PMI 30 days old should be filled when fill_limit=45"
        )

    def test_default_fill_limit_is_45(self) -> None:
        """Default call (no fill_limit arg) uses limit=45."""
        # PMI on 2024-01-01; FRED runs 50 days (beyond default of 45)
        te_obs = _te_df("2024-01-01")
        fred = _fred_df("2024-01-01", "2024-02-20")  # ~50 days

        # Suppress the expected staleness warning; we only care about NaN value
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = assemble_regime_data(_empty_macro(), fred, te_obs)

        assert "pmi" in result.columns
        assert pd.isna(result["pmi"].iloc[-1]), (
            "Default fill_limit=45 should leave PMI NaN when data is 50 days old"
        )


# ---------------------------------------------------------------------------
# TestStalenessWarning
# ---------------------------------------------------------------------------


class TestStalenessWarning:
    """A UserWarning is emitted for each column that is stale at the tail."""

    def test_warns_for_stale_pmi(self) -> None:
        """Stale PMI triggers UserWarning mentioning the column name."""
        te_obs = _te_df("2024-01-01")
        fred = _fred_df("2024-01-01", "2024-04-01")

        with pytest.warns(UserWarning, match="pmi"):
            assemble_regime_data(_empty_macro(), fred, te_obs, fill_limit=45)

    def test_warning_includes_fill_limit_and_last_valid_date(self) -> None:
        """Warning message contains fill_limit value and actual observation date."""
        te_obs = _te_df("2024-01-01")
        fred = _fred_df("2024-01-01", "2024-04-01")

        with pytest.warns(UserWarning) as record:
            assemble_regime_data(_empty_macro(), fred, te_obs, fill_limit=45)

        pmi_warnings = [w for w in record if "pmi" in str(w.message)]
        assert len(pmi_warnings) >= 1
        msg = str(pmi_warnings[0].message)
        assert "fill_limit=45" in msg
        # Warning reports the actual observation date, not the last filled date
        assert "2024-01-01" in msg
        assert "trailing NaN" in msg

    def test_warns_for_each_stale_column(self) -> None:
        """One warning is emitted per stale column."""
        # Both PMI (TE) and spread_2s10s (FRED) will be stale.
        # hy_oas (BAMLH0A0HYM2) provides daily data to extend the date index
        # 91 days beyond the single T10Y2Y / PMI observation on 2024-01-01.
        pmi_obs = _te_df("2024-01-01")

        # T10Y2Y (→ spread_2s10s): only 2024-01-01
        # BAMLH0A0HYM2 (→ hy_oas): fresh daily data → extends the merged date range
        fred_idx = pd.date_range("2024-01-01", "2024-04-01", freq="D")
        fred = pd.DataFrame(
            {
                "T10Y2Y": [1.0] + [np.nan] * (len(fred_idx) - 1),
                "BAMLH0A0HYM2": np.full(len(fred_idx), 3.5),  # fresh hy_oas
            },
            index=fred_idx,
        )

        with pytest.warns(UserWarning) as record:
            assemble_regime_data(_empty_macro(), fred, pmi_obs, fill_limit=45)

        stale_cols = {str(w.message).split("'")[1] for w in record}
        assert "pmi" in stale_cols
        assert "spread_2s10s" in stale_cols


# ---------------------------------------------------------------------------
# TestNoRegressionForFreshData
# ---------------------------------------------------------------------------


class TestNoRegressionForFreshData:
    """No warnings and correct shape when all indicators are fresh."""

    def test_no_warning_for_fresh_data(self) -> None:
        """Fresh indicators (within fill_limit) produce no UserWarning."""
        # PMI observation 10 days before end of FRED series
        end = "2024-02-10"
        pmi_date = "2024-02-01"
        te_obs = _te_df(pmi_date)
        fred = _fred_df("2024-01-01", end)

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            # Should not raise
            assemble_regime_data(_empty_macro(), fred, te_obs, fill_limit=45)

    def test_fresh_data_result_has_no_trailing_nans(self) -> None:
        """All columns are non-NaN at the final row when data is fresh."""
        te_obs = _te_df("2024-02-01")
        fred = _fred_df("2024-01-01", "2024-02-10")

        result = assemble_regime_data(_empty_macro(), fred, te_obs, fill_limit=45)

        assert result.iloc[-1].notna().all(), (
            "All columns should be non-NaN at the tail for fresh data"
        )

    def test_empty_inputs_return_empty_dataframe(self) -> None:
        """All-empty inputs return an empty DataFrame without error."""
        result = assemble_regime_data(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            fill_limit=45,
        )
        assert isinstance(result, pd.DataFrame)
        assert result.empty


# ---------------------------------------------------------------------------
# TestIndexValidationAndCoercion
# ---------------------------------------------------------------------------


class TestIndexValidationAndCoercion:
    """Non-DatetimeIndex inputs are coerced with a warning; bad indexes raise."""

    def test_integer_year_index_warns_and_aligns(self) -> None:
        """Integer-year index on fred_data → UserWarning + no silent NaN columns."""
        # Build a FRED DataFrame with integer-year index instead of DatetimeIndex.
        # Columns must match _FRED_REGIME_MAP keys so they get picked up.
        int_index = [2023, 2024]
        fred_int = pd.DataFrame({"T10Y2Y": [0.5, 0.8]}, index=int_index)

        te_obs = _te_df("2023-06-01")

        with pytest.warns(UserWarning, match="Index"):
            result = assemble_regime_data(
                _empty_macro(), fred_int, te_obs, fill_limit=45
            )

        # The join must have produced at least one non-NaN spread_2s10s value.
        assert "spread_2s10s" in result.columns
        assert result["spread_2s10s"].notna().any(), (
            "Integer-year index should coerce and align — not produce all-NaN column"
        )

    def test_string_date_index_warns_and_aligns(self) -> None:
        """String-date index on te_observations → UserWarning + data aligns."""
        fred = _fred_df("2024-01-01", "2024-02-10")

        # TE DataFrame with string index — must match _TE_REGIME_MAP key.
        str_index = pd.Index(["2024-01-15"])
        te_str = pd.DataFrame({"manufacturing_pmi": [52.0]}, index=str_index)

        with pytest.warns(UserWarning, match="Index"):
            result = assemble_regime_data(_empty_macro(), fred, te_str, fill_limit=45)

        assert "pmi" in result.columns
        assert result["pmi"].notna().any(), (
            "String-date index should coerce and align — not produce all-NaN column"
        )

    def test_unconvertible_index_raises_value_error(self) -> None:
        """Index of random strings that cannot be parsed as dates → ValueError."""
        fred = _fred_df("2024-01-01", "2024-01-10")

        bad_index = pd.Index(["not-a-date", "also-bad"])
        te_bad = pd.DataFrame({"manufacturing_pmi": [52.0, 53.0]}, index=bad_index)

        with pytest.raises(ValueError, match="could not be coerced to DatetimeIndex"):
            assemble_regime_data(_empty_macro(), fred, te_bad, fill_limit=45)

    def test_all_datetime_index_no_warning(self) -> None:
        """All-DatetimeIndex inputs produce no coercion UserWarning."""
        fred = _fred_df("2024-01-01", "2024-02-10")
        te_obs = _te_df("2024-01-15")

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            # Should not raise — all parts already have DatetimeIndex.
            result = assemble_regime_data(_empty_macro(), fred, te_obs, fill_limit=45)

        assert not result.empty
