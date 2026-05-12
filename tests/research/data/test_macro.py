"""Tests for research/data/_macro.py."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from research.data._macro import (
    FRED_SERIES_IDS,
    assemble_bond_observations,
    assemble_fred_series,
    assemble_macro_data,
    assemble_macro_timeseries,
    assemble_te_observations,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_session_all(rows: list[Any]) -> MagicMock:
    """Session whose execute().all() returns *rows*."""
    session = MagicMock()
    session.execute.return_value.all.return_value = rows
    return session


# ---------------------------------------------------------------------------
# TestAssembleMacroData
# ---------------------------------------------------------------------------


class TestAssembleMacroData:
    """Tests for assemble_macro_data()."""

    def test_when_no_data_single_row_fallback_returned(self) -> None:
        """when no data, single-row fallback with gdp_growth and yield_spread."""
        session = MagicMock()
        session.execute.return_value.scalar_one_or_none.return_value = None
        session.execute.return_value.scalars.return_value.all.return_value = []

        with patch("research.data._macro.assemble_macro_timeseries") as mock_ts:
            mock_ts.return_value = pd.DataFrame()
            result = assemble_macro_data(session, country="USA")

        assert len(result) == 1
        assert "gdp_growth" in result.columns
        assert "yield_spread" in result.columns

    def test_when_timeseries_has_two_or_more_rows_timeseries_returned(self) -> None:
        """when timeseries >=2 rows, full timeseries returned instead of fallback."""
        idx = pd.to_datetime(["2024-01-01", "2024-04-01"])
        ts_df = pd.DataFrame(
            {"gdp_growth": [2.1, 2.5], "yield_spread": [0.3, 0.5]},
            index=idx,
        )
        session = MagicMock()

        with patch("research.data._macro.assemble_macro_timeseries") as mock_ts:
            mock_ts.return_value = ts_df
            result = assemble_macro_data(session, country="USA")

        assert len(result) == 2
        pd.testing.assert_frame_equal(result, ts_df)


# ---------------------------------------------------------------------------
# TestAssembleMacroTimeseries
# ---------------------------------------------------------------------------


class TestAssembleMacroTimeseries:
    """Tests for assemble_macro_timeseries()."""

    def test_when_no_rows_empty_dataframe_returned(self) -> None:
        """when all three queries return empty, empty DataFrame returned."""
        session = MagicMock()
        session.execute.return_value.all.return_value = []

        result = assemble_macro_timeseries(session, country="USA")

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_when_gdp_rows_present_gdp_growth_column_present(self) -> None:
        """when GDP rows present, gdp_growth column present in result."""
        import datetime

        date_val = datetime.date(2024, 1, 1)
        gdp_row = (date_val, 2.5)

        call_count = 0

        def execute_side_effect(stmt: Any) -> MagicMock:
            del stmt
            nonlocal call_count
            call_count += 1
            mock = MagicMock()
            # First call = GDP, second = bonds, third = IlSole
            if call_count == 1:
                mock.all.return_value = [gdp_row]
            else:
                mock.all.return_value = []
            return mock

        session = MagicMock()
        session.execute.side_effect = execute_side_effect

        result = assemble_macro_timeseries(session, country="USA")

        assert "gdp_growth" in result.columns
        assert result["gdp_growth"].iloc[0] == pytest.approx(2.5)

    def test_when_bond_rows_present_yield_spread_computed(self) -> None:
        """when 10Y and 2Y bond rows present, yield_spread = 10Y - 2Y."""
        import datetime

        date_val = datetime.date(2024, 3, 1)
        bond_row_10y = (date_val, "10Y", 4.5)
        bond_row_2y = (date_val, "2Y", 4.0)

        call_count = 0

        def execute_side_effect(stmt: Any) -> MagicMock:
            del stmt
            nonlocal call_count
            call_count += 1
            mock = MagicMock()
            if call_count == 1:
                mock.all.return_value = []  # GDP
            elif call_count == 2:
                mock.all.return_value = [bond_row_10y, bond_row_2y]
            else:
                mock.all.return_value = []  # IlSole
            return mock

        session = MagicMock()
        session.execute.side_effect = execute_side_effect

        result = assemble_macro_timeseries(session, country="USA")

        assert "yield_spread" in result.columns
        assert result["yield_spread"].iloc[0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# TestAssembleTeObservations
# ---------------------------------------------------------------------------


class TestAssembleTeObservations:
    """Tests for assemble_te_observations()."""

    def test_when_no_rows_empty_dataframe_returned(self) -> None:
        """when no rows, empty DataFrame returned."""
        session = _mock_session_all([])
        result = assemble_te_observations(session, country="USA")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_when_rows_present_pivot_by_indicator_key(self) -> None:
        """when rows with different indicator_keys on same date, columns are keys."""
        import datetime

        date_val = datetime.date(2024, 6, 1)
        rows = [
            (date_val, "manufacturing_pmi", 52.3),
            (date_val, "gdp_growth_rate", 1.8),
        ]
        session = _mock_session_all(rows)
        result = assemble_te_observations(session, country="USA")

        assert "manufacturing_pmi" in result.columns
        assert "gdp_growth_rate" in result.columns
        assert len(result) == 1


# ---------------------------------------------------------------------------
# TestAssembleBondObservations
# ---------------------------------------------------------------------------


class TestAssembleBondObservations:
    """Tests for assemble_bond_observations()."""

    def test_when_no_rows_empty_dataframe_returned(self) -> None:
        """when no rows, empty DataFrame returned."""
        session = _mock_session_all([])
        result = assemble_bond_observations(session, country="USA")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_when_rows_present_pivot_by_maturity(self) -> None:
        """when rows with 2Y and 10Y maturities, columns are maturity strings."""
        import datetime

        date_val = datetime.date(2024, 6, 1)
        rows = [
            (date_val, "2Y", 4.0),
            (date_val, "10Y", 4.5),
        ]
        session = _mock_session_all(rows)
        result = assemble_bond_observations(session, country="USA")

        assert "2Y" in result.columns
        assert "10Y" in result.columns
        assert len(result) == 1


# ---------------------------------------------------------------------------
# TestAssembleFredSeries
# ---------------------------------------------------------------------------


class TestAssembleFredSeries:
    """Tests for assemble_fred_series()."""

    def test_when_no_rows_empty_dataframe_with_default_columns_returned(self) -> None:
        """when no rows, empty DataFrame with default FRED series IDs as columns."""
        session = _mock_session_all([])
        result = assemble_fred_series(session)

        assert isinstance(result, pd.DataFrame)
        assert result.empty
        assert list(result.columns) == FRED_SERIES_IDS

    def test_when_rows_present_pivot_by_series_id(self) -> None:
        """when rows present, columns are the series IDs."""
        import datetime

        date_val = datetime.date(2024, 1, 15)
        rows = [
            ("VIXCLS", date_val, 18.5),
            ("DGS10", date_val, 4.2),
        ]
        session = _mock_session_all(rows)
        result = assemble_fred_series(session, series_ids=["VIXCLS", "DGS10"])

        assert "VIXCLS" in result.columns
        assert "DGS10" in result.columns
        assert len(result) == 1


# ---------------------------------------------------------------------------
# TestFredSeriesIds
# ---------------------------------------------------------------------------


class TestFredSeriesIds:
    """Tests for the FRED_SERIES_IDS constant."""

    def test_when_imported_known_series_present(self) -> None:
        """when imported, DGS3MO, VIXCLS, T10Y2Y are present in FRED_SERIES_IDS."""
        assert "DGS3MO" in FRED_SERIES_IDS
        assert "VIXCLS" in FRED_SERIES_IDS
        assert "T10Y2Y" in FRED_SERIES_IDS
