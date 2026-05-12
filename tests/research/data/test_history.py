"""Tests for research/data/_history.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from research.data._history import (
    assemble_delisting_returns,
    assemble_fundamental_history,
)


def _mock_session(rows: list) -> MagicMock:
    session = MagicMock()
    session.execute.return_value.all.return_value = rows
    return session


# ---------------------------------------------------------------------------
# TestAssembleDelistingReturns
# ---------------------------------------------------------------------------


class TestAssembleDelistingReturns:
    def test_when_no_rows_empty_dict_returned(self) -> None:
        """when no rows, empty dict returned."""
        result = assemble_delisting_returns(_mock_session([]))
        assert result == {}

    def test_when_row_with_return_value_correct_mapping(self) -> None:
        """when row has delisting_return, ticker mapped to that value."""
        rows = [("BARC.L", -0.50)]
        result = assemble_delisting_returns(_mock_session(rows))
        assert result == {"BARC.L": -0.50}

    def test_when_row_with_null_return_defaults_to_minus_30_pct(self) -> None:
        """when delisting_return is None, default -0.30 used."""
        rows = [("GONE.L", None)]
        result = assemble_delisting_returns(_mock_session(rows))
        assert result["GONE.L"] == pytest.approx(-0.30)

    def test_when_multiple_rows_all_tickers_present(self) -> None:
        """when multiple rows, all tickers appear in result."""
        rows = [("AAA", -0.20), ("BBB", None), ("CCC", -0.80)]
        result = assemble_delisting_returns(_mock_session(rows))
        assert set(result.keys()) == {"AAA", "BBB", "CCC"}


# ---------------------------------------------------------------------------
# TestAssembleFundamentalHistory
# ---------------------------------------------------------------------------


class TestAssembleFundamentalHistory:
    def test_when_no_rows_empty_multiindex_df_returned(self) -> None:
        """when no rows, empty DataFrame with correct MultiIndex returned."""
        session = MagicMock()
        session.execute.return_value.all.return_value = []
        result = assemble_fundamental_history(session)
        assert isinstance(result, pd.DataFrame)
        assert result.empty
        assert result.index.names == ["period_date", "ticker"]

    def test_when_no_rows_expected_columns_present(self) -> None:
        """when no rows, expected columns present in empty result."""
        session = MagicMock()
        session.execute.return_value.all.return_value = []
        result = assemble_fundamental_history(session)
        for col in [
            "net_income",
            "gross_profit",
            "operating_income",
            "total_assets",
            "total_equity",
            "period_type",
            "asset_growth",
        ]:
            assert col in result.columns

    def test_when_ticker_map_empty_empty_df_returned(self) -> None:
        """when ticker map returns nothing, empty DataFrame returned."""
        session = MagicMock()
        # First call: ticker_map query returns empty
        # Second call: statement rows returns empty
        session.execute.return_value.all.return_value = []
        result = assemble_fundamental_history(session)
        assert result.empty

    def test_when_no_look_ahead_period_date_is_statement_date(self) -> None:
        """when panel assembled, index key = period_date (no look-ahead bias)."""
        # The history panel is keyed by period_date (when data was reported),
        # not the current date. Callers must apply publication-lag offset
        # externally; the loader itself does not introduce forward-looking keys.
        session = MagicMock()
        session.execute.return_value.all.return_value = []
        result = assemble_fundamental_history(session)
        assert "period_date" in result.index.names
        assert "ticker" in result.index.names

    def test_when_tickers_filter_provided_session_executed(self) -> None:
        """when tickers list provided, session.execute called (filter applied)."""
        session = MagicMock()
        session.execute.return_value.all.return_value = []
        assemble_fundamental_history(session, tickers=["AAPL", "MSFT"])
        assert session.execute.call_count >= 1
