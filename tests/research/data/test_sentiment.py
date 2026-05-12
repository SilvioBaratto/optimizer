"""Tests for research/data/_sentiment.py."""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock

import pandas as pd

from research.data._sentiment import assemble_sentiment


def _mock_session(rows: list) -> MagicMock:
    session = MagicMock()
    session.execute.return_value.all.return_value = rows
    return session


class TestAssembleSentiment:
    def test_when_no_rows_empty_dataframe_returned(self) -> None:
        """when no rows, empty dataframe returned."""
        result = assemble_sentiment(_mock_session([]))
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_when_all_scores_none_empty_dataframe_returned(self) -> None:
        """when all sentiment_score values are None, empty dataframe returned."""
        rows = [(datetime.date(2024, 1, 1), "USA", None)]
        result = assemble_sentiment(_mock_session(rows))
        assert result.empty

    def test_when_rows_present_date_index_and_country_columns(self) -> None:
        """when rows present, result indexed by date with country columns."""
        rows = [
            (datetime.date(2024, 1, 1), "USA", 0.5),
            (datetime.date(2024, 1, 1), "Germany", -0.2),
        ]
        result = assemble_sentiment(_mock_session(rows))
        assert isinstance(result.index, pd.DatetimeIndex)
        assert "USA" in result.columns
        assert "Germany" in result.columns

    def test_when_single_row_correct_value_stored(self) -> None:
        """when single row, correct sentiment score stored in result."""
        rows = [(datetime.date(2024, 3, 15), "France", 0.75)]
        result = assemble_sentiment(_mock_session(rows))
        assert result.loc[pd.Timestamp("2024-03-15"), "France"] == 0.75

    def test_when_multiple_dates_result_sorted_by_date(self) -> None:
        """when multiple dates, result sorted ascending by date."""
        rows = [
            (datetime.date(2024, 1, 3), "USA", 0.1),
            (datetime.date(2024, 1, 1), "USA", 0.3),
            (datetime.date(2024, 1, 2), "USA", 0.2),
        ]
        result = assemble_sentiment(_mock_session(rows))
        assert list(result.index) == sorted(result.index)

    def test_when_start_date_provided_query_executed_with_filter(self) -> None:
        """when start_date provided, session.execute called (filter applied)."""
        session = _mock_session([])
        assemble_sentiment(session, start_date=datetime.date(2024, 1, 1))
        session.execute.assert_called_once()

    def test_when_duplicate_country_on_same_date_first_value_kept(self) -> None:
        """when duplicate (date, country) pairs, first value kept."""
        rows = [
            (datetime.date(2024, 1, 1), "USA", 0.5),
            (datetime.date(2024, 1, 1), "USA", 0.9),
        ]
        result = assemble_sentiment(_mock_session(rows))
        assert result.loc[pd.Timestamp("2024-01-01"), "USA"] == 0.5
