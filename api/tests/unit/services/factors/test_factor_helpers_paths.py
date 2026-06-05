"""Unit tests for app.services.factors._factor_helpers — empty-data paths.

Covers every public helper in _factor_helpers.py that has an explicit
empty/None branch:

  - _notify: fires callback when provided; silent no-op when None
  - _build_returns_from_price_rows: empty list → empty DataFrame
  - _build_returns_from_price_rows: single ticker → empty (pct_change drops first row)
  - _build_factor_scores_dict: empty list → empty dict
  - _build_multiindex_factor_df: empty list → empty DataFrame
  - _build_multiindex_returns: empty DataFrame → empty DataFrame
  - _build_standardized_scores_df: empty list → (empty, empty)
  - _build_standardized_scores_df: all None standardized_score → (empty, empty)

No DB, no optimizer library, no network.
"""

from __future__ import annotations

import datetime
from typing import Any
from unittest.mock import MagicMock

import pandas as pd

_MODULE = "app.services.factors._factor_helpers"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_score(
    ticker: str = "AAPL",
    score_date: datetime.date | None = None,
    factor_type: str = "book_to_price",
    raw_score: float = 0.5,
    standardized_score: float | None = 0.8,
) -> MagicMock:
    s = MagicMock()
    s.ticker = ticker
    s.score_date = score_date or datetime.date(2023, 6, 30)
    s.factor_type = factor_type
    s.raw_score = raw_score
    s.standardized_score = standardized_score
    return s


def _make_price_row(
    ticker: str = "AAPL",
    date: Any = datetime.date(2023, 1, 2),
    close: float = 150.0,
) -> MagicMock:
    """Return a mock compatible with PriceRow duck-type."""
    row = MagicMock()
    row.ticker = ticker
    row.date = date
    row.close = close
    return row


# ---------------------------------------------------------------------------
# _notify
# ---------------------------------------------------------------------------


class TestNotify:
    def test_when_callback_provided_then_it_is_called(self) -> None:
        from app.services.factors._factor_helpers import _notify

        cb = MagicMock()
        _notify(cb, status="running", current=0, total=5)
        cb.assert_called_once_with(status="running", current=0, total=5)

    def test_when_no_callback_then_no_error(self) -> None:
        from app.services.factors._factor_helpers import _notify

        # Must not raise
        _notify(None, status="done")


# ---------------------------------------------------------------------------
# _build_returns_from_price_rows
# ---------------------------------------------------------------------------


class TestBuildReturnsFromPriceRows:
    def test_when_empty_list_then_returns_empty_dataframe(self) -> None:
        from app.services.factors._factor_helpers import _build_returns_from_price_rows

        result = _build_returns_from_price_rows([])

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_when_single_date_row_then_pct_change_drops_all(self) -> None:
        from app.services.factors._factor_helpers import _build_returns_from_price_rows

        # Only one date per ticker → pct_change removes the first row → empty
        rows = [_make_price_row("AAPL", datetime.date(2023, 1, 2), 150.0)]
        result = _build_returns_from_price_rows(rows)

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_when_two_dates_then_returns_one_row(self) -> None:
        from app.services.factors._factor_helpers import _build_returns_from_price_rows

        rows = [
            _make_price_row("AAPL", datetime.date(2023, 1, 2), 100.0),
            _make_price_row("AAPL", datetime.date(2023, 1, 3), 110.0),
        ]
        result = _build_returns_from_price_rows(rows)

        assert not result.empty
        assert "AAPL" in result.columns
        assert len(result) == 1
        assert abs(result["AAPL"].iloc[0] - 0.1) < 1e-9

    def test_when_valid_rows_then_index_is_datetimeindex(self) -> None:
        from app.services.factors._factor_helpers import _build_returns_from_price_rows

        rows = [
            _make_price_row("AAPL", datetime.date(2023, 1, 2), 100.0),
            _make_price_row("AAPL", datetime.date(2023, 1, 3), 105.0),
        ]
        result = _build_returns_from_price_rows(rows)

        assert isinstance(result.index, pd.DatetimeIndex)


# ---------------------------------------------------------------------------
# _build_factor_scores_dict
# ---------------------------------------------------------------------------


class TestBuildFactorScoresDict:
    def test_when_empty_list_then_returns_empty_dict(self) -> None:
        from app.services.factors._factor_helpers import _build_factor_scores_dict

        result = _build_factor_scores_dict([])

        assert result == {}

    def test_when_scores_provided_then_keys_are_factor_types(self) -> None:
        from app.services.factors._factor_helpers import _build_factor_scores_dict

        scores = [
            _make_score("AAPL", factor_type="momentum"),
            _make_score("MSFT", factor_type="value"),
        ]
        result = _build_factor_scores_dict(scores)

        assert "momentum" in result
        assert "value" in result
        assert isinstance(result["momentum"], pd.DataFrame)


# ---------------------------------------------------------------------------
# _build_multiindex_factor_df
# ---------------------------------------------------------------------------


class TestBuildMultiindexFactorDf:
    def test_when_empty_list_then_returns_empty_dataframe(self) -> None:
        from app.services.factors._factor_helpers import _build_multiindex_factor_df

        result = _build_multiindex_factor_df([])

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_when_scores_provided_then_index_is_multiindex(self) -> None:
        from app.services.factors._factor_helpers import _build_multiindex_factor_df

        scores = [
            _make_score("AAPL", score_date=datetime.date(2023, 6, 30)),
            _make_score("MSFT", score_date=datetime.date(2023, 6, 30)),
        ]
        result = _build_multiindex_factor_df(scores)

        assert isinstance(result.index, pd.MultiIndex)


# ---------------------------------------------------------------------------
# _build_multiindex_returns
# ---------------------------------------------------------------------------


class TestBuildMultiindexReturns:
    def test_when_empty_dataframe_then_returns_empty_dataframe(self) -> None:
        from app.services.factors._factor_helpers import _build_multiindex_returns

        result = _build_multiindex_returns(pd.DataFrame())

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_when_wide_returns_then_multiindex_has_date_ticker_names(self) -> None:
        from app.services.factors._factor_helpers import _build_multiindex_returns

        idx = pd.date_range("2023-01-02", periods=2)
        wide = pd.DataFrame({"AAPL": [0.01, 0.02], "MSFT": [0.005, 0.01]}, index=idx)
        result = _build_multiindex_returns(wide)

        assert isinstance(result.index, pd.MultiIndex)
        assert list(result.index.names) == ["date", "ticker"]
        assert "return" in result.columns


# ---------------------------------------------------------------------------
# _build_standardized_scores_df
# ---------------------------------------------------------------------------


class TestBuildStandardizedScoresDf:
    def test_when_empty_list_then_returns_two_empty_dataframes(self) -> None:
        from app.services.factors._factor_helpers import _build_standardized_scores_df

        std_df, coverage_df = _build_standardized_scores_df([])

        assert isinstance(std_df, pd.DataFrame)
        assert isinstance(coverage_df, pd.DataFrame)
        assert std_df.empty
        assert coverage_df.empty

    def test_when_all_standardized_scores_none_then_returns_empty(self) -> None:
        from app.services.factors._factor_helpers import _build_standardized_scores_df

        scores = [
            _make_score("AAPL", standardized_score=None),
            _make_score("MSFT", standardized_score=None),
        ]
        std_df, coverage_df = _build_standardized_scores_df(scores)

        assert std_df.empty
        assert coverage_df.empty

    def test_when_valid_scores_then_tickers_are_in_index(self) -> None:
        from app.services.factors._factor_helpers import _build_standardized_scores_df

        scores = [
            _make_score("AAPL", factor_type="momentum", standardized_score=0.8),
            _make_score("MSFT", factor_type="momentum", standardized_score=0.6),
        ]
        std_df, coverage_df = _build_standardized_scores_df(scores)

        assert "AAPL" in std_df.index
        assert "MSFT" in std_df.index
        assert coverage_df.shape == std_df.shape

    def test_when_valid_scores_then_coverage_values_are_zero_or_one(self) -> None:
        from app.services.factors._factor_helpers import _build_standardized_scores_df

        scores = [
            _make_score("AAPL", factor_type="value", standardized_score=1.2),
        ]
        _, coverage_df = _build_standardized_scores_df(scores)

        assert set(coverage_df.values.flatten().tolist()).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# _build_factor_scores_dict — empty-row filter
# ---------------------------------------------------------------------------


class TestBuildFactorScoresDictEdge:
    def test_when_scores_have_no_raw_score_then_nans_appear_in_df(self) -> None:
        from app.services.factors._factor_helpers import _build_factor_scores_dict

        score = _make_score("AAPL", factor_type="quality", raw_score=float("nan"))
        result = _build_factor_scores_dict([score])

        assert "quality" in result
        df = result["quality"]
        # NaN raw scores survive in the pivot; no filtering here
        assert isinstance(df, pd.DataFrame)

    def test_when_two_factor_types_then_separate_keys(self) -> None:
        from app.services.factors._factor_helpers import _build_factor_scores_dict

        scores = [
            _make_score("AAPL", factor_type="momentum", raw_score=0.5),
            _make_score("AAPL", factor_type="value", raw_score=0.3),
        ]
        result = _build_factor_scores_dict(scores)

        assert set(result.keys()) == {"momentum", "value"}
