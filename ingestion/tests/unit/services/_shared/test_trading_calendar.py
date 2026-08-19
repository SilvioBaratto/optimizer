"""Unit tests for ``_shared/trading_calendar`` (issue #826).

NOTE on AC: this module has NO ``is_trading_day(date) -> bool`` function. The
public surface is ``parse_period_years``, ``get_expected_trading_sessions``
(backed by the real ``exchange_calendars`` library — pure computation, no
network), and ``has_sufficient_history``. The AC's 'weekend -> False / holiday
-> False / business day -> True' intent is exercised through
``get_expected_trading_sessions``: a one-year NYSE window yields ~252 sessions,
not 365 — proving weekends AND holidays are excluded by the calendar our code
delegates to.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from app.services._shared.trading_calendar import (
    get_expected_trading_sessions,
    has_sufficient_history,
    parse_period_years,
)


class TestParsePeriodYears:
    @pytest.mark.parametrize("period,years", [("5y", 5), ("1y", 1), ("10y", 10)])
    def test_when_year_period_then_parsed(self, period: str, years: int) -> None:
        assert parse_period_years(period) == years

    @pytest.mark.parametrize("period", ["6mo", "max", "1d", "ytd", ""])
    def test_when_non_year_period_then_none(self, period: str) -> None:
        assert parse_period_years(period) is None


class TestExpectedTradingSessions:
    _REF = date(2024, 12, 31)

    def test_when_unknown_exchange_then_none(self) -> None:
        assert get_expected_trading_sessions("Bogus Exchange", "1y", self._REF) is None

    def test_when_non_year_period_then_none(self) -> None:
        assert get_expected_trading_sessions("NYSE", "6mo", self._REF) is None

    def test_when_one_year_nyse_then_excludes_weekends_and_holidays(self) -> None:
        # A calendar year has 365/366 days; NYSE trades ~250-253 of them.
        # A count well below 260 proves weekends + holidays are excluded.
        sessions = get_expected_trading_sessions("NYSE", "1y", self._REF)
        assert sessions is not None
        assert 240 <= sessions <= 256

    def test_when_london_exchange_then_resolves_mic(self) -> None:
        sessions = get_expected_trading_sessions(
            "London Stock Exchange", "1y", self._REF
        )
        assert sessions is not None
        assert 240 <= sessions <= 256

    # ------------------------------------------------------------------
    # Uncovered branch: lines 54-56 — Feb-29 leap-year fallback (day=28)
    # ------------------------------------------------------------------

    def test_when_feb29_reference_date_then_falls_back_to_day28(self) -> None:
        # 2024-02-29 is a real leap day.  Subtracting 1 year raises ValueError
        # (2023 has no Feb 29), so the except branch replaces day with 28.
        result = get_expected_trading_sessions("NYSE", "1y", date(2024, 2, 29))
        # The fallback must produce a valid integer — not None.
        assert result is not None
        assert result > 0

    # ------------------------------------------------------------------
    # Uncovered branch: line 74 — start >= end returns None
    # ------------------------------------------------------------------

    def test_when_calendar_bounds_force_start_gte_end_then_none(self) -> None:
        # Construct a mock calendar whose first_session is far in the future,
        # forcing start > reference_date after the max(start, cal_start) clamp.
        # Use plain date objects to exercise the `not hasattr(..., "date")`
        # branch in the hasattr guard.
        future_date = date(2099, 1, 1)
        mock_cal = MagicMock()
        mock_cal.first_session = future_date  # plain date — no .date() method
        mock_cal.last_session = date(2099, 12, 31)

        with patch(
            "app.services._shared.trading_calendar.xcals.get_calendar",
            return_value=mock_cal,
        ):
            result = get_expected_trading_sessions("NYSE", "1y", self._REF)

        assert result is None

    # ------------------------------------------------------------------
    # Uncovered branch: lines 77-84 — xcals.get_calendar raises → None
    # ------------------------------------------------------------------

    def test_when_get_calendar_raises_then_returns_none(self) -> None:
        with patch(
            "app.services._shared.trading_calendar.xcals.get_calendar",
            side_effect=Exception("boom"),
        ):
            result = get_expected_trading_sessions("NYSE", "1y", self._REF)

        assert result is None


class TestHasSufficientHistory:
    def test_when_exchange_none_then_skipped_true(self) -> None:
        assert has_sufficient_history(0, None, "5y") == (True, None, None)

    def test_when_unknown_exchange_then_skipped_true(self) -> None:
        assert has_sufficient_history(0, "Bogus", "5y") == (True, None, None)

    def test_when_non_year_period_then_skipped_true(self) -> None:
        assert has_sufficient_history(0, "NYSE", "6mo") == (True, None, None)

    def test_when_rows_meet_minimum_then_sufficient(self) -> None:
        sufficient, expected, minimum = has_sufficient_history(10_000, "NYSE", "5y")
        assert expected is not None and minimum is not None
        assert sufficient is True
        assert minimum == int(expected * 0.95)

    def test_when_rows_below_minimum_then_insufficient(self) -> None:
        sufficient, expected, minimum = has_sufficient_history(1, "NYSE", "5y")
        assert sufficient is False
        assert expected is not None
        assert minimum is not None and minimum > 1
