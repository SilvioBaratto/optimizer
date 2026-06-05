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
