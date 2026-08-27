"""SPEC B2 — market-wide calendars (earnings / IPO / splits / economic events).

Client parsing patches ``yf.Calendars``; the repo runs against real SQLite; the
bulk service is driven with a mocked repo + session.
"""

from __future__ import annotations

import datetime as dt
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pandas as pd

from app.repositories.market_data.calendars_repository import CalendarsRepository
from app.services.market_data.calendars_service import run_calendars_fetch
from app.services.market_data.yfinance.market.calendars import CalendarsClient


def _client() -> CalendarsClient:
    return CalendarsClient(
        rate_limiter=MagicMock(),
        circuit_breaker=MagicMock(),
        default_max_retries=1,
    )


_CAL = "app.services.market_data.yfinance.market.calendars.yf.Calendars"


def _page(n: int, start: int = 0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Symbol": [f"T{i}" for i in range(start, start + n)],
            "Event Start Date": ["2026-10-30"] * n,
        }
    )


class TestCalendarsClient:
    def test_fetch_earnings_paginates_and_disables_most_active(self) -> None:
        cal = MagicMock()
        # full 100-row page, then a short page -> stop after two calls.
        cal.get_earnings_calendar.side_effect = [_page(100, 0), _page(3, 100)]
        with patch(_CAL, return_value=cal):
            out = _client().fetch_earnings()

        assert out is not None and len(out) == 103
        assert cal.get_earnings_calendar.call_count == 2
        k0 = cal.get_earnings_calendar.call_args_list[0].kwargs
        assert k0["offset"] == 0
        assert k0["limit"] == 100
        assert k0["filter_most_active"] is False
        assert cal.get_earnings_calendar.call_args_list[1].kwargs["offset"] == 100

    def test_short_first_page_stops_immediately(self) -> None:
        cal = MagicMock()
        cal.get_ipo_info_calendar.return_value = _page(5)
        with patch(_CAL, return_value=cal):
            out = _client().fetch_ipos()
        assert len(out) == 5
        assert cal.get_ipo_info_calendar.call_count == 1

    def test_empty_first_page_returns_empty_list(self) -> None:
        cal = MagicMock()
        cal.get_splits_calendar.return_value = pd.DataFrame()
        with patch(_CAL, return_value=cal):
            assert _client().fetch_splits() == []


class TestRepository:
    """Uses the humanized label columns yfinance actually emits (reset_index of
    the label-keyed DataFrames), not the raw includeFields names — matching the
    production shape that the earlier raw-name tests missed."""

    def test_earnings_real_labels(self, db_session) -> None:
        repo = CalendarsRepository(db_session)
        rows = [
            {
                "Symbol": "AAPL",
                "Company": "Apple",
                "Event Start Date": "2026-10-30",
                "EPS Estimate": 1.5,
                "Reported EPS": 1.6,
                "Surprise(%)": 6.7,
            }
        ]
        assert repo.upsert_earnings(rows) == 1
        assert repo.upsert_earnings(rows) == 1  # idempotent
        db_session.flush()

        from portopt_db.models.market_data.calendars import EarningsCalendar

        got = db_session.query(EarningsCalendar).all()
        assert len(got) == 1
        assert got[0].ticker == "AAPL"
        assert got[0].event_date == dt.date(2026, 10, 30)
        assert got[0].company_name == "Apple"
        assert float(got[0].eps_estimate) == 1.5

    def test_economic_real_labels_and_region(self, db_session) -> None:
        repo = CalendarsRepository(db_session)
        n = repo.upsert_economic_events(
            [
                {
                    "Event": "CPI",
                    "Region": "US",
                    "Event Time": "2026-09-11",
                    "Actual": "3.2%",
                    "Expected": "3.1%",
                    "Last": "3.0%",
                }
            ]
        )
        db_session.flush()
        assert n == 1

        from portopt_db.models.market_data.calendars import EconomicEventCalendar

        got = db_session.query(EconomicEventCalendar).one()
        assert got.country == "US"  # from "Region", not the "?" fallback
        assert got.event_date == dt.date(2026, 9, 11)
        assert got.forecast == "3.1%"
        assert got.prior == "3.0%"

    def test_ipo_real_labels(self, db_session) -> None:
        repo = CalendarsRepository(db_session)
        n = repo.upsert_ipos(
            [
                {
                    "Symbol": "NEWCO",
                    "Company Name": "New Co",
                    "Exchange": "NMS",
                    "Date": "2026-07-01",
                    "Price From": "18.00",
                    "Currency Name": "USD",
                    "Shares": 1_000_000,
                }
            ]
        )
        db_session.flush()
        assert n == 1

        from portopt_db.models.market_data.calendars import IpoCalendar

        got = db_session.query(IpoCalendar).one()
        assert got.ipo_date == dt.date(2026, 7, 1)
        assert got.company_name == "New Co"
        assert got.currency == "USD"
        assert got.price_range == "18.00"

    def test_splits_real_labels(self, db_session) -> None:
        repo = CalendarsRepository(db_session)
        n = repo.upsert_splits(
            [
                {
                    "Symbol": "SPLIT",
                    "Company Name": "Split Co",
                    "Payable On": "2026-08-15",
                    "Optionable": True,
                }
            ]
        )
        db_session.flush()
        assert n == 1

        from portopt_db.models.market_data.calendars import SplitCalendar

        got = db_session.query(SplitCalendar).one()
        assert got.split_date == dt.date(2026, 8, 15)
        assert got.company_name == "Split Co"
        assert got.ratio is None  # Optionable is not a ratio


def _fake_dbm() -> MagicMock:
    @contextmanager
    def _cm():
        yield MagicMock(name="session")

    dbm = MagicMock()
    dbm.get_session = _cm
    return dbm


class TestBulkFetch:
    def test_all_four_calendars_upserted(self) -> None:
        repo = MagicMock(name="repo")
        repo.upsert_earnings.return_value = 2
        repo.upsert_ipos.return_value = 1
        repo.upsert_splits.return_value = 3
        repo.upsert_economic_events.return_value = 4
        yf = MagicMock()
        yf.calendars.fetch_earnings.return_value = [{"ticker": "AAPL"}]
        yf.calendars.fetch_ipos.return_value = [{"ticker": "NEW"}]
        yf.calendars.fetch_splits.return_value = [{"ticker": "SPL"}]
        yf.calendars.fetch_economic_events.return_value = [{"event": "CPI"}]

        with (
            patch(
                "app.repositories.market_data.calendars_repository.CalendarsRepository",
                return_value=repo,
            ),
            patch("app.database.database_manager", _fake_dbm()),
        ):
            result = run_calendars_fetch(yf)

        assert result["counts"] == {
            "earnings": 2,
            "ipos": 1,
            "splits": 3,
            "economic_events": 4,
        }
        assert result["error_count"] == 0

    def test_one_calendar_failure_isolated(self) -> None:
        repo = MagicMock(name="repo")
        repo.upsert_earnings.return_value = 0
        repo.upsert_ipos.return_value = 0
        repo.upsert_splits.return_value = 0
        repo.upsert_economic_events.return_value = 0
        yf = MagicMock()
        yf.calendars.fetch_earnings.side_effect = RuntimeError("boom")
        yf.calendars.fetch_ipos.return_value = []
        yf.calendars.fetch_splits.return_value = []
        yf.calendars.fetch_economic_events.return_value = []

        with (
            patch(
                "app.repositories.market_data.calendars_repository.CalendarsRepository",
                return_value=repo,
            ),
            patch("app.database.database_manager", _fake_dbm()),
        ):
            result = run_calendars_fetch(yf)

        assert result["error_count"] == 1


class TestSchedulerStep:
    def test_composes_run_step(self) -> None:
        M = "app.services.jobs.scheduler"
        with (
            patch(f"{M}._run_step", return_value=True) as run_step,
            patch(
                "app.services.market_data.yfinance.get_yfinance_client",
                return_value=MagicMock(),
            ),
        ):
            from app.services.jobs.scheduler import _calendars_jobs, run_calendars_step

            assert run_calendars_step() is True

        assert run_step.call_args.args[0] == "calendars"
        assert run_step.call_args.args[1] is _calendars_jobs
