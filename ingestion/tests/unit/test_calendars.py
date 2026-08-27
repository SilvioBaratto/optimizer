"""SPEC B2 — market-wide calendars (earnings / IPO / splits / economic events).

Client parsing patches ``yf.Calendars``; the repo runs against real SQLite; the
bulk service is driven with a mocked repo + session.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
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


class TestCalendarsClient:
    def test_fetch_earnings_returns_records(self) -> None:
        df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT"],
                "companyshortname": ["Apple", "Microsoft"],
                "startdatetime": ["2026-10-30", "2026-10-28"],
                "epsestimate": [1.5, 3.1],
            }
        )
        cal = SimpleNamespace(earnings_calendar=df)
        with patch(
            "app.services.market_data.yfinance.market.calendars.yf.Calendars",
            return_value=cal,
        ):
            out = _client().fetch_earnings()

        assert out is not None
        assert {r["ticker"] for r in out} == {"AAPL", "MSFT"}

    def test_empty_dataframe_returns_empty_list(self) -> None:
        cal = SimpleNamespace(splits_calendar=pd.DataFrame())
        with patch(
            "app.services.market_data.yfinance.market.calendars.yf.Calendars",
            return_value=cal,
        ):
            assert _client().fetch_splits() == []


class TestRepository:
    def test_earnings_upsert_idempotent(self, db_session) -> None:
        repo = CalendarsRepository(db_session)
        rows = [
            {
                "ticker": "AAPL",
                "companyshortname": "Apple",
                "startdatetime": "2026-10-30",
                "epsestimate": 1.5,
                "epsactual": 1.6,
                "epssurprisepct": 6.7,
            }
        ]
        assert repo.upsert_earnings(rows) == 1
        assert repo.upsert_earnings(rows) == 1  # idempotent
        db_session.flush()

        from app.models.market_data.calendars import EarningsCalendar

        got = db_session.query(EarningsCalendar).all()
        assert len(got) == 1
        assert got[0].ticker == "AAPL"
        assert float(got[0].eps_estimate) == 1.5

    def test_economic_events_upsert(self, db_session) -> None:
        repo = CalendarsRepository(db_session)
        n = repo.upsert_economic_events(
            [
                {
                    "event": "CPI",
                    "country": "US",
                    "startdatetime": "2026-09-11",
                    "actual": "3.2%",
                    "forecast": "3.1%",
                }
            ]
        )
        db_session.flush()
        assert n == 1


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
