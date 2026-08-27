"""CalendarsRepository upserts (real yfinance humanized-label columns)."""

from __future__ import annotations

import datetime as dt

from portopt_db.models.market_data.calendars import (
    EarningsCalendar,
    EconomicEventCalendar,
    IpoCalendar,
    SplitCalendar,
)
from portopt_db.repositories.market_data.calendars_repository import CalendarsRepository


def test_earnings_real_labels(db_session) -> None:
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

    got = db_session.query(EarningsCalendar).all()
    assert len(got) == 1
    assert got[0].ticker == "AAPL"
    assert got[0].event_date == dt.date(2026, 10, 30)
    assert got[0].company_name == "Apple"
    assert float(got[0].eps_estimate) == 1.5


def test_economic_real_labels_and_region(db_session) -> None:
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

    got = db_session.query(EconomicEventCalendar).one()
    assert got.country == "US"  # from "Region", not the "?" fallback
    assert got.event_date == dt.date(2026, 9, 11)
    assert got.forecast == "3.1%"
    assert got.prior == "3.0%"


def test_ipo_real_labels(db_session) -> None:
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

    got = db_session.query(IpoCalendar).one()
    assert got.ipo_date == dt.date(2026, 7, 1)
    assert got.company_name == "New Co"
    assert got.currency == "USD"
    assert got.price_range == "18.00"


def test_splits_real_labels(db_session) -> None:
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

    got = db_session.query(SplitCalendar).one()
    assert got.split_date == dt.date(2026, 8, 15)
    assert got.company_name == "Split Co"
    assert got.ratio is None  # Optionable is not a ratio
