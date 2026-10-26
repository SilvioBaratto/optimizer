"""CalendarsRepository upserts (real yfinance humanized-label columns)."""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from portopt_db.models.market_data.calendars import (
    EarningsCalendar,
    EconomicEventCalendar,
    IpoCalendar,
    SplitCalendar,
)
from portopt_db.repositories.market_data._calendar_timezones import (
    REGION_TIMEZONES,
    region_timezone,
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
    assert got.prior == "3.0%"


def test_ipo_real_labels(db_session) -> None:
    repo = CalendarsRepository(db_session)
    # yfinance's real IPO labels: "Company" / "Currency" (not "* Name").
    n = repo.upsert_ipos(
        [
            {
                "Symbol": "NEWCO",
                "Company": "New Co",
                "Exchange": "NMS",
                "Date": "2026-07-01",
                "Currency": "USD",
            }
        ]
    )
    db_session.flush()
    assert n == 1

    got = db_session.query(IpoCalendar).one()
    assert got.ipo_date == dt.date(2026, 7, 1)
    assert got.company_name == "New Co"
    assert got.exchange == "NMS"
    assert got.currency == "USD"


def test_splits_real_labels(db_session) -> None:
    repo = CalendarsRepository(db_session)
    # Real labels: "Company"; ratio is Old Share Worth : Share Worth (1-for-5).
    n = repo.upsert_splits(
        [
            {
                "Symbol": "SPLIT",
                "Company": "Split Co",
                "Payable On": "2026-08-15",
                "Optionable": True,
                "Old Share Worth": 5,
                "Share Worth": 1,
            }
        ]
    )
    db_session.flush()
    assert n == 1

    got = db_session.query(SplitCalendar).one()
    assert got.split_date == dt.date(2026, 8, 15)
    assert got.company_name == "Split Co"
    assert got.ratio == "5:1"


@pytest.mark.parametrize(
    ("region", "instant_utc", "local_date"),
    [
        # Japan CPI is released 08:30 JST, which Yahoo reports as 23:30Z the day before.
        ("JP", "2026-11-19 23:30", dt.date(2026, 11, 20)),
        ("KR", "2026-11-04 23:00", dt.date(2026, 11, 5)),
        ("AU", "2026-02-18 22:00", dt.date(2026, 2, 19)),
        ("NZ", "2026-09-16 21:45", dt.date(2026, 9, 17)),
        # Same-day releases stay on their UTC date.
        ("US", "2026-09-11 12:30", dt.date(2026, 9, 11)),
        ("GB", "2026-09-11 06:00", dt.date(2026, 9, 11)),
    ],
)
def test_economic_event_date_is_the_local_release_date(
    db_session, region: str, instant_utc: str, local_date: dt.date
) -> None:
    repo = CalendarsRepository(db_session)
    repo.upsert_economic_events(
        [
            {
                "Event": "CPI",
                "Region": region,
                "Event Time": pd.Timestamp(instant_utc, tz="UTC"),
            }
        ]
    )
    db_session.flush()

    assert db_session.query(EconomicEventCalendar).one().event_date == local_date


def test_economic_event_at_utc_midnight_is_a_date_only_placeholder(
    db_session,
) -> None:
    # Yahoo stamps events with no release time at 00:00Z; converting that to a
    # western zone would shift the event to the previous day.
    repo = CalendarsRepository(db_session)
    repo.upsert_economic_events(
        [
            {
                "Event": "Bank Holiday",
                "Region": "AR",
                "Event Time": pd.Timestamp("2026-09-11 00:00", tz="UTC"),
            }
        ]
    )
    db_session.flush()

    got = db_session.query(EconomicEventCalendar).one()
    assert got.event_date == dt.date(2026, 9, 11)


def test_economic_event_with_unknown_region_keeps_the_utc_date(db_session) -> None:
    repo = CalendarsRepository(db_session)
    repo.upsert_economic_events(
        [{"Event": "CPI", "Event Time": pd.Timestamp("2026-11-19 23:30", tz="UTC")}]
    )
    db_session.flush()

    got = db_session.query(EconomicEventCalendar).one()
    assert got.country == "?"
    assert got.event_date == dt.date(2026, 11, 19)


def test_region_timezones_all_resolve_to_known_zones() -> None:
    for region, name in REGION_TIMEZONES.items():
        assert ZoneInfo(name) is not None, region
        assert region_timezone(region) == ZoneInfo(name)
    assert region_timezone(" jp ") == ZoneInfo("Asia/Tokyo")
    assert region_timezone("?") is None
    assert region_timezone(None) is None


def _utc_today() -> dt.date:
    return dt.datetime.now(dt.timezone.utc).date()


def _earnings_dates(db_session, ticker: str) -> list[dt.date]:
    rows = db_session.query(EarningsCalendar).filter_by(ticker=ticker).all()
    return sorted(r.event_date for r in rows)


def test_rescheduled_earnings_replace_the_stale_upcoming_row(db_session) -> None:
    repo = CalendarsRepository(db_session)
    old = _utc_today() + dt.timedelta(days=3)
    new = _utc_today() + dt.timedelta(days=5)
    repo.upsert_earnings(
        [{"Symbol": "AAPL", "Event Start Date": str(old), "EPS Estimate": 1.5}]
    )
    db_session.flush()

    n = repo.upsert_earnings(
        [{"Symbol": "AAPL", "Event Start Date": str(new), "EPS Estimate": 1.6}]
    )
    db_session.flush()

    assert n == 1
    assert _earnings_dates(db_session, "AAPL") == [new]


def test_earnings_rerun_with_unchanged_dates_deletes_nothing(db_session) -> None:
    repo = CalendarsRepository(db_session)
    rows = [
        {
            "Symbol": "AAPL",
            "Event Start Date": str(_utc_today() + dt.timedelta(days=3)),
        },
        {
            "Symbol": "AAPL",
            "Event Start Date": str(_utc_today() + dt.timedelta(days=80)),
        },
    ]
    repo.upsert_earnings(rows)
    repo.upsert_earnings(rows)
    db_session.flush()

    assert len(_earnings_dates(db_session, "AAPL")) == 2


def test_earnings_reschedule_keeps_realized_history_and_other_tickers(
    db_session,
) -> None:
    repo = CalendarsRepository(db_session)
    today = _utc_today()
    past_reported = today - dt.timedelta(days=20)
    past_unreported = today - dt.timedelta(days=3)
    repo.upsert_earnings(
        [
            {
                "Symbol": "AAPL",
                "Event Start Date": str(past_reported),
                "Reported EPS": 1.2,
            },
            {"Symbol": "AAPL", "Event Start Date": str(past_unreported)},
            {
                "Symbol": "AAPL",
                "Event Start Date": str(today + dt.timedelta(days=2)),
                "Reported EPS": 1.4,
            },
            {"Symbol": "AAPL", "Event Start Date": str(today + dt.timedelta(days=3))},
            # Not in the next payload (e.g. cut by the fetch's row cap): untouched.
            {"Symbol": "MSFT", "Event Start Date": str(today + dt.timedelta(days=3))},
        ]
    )
    db_session.flush()

    new = today + dt.timedelta(days=5)
    repo.upsert_earnings([{"Symbol": "AAPL", "Event Start Date": str(new)}])
    db_session.flush()

    assert _earnings_dates(db_session, "AAPL") == [
        past_reported,
        past_unreported,
        today + dt.timedelta(days=2),
        new,
    ]
    assert _earnings_dates(db_session, "MSFT") == [today + dt.timedelta(days=3)]


def test_rescheduled_ipo_replaces_the_stale_upcoming_row(db_session) -> None:
    repo = CalendarsRepository(db_session)
    old = _utc_today() + dt.timedelta(days=10)
    new = _utc_today() + dt.timedelta(days=17)
    repo.upsert_ipos([{"Symbol": "NEWCO", "Date": str(old)}])
    db_session.flush()

    repo.upsert_ipos([{"Symbol": "NEWCO", "Date": str(new)}])
    db_session.flush()

    assert [r.ipo_date for r in db_session.query(IpoCalendar).all()] == [new]


def test_rescheduled_split_replaces_the_stale_upcoming_row(db_session) -> None:
    repo = CalendarsRepository(db_session)
    old = _utc_today() + dt.timedelta(days=10)
    new = _utc_today() + dt.timedelta(days=12)
    repo.upsert_splits([{"Symbol": "SPLIT", "Payable On": str(old)}])
    db_session.flush()

    repo.upsert_splits([{"Symbol": "SPLIT", "Payable On": str(new)}])
    db_session.flush()

    assert [r.split_date for r in db_session.query(SplitCalendar).all()] == [new]


def test_economic_events_of_one_series_are_never_treated_as_reschedules(
    db_session,
) -> None:
    # The same (event, country) recurs inside the fetch window and the feed is
    # capped newest-first, so a row missing from a payload says nothing about it.
    repo = CalendarsRepository(db_session)
    first = _utc_today() + dt.timedelta(days=10)
    second = _utc_today() + dt.timedelta(days=40)
    repo.upsert_economic_events(
        [{"Event": "CPI", "Region": "US", "Event Time": str(first)}]
    )
    repo.upsert_economic_events(
        [{"Event": "CPI", "Region": "US", "Event Time": str(second)}]
    )
    db_session.flush()

    got = db_session.query(EconomicEventCalendar).all()
    assert sorted(r.event_date for r in got) == [first, second]


def _economic_payload(event: str, region: str, instant_utc: str) -> dict[str, object]:
    return {
        "Event": event,
        "Region": region,
        "Event Time": pd.Timestamp(instant_utc, tz="UTC"),
    }


def _stored_economic_events(db_session) -> list[tuple[str, str, dt.date]]:
    return sorted(
        (row.event, row.country, row.event_date)
        for row in db_session.query(EconomicEventCalendar).all()
    )


@pytest.mark.parametrize(
    ("region", "instant_utc", "legacy_date", "local_date"),
    [
        # East of UTC: the old UTC-date key sits one day before the local date.
        ("JP", "2026-11-19 23:30", dt.date(2026, 11, 19), dt.date(2026, 11, 20)),
        # West of UTC: a late-evening release was keyed one day after its local date.
        ("US", "2026-11-20 02:00", dt.date(2026, 11, 20), dt.date(2026, 11, 19)),
    ],
)
def test_economic_event_keyed_on_its_utc_date_is_replaced_not_duplicated(
    db_session,
    region: str,
    instant_utc: str,
    legacy_date: dt.date,
    local_date: dt.date,
) -> None:
    # Rows ingested before events were keyed on the local release date sit under
    # the UTC date; the refetch window re-sends the same events, which must
    # replace those rows instead of living next to them forever.
    db_session.add(
        EconomicEventCalendar(
            event="CPI", country=region, event_date=legacy_date, prior="old"
        )
    )
    db_session.flush()

    repo = CalendarsRepository(db_session)
    payload = {**_economic_payload("CPI", region, instant_utc), "Last": "new"}
    repo.upsert_economic_events([payload])
    repo.upsert_economic_events([payload])  # a second refetch changes nothing
    db_session.flush()

    assert _stored_economic_events(db_session) == [("CPI", region, local_date)]
    assert db_session.query(EconomicEventCalendar).one().prior == "new"


def test_economic_event_legacy_cleanup_leaves_other_events_alone(db_session) -> None:
    legacy_date = dt.date(2026, 11, 19)
    for event, country in [("CPI", "JP"), ("GDP", "JP"), ("CPI", "KR")]:
        db_session.add(
            EconomicEventCalendar(event=event, country=country, event_date=legacy_date)
        )
    db_session.flush()

    CalendarsRepository(db_session).upsert_economic_events(
        [_economic_payload("CPI", "JP", "2026-11-19 23:30")]
    )
    db_session.flush()

    assert _stored_economic_events(db_session) == [
        ("CPI", "JP", dt.date(2026, 11, 20)),
        ("CPI", "KR", legacy_date),
        ("GDP", "JP", legacy_date),
    ]


def test_economic_event_legacy_cleanup_keeps_a_genuine_occurrence_in_the_payload(
    db_session,
) -> None:
    # A daily event: the 11-19 occurrence (12:00 JST) legitimately owns the key
    # that the 11-20 occurrence (08:30 JST = 23:30Z) used to be stored under.
    CalendarsRepository(db_session).upsert_economic_events(
        [
            _economic_payload("Bond Auction", "JP", "2026-11-19 03:00"),
            _economic_payload("Bond Auction", "JP", "2026-11-19 23:30"),
        ]
    )
    db_session.flush()

    assert _stored_economic_events(db_session) == [
        ("Bond Auction", "JP", dt.date(2026, 11, 19)),
        ("Bond Auction", "JP", dt.date(2026, 11, 20)),
    ]


def test_economic_event_legacy_cleanup_spans_delete_chunks(db_session) -> None:
    legacy_date = dt.date(2026, 11, 19)
    names = [f"Indicator {i}" for i in range(400)]
    for name in names:
        db_session.add(
            EconomicEventCalendar(event=name, country="JP", event_date=legacy_date)
        )
    db_session.flush()

    CalendarsRepository(db_session).upsert_economic_events(
        [_economic_payload(name, "JP", "2026-11-19 23:30") for name in names]
    )
    db_session.flush()

    assert _stored_economic_events(db_session) == sorted(
        (name, "JP", dt.date(2026, 11, 20)) for name in names
    )
