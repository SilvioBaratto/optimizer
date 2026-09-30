"""MarketSummaryRepository upsert and read methods."""

from __future__ import annotations

import datetime as dt

from portopt_db.models.market_data.market_summary import MarketSummary
from portopt_db.repositories.market_data.market_summary_repository import (
    MarketSummaryRepository,
)

_AS_OF = dt.date(2026, 8, 27)
_AS_OF_2 = dt.date(2026, 8, 28)

_ROW = {
    "symbol": "^GSPC",
    "short_name": "S&P 500",
    "price": 5600.5,
    "change": 12.3,
    "change_percent": 0.42,
    "previous_close": 5588.2,
    "market_state": "REGULAR",
}


def test_upsert_idempotent(db_session) -> None:
    """Upserting the same row twice must still result in exactly one row."""
    repo = MarketSummaryRepository(db_session)
    rows = [_ROW]
    assert repo.upsert_summaries("US", _AS_OF, rows) == 1
    assert repo.upsert_summaries("US", _AS_OF, rows) == 1  # idempotent
    db_session.flush()

    got = db_session.query(MarketSummary).all()
    assert len(got) == 1
    assert float(got[0].price) == 5600.5


def test_get_summaries_returns_rows_for_market_and_date(db_session) -> None:
    """get_summaries must return all rows for the given market/date pair."""
    repo = MarketSummaryRepository(db_session)
    repo.upsert_summaries("us_market", _AS_OF, [_ROW, {**_ROW, "symbol": "^DJI"}])
    repo.upsert_summaries("us_market", _AS_OF_2, [_ROW])
    db_session.flush()

    rows = repo.get_summaries("us_market", _AS_OF)
    assert len(rows) == 2
    symbols = {r.symbol for r in rows}
    assert symbols == {"^GSPC", "^DJI"}


def test_get_summaries_filters_by_market(db_session) -> None:
    """get_summaries must not return rows belonging to a different market."""
    repo = MarketSummaryRepository(db_session)
    # Seed both markets on the same date so an absent filter would leak the FTSE row.
    repo.upsert_summaries("gb_market", _AS_OF, [{**_ROW, "symbol": "^FTSE"}])
    repo.upsert_summaries("us_market", _AS_OF, [_ROW])
    db_session.flush()

    rows = repo.get_summaries("us_market", _AS_OF)
    assert len(rows) == 1
    assert rows[0].symbol == "^GSPC"
    assert all(r.market == "us_market" for r in rows)


def test_get_summaries_empty_when_no_rows(db_session) -> None:
    """get_summaries must return an empty sequence when no rows match."""
    repo = MarketSummaryRepository(db_session)
    rows = repo.get_summaries("xx_market", dt.date(1990, 1, 1))
    assert list(rows) == []


def test_get_latest_as_of_returns_max_date(db_session) -> None:
    """get_latest_as_of must return the most recent as_of date for the market."""
    repo = MarketSummaryRepository(db_session)
    repo.upsert_summaries("us_market", _AS_OF, [_ROW])
    repo.upsert_summaries("us_market", _AS_OF_2, [{**_ROW, "symbol": "^DJI"}])
    db_session.flush()

    latest = repo.get_latest_as_of("us_market")
    assert latest == _AS_OF_2


def test_get_latest_as_of_returns_none_when_no_rows(db_session) -> None:
    """get_latest_as_of must return None when the market has no rows."""
    repo = MarketSummaryRepository(db_session)
    result = repo.get_latest_as_of("zz_market")
    assert result is None
