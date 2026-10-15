"""FxRatesRepository upserts + range reads on SQLite."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

from portopt_db.models.market_data.fx_rates import FxRate
from portopt_db.repositories.market_data.fx_rates_repository import FxRatesRepository


def _row(base, quote, d, rate):
    return {
        "base_currency": base,
        "quote_currency": quote,
        "date": d,
        "rate": Decimal(str(rate)),
    }


def test_upsert_writes_rows_and_is_idempotent(db_session) -> None:
    """Upserting the same payload twice converges to one row per key."""
    repo = FxRatesRepository(db_session)
    rows = [
        _row("EUR", "USD", dt.date(2026, 10, 1), 0.91),
        _row("EUR", "GBP", dt.date(2026, 10, 1), 1.17),
    ]
    assert repo.upsert_rates(rows) == 2
    db_session.flush()
    assert db_session.query(FxRate).count() == 2

    # Re-running the same payload converges to the same two rows.
    assert repo.upsert_rates(rows) == 2
    db_session.flush()
    assert db_session.query(FxRate).count() == 2


def test_upsert_updates_rate_on_conflict(db_session) -> None:
    """A second upsert for an existing key overwrites the stored rate."""
    repo = FxRatesRepository(db_session)
    d = dt.date(2026, 10, 1)
    repo.upsert_rates([_row("EUR", "USD", d, 0.91)])
    db_session.flush()
    repo.upsert_rates([_row("EUR", "USD", d, 0.93)])
    db_session.flush()

    got = db_session.query(FxRate).one()
    assert Decimal(str(got.rate)) == Decimal("0.93")


def test_upsert_dedupes_within_one_payload(db_session) -> None:
    """Two rows with the same key in one call collapse to the last value."""
    repo = FxRatesRepository(db_session)
    d = dt.date(2026, 10, 1)
    n = repo.upsert_rates([_row("EUR", "USD", d, 0.91), _row("EUR", "USD", d, 0.95)])
    db_session.flush()
    assert n == 1
    got = db_session.query(FxRate).one()
    assert Decimal(str(got.rate)) == Decimal("0.95")


def test_upsert_empty_returns_zero(db_session) -> None:
    """An empty payload writes nothing and returns zero."""
    repo = FxRatesRepository(db_session)
    assert repo.upsert_rates([]) == 0


def test_upsert_skips_incomplete_rows(db_session) -> None:
    """Rows missing a required field (e.g. rate) are dropped, not written."""
    repo = FxRatesRepository(db_session)
    rows = [
        {"base_currency": "EUR", "quote_currency": "USD", "date": dt.date(2026, 10, 1)},
        _row("EUR", "GBP", dt.date(2026, 10, 1), 1.17),
    ]
    assert repo.upsert_rates(rows) == 1
    db_session.flush()
    assert db_session.query(FxRate).count() == 1


def test_get_rates_filters_by_base_quotes_and_range(db_session) -> None:
    """``get_rates`` restricts to the base, the quote set, and the date window."""
    repo = FxRatesRepository(db_session)
    repo.upsert_rates(
        [
            _row("EUR", "USD", dt.date(2026, 10, 1), 0.91),
            _row("EUR", "USD", dt.date(2026, 10, 2), 0.92),
            _row("EUR", "USD", dt.date(2026, 10, 3), 0.93),
            _row("EUR", "GBP", dt.date(2026, 10, 2), 1.17),
            _row("USD", "EUR", dt.date(2026, 10, 2), 1.08),
        ]
    )
    db_session.flush()

    got = repo.get_rates(
        "EUR",
        ["USD"],
        dt.date(2026, 10, 1),
        dt.date(2026, 10, 2),
    )
    assert [g.date for g in got] == [dt.date(2026, 10, 1), dt.date(2026, 10, 2)]
    assert all(g.base_currency == "EUR" and g.quote_currency == "USD" for g in got)


def test_get_rates_without_filters_returns_all_for_base(db_session) -> None:
    """Omitting quotes/range returns every stored pair for the base currency."""
    repo = FxRatesRepository(db_session)
    repo.upsert_rates(
        [
            _row("EUR", "USD", dt.date(2026, 10, 1), 0.91),
            _row("EUR", "GBP", dt.date(2026, 10, 1), 1.17),
            _row("USD", "EUR", dt.date(2026, 10, 1), 1.08),
        ]
    )
    db_session.flush()

    got = repo.get_rates("EUR")
    assert {g.quote_currency for g in got} == {"USD", "GBP"}
