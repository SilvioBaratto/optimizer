"""A full NAV re-mark replaces the persisted series instead of only overwriting it."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from datetime import date
from decimal import Decimal

import pytest
from portopt_db.models import Base, NavHistory
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from test_scheduler_nav_mark import (
    _PID,
    _benchmark_rows,
    _seed_benchmark_book,
    _set_benchmark,
)

from fund.scheduler import run_nav_mark

_ASOF = date(2026, 10, 12)
_STALE = date(2026, 9, 29)


@pytest.fixture
def session_factory(tmp_path):
    """A file-based SQLite engine so the mark and its heartbeat get own connections."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'rebuild.db'}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    Base.metadata.create_all(bind=engine)

    @contextmanager
    def _factory() -> Generator[Session, None, None]:
        session = Session(bind=engine)
        try:
            yield session
        finally:
            session.close()

    try:
        yield _factory
    finally:
        engine.dispose()


def _plant_stale_row(factory, *, ticker: str) -> None:
    """A row on a date the book's own series never emits, as an old benchmark leaves."""
    with factory() as session:
        session.add(
            NavHistory(
                portfolio_id=_PID,
                date=_STALE,
                nav=Decimal("1"),
                benchmark=Decimal("2"),
                benchmark_ticker=ticker,
            )
        )
        session.commit()


def test_a_benchmark_change_drops_rows_on_dates_the_new_series_does_not_emit(
    session_factory,
):
    """Rows only the old benchmark traded must not outlive the benchmark switch.

    They would keep the old tag, so the mark would see a foreign benchmark and rebuild
    in full on every later run, and their NAV level would sit stale beside the rewritten
    bars.
    """
    _seed_benchmark_book(session_factory, benchmark="SPY")
    assert run_nav_mark(session_factory=session_factory, asof=_ASOF)
    _plant_stale_row(session_factory, ticker="SPY")
    _set_benchmark(session_factory, "ACWI")

    assert run_nav_mark(session_factory=session_factory, asof=_ASOF)

    rows = _benchmark_rows(session_factory)
    assert _STALE not in rows
    assert {ticker for _, ticker in rows.values()} == {"ACWI"}


def test_an_explicit_full_mark_drops_rows_the_rebuilt_series_does_not_emit(
    session_factory,
):
    _seed_benchmark_book(session_factory, benchmark="SPY")
    assert run_nav_mark(session_factory=session_factory, asof=_ASOF)
    _plant_stale_row(session_factory, ticker="SPY")

    assert run_nav_mark(session_factory=session_factory, asof=_ASOF, full=True)

    assert _STALE not in _benchmark_rows(session_factory)


def test_an_incremental_mark_keeps_rows_outside_its_window(session_factory):
    _seed_benchmark_book(session_factory, benchmark="SPY")
    assert run_nav_mark(session_factory=session_factory, asof=_ASOF)
    _plant_stale_row(session_factory, ticker="SPY")

    assert run_nav_mark(session_factory=session_factory, asof=_ASOF)

    assert _STALE in _benchmark_rows(session_factory)
