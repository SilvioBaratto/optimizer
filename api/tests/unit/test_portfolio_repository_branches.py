"""Branch coverage for PortfolioRepository (target: ≥80%).

Missed lines targeted
---------------------
44-50   get_or_create  — existing branch + create branch
61-62   get_by_id      — found + not-found
75      get_by_id_or_name — valid UUID path + name fallback path + not-found
83-90   get_distinct_benchmark_tickers — empty / all-None / populated
227-235 get_snapshots  — start_date filter, end_date filter, no-date-filter
247-256 upsert_positions — SKIPPED (pg_insert ON CONFLICT, PG-only)
279-287 delete_stale_positions — empty set short-circuit + row deleted
374-393 add_event      — with metadata + without metadata
411-452 add_event_idempotent — SKIPPED (external_id path uses pg_insert ON CONFLICT, PG-only)
            (external_id=None path delegates to add_event — covered via add_event tests)
462-470 get_events     — portfolio_id filter, event_type filter, combined, no-filter
478-483 count_events   — with/without filters

PG-only methods (skipped)
--------------------------
- upsert_positions (lines 241-272): uses RepositoryBase._upsert which calls
  pg_insert + ON CONFLICT — not supported on SQLite.
- add_event_idempotent with external_id (lines 420-452): calls pg_insert with
  ON CONFLICT DO NOTHING on ``uq_activity_event_external_id`` constraint —
  not supported on SQLite.  The external_id=None branch is a thin delegate to
  add_event() which is fully covered by TestAddEvent below.

Already covered elsewhere (not duplicated)
-------------------------------------------
- create_snapshot / _get_snapshot / _delete_snapshot_children / _add_snapshot_children:
  covered by tests/unit/test_portfolio_snapshot_upsert.py
- get_latest_snapshot: covered by tests/unit/test_portfolio_snapshot_upsert.py
- upsert_account_snapshot / get_latest_account_snapshot:
  covered by tests/unit/test_broker_account_snapshot_upsert.py
- get_all_active / get_by_name: covered by tests/unit/test_portfolio_routes.py
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

import pytest

from app.models.portfolio.portfolio import (
    ActivityEvent,
    BrokerPosition,
    Portfolio,
    PortfolioSnapshot,
)
from app.repositories.portfolio.portfolio_repository import PortfolioRepository
from tests._fixtures import seed_portfolio
from tests._fixtures._helpers import add_and_flush

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _repo(session) -> PortfolioRepository:
    return PortfolioRepository(session)


def _make_portfolio(
    session,
    name: str = "test-p",
    *,
    currency: str = "EUR",
    benchmark_ticker: str = "SPY",
    is_active: bool = True,
) -> Portfolio:
    p = Portfolio(
        name=name,
        currency=currency,
        benchmark_ticker=benchmark_ticker,
        is_active=is_active,
    )
    return add_and_flush(session, p)


def _make_snapshot(
    session,
    portfolio: Portfolio,
    snap_date: date = date(2024, 3, 1),
    snap_type: str = "optimization",
) -> PortfolioSnapshot:
    return add_and_flush(
        session,
        PortfolioSnapshot(
            portfolio_id=portfolio.id,
            snapshot_date=snap_date,
            snapshot_type=snap_type,
            holding_count=1,
        ),
    )


def _make_event(
    session,
    portfolio: Portfolio | None = None,
    event_type: str = "trade",
    title: str = "Test Event",
) -> ActivityEvent:
    ev = ActivityEvent(
        portfolio_id=portfolio.id if portfolio else None,
        event_type=event_type,
        title=title,
    )
    return add_and_flush(session, ev)


# ---------------------------------------------------------------------------
# get_by_id — lines 61-62
# ---------------------------------------------------------------------------


class TestGetById:
    def test_when_id_exists_returns_portfolio(self, db_session):
        p = _make_portfolio(db_session, "by-id-found")
        result = _repo(db_session).get_by_id(p.id)
        assert result is not None
        assert result.id == p.id

    def test_when_id_not_found_returns_none(self, db_session):
        result = _repo(db_session).get_by_id(uuid.uuid4())
        assert result is None


# ---------------------------------------------------------------------------
# get_by_id_or_name — line 75
# ---------------------------------------------------------------------------


class TestGetByIdOrName:
    def test_when_valid_uuid_string_resolves_by_id(self, db_session):
        p = _make_portfolio(db_session, "uuid-resolve")
        result = _repo(db_session).get_by_id_or_name(str(p.id))
        assert result is not None
        assert result.id == p.id

    def test_when_non_uuid_string_resolves_by_name(self, db_session):
        _make_portfolio(db_session, "name-resolve")
        result = _repo(db_session).get_by_id_or_name("name-resolve")
        assert result is not None
        assert result.name == "name-resolve"

    def test_when_uuid_not_found_returns_none(self, db_session):
        result = _repo(db_session).get_by_id_or_name(str(uuid.uuid4()))
        assert result is None

    def test_when_name_not_found_returns_none(self, db_session):
        result = _repo(db_session).get_by_id_or_name("no-such-portfolio")
        assert result is None


# ---------------------------------------------------------------------------
# get_or_create — lines 44-50
# ---------------------------------------------------------------------------


class TestGetOrCreate:
    def test_when_not_existing_creates_new_portfolio(self, db_session):
        repo = _repo(db_session)
        p = repo.get_or_create("brand-new", currency="USD", benchmark_ticker="QQQ", is_active=True)
        assert p.id is not None
        assert p.name == "brand-new"

    def test_when_already_existing_returns_same_id(self, db_session):
        _make_portfolio(db_session, "already-here")
        repo = _repo(db_session)
        p1 = repo.get_or_create("already-here")
        p2 = repo.get_or_create("already-here")
        assert p1.id == p2.id

    def test_create_path_flushes_id(self, db_session):
        p = _repo(db_session).get_or_create(
            "flushed", currency="EUR", benchmark_ticker="SPY", is_active=True
        )
        assert p.id is not None


# ---------------------------------------------------------------------------
# get_distinct_benchmark_tickers — lines 83-90
# ---------------------------------------------------------------------------


class TestGetDistinctBenchmarkTickers:
    def test_when_no_active_portfolios_returns_empty_list(self, db_session):
        result = _repo(db_session).get_distinct_benchmark_tickers()
        assert result == []

    def test_when_no_active_portfolios_with_ticker_returns_empty(self, db_session):
        # An inactive portfolio with benchmark_ticker should be excluded
        _make_portfolio(db_session, "no-bench-inactive", benchmark_ticker="IWM", is_active=False)
        result = _repo(db_session).get_distinct_benchmark_tickers()
        assert result == []

    def test_returns_sorted_distinct_tickers(self, db_session):
        _make_portfolio(db_session, "p-spy", benchmark_ticker="SPY")
        _make_portfolio(db_session, "p-qqq", benchmark_ticker="QQQ")
        _make_portfolio(db_session, "p-spy2", benchmark_ticker="spy")  # duplicate lowercase
        result = _repo(db_session).get_distinct_benchmark_tickers()
        # Both "SPY" and "spy" normalise to "SPY" — distinct on uppercased
        assert result == ["QQQ", "SPY"]

    def test_inactive_portfolios_excluded(self, db_session):
        _make_portfolio(db_session, "inactive-bench", benchmark_ticker="IWM", is_active=False)
        result = _repo(db_session).get_distinct_benchmark_tickers()
        assert "IWM" not in result


# ---------------------------------------------------------------------------
# get_snapshots — lines 227-235
# ---------------------------------------------------------------------------


class TestGetSnapshots:
    def test_when_no_snapshots_returns_empty(self, db_session):
        p = _make_portfolio(db_session, "snap-empty")
        result = _repo(db_session).get_snapshots(p.id)
        assert list(result) == []

    def test_returns_all_snapshots_for_portfolio(self, db_session):
        p = _make_portfolio(db_session, "snap-all")
        _make_snapshot(db_session, p, date(2024, 1, 1))
        _make_snapshot(db_session, p, date(2024, 2, 1))
        result = _repo(db_session).get_snapshots(p.id)
        assert len(result) == 2

    def test_start_date_filter_excludes_older_snapshots(self, db_session):
        p = _make_portfolio(db_session, "snap-start")
        _make_snapshot(db_session, p, date(2024, 1, 1))
        _make_snapshot(db_session, p, date(2024, 6, 1))
        result = _repo(db_session).get_snapshots(p.id, start_date=date(2024, 3, 1))
        assert len(result) == 1
        assert result[0].snapshot_date == date(2024, 6, 1)

    def test_end_date_filter_excludes_newer_snapshots(self, db_session):
        p = _make_portfolio(db_session, "snap-end")
        _make_snapshot(db_session, p, date(2024, 1, 1))
        _make_snapshot(db_session, p, date(2024, 6, 1))
        result = _repo(db_session).get_snapshots(p.id, end_date=date(2024, 3, 1))
        assert len(result) == 1
        assert result[0].snapshot_date == date(2024, 1, 1)

    def test_both_date_filters_applied_together(self, db_session):
        p = _make_portfolio(db_session, "snap-both")
        _make_snapshot(db_session, p, date(2024, 1, 1))
        _make_snapshot(db_session, p, date(2024, 4, 1))
        _make_snapshot(db_session, p, date(2024, 8, 1))
        result = _repo(db_session).get_snapshots(
            p.id, start_date=date(2024, 2, 1), end_date=date(2024, 6, 1)
        )
        assert len(result) == 1
        assert result[0].snapshot_date == date(2024, 4, 1)

    def test_limit_caps_result_count(self, db_session):
        p = _make_portfolio(db_session, "snap-limit")
        for m in range(1, 6):
            _make_snapshot(
                db_session, p, date(2024, m, 1), snap_type=f"type{m}"
            )
        result = _repo(db_session).get_snapshots(p.id, limit=2)
        assert len(result) == 2

    def test_result_ordered_by_date_descending(self, db_session):
        p = _make_portfolio(db_session, "snap-order")
        _make_snapshot(db_session, p, date(2024, 1, 1))
        _make_snapshot(db_session, p, date(2024, 12, 1), snap_type="rebalance")
        result = _repo(db_session).get_snapshots(p.id)
        assert result[0].snapshot_date > result[1].snapshot_date


# ---------------------------------------------------------------------------
# get_positions — lines 289-298 (already partially covered; adds empty branch)
# ---------------------------------------------------------------------------


class TestGetPositions:
    def test_when_no_positions_returns_empty(self, db_session):
        p = _make_portfolio(db_session, "pos-empty")
        result = _repo(db_session).get_positions(p.id)
        assert list(result) == []

    def test_returns_positions_for_portfolio(self, db_session):
        seed = seed_portfolio(db_session)
        result = _repo(db_session).get_positions(seed.portfolio.id)
        assert len(result) == 1
        assert result[0].ticker == "AAPL_US_EQ"

    def test_isolates_by_portfolio_id(self, db_session):
        seed_portfolio(db_session, name="pos-iso-1")
        s2 = seed_portfolio(db_session, name="pos-iso-2")
        result = _repo(db_session).get_positions(s2.portfolio.id)
        ids = {r.portfolio_id for r in result}
        assert ids == {s2.portfolio.id}

    def test_ordered_by_ticker(self, db_session):
        p = _make_portfolio(db_session, "pos-order")
        for ticker in ("ZZZ", "AAA", "MMM"):
            add_and_flush(
                db_session,
                BrokerPosition(
                    portfolio_id=p.id,
                    ticker=ticker,
                    quantity=1.0,
                    average_price=10.0,
                    synced_at=datetime.now(timezone.utc),
                ),
            )
        tickers = [r.ticker for r in _repo(db_session).get_positions(p.id)]
        assert tickers == ["AAA", "MMM", "ZZZ"]


# ---------------------------------------------------------------------------
# delete_stale_positions — lines 274-287
# ---------------------------------------------------------------------------


class TestDeleteStalePositions:
    def test_when_current_tickers_empty_returns_zero_without_deleting(
        self, db_session
    ):
        seed = seed_portfolio(db_session, name="stale-empty-guard")
        repo = _repo(db_session)
        deleted = repo.delete_stale_positions(seed.portfolio.id, set())
        assert deleted == 0
        # Row must still be there
        remaining = repo.get_positions(seed.portfolio.id)
        assert len(remaining) == 1

    def test_deletes_positions_not_in_current_tickers(self, db_session):
        seed = seed_portfolio(db_session, name="stale-delete")
        repo = _repo(db_session)
        # The seeded position ticker is "AAPL_US_EQ"; pass a different current set
        deleted = repo.delete_stale_positions(
            seed.portfolio.id, {"MSFT_US_EQ"}
        )
        assert deleted == 1
        remaining = repo.get_positions(seed.portfolio.id)
        assert len(remaining) == 0

    def test_keeps_tickers_in_current_set(self, db_session):
        seed = seed_portfolio(db_session, name="stale-keep")
        repo = _repo(db_session)
        deleted = repo.delete_stale_positions(
            seed.portfolio.id, {"AAPL_US_EQ"}
        )
        assert deleted == 0
        remaining = repo.get_positions(seed.portfolio.id)
        assert len(remaining) == 1


# ---------------------------------------------------------------------------
# add_event — lines 365-393
# ---------------------------------------------------------------------------


class TestAddEvent:
    def test_when_no_metadata_creates_event_without_details(self, db_session):
        p = _make_portfolio(db_session, "event-no-meta")
        repo = _repo(db_session)
        ev = repo.add_event("trade", "A trade", portfolio_id=p.id)
        assert ev.id is not None
        assert ev.event_type == "trade"
        assert ev.metadata_ is None

    def test_when_metadata_provided_creates_detail_rows(self, db_session):
        p = _make_portfolio(db_session, "event-with-meta")
        repo = _repo(db_session)
        ev = repo.add_event(
            "rebalance",
            "Rebalanced",
            portfolio_id=p.id,
            metadata={"turnover": 0.12, "count": 5},
        )
        assert ev.id is not None
        assert ev.metadata_ is not None
        assert ev.metadata_["turnover"] == pytest.approx(0.12)
        assert ev.metadata_["count"] == 5

    def test_portfolio_id_none_allowed(self, db_session):
        repo = _repo(db_session)
        ev = repo.add_event("system", "System event")
        assert ev.portfolio_id is None

    def test_description_stored(self, db_session):
        p = _make_portfolio(db_session, "event-desc")
        repo = _repo(db_session)
        ev = repo.add_event("info", "Info title", portfolio_id=p.id, description="details")
        assert ev.description == "details"


# ---------------------------------------------------------------------------
# add_event_idempotent — external_id=None delegates to add_event
# ---------------------------------------------------------------------------


class TestAddEventIdempotentNoBranch:
    """external_id=None path only — the ON CONFLICT path is PG-only (skipped)."""

    def test_when_external_id_none_delegates_to_add_event(self, db_session):
        p = _make_portfolio(db_session, "idempotent-none")
        repo = _repo(db_session)
        ev = repo.add_event_idempotent(
            "trade", "A trade", portfolio_id=p.id, external_id=None
        )
        assert ev is not None
        assert ev.event_type == "trade"

    def test_when_external_id_none_with_metadata_works(self, db_session):
        p = _make_portfolio(db_session, "idempotent-none-meta")
        repo = _repo(db_session)
        ev = repo.add_event_idempotent(
            "info",
            "Info",
            portfolio_id=p.id,
            external_id=None,
            metadata={"key": "value"},
        )
        assert ev is not None
        assert ev.metadata_ == {"key": "value"}


# ---------------------------------------------------------------------------
# get_events — lines 454-470
# ---------------------------------------------------------------------------


class TestGetEvents:
    def test_when_empty_returns_empty(self, db_session):
        result = _repo(db_session).get_events()
        assert list(result) == []

    def test_no_filter_returns_all_events(self, db_session):
        p = _make_portfolio(db_session, "ev-all")
        _make_event(db_session, p, "trade", "T1")
        _make_event(db_session, p, "rebalance", "R1")
        result = _repo(db_session).get_events()
        assert len(result) == 2

    def test_portfolio_id_filter_returns_only_matching(self, db_session):
        p1 = _make_portfolio(db_session, "ev-pid-1")
        p2 = _make_portfolio(db_session, "ev-pid-2")
        _make_event(db_session, p1, "trade", "P1 event")
        _make_event(db_session, p2, "trade", "P2 event")
        result = _repo(db_session).get_events(portfolio_id=p1.id)
        assert len(result) == 1
        assert result[0].portfolio_id == p1.id

    def test_event_type_filter_returns_only_matching(self, db_session):
        p = _make_portfolio(db_session, "ev-type")
        _make_event(db_session, p, "trade", "Trade")
        _make_event(db_session, p, "dividend", "Div")
        result = _repo(db_session).get_events(event_type="dividend")
        assert len(result) == 1
        assert result[0].event_type == "dividend"

    def test_portfolio_and_type_filter_combined(self, db_session):
        p1 = _make_portfolio(db_session, "ev-combined-1")
        p2 = _make_portfolio(db_session, "ev-combined-2")
        _make_event(db_session, p1, "trade", "P1 trade")
        _make_event(db_session, p1, "dividend", "P1 div")
        _make_event(db_session, p2, "trade", "P2 trade")
        result = _repo(db_session).get_events(portfolio_id=p1.id, event_type="trade")
        assert len(result) == 1
        assert result[0].title == "P1 trade"

    def test_limit_caps_result(self, db_session):
        p = _make_portfolio(db_session, "ev-limit")
        for i in range(5):
            _make_event(db_session, p, "info", f"Ev{i}")
        result = _repo(db_session).get_events(limit=2)
        assert len(result) == 2

    def test_offset_skips_results(self, db_session):
        p = _make_portfolio(db_session, "ev-offset")
        for i in range(4):
            _make_event(db_session, p, "info", f"Ev{i}")
        result_all = _repo(db_session).get_events(limit=10)
        result_offset = _repo(db_session).get_events(limit=10, offset=2)
        assert len(result_offset) == len(result_all) - 2


# ---------------------------------------------------------------------------
# count_events — lines 472-483
# ---------------------------------------------------------------------------


class TestCountEvents:
    def test_when_empty_returns_zero(self, db_session):
        assert _repo(db_session).count_events() == 0

    def test_counts_all_events(self, db_session):
        p = _make_portfolio(db_session, "cnt-all")
        _make_event(db_session, p, "trade", "T1")
        _make_event(db_session, p, "trade", "T2")
        assert _repo(db_session).count_events() == 2

    def test_portfolio_id_filter_counts_only_matching(self, db_session):
        p1 = _make_portfolio(db_session, "cnt-pid-1")
        p2 = _make_portfolio(db_session, "cnt-pid-2")
        _make_event(db_session, p1, "trade", "P1")
        _make_event(db_session, p2, "trade", "P2")
        assert _repo(db_session).count_events(portfolio_id=p1.id) == 1

    def test_event_type_filter_counts_only_matching(self, db_session):
        p = _make_portfolio(db_session, "cnt-type")
        _make_event(db_session, p, "trade", "T")
        _make_event(db_session, p, "dividend", "D")
        assert _repo(db_session).count_events(event_type="trade") == 1

    def test_combined_filters_count(self, db_session):
        p1 = _make_portfolio(db_session, "cnt-combined-1")
        p2 = _make_portfolio(db_session, "cnt-combined-2")
        _make_event(db_session, p1, "trade", "P1-T")
        _make_event(db_session, p1, "info", "P1-I")
        _make_event(db_session, p2, "trade", "P2-T")
        count = _repo(db_session).count_events(portfolio_id=p1.id, event_type="trade")
        assert count == 1

    def test_portfolio_id_none_filter_counts_all(self, db_session):
        p = _make_portfolio(db_session, "cnt-none-pid")
        _make_event(db_session, p, "info", "E1")
        _make_event(db_session, None, "info", "E2")  # global event
        assert _repo(db_session).count_events() == 2
