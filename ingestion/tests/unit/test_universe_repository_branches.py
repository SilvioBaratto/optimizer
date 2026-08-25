"""Branch coverage for UniverseRepository (target: ≥80%).

Missed lines targeted
---------------------
22, 25-39   save_exchange  — update path (existing) + create path
44-80       save_instruments_batch — SKIPPED (pg_insert ON CONFLICT, PG-only)
108-116     mark_delisted  — found+updated / not-found / already-delisted branches
120-125     get_active_tickers
128-139     clear_all
141-148     get_instrument_count / get_exchange_count
158-167     get_instruments — exchange_name filter, search filter, skip/limit, no-filter
170         get_exchanges — empty + populated

PG-only methods (skipped)
--------------------------
- save_instruments_batch (lines 41-80): uses pg_insert + ON CONFLICT DO UPDATE
  with a named constraint that does not exist in SQLite — SQLite raises
  ``OperationalError: no such constraint``.  Document-only, no test.
"""

from __future__ import annotations

from datetime import date

import pytest

from app.models.universe.universe import Exchange, Instrument
from app.repositories.universe.universe_repository import UniverseRepository
from tests._fixtures import seed_universe
from tests._fixtures._helpers import add_and_flush

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_exchange(session, name: str = "NYSE") -> Exchange:
    return add_and_flush(session, Exchange(name=name))


def _make_instrument(session, exchange: Exchange, ticker: str = "AAPL") -> Instrument:
    return add_and_flush(
        session,
        Instrument(ticker=ticker, short_name=f"{ticker} Inc.", exchange=exchange),
    )


def _repo(session) -> UniverseRepository:
    return UniverseRepository(session)


# ---------------------------------------------------------------------------
# save_exchange — lines 22, 25-39
# ---------------------------------------------------------------------------


class TestSaveExchange:
    def test_when_exchange_does_not_exist_creates_new_row(self, db_session):
        repo = _repo(db_session)
        ex = repo.save_exchange({"name": "NASDAQ", "id": 7})

        assert ex.name == "NASDAQ"
        assert ex.t212_id == 7
        assert ex.id is not None

    def test_when_exchange_already_exists_updates_t212_id(self, db_session):
        # Seed it first
        _make_exchange(db_session, "LSE")
        repo = _repo(db_session)

        updated = repo.save_exchange({"name": "LSE", "id": 42})

        assert updated.name == "LSE"
        assert updated.t212_id == 42
        # Only one row must exist
        count = repo.get_exchange_count()
        assert count == 1

    def test_when_exchange_exists_returns_same_object(self, db_session):
        ex1 = _make_exchange(db_session, "TSX")
        repo = _repo(db_session)

        ex2 = repo.save_exchange({"name": "TSX", "id": 99})

        assert ex1.id == ex2.id

    def test_when_called_repeatedly_then_one_row_and_latest_t212_id(self, db_session):
        """R1/§5.4 (T1.3): repeated saves converge to a single row via
        INSERT ... ON CONFLICT DO UPDATE on exchanges.name — id stable,
        latest t212_id reflected, no duplicate."""
        repo = _repo(db_session)
        first = repo.save_exchange({"name": "XETRA", "id": 1})
        first_id = first.id

        again = repo.save_exchange({"name": "XETRA", "id": 2})

        assert repo.get_exchange_count() == 1
        assert again.id == first_id
        assert again.t212_id == 2


# ---------------------------------------------------------------------------
# get_exchange_count / get_instrument_count — lines 141-148
# ---------------------------------------------------------------------------


class TestCounts:
    def test_exchange_count_zero_when_empty(self, db_session):
        assert _repo(db_session).get_exchange_count() == 0

    def test_exchange_count_reflects_seeded_row(self, db_session):
        seed_universe(db_session)
        assert _repo(db_session).get_exchange_count() == 1

    def test_instrument_count_zero_when_empty(self, db_session):
        assert _repo(db_session).get_instrument_count() == 0

    def test_instrument_count_reflects_seeded_row(self, db_session):
        seed_universe(db_session)
        assert _repo(db_session).get_instrument_count() == 1

    def test_counts_accumulate_across_multiple_seeds(self, db_session):
        ex = _make_exchange(db_session, "XETRA")
        _make_instrument(db_session, ex, "SAP")
        _make_instrument(db_session, ex, "BMW")
        repo = _repo(db_session)
        assert repo.get_exchange_count() == 1
        assert repo.get_instrument_count() == 2


# ---------------------------------------------------------------------------
# get_exchanges — line 170
# ---------------------------------------------------------------------------


class TestGetExchanges:
    def test_when_empty_returns_empty_sequence(self, db_session):
        result = _repo(db_session).get_exchanges()
        assert list(result) == []

    def test_when_seeded_returns_all_exchanges_ordered_by_name(self, db_session):
        _make_exchange(db_session, "Zeta")
        _make_exchange(db_session, "Alpha")
        result = list(_repo(db_session).get_exchanges())
        assert [e.name for e in result] == ["Alpha", "Zeta"]

    def test_returns_exchange_objects(self, db_session):
        seed_universe(db_session)
        result = _repo(db_session).get_exchanges()
        assert all(isinstance(e, Exchange) for e in result)


# ---------------------------------------------------------------------------
# get_instruments — lines 158-167
# ---------------------------------------------------------------------------


class TestGetInstruments:
    def test_when_empty_returns_empty_sequence(self, db_session):
        result = _repo(db_session).get_instruments()
        assert list(result) == []

    def test_no_filter_returns_all_instruments(self, db_session):
        seed = seed_universe(db_session)
        _make_instrument(db_session, seed.exchange, "MSFT")
        result = _repo(db_session).get_instruments()
        assert len(result) == 2

    def test_exchange_name_filter_returns_only_matching_exchange(self, db_session):
        ex1 = _make_exchange(db_session, "Exchange A")
        ex2 = _make_exchange(db_session, "Exchange B")
        _make_instrument(db_session, ex1, "T1")
        _make_instrument(db_session, ex2, "T2")

        result = list(_repo(db_session).get_instruments(exchange_name="Exchange A"))
        assert len(result) == 1
        assert result[0].ticker == "T1"

    def test_exchange_name_filter_none_returns_all(self, db_session):
        ex = _make_exchange(db_session, "E1")
        _make_instrument(db_session, ex, "X1")
        _make_instrument(db_session, ex, "X2")
        result = _repo(db_session).get_instruments(exchange_name=None)
        assert len(result) == 2

    def test_search_filter_matches_ticker_substring(self, db_session):
        ex = _make_exchange(db_session, "Search Exchange")
        _make_instrument(db_session, ex, "AMZN")
        _make_instrument(db_session, ex, "GOOG")
        result = list(_repo(db_session).get_instruments(search="AMZ"))
        assert len(result) == 1
        assert result[0].ticker == "AMZN"

    def test_search_filter_matches_name_substring(self, db_session):
        ex = _make_exchange(db_session, "Name Search Exchange")
        inst = add_and_flush(
            db_session,
            Instrument(
                ticker="XYZ",
                short_name="XYZ Corp",
                name="XYZ Holdings Inc.",
                exchange=ex,
            ),
        )
        result = list(_repo(db_session).get_instruments(search="Holdings"))
        assert len(result) == 1
        assert result[0].id == inst.id

    def test_search_filter_none_skips_where_clause(self, db_session):
        ex = _make_exchange(db_session, "No-Search Exchange")
        _make_instrument(db_session, ex, "A1")
        _make_instrument(db_session, ex, "A2")
        result = _repo(db_session).get_instruments(search=None)
        assert len(result) == 2

    def test_skip_offset_skips_first_row(self, db_session):
        ex = _make_exchange(db_session, "Pagination Exchange")
        _make_instrument(db_session, ex, "P1")
        _make_instrument(db_session, ex, "P2")
        _make_instrument(db_session, ex, "P3")
        result = _repo(db_session).get_instruments(skip=1, limit=10)
        assert len(result) == 2

    def test_limit_caps_result_count(self, db_session):
        ex = _make_exchange(db_session, "Limit Exchange")
        for i in range(5):
            _make_instrument(db_session, ex, f"LIM{i}")
        result = _repo(db_session).get_instruments(limit=2)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# mark_delisted — lines 108-116
# ---------------------------------------------------------------------------


class TestMarkDelisted:
    def test_when_instrument_exists_and_active_marks_delisted_returns_true(
        self, db_session
    ):
        seed = seed_universe(db_session, ticker="DELME")
        repo = _repo(db_session)

        result = repo.mark_delisted("DELME", seed.exchange.id, date(2024, 6, 1), -0.30)

        assert result is True
        db_session.refresh(seed.instrument)
        assert seed.instrument.delisted_at == date(2024, 6, 1)
        assert seed.instrument.delisting_return == pytest.approx(-0.30)

    def test_when_instrument_not_found_returns_false(self, db_session):
        ex = _make_exchange(db_session, "Delist Exchange")
        repo = _repo(db_session)

        result = repo.mark_delisted("NOBODY", ex.id, date(2024, 6, 1))

        assert result is False

    def test_when_already_delisted_does_not_update_again_returns_false(
        self, db_session
    ):
        ex = _make_exchange(db_session, "Already Delisted Exchange")
        inst = add_and_flush(
            db_session,
            Instrument(
                ticker="OLD",
                short_name="Old Corp",
                exchange=ex,
                delisted_at=date(2023, 1, 1),
                delisting_return=-1.0,
            ),
        )
        repo = _repo(db_session)

        result = repo.mark_delisted("OLD", ex.id, date(2024, 6, 1))

        assert result is False
        db_session.refresh(inst)
        # Original date must be unchanged
        assert inst.delisted_at == date(2023, 1, 1)


# ---------------------------------------------------------------------------
# get_active_tickers — lines 120-125
# ---------------------------------------------------------------------------


class TestGetActiveTickers:
    def test_when_no_instruments_returns_empty_set(self, db_session):
        ex = _make_exchange(db_session, "Empty Active Exchange")
        result = _repo(db_session).get_active_tickers(ex.id)
        assert result == set()

    def test_returns_tickers_without_delisted_at(self, db_session):
        ex = _make_exchange(db_session, "Active Exchange")
        _make_instrument(db_session, ex, "LIVE")
        add_and_flush(
            db_session,
            Instrument(
                ticker="DEAD",
                short_name="Dead Corp",
                exchange=ex,
                delisted_at=date(2023, 1, 1),
            ),
        )
        result = _repo(db_session).get_active_tickers(ex.id)
        assert result == {"LIVE"}

    def test_all_active_instruments_included(self, db_session):
        ex = _make_exchange(db_session, "All Active Exchange")
        _make_instrument(db_session, ex, "AA1")
        _make_instrument(db_session, ex, "AA2")
        result = _repo(db_session).get_active_tickers(ex.id)
        assert result == {"AA1", "AA2"}

    def test_isolates_by_exchange_id(self, db_session):
        ex1 = _make_exchange(db_session, "Isolated Ex1")
        ex2 = _make_exchange(db_session, "Isolated Ex2")
        _make_instrument(db_session, ex1, "EX1T")
        _make_instrument(db_session, ex2, "EX2T")
        result = _repo(db_session).get_active_tickers(ex1.id)
        assert result == {"EX1T"}


# ---------------------------------------------------------------------------
# clear_all — lines 128-139
# ---------------------------------------------------------------------------


class TestClearAll:
    def test_when_empty_returns_zero_zero(self, db_session):
        ex_count, inst_count = _repo(db_session).clear_all()
        assert ex_count == 0
        assert inst_count == 0

    def test_when_seeded_returns_correct_counts(self, db_session):
        seed_universe(db_session)
        ex_count, inst_count = _repo(db_session).clear_all()
        assert ex_count == 1
        assert inst_count == 1

    def test_after_clear_counts_are_zero(self, db_session):
        seed_universe(db_session)
        repo = _repo(db_session)
        repo.clear_all()
        assert repo.get_exchange_count() == 0
        assert repo.get_instrument_count() == 0

    def test_returns_tuple_of_two_ints(self, db_session):
        ex = _make_exchange(db_session, "Tuple Exchange")
        _make_instrument(db_session, ex, "T1")
        _make_instrument(db_session, ex, "T2")
        result = _repo(db_session).clear_all()
        assert result == (1, 2)


class TestGetActiveTickersTypeScope:
    """Regression (review-critical): the two-pass build must reconcile delisting
    per instrument_type, or the ETF pass mass-delists stocks on shared exchanges."""

    def test_scopes_by_instrument_type(self, db_session) -> None:
        from app.models.universe.universe import Instrument
        from app.repositories.universe.universe_repository import UniverseRepository

        repo = UniverseRepository(db_session)
        ex = repo.save_exchange({"name": "NASDAQ", "id": 1})
        db_session.add_all(
            [
                Instrument(
                    ticker="AAPL_US",
                    short_name="AAPL",
                    exchange_id=ex.id,
                    instrument_type="STOCK",
                    asset_class="equity",
                ),
                Instrument(
                    ticker="AGG_US",
                    short_name="AGG",
                    exchange_id=ex.id,
                    instrument_type="ETF",
                    asset_class="fixed_income",
                ),
            ]
        )
        db_session.flush()

        assert repo.get_active_tickers(ex.id, instrument_type="ETF") == {"AGG_US"}
        assert repo.get_active_tickers(ex.id, instrument_type="STOCK") == {"AAPL_US"}
        assert repo.get_active_tickers(ex.id) == {"AAPL_US", "AGG_US"}
