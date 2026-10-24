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

save_instruments_batch
----------------------
Built with ``pg_insert(...).on_conflict_do_update(constraint=...)``. SQLite
(>= 3.35) compiles that as a target-less ``ON CONFLICT DO UPDATE``, which fires
on the table's one non-PK unique constraint (ticker + exchange), so the venue
change and per-row upsert semantics run against the in-memory engine. What
SQLite cannot show (the conflict update's column expressions) is checked by
compiling the emitted statement for the PostgreSQL dialect.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from unittest.mock import MagicMock

import pytest
from _fixtures import add_and_flush, seed_universe
from sqlalchemy import select, update
from sqlalchemy.dialects import postgresql

from portopt_db.models.universe.universe import Exchange, Instrument
from portopt_db.repositories.universe.universe_repository import UniverseRepository


def _make_exchange(session, name: str = "NYSE") -> Exchange:
    return add_and_flush(session, Exchange(name=name))


def _make_instrument(session, exchange: Exchange, ticker: str = "AAPL") -> Instrument:
    return add_and_flush(
        session,
        Instrument(ticker=ticker, short_name=f"{ticker} Inc.", exchange=exchange),
    )


def _repo(session) -> UniverseRepository:
    return UniverseRepository(session)


class TestSaveExchange:
    def test_when_exchange_does_not_exist_creates_new_row(self, db_session):
        repo = _repo(db_session)
        ex = repo.save_exchange({"name": "NASDAQ", "id": 7})

        assert ex.name == "NASDAQ"
        assert ex.t212_id == 7
        assert ex.id is not None

    def test_when_exchange_already_exists_updates_t212_id(self, db_session):
        _make_exchange(db_session, "LSE")
        repo = _repo(db_session)

        updated = repo.save_exchange({"name": "LSE", "id": 42})

        assert updated.name == "LSE"
        assert updated.t212_id == 42
        count = repo.get_exchange_count()
        assert count == 1

    def test_when_exchange_exists_returns_same_object(self, db_session):
        ex1 = _make_exchange(db_session, "TSX")
        repo = _repo(db_session)

        ex2 = repo.save_exchange({"name": "TSX", "id": 99})

        assert ex1.id == ex2.id

    def test_when_called_repeatedly_then_one_row_and_latest_t212_id(self, db_session):
        """Repeated saves converge to a single row via INSERT ... ON CONFLICT DO UPDATE
        on exchanges.name — id stable, latest t212_id reflected, no duplicate."""
        repo = _repo(db_session)
        first = repo.save_exchange({"name": "XETRA", "id": 1})
        first_id = first.id

        again = repo.save_exchange({"name": "XETRA", "id": 2})

        assert repo.get_exchange_count() == 1
        assert again.id == first_id
        assert again.t212_id == 2


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

    def test_pages_walk_the_universe_in_ticker_order_without_gaps_or_repeats(
        self, db_session
    ):
        """OFFSET/LIMIT without ORDER BY lets pages overlap or skip rows."""
        ex = _make_exchange(db_session, "Ordered Exchange")
        for ticker in ("MMM", "ZZZ", "AAA", "QQQ", "CCC"):
            _make_instrument(db_session, ex, ticker)
        repo = _repo(db_session)

        pages = [
            [i.ticker for i in repo.get_instruments(skip=skip, limit=2)]
            for skip in (0, 2, 4)
        ]

        assert pages == [["AAA", "CCC"], ["MMM", "QQQ"], ["ZZZ"]]

    def test_limit_caps_result_count(self, db_session):
        ex = _make_exchange(db_session, "Limit Exchange")
        for i in range(5):
            _make_instrument(db_session, ex, f"LIM{i}")
        result = _repo(db_session).get_instruments(limit=2)
        assert len(result) == 2


class TestSaveInstrumentsBatchConflictUpdate:
    @staticmethod
    def _upsert_sql() -> str:
        session = MagicMock()
        UniverseRepository(session).save_instruments_batch(
            [{"ticker": "AGG", "shortName": "AGG", "isin": None}], "exchange-id"
        )
        stmt = session.execute.call_args.args[0]
        return " ".join(str(stmt.compile(dialect=postgresql.dialect())).split())

    def test_null_incoming_isin_does_not_overwrite_the_stored_one(self):
        # The screener-backed build carries no ISIN, so a plain
        # ``isin = excluded.isin`` would blank every ISIN the profile backfill
        # had filled on each weekly rebuild.
        assert "isin = coalesce(excluded.isin, instruments.isin)" in (
            self._upsert_sql().lower()
        )


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

    def test_when_no_return_given_then_terminal_return_stays_unknown(self, db_session):
        """Absence from a universe snapshot proves nothing about the loss taken.

        Stamping a synthetic -30% would let a survivorship correction overwrite
        the last real return of a stock that is still trading, so the column
        must stay NULL unless the caller knows the actual terminal return.
        """
        seed = seed_universe(db_session, ticker="GONE")

        _repo(db_session).mark_delisted("GONE", seed.exchange.id, date(2024, 6, 1))

        db_session.refresh(seed.instrument)
        assert seed.instrument.delisted_at == date(2024, 6, 1)
        assert seed.instrument.delisting_return is None

    def test_when_return_given_then_it_is_stored_verbatim(self, db_session):
        seed = seed_universe(db_session, ticker="BANKRUPT")

        _repo(db_session).mark_delisted(
            "BANKRUPT", seed.exchange.id, date(2024, 6, 1), delisting_return=-1.0
        )

        db_session.refresh(seed.instrument)
        assert seed.instrument.delisting_return == pytest.approx(-1.0)

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
        assert inst.delisted_at == date(2023, 1, 1)


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
        from portopt_db.models.universe.universe import Instrument
        from portopt_db.repositories.universe.universe_repository import (
            UniverseRepository,
        )

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


class TestSaveInstrumentsBatchKeepsBackfilledIsin:
    """The screener source carries no ISIN, so a rebuild upserts ``isin=NULL``;
    the conflict update must not erase an ISIN backfilled from ticker_profiles.

    ``save_instruments_batch`` is PG-only (ON CONFLICT on a named constraint),
    so the statement it emits is captured and compiled for PostgreSQL."""

    def _upsert_sql(self, db_session, monkeypatch) -> str:
        from sqlalchemy.dialects import postgresql

        statements: list = []
        real_execute = db_session.execute

        def _capture_upsert(stmt, *args, **kwargs):
            # The venue-change lookup is a real SELECT; only the upsert, which
            # SQLite cannot run, is captured instead of executed.
            if getattr(stmt, "is_insert", False):
                statements.append(stmt)
                return None
            return real_execute(stmt, *args, **kwargs)

        monkeypatch.setattr(db_session, "execute", _capture_upsert)
        _repo(db_session).save_instruments_batch(
            [{"ticker": "AAPL_US", "shortName": "AAPL", "isin": None}],
            exchange_id=uuid.uuid4(),
        )
        assert len(statements) == 1
        return str(statements[0].compile(dialect=postgresql.dialect()))

    def test_null_incoming_isin_does_not_overwrite_stored_isin(
        self, db_session, monkeypatch
    ) -> None:
        sql = self._upsert_sql(db_session, monkeypatch)

        assert "isin = coalesce(excluded.isin, instruments.isin)" in sql


def _batch_row(ticker: str, **overrides) -> dict:
    return {"ticker": ticker, "shortName": ticker, "type": "STOCK", **overrides}


def _rows_for(session, ticker: str) -> list[Instrument]:
    return list(
        session.execute(select(Instrument).where(Instrument.ticker == ticker))
        .scalars()
        .all()
    )


class TestSaveInstrumentsBatchVenueChange:
    """A symbol that moves venue must stay ONE instrument row.

    Ticker-keyed readers (price lookups, the yfinance sweep) assume a symbol is
    unique, so a second (ticker, new exchange) row beside the stale one makes
    them raise ``MultipleResultsFound`` or fetch the symbol twice.
    """

    def test_when_symbol_moves_exchange_then_existing_row_is_rehomed(self, db_session):
        repo = _repo(db_session)
        nasdaq = repo.save_exchange({"name": "NASDAQ", "id": 1})
        nyse = repo.save_exchange({"name": "NYSE", "id": 2})
        repo.save_instruments_batch([_batch_row("XYZ")], nasdaq.id)
        original_id = _rows_for(db_session, "XYZ")[0].id

        repo.save_instruments_batch([_batch_row("XYZ")], nyse.id)

        rows = _rows_for(db_session, "XYZ")
        assert len(rows) == 1
        assert rows[0].id == original_id
        assert rows[0].exchange_id == nyse.id

    def test_when_stale_row_was_marked_delisted_then_move_reactivates_it(
        self, db_session
    ):
        repo = _repo(db_session)
        nasdaq = repo.save_exchange({"name": "NASDAQ", "id": 1})
        nyse = repo.save_exchange({"name": "NYSE", "id": 2})
        repo.save_instruments_batch([_batch_row("XYZ")], nasdaq.id)
        repo.mark_delisted("XYZ", nasdaq.id, date(2024, 6, 1))

        repo.save_instruments_batch([_batch_row("XYZ")], nyse.id)

        rows = _rows_for(db_session, "XYZ")
        assert len(rows) == 1
        assert rows[0].exchange_id == nyse.id
        assert rows[0].delisted_at is None

    def test_when_symbol_already_on_target_exchange_then_other_venue_is_untouched(
        self, db_session
    ):
        repo = _repo(db_session)
        nasdaq = repo.save_exchange({"name": "NASDAQ", "id": 1})
        nyse = repo.save_exchange({"name": "NYSE", "id": 2})
        # Legacy state: the symbol already has a row on both venues.
        for exchange in (nasdaq, nyse):
            add_and_flush(
                db_session,
                Instrument(ticker="XYZ", short_name="XYZ", exchange_id=exchange.id),
            )

        repo.save_instruments_batch([_batch_row("XYZ")], nyse.id)

        venues = sorted(r.exchange_id for r in _rows_for(db_session, "XYZ"))
        assert venues == sorted([nasdaq.id, nyse.id])

    def test_when_predecessor_is_ambiguous_then_no_existing_row_is_moved(
        self, db_session
    ):
        repo = _repo(db_session)
        nasdaq = repo.save_exchange({"name": "NASDAQ", "id": 1})
        amex = repo.save_exchange({"name": "AMEX", "id": 2})
        nyse = repo.save_exchange({"name": "NYSE", "id": 3})
        # Legacy state: two stale rows elsewhere, so neither is provably the
        # predecessor of the NYSE listing.
        stale_ids = {}
        for exchange in (nasdaq, amex):
            row = Instrument(ticker="XYZ", short_name="XYZ", exchange_id=exchange.id)
            add_and_flush(db_session, row)
            stale_ids[exchange.id] = row.id

        repo.save_instruments_batch([_batch_row("XYZ")], nyse.id)

        rows = _rows_for(db_session, "XYZ")
        assert {r.exchange_id for r in rows} == {nasdaq.id, amex.id, nyse.id}
        for row in rows:
            if row.exchange_id in stale_ids:
                assert row.id == stale_ids[row.exchange_id]

    def test_unrelated_tickers_on_other_exchanges_stay_put(self, db_session):
        repo = _repo(db_session)
        nasdaq = repo.save_exchange({"name": "NASDAQ", "id": 1})
        nyse = repo.save_exchange({"name": "NYSE", "id": 2})
        repo.save_instruments_batch([_batch_row("AAA")], nasdaq.id)

        repo.save_instruments_batch([_batch_row("BBB")], nyse.id)

        assert _rows_for(db_session, "AAA")[0].exchange_id == nasdaq.id
        assert _rows_for(db_session, "BBB")[0].exchange_id == nyse.id


def _reload(session, ticker: str) -> Instrument:
    return session.execute(
        select(Instrument)
        .where(Instrument.ticker == ticker)
        .execution_options(populate_existing=True)
    ).scalar_one()


class TestSaveInstrumentsBatchResave:
    """Re-saving a universe must not erase data the source did not resend.

    The yfinance universe source carries no ISIN (it is backfilled later from
    ticker profiles), so a blind overwrite would blank every ISIN each build.
    """

    def test_when_resaved_without_isin_then_stored_isin_is_kept(self, db_session):
        repo = _repo(db_session)
        ex = repo.save_exchange({"name": "NASDAQ", "id": 1})
        repo.save_instruments_batch([_batch_row("AAPL")], ex.id)
        db_session.execute(
            update(Instrument)
            .where(Instrument.ticker == "AAPL")
            .values(isin="US0378331005")
        )

        repo.save_instruments_batch([_batch_row("AAPL", isin=None)], ex.id)

        assert _reload(db_session, "AAPL").isin == "US0378331005"

    def test_when_resaved_with_isin_then_new_isin_replaces_old(self, db_session):
        repo = _repo(db_session)
        ex = repo.save_exchange({"name": "NASDAQ", "id": 1})
        repo.save_instruments_batch([_batch_row("AAPL", isin="OLD000000000")], ex.id)

        repo.save_instruments_batch([_batch_row("AAPL", isin="US0378331005")], ex.id)

        assert _reload(db_session, "AAPL").isin == "US0378331005"

    def test_when_resaved_without_name_or_currency_then_stored_values_are_kept(
        self, db_session
    ):
        repo = _repo(db_session)
        ex = repo.save_exchange({"name": "NASDAQ", "id": 1})
        repo.save_instruments_batch(
            [_batch_row("AAPL", name="Apple Inc.", currencyCode="USD")], ex.id
        )

        repo.save_instruments_batch([_batch_row("AAPL")], ex.id)

        row = _reload(db_session, "AAPL")
        assert row.name == "Apple Inc."
        assert row.currency_code == "USD"

    def test_when_resaved_then_updated_at_is_refreshed(self, db_session):
        repo = _repo(db_session)
        ex = repo.save_exchange({"name": "NASDAQ", "id": 1})
        repo.save_instruments_batch([_batch_row("AAPL")], ex.id)
        db_session.execute(
            update(Instrument)
            .where(Instrument.ticker == "AAPL")
            .values(updated_at=datetime(2000, 1, 1))
        )

        repo.save_instruments_batch([_batch_row("AAPL")], ex.id)

        assert _reload(db_session, "AAPL").updated_at.year > 2000
