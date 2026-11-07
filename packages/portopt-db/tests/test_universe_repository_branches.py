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

from portopt_db.models.universe.pinned_instrument import PinnedInstrument
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


class TestGetExchangeByName:
    """``get_exchange_by_name`` is a pure read that never creates or edits a row."""

    def test_hit_returns_the_exchange(self, db_session):
        """An existing exchange is returned by its exact name."""
        existing = _make_exchange(db_session, "Borsa Italiana")

        found = _repo(db_session).get_exchange_by_name("Borsa Italiana")

        assert found is not None
        assert found.id == existing.id

    def test_miss_returns_none_and_creates_nothing(self, db_session):
        """An unknown name yields None and leaves the table empty."""
        repo = _repo(db_session)

        assert repo.get_exchange_by_name("Nowhere") is None
        assert repo.get_exchange_count() == 0

    def test_lookup_leaves_t212_id_unchanged(self, db_session):
        """Unlike save_exchange, a lookup does not overwrite a stored t212_id."""
        existing = add_and_flush(db_session, Exchange(name="LSE", t212_id=42))
        repo = _repo(db_session)

        repo.get_exchange_by_name("LSE")
        db_session.refresh(existing)

        assert existing.t212_id == 42


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

    def test_pinned_instrument_is_never_marked_delisted(self, db_session):
        """A pin overrides the "absent from the snapshot" signal."""
        ex = _make_exchange(db_session, "Pinned Venue")
        etf = add_and_flush(
            db_session,
            Instrument(
                ticker="VWCE.DE",
                short_name="VWCE",
                yfinance_ticker="VWCE.DE",
                exchange=ex,
            ),
        )
        add_and_flush(db_session, PinnedInstrument(ticker="VWCE.DE", source="manual"))

        result = _repo(db_session).mark_delisted("VWCE.DE", ex.id, date(2024, 6, 1))

        assert result is False
        db_session.refresh(etf)
        assert etf.delisted_at is None
        assert etf.delisting_return is None

    def test_unpinned_instrument_is_still_marked_beside_a_pinned_one(self, db_session):
        """Only the pinned ticker is spared; its neighbour is delisted as before."""
        ex = _make_exchange(db_session, "Mixed Venue")
        for ticker in ("KEEP.DE", "DROP.DE"):
            add_and_flush(
                db_session,
                Instrument(
                    ticker=ticker,
                    short_name=ticker,
                    yfinance_ticker=ticker,
                    exchange=ex,
                ),
            )
        add_and_flush(db_session, PinnedInstrument(ticker="KEEP.DE", source="manual"))
        repo = _repo(db_session)

        assert repo.mark_delisted("KEEP.DE", ex.id, date(2024, 6, 1)) is False
        assert repo.mark_delisted("DROP.DE", ex.id, date(2024, 6, 1)) is True

    def test_pin_matches_the_yfinance_ticker_not_the_screener_ticker(self, db_session):
        """The pin key is the Yahoo symbol, whatever the row's own ticker says."""
        ex = _make_exchange(db_session, "Alias Venue")
        add_and_flush(
            db_session,
            Instrument(
                ticker="VWCEd_EQ",
                short_name="VWCE",
                yfinance_ticker="VWCE.DE",
                exchange=ex,
            ),
        )
        add_and_flush(db_session, PinnedInstrument(ticker="VWCE.DE", source="manual"))

        assert (
            _repo(db_session).mark_delisted("VWCEd_EQ", ex.id, date(2024, 6, 1))
            is False
        )


class TestPinnedSiblingRows:
    """A pin protects the row that carries its symbol, not every row that shares it.

    The Yahoo-ticker backfill left broker-era rows (``VUSAl_EQ``) beside the
    canonical ones (``VUSA.L``). They share a ``yfinance_ticker``, so protecting
    all of them kept a stale twin active for as long as the pin existed and had
    the symbol ingested twice.
    """

    @staticmethod
    def _pair(db_session):
        ex = _make_exchange(db_session, "Sibling Venue")
        canonical = add_and_flush(
            db_session,
            Instrument(
                ticker="VUSA.L",
                short_name="VUSA",
                yfinance_ticker="VUSA.L",
                exchange=ex,
            ),
        )
        sibling = add_and_flush(
            db_session,
            Instrument(
                ticker="VUSAl_EQ",
                short_name="VUSA",
                yfinance_ticker="VUSA.L",
                exchange=ex,
            ),
        )
        add_and_flush(db_session, PinnedInstrument(ticker="VUSA.L", source="manual"))
        return ex, canonical, sibling

    def test_the_canonical_row_is_protected_and_its_sibling_is_not(self, db_session):
        """Only the row whose ticker is the pinned symbol is spared from delisting."""
        ex, canonical, sibling = self._pair(db_session)
        repo = _repo(db_session)

        assert repo.mark_delisted("VUSA.L", ex.id, date(2026, 11, 1)) is False
        assert repo.mark_delisted("VUSAl_EQ", ex.id, date(2026, 11, 1)) is True

        db_session.refresh(canonical)
        db_session.refresh(sibling)
        assert canonical.delisted_at is None
        assert sibling.delisted_at == date(2026, 11, 1)

    def test_only_the_canonical_row_is_reported_as_pinned(self, db_session):
        """The universe build must not subtract a stale twin from its snapshot."""
        ex, _, _ = self._pair(db_session)

        assert _repo(db_session).get_pinned_tickers(ex.id) == {"VUSA.L"}


class TestGetPinnedTickers:
    """``get_pinned_tickers`` speaks the same ticker space as ``get_active_tickers``."""

    def test_returns_instrument_tickers_of_pinned_rows_on_the_exchange(
        self, db_session
    ):
        """The result holds ``Instrument.ticker`` values, keyed via the Yahoo pin."""
        ex = _make_exchange(db_session, "Pin Venue")
        add_and_flush(
            db_session,
            Instrument(
                ticker="VWCEd_EQ",
                short_name="VWCE",
                yfinance_ticker="VWCE.DE",
                instrument_type="ETF",
                exchange=ex,
            ),
        )
        add_and_flush(
            db_session,
            Instrument(
                ticker="OTHER",
                short_name="OTHER",
                yfinance_ticker="OTHER.DE",
                instrument_type="ETF",
                exchange=ex,
            ),
        )
        add_and_flush(db_session, PinnedInstrument(ticker="VWCE.DE", source="manual"))

        assert _repo(db_session).get_pinned_tickers(ex.id) == {"VWCEd_EQ"}

    def test_scoped_by_exchange(self, db_session):
        """A pinned row on another venue is not reported for this one."""
        here = _make_exchange(db_session, "Here")
        there = _make_exchange(db_session, "There")
        add_and_flush(
            db_session,
            Instrument(
                ticker="X", short_name="X", yfinance_ticker="X.L", exchange=there
            ),
        )
        add_and_flush(db_session, PinnedInstrument(ticker="X.L", source="manual"))

        assert _repo(db_session).get_pinned_tickers(here.id) == set()
        assert _repo(db_session).get_pinned_tickers(there.id) == {"X"}

    def test_scoped_by_instrument_type(self, db_session):
        """The stock and ETF build passes each see only their own pinned rows."""
        ex = _make_exchange(db_session, "Typed")
        for ticker, itype in (("S.DE", "STOCK"), ("E.DE", "ETF")):
            add_and_flush(
                db_session,
                Instrument(
                    ticker=ticker,
                    short_name=ticker,
                    yfinance_ticker=ticker,
                    instrument_type=itype,
                    exchange=ex,
                ),
            )
            add_and_flush(db_session, PinnedInstrument(ticker=ticker, source="manual"))
        repo = _repo(db_session)

        assert repo.get_pinned_tickers(ex.id, instrument_type="ETF") == {"E.DE"}
        assert repo.get_pinned_tickers(ex.id, instrument_type="STOCK") == {"S.DE"}
        assert repo.get_pinned_tickers(ex.id) == {"E.DE", "S.DE"}

    def test_no_pins_returns_empty_set(self, db_session):
        """Without pins nothing is reported."""
        ex = _make_exchange(db_session, "Unpinned")
        _make_instrument(db_session, ex, "AAA")

        assert _repo(db_session).get_pinned_tickers(ex.id) == set()


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


_PROTECTED_COLUMNS = (
    "asset_class",
    "fi_subclass",
    "duration_bucket",
    "currency_code",
    "yfinance_ticker",
)


def _screener_row(ticker: str, **overrides) -> dict:
    """A Screener-style batch row that would downgrade a hand-classified ETF."""
    return _batch_row(
        ticker,
        type="ETF",
        assetClass="equity",
        fiSubclass="government",
        durationBucket="long",
        currencyCode="USD",
        yfinanceTicker=f"{ticker}.SCR",
        **overrides,
    )


def _seed_commodity_etf(session, exchange, *, pinned: bool, ticker: str) -> None:
    """A hand-added commodity ETF, optionally pinned under source ``manual``."""
    add_and_flush(
        session,
        Instrument(
            ticker=ticker,
            short_name=ticker,
            instrument_type="ETF",
            asset_class="commodity",
            fi_subclass=None,
            duration_bucket=None,
            currency_code="GBp",
            yfinance_ticker=f"{ticker}.L",
            exchange_id=exchange.id,
        ),
    )
    if pinned:
        add_and_flush(session, PinnedInstrument(ticker=f"{ticker}.L", source="manual"))


class TestSaveInstrumentsBatchPinnedRows:
    """A Screener upsert must not rewrite the classification of a pinned row."""

    @staticmethod
    def _upsert_sql() -> str:
        session = MagicMock()
        UniverseRepository(session).save_instruments_batch(
            [{"ticker": "AGG", "shortName": "AGG"}], "exchange-id"
        )
        stmt = session.execute.call_args.args[0]
        return " ".join(str(stmt.compile(dialect=postgresql.dialect())).split())

    def test_statement_guards_each_protected_column_with_a_pin_check(self):
        """The PostgreSQL statement wraps all five columns in CASE WHEN EXISTS."""
        sql = self._upsert_sql().lower()
        pin_check = (
            "((exists (select * from pinned_instruments "
            "where pinned_instruments.ticker = instruments.yfinance_ticker)) "
            "and (instruments.ticker = instruments.yfinance_ticker "
            "or not (exists (select * from instruments as instruments_1 "
            "where instruments_1.yfinance_ticker = instruments.yfinance_ticker "
            "and instruments_1.ticker = instruments_1.yfinance_ticker))))"
        )

        for column in _PROTECTED_COLUMNS:
            kept = (
                "coalesce(instruments.currency_code"
                if column == "currency_code"
                else f"instruments.{column} else"
            )
            assert f"{column} = case when {pin_check} then {kept}" in sql
        assert sql.count(pin_check) == len(_PROTECTED_COLUMNS)

    def test_statement_leaves_other_columns_overwritten_as_before(self):
        """Unprotected columns keep their plain excluded/coalesce form."""
        sql = self._upsert_sql().lower()

        assert "short_name = excluded.short_name" in sql
        assert "isin = coalesce(excluded.isin, instruments.isin)" in sql
        assert "delisted_at = %(param_" in sql
        assert "instrument_type = excluded.instrument_type" in sql

    def test_pinned_row_keeps_its_classification_and_listing_fields(self, db_session):
        """A Screener row for a pinned ticker changes none of the five fields."""
        repo = _repo(db_session)
        lse = repo.save_exchange({"name": "London Stock Exchange", "id": 1})
        _seed_commodity_etf(db_session, lse, pinned=True, ticker="EGLN")

        repo.save_instruments_batch([_screener_row("EGLN")], lse.id)

        row = _reload(db_session, "EGLN")
        assert row.asset_class == "commodity"
        assert row.fi_subclass is None
        assert row.duration_bucket is None
        assert row.currency_code == "GBp"
        assert row.yfinance_ticker == "EGLN.L"

    def test_pinned_row_is_still_revived_and_named(self, db_session):
        """Delisting is cleared and the coalesced name/ISIN still fill in."""
        repo = _repo(db_session)
        lse = repo.save_exchange({"name": "London Stock Exchange", "id": 1})
        _seed_commodity_etf(db_session, lse, pinned=True, ticker="EGLN")
        db_session.execute(
            update(Instrument)
            .where(Instrument.ticker == "EGLN")
            .values(delisted_at=date(2026, 10, 1), delisting_return=-0.2)
        )

        repo.save_instruments_batch(
            [_screener_row("EGLN", name="iShares Physical Gold", isin="IE00B4ND3602")],
            lse.id,
        )

        row = _reload(db_session, "EGLN")
        assert row.delisted_at is None
        assert row.delisting_return is None
        assert row.name == "iShares Physical Gold"
        assert row.isin == "IE00B4ND3602"

    def test_unpinned_row_is_updated_exactly_as_before(self, db_session):
        """Without a pin the Screener values win on all five fields."""
        repo = _repo(db_session)
        lse = repo.save_exchange({"name": "London Stock Exchange", "id": 1})
        _seed_commodity_etf(db_session, lse, pinned=False, ticker="EGLN")

        repo.save_instruments_batch([_screener_row("EGLN")], lse.id)

        row = _reload(db_session, "EGLN")
        assert row.asset_class == "equity"
        assert row.fi_subclass == "government"
        assert row.duration_bucket == "long"
        assert row.currency_code == "USD"
        assert row.yfinance_ticker == "EGLN.SCR"

    def test_pin_is_matched_per_row_not_per_batch(self, db_session):
        """One batch holding a pinned and an unpinned ticker treats each on its own."""
        repo = _repo(db_session)
        lse = repo.save_exchange({"name": "London Stock Exchange", "id": 1})
        _seed_commodity_etf(db_session, lse, pinned=True, ticker="EGLN")
        _seed_commodity_etf(db_session, lse, pinned=False, ticker="IGLN")

        repo.save_instruments_batch(
            [_screener_row("EGLN"), _screener_row("IGLN")], lse.id
        )

        assert _reload(db_session, "EGLN").asset_class == "commodity"
        assert _reload(db_session, "IGLN").asset_class == "equity"

    def test_pinned_row_with_null_currency_takes_the_screener_value(self, db_session):
        """A stored NULL currency is filled rather than frozen at NULL."""
        repo = _repo(db_session)
        lse = repo.save_exchange({"name": "London Stock Exchange", "id": 1})
        _seed_commodity_etf(db_session, lse, pinned=True, ticker="EGLN")
        db_session.execute(
            update(Instrument)
            .where(Instrument.ticker == "EGLN")
            .values(currency_code=None)
        )

        repo.save_instruments_batch([_screener_row("EGLN")], lse.id)

        assert _reload(db_session, "EGLN").currency_code == "USD"

    def test_pinned_row_follows_the_screener_to_its_new_venue(self, db_session):
        """A pinned symbol the Screener lists elsewhere is moved, not duplicated.

        A second row would be a second active instrument for one symbol: the pin
        matches by symbol text, so neither row could ever be delisted and the
        sweep would fetch the symbol under two ids. Moving keeps the instrument
        id, and with it the price history, and the pin still shields the
        hand-set classification.
        """
        repo = _repo(db_session)
        lse = repo.save_exchange({"name": "London Stock Exchange", "id": 1})
        xetra = repo.save_exchange({"name": "Deutsche Börse Xetra", "id": 2})
        _seed_commodity_etf(db_session, lse, pinned=True, ticker="EGLN")
        original_id = _rows_for(db_session, "EGLN")[0].id

        repo.save_instruments_batch([_screener_row("EGLN")], xetra.id)

        rows = _rows_for(db_session, "EGLN")
        assert [(r.id, r.exchange_id) for r in rows] == [(original_id, xetra.id)]
        assert rows[0].asset_class == "commodity"
        assert rows[0].yfinance_ticker == "EGLN.L"

    def test_unpinned_row_is_still_rehomed(self, db_session):
        """The venue-change repair keeps working for rows nobody pinned."""
        repo = _repo(db_session)
        lse = repo.save_exchange({"name": "London Stock Exchange", "id": 1})
        xetra = repo.save_exchange({"name": "Deutsche Börse Xetra", "id": 2})
        _seed_commodity_etf(db_session, lse, pinned=False, ticker="EGLN")
        original_id = _rows_for(db_session, "EGLN")[0].id

        repo.save_instruments_batch([_batch_row("EGLN", type="ETF")], xetra.id)

        rows = _rows_for(db_session, "EGLN")
        assert [(r.id, r.exchange_id) for r in rows] == [(original_id, xetra.id)]


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
