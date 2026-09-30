"""Tests for MarketJournal model and MarketJournalRepository.

Covers model round-trips (JSON dict equality, defaults), metadata registration,
UniqueConstraint enforcement, upsert idempotency, updated_at advancement on
re-upsert with created_at preserved, and read-filter correctness.
"""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import IntegrityError

from portopt_db.models.market_data.market_journal import MarketJournal
from portopt_db.repositories.market_data.market_journal_repository import (
    MarketJournalRepository,
)

_AS_OF_A = dt.date(2026, 9, 1)
_AS_OF_B = dt.date(2026, 9, 2)
_AS_OF_C = dt.date(2026, 9, 3)

_MACRO = {"CPI": 3.2, "FEDFUNDS": 5.25}
_MOVES = {"^GSPC": -0.5, "^DJI": -0.3}
_THEMES = {"macro": ["inflation", "rates"], "ticker": ["AAPL earnings"]}
_COUNTS = {"fred_rows": 12, "summary_rows": 8, "news_rows": 45}
_NARRATIVE = "CPI held steady; S&P 500 slipped 0.5%."


class TestMarketJournalMetadata:
    """market_journal must be visible on Base.metadata after the package import."""

    def test_table_registered_on_metadata(self, test_engine) -> None:
        """Verify market_journal is created by Base.metadata.create_all."""
        tables = set(inspect(test_engine).get_table_names())
        assert "market_journal" in tables

    def test_expected_indexes_exist(self, test_engine) -> None:
        """Verify ix_market_journal_as_of is created."""
        names = {
            idx["name"] for idx in inspect(test_engine).get_indexes("market_journal")
        }
        assert "ix_market_journal_as_of" in names

    def test_unique_constraint_exists(self, test_engine) -> None:
        """Verify uq_market_journal_as_of unique constraint is created."""
        constraints = {
            uc["name"]
            for uc in inspect(test_engine).get_unique_constraints("market_journal")
        }
        assert "uq_market_journal_as_of" in constraints


class TestMarketJournalRoundTrip:
    """ORM model round-trips JSON dicts and scalar columns correctly."""

    def test_json_dicts_survive_flush(self, db_session) -> None:
        """Verify all JSON columns are stored and retrieved with dict equality."""
        row = MarketJournal(
            as_of=_AS_OF_A,
            macro_deltas=_MACRO,
            market_moves=_MOVES,
            news_themes=_THEMES,
            source_counts=_COUNTS,
            narrative=_NARRATIVE,
        )
        db_session.add(row)
        db_session.flush()
        db_session.expire(row)

        fetched = db_session.execute(
            select(MarketJournal).where(MarketJournal.as_of == _AS_OF_A)
        ).scalar_one()
        assert fetched.macro_deltas == _MACRO
        assert fetched.market_moves == _MOVES
        assert fetched.news_themes == _THEMES
        assert fetched.source_counts == _COUNTS
        assert fetched.narrative == _NARRATIVE

    def test_region_defaults_to_us(self, db_session) -> None:
        """Verify region column carries 'US' when not explicitly set."""
        row = MarketJournal(
            as_of=dt.date(2026, 8, 1),
            macro_deltas={},
            market_moves={},
            news_themes={},
            source_counts={},
            narrative="",
        )
        db_session.add(row)
        db_session.flush()
        db_session.expire(row)

        fetched = db_session.execute(
            select(MarketJournal).where(MarketJournal.as_of == dt.date(2026, 8, 1))
        ).scalar_one()
        assert fetched.region == "US"

    def test_timestamps_populated_after_flush(self, db_session) -> None:
        """Verify created_at and updated_at are non-None after the first flush."""
        row = MarketJournal(
            as_of=dt.date(2026, 7, 1),
            macro_deltas={},
            market_moves={},
            news_themes={},
            source_counts={},
            narrative="",
        )
        db_session.add(row)
        db_session.flush()
        assert row.created_at is not None
        assert row.updated_at is not None


class TestMarketJournalUniqueConstraint:
    """Duplicate as_of must raise; distinct as_of must succeed."""

    def test_duplicate_as_of_raises_integrity_error(self, db_session) -> None:
        """Two rows with the same as_of must violate uq_market_journal_as_of."""
        for _ in range(2):
            db_session.add(
                MarketJournal(
                    as_of=dt.date(2026, 6, 1),
                    macro_deltas={},
                    market_moves={},
                    news_themes={},
                    source_counts={},
                    narrative="",
                )
            )
        with pytest.raises(IntegrityError):
            db_session.flush()

    def test_distinct_as_of_succeeds(self, db_session) -> None:
        """Two rows with different as_of values must not raise."""
        for d in (dt.date(2026, 5, 1), dt.date(2026, 5, 2)):
            db_session.add(
                MarketJournal(
                    as_of=d,
                    macro_deltas={},
                    market_moves={},
                    news_themes={},
                    source_counts={},
                    narrative="",
                )
            )
        db_session.flush()  # must not raise


class TestMarketJournalRepositoryUpsert:
    """MarketJournalRepository.upsert_journal idempotency and conflict behaviour."""

    def test_upsert_writes_one_row(self, db_session) -> None:
        """A single upsert_journal call must produce exactly one row."""
        repo = MarketJournalRepository(db_session)
        n = repo.upsert_journal(
            _AS_OF_A,
            macro_deltas=_MACRO,
            market_moves=_MOVES,
            news_themes=_THEMES,
            narrative=_NARRATIVE,
            source_counts=_COUNTS,
        )
        db_session.flush()
        assert n == 1
        rows = db_session.execute(select(MarketJournal)).scalars().all()
        assert len(rows) == 1

    def test_upsert_twice_still_one_row(self, db_session) -> None:
        """Two upserts for the same as_of must converge to one row with the latest data."""
        repo = MarketJournalRepository(db_session)
        repo.upsert_journal(
            _AS_OF_B,
            macro_deltas={},
            market_moves={},
            news_themes={},
            narrative="v1",
            source_counts={},
        )
        db_session.flush()
        repo.upsert_journal(
            _AS_OF_B,
            macro_deltas={"x": 1},
            market_moves={"^GSPC": -0.4},
            news_themes={"macro": ["rates"]},
            narrative="v2",
            source_counts={"a": 1},
            region="EU",
        )
        db_session.flush()

        rows = (
            db_session.execute(
                select(MarketJournal).where(MarketJournal.as_of == _AS_OF_B)
            )
            .scalars()
            .all()
        )
        assert len(rows) == 1
        # Exercise every column in _UPDATE_COLUMNS: a check on the narrative alone would
        # miss a column silently dropped from the update set.
        assert rows[0].narrative == "v2"
        assert rows[0].macro_deltas == {"x": 1}
        assert rows[0].market_moves == {"^GSPC": -0.4}
        assert rows[0].news_themes == {"macro": ["rates"]}
        assert rows[0].source_counts == {"a": 1}
        assert rows[0].region == "EU"

    def test_updated_at_advances_created_at_preserved(self, db_session) -> None:
        """On conflict, updated_at must advance while created_at stays frozen."""
        _PAST = dt.datetime(2000, 1, 1, 0, 0, 0)
        repo = MarketJournalRepository(db_session)
        repo.upsert_journal(
            _AS_OF_C,
            macro_deltas={},
            market_moves={},
            news_themes={},
            narrative="first",
            source_counts={},
        )
        db_session.flush()

        row = db_session.execute(
            select(MarketJournal).where(MarketJournal.as_of == _AS_OF_C)
        ).scalar_one()
        created_at = row.created_at

        # Force updated_at back to a known sentinel before the second upsert.
        db_session.execute(
            text("UPDATE market_journal SET updated_at = :ts WHERE as_of = :d"),
            {"ts": _PAST, "d": _AS_OF_C},
        )
        db_session.flush()
        db_session.expire(row)

        repo.upsert_journal(
            _AS_OF_C,
            macro_deltas={"y": 9},
            market_moves={},
            news_themes={},
            narrative="second",
            source_counts={},
        )
        db_session.flush()
        db_session.expire(row)

        row = db_session.execute(
            select(MarketJournal).where(MarketJournal.as_of == _AS_OF_C)
        ).scalar_one()
        assert row.narrative == "second"
        assert row.created_at == created_at, "created_at must not change on conflict"
        assert row.updated_at > _PAST, "updated_at must advance on conflict"


class TestMarketJournalRepositoryReads:
    """get_for_date and get_range filtering and ordering."""

    def _seed(self, db_session) -> None:
        """Insert three rows for _AS_OF_A, _AS_OF_B, _AS_OF_C."""
        repo = MarketJournalRepository(db_session)
        for d in (_AS_OF_A, _AS_OF_B, _AS_OF_C):
            repo.upsert_journal(
                d,
                macro_deltas={},
                market_moves={},
                news_themes={},
                narrative=str(d),
                source_counts={},
            )
        db_session.flush()

    def test_get_for_date_returns_correct_row(self, db_session) -> None:
        """get_for_date must return the row matching the exact as_of date."""
        self._seed(db_session)
        repo = MarketJournalRepository(db_session)
        row = repo.get_for_date(_AS_OF_B)
        assert row is not None
        assert row.as_of == _AS_OF_B
        assert row.narrative == str(_AS_OF_B)

    def test_get_for_date_missing_returns_none(self, db_session) -> None:
        """get_for_date must return None when no row exists for the date."""
        repo = MarketJournalRepository(db_session)
        assert repo.get_for_date(dt.date(1990, 1, 1)) is None

    def test_get_range_returns_chronological_slice(self, db_session) -> None:
        """get_range must return rows ordered by as_of ascending within the window."""
        self._seed(db_session)
        repo = MarketJournalRepository(db_session)
        rows = repo.get_range(_AS_OF_A, _AS_OF_B)
        assert len(rows) == 2
        assert rows[0].as_of == _AS_OF_A
        assert rows[1].as_of == _AS_OF_B

    def test_get_range_inclusive_on_both_ends(self, db_session) -> None:
        """get_range bounds are both inclusive."""
        self._seed(db_session)
        repo = MarketJournalRepository(db_session)
        rows = repo.get_range(_AS_OF_A, _AS_OF_C)
        assert len(rows) == 3

    def test_get_range_empty_when_no_rows_match(self, db_session) -> None:
        """get_range must return an empty sequence when no rows fall in the window."""
        repo = MarketJournalRepository(db_session)
        rows = repo.get_range(dt.date(1980, 1, 1), dt.date(1980, 12, 31))
        assert list(rows) == []
