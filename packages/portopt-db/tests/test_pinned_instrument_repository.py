"""``PinnedInstrumentRepository`` on SQLite.

The repository is the single writer of ``pinned_instruments`` for the fund's
mandate save, the ``add-instrument`` command and the universe build. These tests
pin the contract the three callers rely on: idempotent ``pin``, per-source
isolation in ``replace_pins``, ticker normalisation, and the delisting repair a
pin performs on an existing instrument row.
"""

from __future__ import annotations

import datetime
import uuid

from _fixtures import get_or_create_exchange
from sqlalchemy import select

from portopt_db.models import Instrument, PinnedInstrument
from portopt_db.repositories.universe import PinnedInstrumentRepository

MANDATE_A = f"mandate:{uuid.uuid4()}"
MANDATE_B = f"mandate:{uuid.uuid4()}"


def _rows(session) -> list[tuple[str, str]]:
    """All pins as sorted ``(ticker, source)`` pairs."""
    stmt = select(PinnedInstrument.ticker, PinnedInstrument.source)
    return sorted((t, s) for t, s in session.execute(stmt).all())


def _instrument(
    session,
    yfinance_ticker: str,
    *,
    ticker: str | None = None,
    delisting_return: float | None = None,
    exchange: str = "Deutsche Börse Xetra",
    delisted_at: datetime.date | None = datetime.date(2026, 10, 1),
) -> Instrument:
    """An instrument whose yfinance symbol is ``yfinance_ticker``.

    By default it left the universe without a recorded terminal return, the state
    the weekly build leaves a line in when the Screener stops listing it.
    """
    row = Instrument(
        ticker=ticker or yfinance_ticker.split(".")[0],
        short_name="Fund",
        yfinance_ticker=yfinance_ticker,
        exchange=get_or_create_exchange(session, exchange),
        delisted_at=delisted_at,
        delisting_return=delisting_return,
    )
    session.add(row)
    session.flush()
    return row


class TestPin:
    """``pin`` adds one owner's claim and repairs a delisted instrument row."""

    def test_pin_inserts_and_reports_added(self, db_session):
        """A first pin inserts the row and returns True."""
        repo = PinnedInstrumentRepository(db_session)

        assert repo.pin("VWCE.DE", "manual") is True
        assert _rows(db_session) == [("VWCE.DE", "manual")]

    def test_repeated_pin_leaves_one_row(self, db_session):
        """Pinning the same (ticker, source) again adds nothing and returns False."""
        repo = PinnedInstrumentRepository(db_session)
        repo.pin("VWCE.DE", "manual")

        assert repo.pin("VWCE.DE", "manual") is False
        assert _rows(db_session) == [("VWCE.DE", "manual")]

    def test_ticker_is_stripped_and_uppercased(self, db_session):
        """Whitespace and case never create a second row for one ticker."""
        repo = PinnedInstrumentRepository(db_session)

        assert repo.pin("  vwce.de ", "manual") is True
        assert repo.pin("VWCE.DE", "manual") is False
        assert _rows(db_session) == [("VWCE.DE", "manual")]

    def test_blank_ticker_is_ignored(self, db_session):
        """A blank ticker cannot be pinned and writes nothing."""
        repo = PinnedInstrumentRepository(db_session)

        assert repo.pin("   ", "manual") is False
        assert _rows(db_session) == []

    def test_same_ticker_under_two_sources_keeps_both(self, db_session):
        """Two owners of one ticker each hold their own row."""
        repo = PinnedInstrumentRepository(db_session)
        repo.pin("SXR8.DE", "manual")
        repo.pin("SXR8.DE", MANDATE_A)

        assert _rows(db_session) == sorted(
            [("SXR8.DE", "manual"), ("SXR8.DE", MANDATE_A)]
        )

    def test_pin_of_unknown_instrument_is_a_noop_for_instruments(self, db_session):
        """Pinning a ticker with no instrument row touches no instruments."""
        repo = PinnedInstrumentRepository(db_session)

        repo.pin("NOROW.DE", "manual")

        assert db_session.execute(select(Instrument)).first() is None

    def test_pin_clears_delisting_on_matching_instrument(self, db_session):
        """Pinning revives an instrument the build had already delisted."""
        row = _instrument(db_session, "EGLN.L")
        repo = PinnedInstrumentRepository(db_session)

        repo.pin("egln.l", "manual")
        db_session.refresh(row)

        assert row.delisted_at is None
        assert row.delisting_return is None

    def test_pin_leaves_other_delisted_instruments_alone(self, db_session):
        """Only the pinned ticker's row is revived."""
        other = _instrument(db_session, "OTHER.DE")
        repo = PinnedInstrumentRepository(db_session)

        repo.pin("EGLN.L", "manual")
        db_session.refresh(other)

        assert other.delisted_at == datetime.date(2026, 10, 1)

    def test_repeat_pin_still_repairs_a_later_delisting(self, db_session):
        """An already-existing pin still revives a row delisted after it was made."""
        row = _instrument(db_session, "EGLN.L")
        repo = PinnedInstrumentRepository(db_session)
        repo.pin("EGLN.L", "manual")
        row.delisted_at = datetime.date(2026, 10, 20)
        db_session.flush()

        assert repo.pin("EGLN.L", "manual") is False
        db_session.refresh(row)

        assert row.delisted_at is None


class TestPinLeavesRealDelistingsAlone:
    """A pin revives a line that left the universe, not a delisting on record."""

    def test_pin_does_not_revive_a_broker_era_sibling_of_the_canonical_row(
        self, db_session
    ):
        """VUSA.L is carried by its canonical row; the old VUSAl_EQ line stays dead."""
        canonical = _instrument(db_session, "VUSA.L", ticker="VUSA.L", delisted_at=None)
        sibling = _instrument(db_session, "VUSA.L", ticker="VUSAl_EQ")
        repo = PinnedInstrumentRepository(db_session)

        repo.pin("VUSA.L", "manual")
        db_session.refresh(canonical)
        db_session.refresh(sibling)

        assert canonical.delisted_at is None
        assert sibling.delisted_at == datetime.date(2026, 10, 1)

    def test_pin_revives_the_canonical_row_but_not_its_sibling(self, db_session):
        """Both lines delisted: only the row whose ticker is the symbol returns."""
        canonical = _instrument(db_session, "VUSA.L", ticker="VUSA.L")
        sibling = _instrument(db_session, "VUSA.L", ticker="VUSAl_EQ")

        PinnedInstrumentRepository(db_session).pin("VUSA.L", "manual")
        db_session.refresh(canonical)
        db_session.refresh(sibling)

        assert canonical.delisted_at is None
        assert sibling.delisted_at == datetime.date(2026, 10, 1)

    def test_pin_revives_the_only_row_of_a_symbol_whatever_its_ticker(self, db_session):
        """A hand-inserted row with a broker-style ticker has no canonical rival."""
        only = _instrument(db_session, "VWCE.DE", ticker="VWCEd_EQ")

        PinnedInstrumentRepository(db_session).pin("VWCE.DE", "manual")
        db_session.refresh(only)

        assert only.delisted_at is None

    def test_pin_keeps_a_delisting_that_carries_a_terminal_return(self, db_session):
        """A recorded return marks a real delisting; no claim undoes it or erases it."""
        dead = _instrument(db_session, "DEAD.L", ticker="DEAD.L", delisting_return=-1.0)

        PinnedInstrumentRepository(db_session).pin("DEAD.L", "manual")
        db_session.refresh(dead)

        assert dead.delisted_at == datetime.date(2026, 10, 1)
        assert dead.delisting_return == -1.0

    def test_pin_does_not_revive_the_old_side_of_a_venue_transfer(self, db_session):
        """The symbol is already carried by an active row on its new venue."""
        old = _instrument(db_session, "XYZ", ticker="XYZ", exchange="NASDAQ")
        new = _instrument(
            db_session, "XYZ", ticker="XYZ", exchange="NYSE", delisted_at=None
        )

        PinnedInstrumentRepository(db_session).pin("XYZ", "manual")
        db_session.refresh(old)
        db_session.refresh(new)

        assert old.delisted_at == datetime.date(2026, 10, 1)
        assert new.delisted_at is None


class TestReplacePins:
    """``replace_pins`` makes one source's set equal a list, never touching others."""

    def test_removes_only_stale_rows_of_the_source(self, db_session):
        """Stale rows of the source go; other sources keep theirs."""
        repo = PinnedInstrumentRepository(db_session)
        repo.pin("OLD.DE", MANDATE_A)
        repo.pin("KEEP.DE", MANDATE_A)
        repo.pin("OLD.DE", "manual")
        repo.pin("OLD.DE", MANDATE_B)

        repo.replace_pins(MANDATE_A, ["KEEP.DE", "NEW.DE"])

        assert _rows(db_session) == sorted(
            [
                ("KEEP.DE", MANDATE_A),
                ("NEW.DE", MANDATE_A),
                ("OLD.DE", "manual"),
                ("OLD.DE", MANDATE_B),
            ]
        )

    def test_empty_list_clears_the_source_only(self, db_session):
        """An empty list releases every pin of the source and nothing else."""
        repo = PinnedInstrumentRepository(db_session)
        repo.pin("A.DE", MANDATE_A)
        repo.pin("B.DE", "manual")

        repo.replace_pins(MANDATE_A, [])

        assert _rows(db_session) == [("B.DE", "manual")]

    def test_normalises_and_deduplicates_tickers(self, db_session):
        """Case, whitespace, duplicates and blanks collapse to the distinct set."""
        repo = PinnedInstrumentRepository(db_session)

        repo.replace_pins(MANDATE_A, [" vwce.de", "VWCE.DE", "  ", "iwda.as"])

        assert _rows(db_session) == [("IWDA.AS", MANDATE_A), ("VWCE.DE", MANDATE_A)]

    def test_is_idempotent(self, db_session):
        """Replacing with the same list twice leaves the same rows."""
        repo = PinnedInstrumentRepository(db_session)
        repo.replace_pins(MANDATE_A, ["A.DE", "B.DE"])

        repo.replace_pins(MANDATE_A, ["A.DE", "B.DE"])

        assert _rows(db_session) == [("A.DE", MANDATE_A), ("B.DE", MANDATE_A)]

    def test_clears_delisting_for_the_new_pins(self, db_session):
        """Newly pinned tickers revive their delisted instrument rows."""
        row = _instrument(db_session, "EGLN.L")
        repo = PinnedInstrumentRepository(db_session)

        repo.replace_pins(MANDATE_A, ["EGLN.L"])
        db_session.refresh(row)

        assert row.delisted_at is None
        assert row.delisting_return is None

    def test_a_later_delisting_survives_the_next_save_of_the_same_set(self, db_session):
        """Only a new claim repairs a delisting; re-saving the mandate does not.

        Every co-pilot directive re-saves the mandate, so reviving on each save
        would undo a delisting that happened after the ticker was allowlisted.
        """
        row = _instrument(db_session, "EGLN.L", delisted_at=None)
        repo = PinnedInstrumentRepository(db_session)
        repo.replace_pins(MANDATE_A, ["EGLN.L"])
        row.delisted_at = datetime.date(2026, 11, 1)
        db_session.flush()

        repo.replace_pins(MANDATE_A, ["EGLN.L"])
        db_session.refresh(row)

        assert row.delisted_at == datetime.date(2026, 11, 1)

    def test_a_ticker_added_to_an_existing_set_is_revived(self, db_session):
        """The claim that is new in this save repairs its row, the others stay."""
        kept = _instrument(db_session, "OLD.L", delisted_at=None)
        new = _instrument(db_session, "NEW.L")
        repo = PinnedInstrumentRepository(db_session)
        repo.replace_pins(MANDATE_A, ["OLD.L"])
        kept.delisted_at = datetime.date(2026, 11, 1)
        db_session.flush()

        repo.replace_pins(MANDATE_A, ["OLD.L", "NEW.L"])
        db_session.refresh(kept)
        db_session.refresh(new)

        assert new.delisted_at is None
        assert kept.delisted_at == datetime.date(2026, 11, 1)


class TestUnpin:
    """``unpin`` releases exactly one (ticker, source) claim."""

    def test_removes_the_row_and_reports_it(self, db_session):
        """Unpinning an existing claim deletes it and returns True."""
        repo = PinnedInstrumentRepository(db_session)
        repo.pin("VWCE.DE", "manual")

        assert repo.unpin(" vwce.de", "manual") is True
        assert _rows(db_session) == []

    def test_missing_row_reports_false(self, db_session):
        """Unpinning a claim that does not exist returns False."""
        repo = PinnedInstrumentRepository(db_session)

        assert repo.unpin("VWCE.DE", "manual") is False

    def test_other_sources_survive(self, db_session):
        """A ticker stays pinned while any other owner still claims it."""
        repo = PinnedInstrumentRepository(db_session)
        repo.pin("VWCE.DE", "manual")
        repo.pin("VWCE.DE", MANDATE_A)

        repo.unpin("VWCE.DE", MANDATE_A)

        assert _rows(db_session) == [("VWCE.DE", "manual")]


class TestReads:
    """The read side the universe build and annotate step consume."""

    def test_pinned_tickers_is_distinct_across_sources(self, db_session):
        """A ticker pinned by two owners appears once."""
        repo = PinnedInstrumentRepository(db_session)
        repo.pin("VWCE.DE", "manual")
        repo.pin("VWCE.DE", MANDATE_A)
        repo.pin("EGLN.L", MANDATE_B)

        assert repo.pinned_tickers() == {"VWCE.DE", "EGLN.L"}

    def test_pinned_tickers_can_be_scoped_to_one_source(self, db_session):
        """The annotation step asks for the operator's own pins only."""
        repo = PinnedInstrumentRepository(db_session)
        repo.pin("VWCE.DE", "manual")
        repo.pin("EGLN.L", MANDATE_A)
        repo.pin("SXR8.DE", "manual")
        repo.pin("SXR8.DE", MANDATE_A)

        assert repo.pinned_tickers(source="manual") == {"VWCE.DE", "SXR8.DE"}
        assert repo.pinned_tickers(source=MANDATE_A) == {"EGLN.L", "SXR8.DE"}
        assert repo.pinned_tickers(source="unknown") == set()

    def test_pinned_tickers_empty(self, db_session):
        """No pins yields an empty set."""
        assert PinnedInstrumentRepository(db_session).pinned_tickers() == set()

    def test_is_pinned_normalises_its_argument(self, db_session):
        """Lookup ignores case and surrounding whitespace; blank is never pinned."""
        repo = PinnedInstrumentRepository(db_session)
        repo.pin("VWCE.DE", "manual")

        assert repo.is_pinned(" vwce.de") is True
        assert repo.is_pinned("EGLN.L") is False
        assert repo.is_pinned("  ") is False
