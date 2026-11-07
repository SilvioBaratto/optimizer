"""Repository for the instruments the universe build must keep (pins)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from sqlalchemy import exists, literal_column, or_, select, update
from sqlalchemy.orm import Session, aliased

from portopt_db.models.universe.pinned_instrument import PinnedInstrument
from portopt_db.models.universe.universe import Instrument
from portopt_db.repository import RepositoryBase


def _normalise(ticker: str) -> str:
    return ticker.strip().upper()


def row_carries_symbol() -> Any:
    """Correlated predicate: the instrument row being written carries its symbol.

    Several rows can share one ``yfinance_ticker``. The row whose own ``ticker``
    is that symbol is the canonical listing; a row keyed by an old broker ticker
    (the ``VUSAl_EQ`` left behind by the Yahoo-ticker backfill) is a sibling. A
    sibling carries the symbol only when no canonical row exists, so a hand-made
    row with a broker-style ticker stays reachable by its pin while a stale twin
    beside a canonical row does not inherit the canonical row's protection.

    The outer columns are literal references for the reason given on
    ``_row_is_pinned`` in the universe repository: the statements using this write
    ``instruments`` unaliased, and inside ``INSERT ... ON CONFLICT DO UPDATE``
    SQLAlchemy does not correlate the ORM columns to the target table.
    """
    canonical = aliased(Instrument)
    return or_(
        literal_column("instruments.ticker")
        == literal_column("instruments.yfinance_ticker"),
        ~exists().where(
            canonical.yfinance_ticker == literal_column("instruments.yfinance_ticker"),
            canonical.ticker == canonical.yfinance_ticker,
        ),
    )


class PinnedInstrumentRepository(RepositoryBase):
    """Read and write ``pinned_instruments`` on an injected sync session.

    Ticker arguments are stripped and upper-cased once here, so callers never
    produce near-duplicate rows. A blank ticker is ignored rather than raised:
    it names nothing to protect. Writes only ``flush()``; the caller owns the
    transaction. Plain ORM select/add/delete keeps it portable to SQLite, where
    the fund tests run.
    """

    def __init__(self, session: Session):
        super().__init__(session)

    def pin(self, ticker: str, source: str) -> bool:
        """Record that ``source`` wants ``ticker`` kept in the universe.

        Idempotent on ``(ticker, source)``. Also revives an ``instruments`` row
        the build had taken out of the universe, on every call, so a pin repairs
        a removal that happened before or after the claim was first made (see
        :meth:`_revive_delisted` for which rows qualify).

        Args:
            ticker: Yahoo ticker; stripped and upper-cased. Blank is ignored.
            source: Owner of the pin, ``manual`` or ``mandate:<portfolio_uuid>``.

        Returns:
            True when a row was added, False when the claim already existed or
            the ticker was blank.
        """
        normalised = _normalise(ticker)
        if not normalised:
            return False
        added = self._find(normalised, source) is None
        if added:
            self.session.add(PinnedInstrument(ticker=normalised, source=source))
        self.session.flush()
        self._revive_delisted({normalised})
        return added

    def replace_pins(self, source: str, tickers: Iterable[str]) -> None:
        """Make the set of tickers pinned by ``source`` equal ``tickers``.

        Rows of other sources are never touched, so a ticker stays protected
        while any other owner still claims it. Tickers are normalised and
        de-duplicated; blanks are dropped. Only tickers this call newly pins
        revive their delisted ``instruments`` rows, as in :meth:`pin`: a save
        that merely repeats an existing claim must not undo a delisting that
        happened after it was made.

        Args:
            source: Owner whose pins are replaced.
            tickers: The complete set ``source`` wants pinned; empty releases
                every pin of ``source``.
        """
        wanted = {n for n in map(_normalise, tickers) if n}
        existing = {
            row.ticker: row
            for row in self.session.execute(
                select(PinnedInstrument).where(PinnedInstrument.source == source)
            ).scalars()
        }
        for ticker, row in existing.items():
            if ticker not in wanted:
                self.session.delete(row)
        for ticker in sorted(wanted - existing.keys()):
            self.session.add(PinnedInstrument(ticker=ticker, source=source))
        self.session.flush()
        self._revive_delisted(wanted - existing.keys())

    def unpin(self, ticker: str, source: str) -> bool:
        """Release ``source``'s claim on ``ticker``.

        Args:
            ticker: Yahoo ticker; stripped and upper-cased.
            source: Owner whose claim is released.

        Returns:
            True when a row was removed, False when there was none.
        """
        normalised = _normalise(ticker)
        row = self._find(normalised, source) if normalised else None
        if row is None:
            return False
        self.session.delete(row)
        self.session.flush()
        return True

    def pinned_tickers(self, source: str | None = None) -> set[str]:
        """Return the distinct pinned tickers.

        Args:
            source: Restrict to the pins of this owner (``manual`` or
                ``mandate:<portfolio_uuid>``); ``None`` spans every source.
        """
        stmt = select(PinnedInstrument.ticker)
        if source is not None:
            stmt = stmt.where(PinnedInstrument.source == source)
        return set(self.session.execute(stmt).scalars())

    def is_pinned(self, ticker: str) -> bool:
        """Return whether any source pins ``ticker`` (stripped, case-insensitive)."""
        normalised = _normalise(ticker)
        if not normalised:
            return False
        return (
            self.session.execute(
                select(PinnedInstrument.id)
                .where(PinnedInstrument.ticker == normalised)
                .limit(1)
            ).first()
            is not None
        )

    def _find(self, ticker: str, source: str) -> PinnedInstrument | None:
        return self.session.execute(
            select(PinnedInstrument)
            .where(PinnedInstrument.ticker == ticker)
            .where(PinnedInstrument.source == source)
        ).scalar_one_or_none()

    def _revive_delisted(self, tickers: set[str]) -> None:
        """Re-activate the rows a claim on ``tickers`` is meant to bring back.

        A row qualifies when it left the universe without a recorded terminal
        return (``delisted_at`` set, ``delisting_return`` NULL: the line is
        absent from the Screener, not necessarily dead), carries the symbol (see
        :func:`row_carries_symbol`) and the symbol has no active row yet. A row
        with a recorded return is a real delisting that no claim reverses, and its
        return is never erased; an old broker-era twin or the superseded side of
        a venue transfer would otherwise come back beside the live row and have the
        symbol fetched twice.
        """
        if not tickers:
            return
        carrier = aliased(Instrument)
        symbol_is_active = exists().where(
            carrier.yfinance_ticker == literal_column("instruments.yfinance_ticker"),
            carrier.delisted_at.is_(None),
        )
        self.session.execute(
            update(Instrument)
            .where(Instrument.yfinance_ticker.in_(tickers))
            .where(Instrument.delisted_at.is_not(None))
            .where(Instrument.delisting_return.is_(None))
            .where(row_carries_symbol())
            .where(~symbol_is_active)
            .values(delisted_at=None)
            .execution_options(synchronize_session=False)
        )
        self.session.flush()
