"""Repository for the instruments the universe build must keep (pins)."""

from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from portopt_db.models.universe.pinned_instrument import PinnedInstrument
from portopt_db.models.universe.universe import Instrument
from portopt_db.repository import RepositoryBase


def _normalise(ticker: str) -> str:
    return ticker.strip().upper()


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

        Idempotent on ``(ticker, source)``. Also revives an already-delisted
        ``instruments`` row for the ticker, on every call, so a pin repairs a
        delisting that happened before or after the claim was first made.

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
        de-duplicated; blanks are dropped. Newly pinned tickers revive any
        delisted ``instruments`` row, as in :meth:`pin`.

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
        self._revive_delisted(wanted)

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

    def pinned_tickers(self) -> set[str]:
        """Return the distinct tickers pinned by any source."""
        return set(self.session.execute(select(PinnedInstrument.ticker)).scalars())

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
        # Filtering on the delisting columns keeps already-active rows from being
        # rewritten (and their updated_at bumped) on every pin.
        if not tickers:
            return
        self.session.execute(
            update(Instrument)
            .where(Instrument.yfinance_ticker.in_(tickers))
            .where(
                Instrument.delisted_at.is_not(None)
                | Instrument.delisting_return.is_not(None)
            )
            .values(delisted_at=None, delisting_return=None)
            .execution_options(synchronize_session=False)
        )
        self.session.flush()
