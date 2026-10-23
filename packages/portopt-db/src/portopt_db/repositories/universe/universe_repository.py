"""Repository for universe data access (exchanges and instruments)."""

import logging
import uuid as uuid_mod
from collections import defaultdict
from collections.abc import Sequence
from datetime import date
from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session, joinedload

from portopt_db.models.market_data.yfinance_data import TickerProfile
from portopt_db.models.universe.universe import Exchange, Instrument
from portopt_db.repository import RepositoryBase

logger = logging.getLogger(__name__)


class UniverseRepository(RepositoryBase):
    def __init__(self, session: Session):
        super().__init__(session)

    def save_exchange(self, exchange_data: dict[str, Any]) -> Exchange:
        """Insert or update an exchange by name, returning the persisted row.

        Written as an idempotent ``INSERT ... ON CONFLICT DO UPDATE`` on the
        unique ``exchanges.name`` column (``index_elements=["name"]``, since the
        column carries a column-level ``unique=True`` with no named constraint)
        rather than a SELECT-then-INSERT, so an at-least-once re-run converges
        to one row without racing the unique index. Only ``t212_id`` (and
        ``updated_at``) are in the conflict update set, preserving the row's id.

        Args:
            exchange_data: Dict with at least ``name`` and optionally ``id``
                (the Trading 212 exchange ID stored as ``t212_id``).

        Returns:
            The persisted Exchange ORM row with identity-map refreshed.
        """
        name = exchange_data.get("name", "")
        t212_id = exchange_data.get("id")

        self._upsert(
            Exchange,
            [{"id": uuid_mod.uuid4(), "name": name, "t212_id": t212_id}],
            index_elements=["name"],
            update_columns=["t212_id", "updated_at"],
        )
        self.session.flush()

        # populate_existing refreshes an already-identity-mapped row so the
        # returned object reflects the just-upserted t212_id, not a stale value.
        return self.session.execute(
            select(Exchange)
            .where(Exchange.name == name)
            .execution_options(populate_existing=True)
        ).scalar_one()

    def save_instruments_batch(
        self, instruments_data: list[dict[str, Any]], exchange_id: Any
    ) -> int:
        """Upsert a batch of instruments for the given exchange.

        Matched on the ``uq_instrument_ticker_exchange`` constraint (ticker +
        exchange). On conflict the mutable fields are overwritten, except that
        ``name``, ``isin`` and ``currency_code`` keep their stored value when the
        batch row has none. ``delisted_at``/``delisting_return`` are cleared so a
        re-listed instrument is automatically re-activated. A symbol that already
        exists on another exchange is first re-homed onto ``exchange_id`` (see
        :meth:`_rehome_moved_symbols`) so a venue change never yields a second
        row. The session is flushed but not committed.

        Args:
            instruments_data: Raw attribute dicts as returned by the T212 API.
            exchange_id: UUID of the owning exchange row.

        Returns:
            Number of rows in the batch (including no-op conflicts).
        """
        if not instruments_data:
            return 0

        rows = []
        for data in instruments_data:
            rows.append(
                {
                    "id": uuid_mod.uuid4(),
                    "ticker": data.get("ticker", ""),
                    "short_name": data.get("shortName", ""),
                    "name": data.get("name"),
                    "isin": data.get("isin"),
                    "instrument_type": data.get("type"),
                    "currency_code": data.get("currencyCode"),
                    "yfinance_ticker": data.get("yfinanceTicker"),
                    "asset_class": data.get("assetClass", "equity"),
                    "fi_subclass": data.get("fiSubclass"),
                    "duration_bucket": data.get("durationBucket"),
                    "exchange_id": exchange_id,
                }
            )

        self._rehome_moved_symbols([row["ticker"] for row in rows], exchange_id)

        stmt = pg_insert(Instrument).values(rows)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_instrument_ticker_exchange",
            set_={
                "short_name": stmt.excluded.short_name,
                # The universe source does not carry an ISIN (it is backfilled
                # from ticker profiles), so a missing value must not erase one
                # that is already stored.
                "name": func.coalesce(stmt.excluded.name, Instrument.name),
                "isin": func.coalesce(stmt.excluded.isin, Instrument.isin),
                "instrument_type": stmt.excluded.instrument_type,
                "currency_code": func.coalesce(
                    stmt.excluded.currency_code, Instrument.currency_code
                ),
                "yfinance_ticker": stmt.excluded.yfinance_ticker,
                "asset_class": stmt.excluded.asset_class,
                "fi_subclass": stmt.excluded.fi_subclass,
                "duration_bucket": stmt.excluded.duration_bucket,
                # Re-activating an instrument clears its delisting status.
                "delisted_at": None,
                "delisting_return": None,
                # Column.onupdate does not fire for ON CONFLICT DO UPDATE.
                "updated_at": func.now(),
            },
        )
        self.session.execute(stmt)
        self.session.flush()
        return len(rows)

    def _rehome_moved_symbols(self, tickers: list[str], exchange_id: Any) -> None:
        """Move rows for symbols that changed venue onto ``exchange_id``.

        The unique key is (ticker, exchange), so upserting a symbol that moved
        (e.g. a NASDAQ -> NYSE listing transfer) would insert a second row next
        to the stale one. Ticker-keyed readers then hit ``MultipleResultsFound``
        and the yfinance sweep fetches the symbol twice. Updating
        ``exchange_id`` in place keeps the instrument id, so its price and
        profile history stay attached.

        A symbol is left alone when it already has a row on ``exchange_id`` (the
        move would violate the unique key) or when several rows exist elsewhere
        (no way to tell which one is the predecessor).
        """
        homed_here = set(
            self.session.execute(
                select(Instrument.ticker)
                .where(Instrument.exchange_id == exchange_id)
                .where(Instrument.ticker.in_(tickers))
            ).scalars()
        )
        candidates = set(tickers) - homed_here
        if not candidates:
            return

        ids_by_ticker: dict[str, list[Any]] = defaultdict(list)
        for instrument_id, ticker in self.session.execute(
            select(Instrument.id, Instrument.ticker)
            .where(Instrument.exchange_id != exchange_id)
            .where(Instrument.ticker.in_(candidates))
        ):
            ids_by_ticker[ticker].append(instrument_id)

        moved_ids = [ids[0] for ids in ids_by_ticker.values() if len(ids) == 1]
        if moved_ids:
            self.session.execute(
                update(Instrument)
                .where(Instrument.id.in_(moved_ids))
                .values(exchange_id=exchange_id)
            )

    def mark_delisted(
        self,
        ticker: str,
        exchange_id: Any,
        delisted_at: date,
        delisting_return: float | None = None,
    ) -> bool:
        """Mark an instrument as delisted.

        Being absent from a universe snapshot is also what a still-trading name
        looks like after a screener floor change, a truncated page or a flipped
        canonical listing, so this method never invents a terminal return.

        Args:
            ticker: Ticker of the instrument within ``exchange_id``.
            exchange_id: UUID of the exchange the instrument belongs to.
            delisted_at: Date the instrument was first found missing from the
                universe snapshot (a detection date, not a last trading date).
            delisting_return: Realised terminal return when it is known, e.g.
                -1.0 for bankruptcy or the actual acquisition premium. ``None``
                leaves the column NULL so consumers resolve the unknown return
                themselves instead of booking a fabricated loss.

        Returns:
            True if the row was updated, False if no matching active instrument
            was found.
        """
        result: CursorResult[Any] = self.session.execute(  # type: ignore[assignment]
            update(Instrument)
            .where(Instrument.ticker == ticker)
            .where(Instrument.exchange_id == exchange_id)
            .where(Instrument.delisted_at.is_(None))  # only if not already marked
            .values(delisted_at=delisted_at, delisting_return=delisting_return)
        )
        self.session.flush()
        return bool(result.rowcount)

    def get_active_tickers(
        self, exchange_id: Any, instrument_type: str | None = None
    ) -> set[str]:
        """Return the set of non-delisted tickers for an exchange.

        ``instrument_type`` scopes the result (e.g. "STOCK" or "ETF"). The
        universe build processes stocks and ETFs in separate passes over
        exchanges that overlap (Xetra/LSE/Paris are in both sets), so delisting
        reconciliation MUST be scoped by type — otherwise the ETF pass would see
        the stock tickers the stock pass just flushed and mark them delisted.
        """
        stmt = (
            select(Instrument.ticker)
            .where(Instrument.exchange_id == exchange_id)
            .where(Instrument.delisted_at.is_(None))
        )
        if instrument_type is not None:
            stmt = stmt.where(Instrument.instrument_type == instrument_type)
        rows = self.session.execute(stmt).all()
        return {r[0] for r in rows}

    def get_active_listings(self) -> set[tuple[str, str]]:
        """Return ``(ticker, exchange name)`` for every non-delisted instrument.

        The universe build uses it to tell existing members from new entrants,
        so members can be retained down to the lower exit band of the
        investability floor instead of churning in and out at the entry band.
        """
        rows = self.session.execute(
            select(Instrument.ticker, Exchange.name)
            .join(Exchange, Instrument.exchange_id == Exchange.id)
            .where(Instrument.delisted_at.is_(None))
        ).all()
        return {(ticker, exchange) for ticker, exchange in rows}

    def set_t212_ticker(
        self, *, ticker: str, exchange_id: Any, t212_ticker: str | None
    ) -> bool:
        """Attach a Trading 212 ticker to a yfinance-sourced instrument.

        Matched by ``(ticker, exchange_id)`` — the unique key. Passing ``None``
        clears a cross-reference that no longer holds. Returns ``True`` if a row
        was updated, ``False`` if none matched.
        """
        result: CursorResult[Any] = self.session.execute(  # type: ignore[assignment]
            update(Instrument)
            .where(Instrument.ticker == ticker)
            .where(Instrument.exchange_id == exchange_id)
            .values(t212_ticker=t212_ticker)
        )
        self.session.flush()
        return bool(result.rowcount)

    def backfill_isin_from_profiles(self) -> int:
        """Copy ISIN from ``ticker_profiles`` onto instruments that lack one.

        The Screener universe build carries no ISIN, but the per-instrument
        profile fetch stores ``info["isin"]`` on ``ticker_profiles``. This copies
        that value onto the canonical ``instruments.isin`` for rows still missing
        one — a single portable UPDATE (correlated subquery; compiles on both
        PostgreSQL and the SQLite test engine). Never overwrites an existing
        value. Returns the number of rows filled.
        """
        profile_isin = (
            select(TickerProfile.isin)
            .where(TickerProfile.instrument_id == Instrument.id)
            .scalar_subquery()
        )
        result: CursorResult[Any] = self.session.execute(  # type: ignore[assignment]
            update(Instrument)
            .where(Instrument.isin.is_(None))
            .where(
                Instrument.id.in_(
                    select(TickerProfile.instrument_id).where(
                        TickerProfile.isin.is_not(None)
                    )
                )
            )
            .values(isin=profile_isin)
        )
        self.session.flush()
        return result.rowcount or 0

    def get_active_instruments(self) -> Sequence[Instrument]:
        """All non-delisted instruments (used by the T212 annotation step)."""
        return (
            self.session.execute(
                select(Instrument).where(Instrument.delisted_at.is_(None))
            )
            .scalars()
            .all()
        )

    def clear_all(self) -> tuple[int, int]:
        """Delete all instruments and exchanges, returning pre-deletion counts.

        Returns:
            Tuple of (exchange_count, instrument_count) recorded before deletion.
        """
        inst_count = self.session.execute(
            select(func.count()).select_from(Instrument)
        ).scalar_one()
        ex_count = self.session.execute(
            select(func.count()).select_from(Exchange)
        ).scalar_one()

        self.session.execute(delete(Instrument))
        self.session.execute(delete(Exchange))
        self.session.flush()

        return ex_count, inst_count

    def get_instrument_count(self) -> int:
        """Return the total number of instruments, including delisted ones."""
        return self.session.execute(
            select(func.count()).select_from(Instrument)
        ).scalar_one()

    def get_exchange_count(self) -> int:
        """Return the total number of exchanges."""
        return self.session.execute(
            select(func.count()).select_from(Exchange)
        ).scalar_one()

    def get_instruments(
        self,
        exchange_name: str | None = None,
        search: str | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> Sequence[Instrument]:
        """Query instruments with optional exchange filter and text search.

        Args:
            exchange_name: Restrict results to instruments on this exchange.
            search: Case-insensitive substring matched against ticker or name.
            skip: Rows to skip before returning (for offset pagination).
            limit: Maximum number of rows to return.

        Returns:
            Instruments eager-loaded with their exchange, deduplicated.
        """
        stmt = select(Instrument).options(joinedload(Instrument.exchange))
        if exchange_name:
            stmt = stmt.join(Exchange).where(Exchange.name == exchange_name)
        if search:
            term = f"%{search.strip()}%"
            stmt = stmt.where(
                (Instrument.ticker.ilike(term)) | (Instrument.name.ilike(term))
            )
        stmt = stmt.offset(skip).limit(limit)
        return self.session.execute(stmt).scalars().unique().all()

    def get_exchanges(self) -> Sequence[Exchange]:
        """Return all exchanges ordered alphabetically by name."""
        return (
            self.session.execute(select(Exchange).order_by(Exchange.name))
            .scalars()
            .all()
        )
