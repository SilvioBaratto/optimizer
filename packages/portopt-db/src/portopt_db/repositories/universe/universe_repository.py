"""Repository for universe data access (exchanges and instruments)."""

import logging
import uuid as uuid_mod
from collections import defaultdict
from collections.abc import Collection, Sequence
from datetime import date, timedelta
from typing import Any

from sqlalchemy import (
    and_,
    case,
    delete,
    exists,
    func,
    literal_column,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session, aliased, joinedload

from portopt_db.models.market_data.yfinance_data import PriceHistory, TickerProfile
from portopt_db.models.universe.pinned_instrument import PinnedInstrument
from portopt_db.models.universe.universe import Exchange, Instrument
from portopt_db.ports import InstrumentClass
from portopt_db.repositories.universe.pinned_instrument_repository import (
    row_carries_symbol,
)
from portopt_db.repository import RepositoryBase

logger = logging.getLogger(__name__)

# A delisted instrument that Yahoo has priced nothing for in this long is no
# longer worth a request in every sweep.
DORMANT_AFTER_DAYS = 365

# The order get_instrument_by_yfinance_ticker picks a listing by. instrument_classes_for
# sorts by it too, so its first row per symbol is the listing that lookup returns.
_LISTING_PREFERENCE = (
    Instrument.delisted_at.is_(None).desc(),
    Instrument.delisted_at.desc(),
    Instrument.created_at.desc(),
    Instrument.ticker,
    Instrument.id,
)


def _row_is_pinned() -> Any:
    """Correlated predicate: the instrument row being written is pinned.

    A pin names a Yahoo symbol, so it protects the row that carries that symbol
    (see :func:`row_carries_symbol`) and not a broker-era twin sharing it.

    The outer column is a literal reference, not ``Instrument.yfinance_ticker``:
    inside an ``INSERT ... ON CONFLICT DO UPDATE`` SQLAlchemy does not correlate
    the target table, so the ORM column would be added to the subquery's FROM and
    the predicate would match whenever any pin exists. Both the upsert and the
    UPDATE that use this write ``instruments`` unaliased, so the name resolves to
    the row being written.
    """
    return and_(
        exists().where(
            PinnedInstrument.ticker == literal_column("instruments.yfinance_ticker")
        ),
        row_carries_symbol(),
    )


class UniverseRepository(RepositoryBase):
    """Exchanges and instruments of the universe, and lookups over them."""

    def __init__(self, session: Session):
        super().__init__(session)

    def save_exchange(self, exchange_data: dict[str, Any]) -> Exchange:
        """Insert an exchange by name if it is new, returning the persisted row.

        Written as an idempotent ``INSERT ... ON CONFLICT DO UPDATE`` on the
        unique ``exchanges.name`` column (``index_elements=["name"]``, since the
        column carries a column-level ``unique=True`` with no named constraint)
        rather than a SELECT-then-INSERT, so an at-least-once re-run converges
        to one row without racing the unique index. Only ``updated_at`` is in the
        conflict update set, preserving the row's id.

        Args:
            exchange_data: Dict with at least ``name``; any other key is ignored.

        Returns:
            The persisted Exchange ORM row with identity-map refreshed.
        """
        name = exchange_data.get("name", "")

        self._upsert(
            Exchange,
            [{"id": uuid_mod.uuid4(), "name": name}],
            index_elements=["name"],
            update_columns=["updated_at"],
        )
        self.session.flush()

        # populate_existing refreshes an already-identity-mapped row so the
        # returned object reflects the just-upserted state, not a stale copy.
        return self.session.execute(
            select(Exchange)
            .where(Exchange.name == name)
            .execution_options(populate_existing=True)
        ).scalar_one()

    def get_exchange_by_name(self, name: str) -> Exchange | None:
        """Return the exchange with this exact name, or ``None``.

        A pure read: unlike ``save_exchange`` it never creates a row, so callers
        that merely need to resolve an existing venue use it instead.
        """
        return self.session.execute(
            select(Exchange).where(Exchange.name == name)
        ).scalar_one_or_none()

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
        :meth:`_rehome_moved_symbols`), pinned or not, so a venue change never
        yields a second row. The session is flushed but not committed.

        Rows pinned in ``pinned_instruments`` are operator-managed: their
        ``asset_class``, ``fi_subclass``, ``duration_bucket``, ``currency_code``
        and ``yfinance_ticker`` keep the stored value, because the Screener's
        defaults (``equity``, its own listing currency) would otherwise
        downgrade a hand-classified instrument such as a commodity ETF. A stored
        NULL ``currency_code`` is still filled from the batch.

        Args:
            instruments_data: Raw attribute dicts from the universe build source.
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
        pinned = _row_is_pinned()
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
                "currency_code": case(
                    (
                        pinned,
                        func.coalesce(
                            Instrument.currency_code, stmt.excluded.currency_code
                        ),
                    ),
                    else_=func.coalesce(
                        stmt.excluded.currency_code, Instrument.currency_code
                    ),
                ),
                "yfinance_ticker": case(
                    (pinned, Instrument.yfinance_ticker),
                    else_=stmt.excluded.yfinance_ticker,
                ),
                "asset_class": case(
                    (pinned, Instrument.asset_class),
                    else_=stmt.excluded.asset_class,
                ),
                "fi_subclass": case(
                    (pinned, Instrument.fi_subclass),
                    else_=stmt.excluded.fi_subclass,
                ),
                "duration_bucket": case(
                    (pinned, Instrument.duration_bucket),
                    else_=stmt.excluded.duration_bucket,
                ),
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
        (no way to tell which one is the predecessor). A pinned row moves like any
        other: the pin names the symbol, not the venue, and a second row would
        leave two active instruments for one symbol that no delisting could ever
        reconcile. The upsert that follows still keeps the pinned row's
        classification.
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
        A pinned instrument (see ``pinned_instruments``) is never marked, so no
        caller can delist a row an operator asked to keep.

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
            was found or the instrument is pinned.
        """
        result: CursorResult[Any] = self.session.execute(  # type: ignore[assignment]
            update(Instrument)
            .where(Instrument.ticker == ticker)
            .where(Instrument.exchange_id == exchange_id)
            .where(Instrument.delisted_at.is_(None))  # only if not already marked
            .where(~_row_is_pinned())
            .values(delisted_at=delisted_at, delisting_return=delisting_return)
        )
        self.session.flush()
        return bool(result.rowcount)

    def get_pinned_tickers(
        self, exchange_id: Any, instrument_type: str | None = None
    ) -> set[str]:
        """Return the ``Instrument.ticker`` of pinned rows on an exchange.

        Same ticker space and ``instrument_type`` scoping as
        :meth:`get_active_tickers`, so the universe build can subtract the result
        from its pre-build snapshot. The pin itself is keyed by the Yahoo symbol
        (``yfinance_ticker``), which may differ from the row's own ticker.
        """
        stmt = (
            select(Instrument.ticker)
            .where(Instrument.exchange_id == exchange_id)
            .where(_row_is_pinned())
        )
        if instrument_type is not None:
            stmt = stmt.where(Instrument.instrument_type == instrument_type)
        return set(self.session.execute(stmt).scalars())

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

    def instrument_classes_for(self, tickers: Collection[str]) -> list[InstrumentClass]:
        """Return the classification of every row that carries one of ``tickers``.

        The rows follow :meth:`YFinanceRepository.get_instrument_by_yfinance_ticker`'s
        order, so a caller that keeps the first row per symbol keeps the listing
        that lookup returns.

        Args:
            tickers: Yahoo symbols to look up; an empty collection matches no row.

        Returns:
            One entry per matching row, sorted by these keys in turn: not delisted
            before delisted, newer ``delisted_at`` first, newer ``created_at``
            first, then ``ticker`` and ``id``; empty when no row matches.
        """
        rows = self.session.execute(
            select(
                Instrument.yfinance_ticker,
                Instrument.id,
                Instrument.instrument_type,
                Instrument.asset_class,
                Instrument.fi_subclass,
            )
            .where(Instrument.yfinance_ticker.in_(list(tickers)))
            .order_by(*_LISTING_PREFERENCE)
        ).all()
        # The IN already drops NULL symbols; the condition narrows the column type.
        return [
            InstrumentClass(
                yfinance_ticker=ticker,
                id=instrument_id,
                instrument_type=instrument_type,
                asset_class=asset_class,
                fi_subclass=fi_subclass,
            )
            for ticker, instrument_id, instrument_type, asset_class, fi_subclass in rows
            if ticker is not None
        ]

    def set_isin_if_missing(self, *, ticker: str, exchange_id: Any, isin: str) -> bool:
        """Store an ISIN on an instrument that has none yet.

        Matched by ``(ticker, exchange_id)``. A stored ISIN is never replaced, for
        the same reason :meth:`backfill_isin_from_profiles` leaves one alone: an
        identifier already pinned to the row is more likely right than a later
        lookup.

        Args:
            ticker: Ticker of the instrument within ``exchange_id``.
            exchange_id: UUID of the exchange the instrument belongs to.
            isin: The ISIN to store; the caller has already vetted it.

        Returns:
            ``True`` if the ISIN was written, ``False`` if no row matched or the
            row already had an ISIN.
        """
        result: CursorResult[Any] = self.session.execute(  # type: ignore[assignment]
            update(Instrument)
            .where(Instrument.ticker == ticker)
            .where(Instrument.exchange_id == exchange_id)
            .where(Instrument.isin.is_(None))
            .values(isin=isin)
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
        value.

        A venue-suffixed symbol (``ASML.AS``, ``7203.T``) is skipped. The ISIN
        scrape can only be tied to a suffix-less symbol; for a suffixed one it
        answers with a secondary line of the issuer (a depositary receipt or a
        preferred share). The fetch no longer stores such codes, but profiles
        written before that still carry them, and copying one onto the instrument
        would pin the wrong security to it for good, since a stored ISIN is never
        overwritten. The same rule as ``has_exchange_suffix`` in ingestion, which
        this package cannot import.

        Returns the number of rows filled.
        """
        profile_isin = (
            select(TickerProfile.isin)
            .where(TickerProfile.instrument_id == Instrument.id)
            .scalar_subquery()
        )
        symbol = func.coalesce(Instrument.yfinance_ticker, Instrument.ticker)
        result: CursorResult[Any] = self.session.execute(  # type: ignore[assignment]
            update(Instrument)
            .where(Instrument.isin.is_(None))
            .where(symbol.not_like("%.%"))
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
        """All non-delisted instruments."""
        return (
            self.session.execute(
                select(Instrument).where(Instrument.delisted_at.is_(None))
            )
            .scalars()
            .all()
        )

    def has_active_instruments(self) -> bool:
        """Return whether at least one non-delisted instrument exists.

        An ``EXISTS`` probe, so the cost stays flat however large the universe
        grows; ``get_active_instruments`` would load every row just to be counted.
        """
        return bool(
            self.session.execute(
                select(exists().where(Instrument.delisted_at.is_(None)))
            ).scalar()
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
            Instruments eager-loaded with their exchange, deduplicated and
            ordered by ticker (then id) so offset pages are stable.
        """
        stmt = select(Instrument).options(joinedload(Instrument.exchange))
        if exchange_name:
            stmt = stmt.join(Exchange).where(Exchange.name == exchange_name)
        if search:
            term = f"%{search.strip()}%"
            stmt = stmt.where(
                (Instrument.ticker.ilike(term)) | (Instrument.name.ilike(term))
            )
        # ticker alone repeats across exchanges, so id breaks ties; without a
        # total order OFFSET pages can overlap or skip rows.
        stmt = stmt.order_by(Instrument.ticker, Instrument.id).offset(skip).limit(limit)
        return self.session.execute(stmt).scalars().unique().all()

    def get_exchanges(self) -> Sequence[Exchange]:
        """Return all exchanges ordered alphabetically by name."""
        return (
            self.session.execute(select(Exchange).order_by(Exchange.name))
            .scalars()
            .all()
        )

    def get_instruments_with_yfinance_ticker(
        self, *, today: date | None = None
    ) -> Sequence[Instrument]:
        """Return the instruments to ingest: those with a non-empty yfinance_ticker.

        A delisted row whose symbol is now carried by an active row (the old side
        of a venue transfer) is left out, so one symbol is never fetched and
        stored under two instrument ids. A delisted row with no active successor
        stays, because ``delisted_at`` records when a universe build stopped
        seeing a name, not when it stopped trading. It leaves once it has been
        delisted for ``DORMANT_AFTER_DAYS`` and has no stored bar from that
        period, i.e. Yahoo has priced nothing for it since; a pinned symbol never
        leaves. Its stored history stays. The exchange is eager-loaded.

        Args:
            today: Reference date for the dormancy cutoff; defaults to the
                current date.
        """
        cutoff = (today or date.today()) - timedelta(days=DORMANT_AFTER_DAYS)
        successor = aliased(Instrument)
        superseded = and_(
            Instrument.delisted_at.is_not(None),
            exists().where(
                successor.yfinance_ticker == Instrument.yfinance_ticker,
                successor.delisted_at.is_(None),
            ),
        )
        dormant = and_(
            Instrument.delisted_at.is_not(None),
            Instrument.delisted_at < cutoff,
            ~exists().where(
                PriceHistory.instrument_id == Instrument.id,
                PriceHistory.date >= cutoff,
            ),
            ~exists().where(PinnedInstrument.ticker == Instrument.yfinance_ticker),
        )
        return (
            self.session.execute(
                select(Instrument)
                .options(joinedload(Instrument.exchange))
                .where(Instrument.yfinance_ticker.isnot(None))
                .where(Instrument.yfinance_ticker != "")
                .where(~superseded)
                .where(~dormant)
            )
            .scalars()
            .unique()
            .all()
        )

    def get_instrument_by_yfinance_ticker(
        self, yfinance_ticker: str
    ) -> Instrument | None:
        """Return the instrument for a yfinance_ticker, with exchange eager-loaded.

        ``instruments`` is unique on ``(ticker, exchange)`` and ``yfinance_ticker``
        is indexed but not unique, so a symbol that moved venues leaves a delisted
        row beside its active one. The active listing wins; among several
        candidates the most recently delisted (or, for active rows, the most
        recently created) does, so the result is deterministic and a genuinely
        delisted holding still resolves.

        Args:
            yfinance_ticker: The Yahoo Finance symbol to look up.

        Returns:
            The chosen instrument, or ``None`` when no row has that symbol.
        """
        return (
            self.session.execute(
                select(Instrument)
                .options(joinedload(Instrument.exchange))
                .where(Instrument.yfinance_ticker == yfinance_ticker)
                .order_by(*_LISTING_PREFERENCE)
                .limit(1)
            )
            .scalars()
            .first()
        )

    def get_benchmark_coverage(
        self,
        tickers: list[str],
    ) -> dict[str, tuple[int, date | None]]:
        """Return ``{ticker: (price_row_count, latest_price_date)}``.

        Tickers without an instrument row (or with zero prices) come back as
        ``(0, None)``; an empty ``tickers`` returns an empty dict.
        """
        if not tickers:
            return {}

        stmt = (
            select(
                Instrument.yfinance_ticker,
                func.count(PriceHistory.id).label("price_rows"),
                func.max(PriceHistory.date).label("latest"),
            )
            .outerjoin(PriceHistory, PriceHistory.instrument_id == Instrument.id)
            .where(Instrument.yfinance_ticker.in_(tickers))
            .group_by(Instrument.yfinance_ticker)
        )
        rows = self.session.execute(stmt).all()
        coverage: dict[str, tuple[int, date | None]] = dict.fromkeys(tickers, (0, None))
        for ticker, count, latest in rows:
            # The column is nullable and the IN filter already excludes NULL; the
            # guard states that for the row type SQLAlchemy 2.1 now infers.
            if ticker is None:
                continue
            coverage[ticker] = (int(count or 0), latest)
        return coverage
