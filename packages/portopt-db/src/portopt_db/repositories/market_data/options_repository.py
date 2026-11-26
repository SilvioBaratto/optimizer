"""Repository for an instrument's option-chain snapshots."""

from collections.abc import Sequence
from datetime import date
from typing import Any, cast
from uuid import UUID

from sqlalchemy import CursorResult, delete, func, select

from portopt_db.models.market_data.yfinance_data import OptionContract
from portopt_db.repository import RepositoryBase


class OptionsRepository(RepositoryBase):
    """Option-chain snapshots of an instrument."""

    def get_options_as_of(self, instrument_id: UUID) -> date | None:
        """Most recent options snapshot date for the instrument, or None."""
        stmt = select(func.max(OptionContract.as_of)).where(
            OptionContract.instrument_id == instrument_id
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def get_options_as_of_bulk(
        self, instrument_ids: Sequence[UUID]
    ) -> dict[UUID, date]:
        """Latest options snapshot date per instrument in one grouped query.

        Avoids a MAX(as_of) round-trip per instrument across the full sweep.
        """
        ids = list(instrument_ids)
        if not ids:
            return {}
        stmt = (
            select(OptionContract.instrument_id, func.max(OptionContract.as_of))
            .where(OptionContract.instrument_id.in_(ids))
            .group_by(OptionContract.instrument_id)
        )
        return {
            row[0]: row[1]
            for row in self.session.execute(stmt).all()
            if row[1] is not None
        }

    def upsert_option_chain(
        self, instrument_id: UUID, rows: list[dict[str, Any]]
    ) -> int:
        """Upsert option-contract snapshot rows (already flattened + typed).

        Deduped in-batch on (as_of, contract_symbol): a single snapshot must not
        touch the same natural key twice (PostgreSQL ON CONFLICT cardinality).
        """
        deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
        for r in rows:
            symbol = r.get("contract_symbol")
            as_of = r.get("as_of")
            if not symbol or as_of is None:
                continue
            deduped[(as_of, symbol)] = {"instrument_id": instrument_id, **r}
        prepared = list(deduped.values())
        if not prepared:
            return 0
        return self._upsert(
            OptionContract,
            prepared,
            constraint_name="uq_option_contract",
        )

    def prune_option_snapshots(self, instrument_id: UUID, *, keep: int) -> int:
        """Delete all but the newest ``keep`` snapshot dates of one instrument.

        Every weekly sweep inserts a fresh full chain (the snapshot date is part
        of the natural key), so without pruning the table grows by millions of
        rows a week. Dates, not rows, are counted: a snapshot is kept or dropped
        whole.

        Args:
            instrument_id: Instrument whose history is trimmed; others are
                untouched.
            keep: Number of most recent snapshot dates to retain; must be at
                least 1 so a bad value can never wipe the history.

        Returns:
            Number of contract rows deleted.

        Raises:
            ValueError: ``keep`` is below 1.
        """
        if keep < 1:
            raise ValueError(f"keep must be at least 1, got {keep}")
        cutoff = (
            self.session.execute(
                select(OptionContract.as_of)
                .where(OptionContract.instrument_id == instrument_id)
                .distinct()
                .order_by(OptionContract.as_of.desc())
                .offset(keep - 1)
                .limit(1)
            )
            .scalars()
            .first()
        )
        if cutoff is None:
            return 0
        result = self.session.execute(
            delete(OptionContract).where(
                OptionContract.instrument_id == instrument_id,
                OptionContract.as_of < cutoff,
            )
        )
        return cast("CursorResult[Any]", result).rowcount
