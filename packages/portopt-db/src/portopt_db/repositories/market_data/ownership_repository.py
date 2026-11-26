"""Repository for an instrument's holders and insider activity."""

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any
from uuid import UUID

import pandas as pd
from sqlalchemy import delete, select

from portopt_db.models.market_data.yfinance_data import (
    InsiderPurchaseSummary,
    InsiderRosterHolder,
    InsiderTransaction,
    InstitutionalHolder,
    MajorHolders,
    MutualFundHolder,
)
from portopt_db.repositories.market_data._mapping import (
    _last_wins,
    _safe_date,
    _safe_float,
    _safe_int,
    _safe_str,
    truncate,
)
from portopt_db.repository import RepositoryBase

_SENTINEL_DATE = date(1970, 1, 1)


def _insider_lot_order(row: dict[str, Any]) -> tuple[Any, ...]:
    """Sort key giving same-key insider lots a source-order-independent order."""
    shares, value = row["shares"], row["value"]
    return (
        shares is None,
        shares or 0,
        value is None,
        value or 0,
        row["ownership"] or "",
        row["position"] or "",
    )


def _latest_reported_per_holder(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one row per holder name, preferring the most recent report date."""
    ordered = sorted(rows, key=lambda r: r["date_reported"] or _SENTINEL_DATE)
    return _last_wins(ordered, "holder_name")


class OwnershipRepository(RepositoryBase):
    """Holders of an instrument and its insiders' trades, summary and roster."""

    def _drop_departed(
        self,
        model: Any,
        name_column: str,
        instrument_id: UUID,
        current: Sequence[str],
    ) -> None:
        """Delete an instrument's rows whose name is not in ``current``.

        The holder and roster tables mirror Yahoo's current list and keep no
        history, so a name that left it must not linger with stale figures. An
        empty ``current`` is ignored: an empty fetch says nothing about who left.
        """
        if not current:
            return
        with self.session.begin_nested():
            self.session.execute(
                delete(model).where(
                    model.instrument_id == instrument_id,
                    getattr(model, name_column).not_in(current),
                )
            )

    def upsert_institutional_holders(
        self, instrument_id: UUID, holders_df: pd.DataFrame
    ) -> int:
        """Upsert institutional holders from DataFrame.

        The table mirrors Yahoo's current list: after a non-empty frame, stored
        holders it no longer names are deleted.
        """
        rows = []
        for _, row_data in holders_df.iterrows():
            name = truncate(InstitutionalHolder, "holder_name", row_data.get("Holder"))
            if not name:
                continue
            rows.append(
                {
                    "instrument_id": instrument_id,
                    "holder_name": name,
                    "date_reported": _safe_date(row_data.get("Date Reported")),
                    "shares": _safe_int(row_data.get("Shares")),
                    "value": _safe_int(row_data.get("Value")),
                    "pct_held": _safe_float(row_data.get("pctHeld")),
                }
            )

        current = _latest_reported_per_holder(rows)
        written = self._upsert(
            InstitutionalHolder,
            current,
            constraint_name="uq_inst_holder_instrument_name",
        )
        self._drop_departed(
            InstitutionalHolder,
            "holder_name",
            instrument_id,
            [r["holder_name"] for r in current],
        )
        return written

    def get_institutional_holders(
        self, instrument_id: UUID
    ) -> Sequence[InstitutionalHolder]:
        """Return institutional holder rows for an instrument, ordered by holder name."""
        stmt = (
            select(InstitutionalHolder)
            .where(InstitutionalHolder.instrument_id == instrument_id)
            .order_by(InstitutionalHolder.holder_name)
        )
        return self.session.execute(stmt).scalars().all()

    def upsert_mutualfund_holders(
        self, instrument_id: UUID, holders_df: pd.DataFrame
    ) -> int:
        """Upsert mutual fund holders from DataFrame.

        The table mirrors Yahoo's current list: after a non-empty frame, stored
        holders it no longer names are deleted.
        """
        rows = []
        for _, row_data in holders_df.iterrows():
            name = truncate(MutualFundHolder, "holder_name", row_data.get("Holder"))
            if not name:
                continue
            rows.append(
                {
                    "instrument_id": instrument_id,
                    "holder_name": name,
                    "date_reported": _safe_date(row_data.get("Date Reported")),
                    "shares": _safe_int(row_data.get("Shares")),
                    "value": _safe_int(row_data.get("Value")),
                    "pct_held": _safe_float(row_data.get("pctHeld")),
                }
            )

        current = _latest_reported_per_holder(rows)
        written = self._upsert(
            MutualFundHolder,
            current,
            constraint_name="uq_mutual_fund_holder_instrument_name",
        )
        self._drop_departed(
            MutualFundHolder,
            "holder_name",
            instrument_id,
            [r["holder_name"] for r in current],
        )
        return written

    def get_mutualfund_holders(self, instrument_id: UUID) -> Sequence[MutualFundHolder]:
        """Return mutual fund holder rows for an instrument, ordered by holder name."""
        stmt = (
            select(MutualFundHolder)
            .where(MutualFundHolder.instrument_id == instrument_id)
            .order_by(MutualFundHolder.holder_name)
        )
        return self.session.execute(stmt).scalars().all()

    def upsert_insider_transactions(
        self, instrument_id: UUID, insiders_df: pd.DataFrame
    ) -> int:
        """Upsert insider transactions from a ``Ticker.insider_transactions`` frame.

        Rows sharing (insider, start date, transaction type) are numbered as
        lots, because the unique key cannot tell separate same-day tranches
        apart, and each lot keeps its own shares and value. A group Yahoo now
        lists with fewer lots loses its surplus stored lots; groups the frame
        does not mention are left as stored. A row whose Transaction and Text
        are both blank is kept under an empty transaction type; only rows
        without an insider name are dropped.

        Args:
            instrument_id: Instrument the transactions belong to.
            insiders_df: Frame as returned by yfinance.

        Returns:
            Number of distinct rows upserted.
        """
        rows: list[dict[str, Any]] = []
        for _, row_data in insiders_df.iterrows():
            name = _safe_str(row_data.get("Insider"), 500)
            tx_type = _safe_str(row_data.get("Transaction"), 200)
            if not tx_type:
                # yfinance 1.3.0 stopped populating the "Transaction" column
                # (now always ""); the human-readable action moved to "Text"
                # (e.g. "Sale at price 290.00 per share."). Without this
                # fallback every row was skipped and insider_transactions
                # silently stayed empty with no error logged.
                tx_type = _safe_str(row_data.get("Text"), 200)
            if not name:
                continue
            rows.append(
                {
                    "instrument_id": instrument_id,
                    "insider_name": name,
                    "position": _safe_str(row_data.get("Position"), 500),
                    # Real grants/vestings arrive with Transaction and Text both
                    # blank; the column is NOT NULL, so keep them under "".
                    "transaction_type": tx_type or "",
                    "shares": _safe_int(row_data.get("Shares")),
                    "value": _safe_int(row_data.get("Value")),
                    "start_date": _safe_date(row_data.get("Start Date"))
                    or _SENTINEL_DATE,
                    "ownership": _safe_str(row_data.get("Ownership"), 50),
                }
            )

        # Yahoo returns several real transactions sharing (instrument, insider,
        # start_date, transaction_type), e.g. two grants of different size on
        # one day. A single INSERT ... ON CONFLICT cannot touch the same
        # conflict key twice, and keeping only one of them would drop the other
        # lots' shares/value for good, so they are numbered with ``lot_index``.
        # Sorting each group first pins every lot to the same slot regardless of
        # the order Yahoo lists them in, keeping refetches idempotent.
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for r in rows:
            key = (
                r["instrument_id"],
                r["insider_name"],
                r["start_date"],
                r["transaction_type"],
            )
            groups.setdefault(key, []).append(r)

        numbered: list[dict[str, Any]] = []
        for group in groups.values():
            group.sort(key=_insider_lot_order)
            for lot_index, r in enumerate(group):
                r["lot_index"] = lot_index
                numbered.append(r)

        written = self._upsert(
            InsiderTransaction,
            numbered,
            constraint_name="uq_insider_tx_row",
        )
        self._drop_surplus_insider_lots(instrument_id, groups)
        return written

    def _drop_surplus_insider_lots(
        self,
        instrument_id: UUID,
        groups: Mapping[tuple[Any, ...], Sequence[dict[str, Any]]],
    ) -> None:
        """Delete stored lots beyond the size Yahoo now reports for a group.

        The upsert only writes lot slots that exist in the fetch, so a sale Yahoo
        stops listing twice would otherwise keep its second lot and be counted
        twice. Only lots at index 1 or higher can be surplus, which keeps the
        lookup small.
        """
        if not groups:
            return
        candidates = self.session.execute(
            select(
                InsiderTransaction.id,
                InsiderTransaction.insider_name,
                InsiderTransaction.start_date,
                InsiderTransaction.transaction_type,
                InsiderTransaction.lot_index,
            ).where(
                InsiderTransaction.instrument_id == instrument_id,
                InsiderTransaction.lot_index >= 1,
            )
        ).all()
        surplus: list[UUID] = []
        for row in candidates:
            group = groups.get(
                (instrument_id, row.insider_name, row.start_date, row.transaction_type)
            )
            if group is not None and row.lot_index >= len(group):
                surplus.append(row.id)
        if not surplus:
            return
        with self.session.begin_nested():
            self.session.execute(
                delete(InsiderTransaction).where(InsiderTransaction.id.in_(surplus))
            )

    def get_insider_transactions(
        self, instrument_id: UUID
    ) -> Sequence[InsiderTransaction]:
        """Return insider transaction rows for an instrument, ordered newest-first."""
        stmt = (
            select(InsiderTransaction)
            .where(InsiderTransaction.instrument_id == instrument_id)
            .order_by(InsiderTransaction.start_date.desc())
        )
        return self.session.execute(stmt).scalars().all()

    def upsert_major_holders(self, instrument_id: UUID, df: pd.DataFrame) -> int:
        """Upsert the 1:1 ownership breakdown from ``major_holders``.

        yfinance returns a label-indexed single-value DataFrame
        (``insidersPercentHeld`` etc.); read the first column as a lookup.
        """
        lookup = df.iloc[:, 0] if df.shape[1] else df.squeeze()

        def _v(key: str) -> Any:
            try:
                return lookup.get(key)
            except (AttributeError, TypeError):
                return None

        row = {
            "instrument_id": instrument_id,
            "insiders_percent_held": _safe_float(_v("insidersPercentHeld")),
            "institutions_percent_held": _safe_float(_v("institutionsPercentHeld")),
            "institutions_float_percent_held": _safe_float(
                _v("institutionsFloatPercentHeld")
            ),
            "institutions_count": _safe_int(_v("institutionsCount")),
        }
        return self._upsert(
            MajorHolders, [row], constraint_name="uq_major_holders_instrument"
        )

    def upsert_insider_purchases(self, instrument_id: UUID, df: pd.DataFrame) -> int:
        """Upsert the 1:1 6-month insider buy/sell summary.

        The label lives in the first column ("Insider Purchases Last 6m") with
        the figure in "Shares"; read known labels defensively.
        """
        label_col = df.columns[0]
        shares_by_label: dict[str, Any] = {}
        for _, r in df.iterrows():
            label = _safe_str(r.get(label_col), 100)
            if label:
                shares_by_label[label] = r.get("Shares")

        def _shares(*labels: str) -> int | None:
            for label in labels:
                if label in shares_by_label:
                    return _safe_int(shares_by_label[label])
            return None

        row = {
            "instrument_id": instrument_id,
            "purchase_shares": _shares("Purchases"),
            "sale_shares": _shares("Sales"),
            "net_shares": _shares(
                "Net Shares Purchased (Sold)", "Net Shares Purchased"
            ),
            "total_insider_shares": _shares("Total Insider Shares Held"),
        }
        return self._upsert(
            InsiderPurchaseSummary,
            [row],
            constraint_name="uq_insider_purchases_instrument",
        )

    def upsert_insider_roster(self, instrument_id: UUID, df: pd.DataFrame) -> int:
        """Upsert individual insiders from ``insider_roster_holders``.

        The table mirrors Yahoo's current roster: after a non-empty frame, stored
        insiders it no longer names are deleted.
        """

        def _col(r: Any, *names: str) -> Any:
            for name in names:
                if name in r:
                    return r.get(name)
            return None

        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for _, r in df.iterrows():
            name = _safe_str(_col(r, "Name", "name"), 500)
            if not name or name in seen:
                continue
            seen.add(name)
            rows.append(
                {
                    "instrument_id": instrument_id,
                    "insider_name": name,
                    "position": _safe_str(_col(r, "Position", "position"), 500),
                    "most_recent_transaction": _safe_str(
                        _col(r, "Most Recent Transaction"), 200
                    ),
                    "latest_transaction_date": _safe_date(
                        _col(r, "Latest Transaction Date")
                    ),
                    "shares_owned_directly": _safe_int(
                        _col(r, "Shares Owned Directly")
                    ),
                }
            )
        written = self._upsert(
            InsiderRosterHolder,
            rows,
            constraint_name="uq_insider_roster_instrument_name",
        )
        self._drop_departed(
            InsiderRosterHolder,
            "insider_name",
            instrument_id,
            [r["insider_name"] for r in rows],
        )
        return written
