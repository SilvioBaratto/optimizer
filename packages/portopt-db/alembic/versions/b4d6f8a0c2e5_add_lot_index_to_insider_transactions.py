"""Add ``lot_index`` to ``insider_transactions`` and widen its natural key.

Yahoo returns distinct insider transactions that render identically (same
insider, day and text, e.g. two grants of different size). The old key
``(instrument_id, insider_name, start_date, transaction_type)`` forced the
writer to keep one of them, so the other lots' shares/value were lost and could
never be recovered by re-ingesting. ``lot_index`` numbers the rows sharing the
other four columns. Existing rows become lot 0 and the dropped lots reappear on
the next insider refetch.

Revision ID: b4d6f8a0c2e5
Revises: a3c5e7f9b1d4
Create Date: 2026-10-23
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "b4d6f8a0c2e5"
down_revision: str | Sequence[str] | None = "a3c5e7f9b1d4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "insider_transactions"
_CONSTRAINT = "uq_insider_tx_row"
_KEY = ["instrument_id", "insider_name", "start_date", "transaction_type"]


def upgrade() -> None:
    """Add ``lot_index`` (default 0) and include it in ``uq_insider_tx_row``."""
    # Literal table name: the chain-replay test reads names from the AST.
    op.add_column(
        "insider_transactions",
        sa.Column("lot_index", sa.SmallInteger(), nullable=False, server_default="0"),
    )
    with op.batch_alter_table(_TABLE) as batch:
        batch.drop_constraint(_CONSTRAINT, type_="unique")
        batch.create_unique_constraint(_CONSTRAINT, [*_KEY, "lot_index"])


def downgrade() -> None:
    """Restore the four-column key; extra lots collapse to the lowest lot_index."""
    op.execute(
        sa.text(
            "DELETE FROM insider_transactions "
            "WHERE lot_index <> (SELECT MIN(t.lot_index) FROM insider_transactions t "
            "WHERE t.instrument_id = insider_transactions.instrument_id "
            "AND t.insider_name = insider_transactions.insider_name "
            "AND t.start_date = insider_transactions.start_date "
            "AND t.transaction_type = insider_transactions.transaction_type)"
        )
    )
    with op.batch_alter_table(_TABLE) as batch:
        batch.drop_constraint(_CONSTRAINT, type_="unique")
        batch.drop_column("lot_index")
        batch.create_unique_constraint(_CONSTRAINT, _KEY)
