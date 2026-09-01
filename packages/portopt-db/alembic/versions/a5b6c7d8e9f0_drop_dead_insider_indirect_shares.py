"""Drop the dead insider_roster.shares_owned_indirectly column.

yfinance's current ``Ticker.insider_roster_holders`` no longer emits a
"Shares Owned Indirectly" column (only "Shares Owned Directly"), so the field can
never be refreshed. The ~184 populated values are legacy data from when Yahoo
still served it. ``shares_owned_directly`` remains live (populated for insiders
who hold directly; NULL for indirect-only institutional holders).

Data note: dropping loses those legacy indirect values and they are **not
re-fetchable** (Yahoo removed the source column). ``downgrade`` re-adds the
column but cannot restore the values.

Revision ID: a5b6c7d8e9f0
Revises: f4a5b6c7d8e9
Create Date: 2026-09-01
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "a5b6c7d8e9f0"
down_revision: str | Sequence[str] | None = "f4a5b6c7d8e9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_column("insider_roster", "shares_owned_indirectly")


def downgrade() -> None:
    op.add_column(
        "insider_roster",
        sa.Column("shares_owned_indirectly", sa.BigInteger(), nullable=True),
    )
