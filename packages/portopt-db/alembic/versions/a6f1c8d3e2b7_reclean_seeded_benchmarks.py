"""Delete the seed-migration benchmark ETFs on already-migrated installs.

``b9c8d7e6f5a4`` first matched only name-less rows, but the seed revisions
(``x4y5z6a7b8c9``, ``a7b8c9d0e1f2``) insert SPY, QQQ, AGG, TLT, GLD and the rest
*with* a name. A database that applied the first version of that revision still
holds them as mis-tagged equity ETFs that the daily pipeline keeps ingesting.
Editing an applied revision cannot reach those databases, so this revision
repeats the cleanup going forward.

Rows are matched on the exact (ticker, isin) pair the seeds wrote, plus the ETF
type. The universe builders cannot produce that pair: the Yahoo screen leaves
``isin`` NULL and Trading 212 lists no US-domiciled ETFs, so a live universe
member that shares one of these tickers is kept. A database that never held the
rows (fresh install, or ``b9c8d7e6f5a4`` already ran in its fixed form) is
untouched.

The delete cascades to ``price_history`` and every other per-instrument table,
which is the retirement b9c8d7e6f5a4 describes. It is irreversible, so
``downgrade`` raises rather than pretend to restore anything.

Revision ID: a6f1c8d3e2b7
Revises: c5d9a1e7b3f8
Create Date: 2026-10-24
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "a6f1c8d3e2b7"
down_revision: str | Sequence[str] | None = "c5d9a1e7b3f8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Same catalogue as b9c8d7e6f5a4; a test pins both to the seed revisions so a
# drift in either copy fails loudly.
_MIGRATION_SEEDED = (
    ("SPY", "US78462F1030"),
    ("QQQ", "US46090E1038"),
    ("IWM", "US4642876555"),
    ("EFA", "US4642874659"),
    ("EEM", "US4642872349"),
    ("AGG", "US4642872265"),
    ("VGK", "US9220428745"),
    # Mistyped code the expanded seed carried before c8d0bb63d826 corrected it;
    # a database that ran that seed still holds rows under it.
    ("VGK", "US9220427671"),
    ("VWO", "US9220428588"),
    ("TLT", "US4642874329"),
    ("GLD", "US78463V1070"),
)


def upgrade() -> None:
    instruments = sa.table(
        "instruments",
        sa.column("ticker"),
        sa.column("isin"),
        sa.column("instrument_type"),
    )
    seeded_pair = sa.or_(
        *(
            sa.and_(instruments.c.ticker == t, instruments.c.isin == i)
            for t, i in _MIGRATION_SEEDED
        )
    )
    op.execute(
        sa.delete(instruments).where(
            sa.and_(instruments.c.instrument_type == "ETF", seeded_pair)
        )
    )


def downgrade() -> None:
    # Irreversible: the deleted benchmark rows and their price history cannot be
    # reconstructed. Restore from a pre-upgrade dump if they are needed.
    raise NotImplementedError(
        "a6f1c8d3e2b7 is one-way: deleted benchmark rows and their price history "
        "cannot be restored. Recover from a dump if the reference indices are needed."
    )
