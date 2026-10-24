"""Remove the seeded reference-index benchmarks.

The reference-index / benchmark feature was retired: benchmarks now come from the
investable ETF universe, not the seeded tickers (SPY/AGG/TLT/…). This deletes
the 12 seeded reference instruments and, via ON DELETE CASCADE, their
price_history and any other per-instrument rows. Two populations exist: rows
from the retired runtime seeder (no name) and rows from the seed migrations
(x4y5z6a7b8c9, a7b8c9d0e1f2), which carry a name and are matched by their exact
(ticker, isin) pair so a real instrument sharing a ticker is never removed.
Databases that applied the earlier name-less-only version of this revision keep
the named rows; a6f1c8d3e2b7 removes them.

The historical seed migrations are left intact so the chain still replays; this
migration simply removes their output. Irreversible (the price history is
dropped) — downgrade is a no-op.

Revision ID: b9c8d7e6f5a4
Revises: f2a3b4c5d6e7
Create Date: 2026-08-25
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "b9c8d7e6f5a4"
down_revision: str | Sequence[str] | None = "f2a3b4c5d6e7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_BENCHMARKS = (
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "AGG",
    "VGK",
    "VWO",
    "TLT",
    "GLD",
    "URTH",
    "VBINX",
)

# (ticker, isin) exactly as the seed migrations x4y5z6a7b8c9 and a7b8c9d0e1f2
# inserted them, name included — which is why name alone cannot identify them.
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
    # Delete only the seeded reference rows. The runtime seeder never set a name;
    # the seed migrations did, so those are matched on their exact (ticker, isin)
    # pair — a real instrument sharing a ticker has a different ISIN and stays.
    instruments = sa.table(
        "instruments",
        sa.column("ticker"),
        sa.column("name"),
        sa.column("isin"),
        sa.column("instrument_type"),
    )
    runtime_seeded = sa.and_(
        instruments.c.name.is_(None), instruments.c.ticker.in_(_BENCHMARKS)
    )
    migration_seeded = sa.and_(
        instruments.c.instrument_type == "ETF",
        sa.or_(
            *(
                sa.and_(instruments.c.ticker == t, instruments.c.isin == i)
                for t, i in _MIGRATION_SEEDED
            )
        ),
    )
    op.execute(sa.delete(instruments).where(sa.or_(runtime_seeded, migration_seeded)))


def downgrade() -> None:
    # Irreversible: the deleted benchmark price history cannot be reconstructed.
    # Restore from a pre-upgrade dump, or re-run the historical seed migrations.
    raise NotImplementedError(
        "b9c8d7e6f5a4 is one-way: deleted benchmark price history cannot be "
        "restored. Recover from a dump if the reference indices are needed."
    )
