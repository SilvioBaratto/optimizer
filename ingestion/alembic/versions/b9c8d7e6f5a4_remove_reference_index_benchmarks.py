"""Remove the seeded reference-index benchmarks.

The reference-index / benchmark feature was retired: benchmarks now come from the
investable ETF universe, not the name-less seeded tickers (SPY/AGG/TLT/…). This
deletes the 12 seeded reference instruments (identified by their NULL name — the
seeder never set one) and, via ON DELETE CASCADE, their price_history and any
other per-instrument rows.

The historical seed migrations (x4y5z6a7b8c9, a7b8c9d0e1f2) are left intact so
the chain still replays; this migration simply removes their output. Irreversible
(the price history is dropped) — downgrade is a no-op.

Revision ID: b9c8d7e6f5a4
Revises: f2a3b4c5d6e7
Create Date: 2026-08-25
"""

from collections.abc import Sequence

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


def upgrade() -> None:
    # Delete only the seeded reference rows: the NULL-name guard makes sure a
    # real T212 instrument that happens to share a ticker is never removed.
    tickers = ", ".join(f"'{t}'" for t in _BENCHMARKS)
    op.execute(f"DELETE FROM instruments WHERE name IS NULL AND ticker IN ({tickers})")


def downgrade() -> None:
    # Irreversible: the deleted benchmark price history cannot be reconstructed.
    # Restore from a pre-upgrade dump, or re-run the historical seed migrations.
    raise NotImplementedError(
        "b9c8d7e6f5a4 is one-way: deleted benchmark price history cannot be "
        "restored. Recover from a dump if the reference indices are needed."
    )
