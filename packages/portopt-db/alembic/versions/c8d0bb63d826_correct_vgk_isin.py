"""Correct the ISIN seeded for VGK (Vanguard FTSE Europe ETF).

``a7b8c9d0e1f2`` seeded VGK with ``US9220427671``. That code embeds CUSIP
922042767, a different security; the fund's CUSIP is 922042874, giving
``US9220428745``. Both pass the ISIN check digit, so the typo was not
detectable by checksum. Databases that already ran the seed keep the wrong code
(the profile backfill only fills NULL ISINs and never overwrites), so this
revision rewrites it. The seed itself now carries the right code for databases
that have not run it yet.

Revision ID: c8d0bb63d826
Revises: d8f0b2c4e6a1
Create Date: 2026-10-24
"""

from collections.abc import Sequence

from alembic import op

revision: str = "c8d0bb63d826"
down_revision: str | Sequence[str] | None = "d8f0b2c4e6a1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Matching on the exact seeded typo keeps a later, legitimate ISIN (from a
# profile fetch) intact and makes the statement safe to replay.
_CORRECT_VGK_ISIN = """
    UPDATE instruments
    SET isin = 'US9220428745'
    WHERE ticker = 'VGK'
      AND isin = 'US9220427671'
"""


def upgrade() -> None:
    """Rewrite the mistyped VGK ISIN left behind by the expanded index seed."""
    op.execute(_CORRECT_VGK_ISIN)


def downgrade() -> None:
    """Re-assert the correction instead of restoring the mistyped code.

    The seed revision below this one now inserts the right ISIN, so the
    mistyped value is not a state worth returning to.
    """
    op.execute(_CORRECT_VGK_ISIN)
