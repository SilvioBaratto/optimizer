"""Backfill SQL NULL over the JSON literal ``null`` in the job and journal columns.

``background_jobs.extra`` / ``result`` and ``portfolio_journal.trades`` /
``allocation`` / ``drift`` stored a Python ``None`` as the JSON literal ``null``
until their models set ``none_as_null=True``. Rows written before that keep the
literal, so ``IS NULL`` / ``IS NOT NULL`` filters on those columns would
misclassify them. This rewrites those cells to SQL NULL.

``CAST(col AS TEXT)`` reads the literal back as ``null`` on PostgreSQL (JSONB) and on
SQLite alike, and does not match the JSON string ``"null"``, which keeps its quotes.
Raw SQL leaves ``updated_at`` untouched, so the rewrite does not look like an edit.

Downgrade stores the literal again in every empty cell. The original mix of literal
and SQL NULL cannot be recovered, but the previous code reads both as ``None``.

Revision ID: b8d0f2a4c6e1
Revises: f4b6d8a0c2e3
Create Date: 2026-11-06
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "b8d0f2a4c6e1"
down_revision: str | Sequence[str] | None = "f4b6d8a0c2e3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_NULLABLE_JSON_COLUMNS: tuple[tuple[str, str], ...] = (
    ("background_jobs", "extra"),
    ("background_jobs", "result"),
    ("portfolio_journal", "trades"),
    ("portfolio_journal", "allocation"),
    ("portfolio_journal", "drift"),
)


def upgrade() -> None:
    """Rewrite JSON-literal ``null`` cells to SQL NULL."""
    for table, column in _NULLABLE_JSON_COLUMNS:
        op.execute(
            sa.text(
                f"UPDATE {table} SET {column} = NULL "  # noqa: S608
                f"WHERE CAST({column} AS TEXT) = 'null'"
            )
        )


def downgrade() -> None:
    """Store the JSON literal ``null`` in every empty cell again."""
    for table, column in _NULLABLE_JSON_COLUMNS:
        op.execute(
            sa.text(
                f"UPDATE {table} SET {column} = 'null' "  # noqa: S608
                f"WHERE {column} IS NULL"
            )
        )
