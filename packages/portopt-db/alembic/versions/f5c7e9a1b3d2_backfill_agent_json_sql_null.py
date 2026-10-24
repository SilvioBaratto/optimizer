"""Backfill SQL NULL over the JSON literal ``null`` in the nullable agent columns.

The agent models stored a Python ``None`` as the JSON literal ``null`` until they set
``none_as_null=True``. Rows written before that keep the literal, so ``IS NULL`` /
``IS NOT NULL`` filters on ``agent_runs.weights`` and on
``agent_decisions.constraint_set`` / ``views`` / ``hitl_decision`` still misclassify
them. This rewrites those cells to SQL NULL. ``human_audit_log`` is append-only (a
trigger blocks UPDATE) and is deliberately left as written.

``CAST(col AS TEXT)`` reads the literal back as ``null`` on PostgreSQL (JSONB) and on
SQLite alike, and does not match the JSON string ``"null"``, which keeps its quotes.
Raw SQL leaves ``updated_at`` untouched, so the rewrite does not look like an edit.

Downgrade stores the literal again in every empty cell. The original mix of literal
and SQL NULL cannot be recovered, but the previous code reads both as ``None``.

Revision ID: f5c7e9a1b3d2
Revises: e4a6c8d0f2b1
Create Date: 2026-10-24
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "f5c7e9a1b3d2"
down_revision: str | Sequence[str] | None = "e4a6c8d0f2b1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_NULLABLE_JSON_COLUMNS: tuple[tuple[str, str], ...] = (
    ("agent_runs", "weights"),
    ("agent_decisions", "constraint_set"),
    ("agent_decisions", "views"),
    ("agent_decisions", "hitl_decision"),
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
