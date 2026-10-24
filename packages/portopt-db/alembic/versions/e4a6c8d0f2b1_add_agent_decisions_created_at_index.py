"""Index ``agent_decisions.created_at`` for the Audit timeline.

The Audit view lists the newest decisions across every run
(``ORDER BY created_at DESC LIMIT n``, polled while the view is open). With only the
``run_id`` index PostgreSQL scans and sorts the whole table for each poll; a
``created_at`` index lets it walk the newest rows and stop. ``human_audit_log``
already carries the equivalent index.

Revision ID: e4a6c8d0f2b1
Revises: d8f0b2c4e6a1
Create Date: 2026-10-24
"""

from collections.abc import Sequence

from alembic import op

revision: str = "e4a6c8d0f2b1"
down_revision: str | Sequence[str] | None = "d8f0b2c4e6a1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create ``ix_agent_decisions_created_at``."""
    op.create_index("ix_agent_decisions_created_at", "agent_decisions", ["created_at"])


def downgrade() -> None:
    """Drop ``ix_agent_decisions_created_at``."""
    op.drop_index("ix_agent_decisions_created_at", table_name="agent_decisions")
