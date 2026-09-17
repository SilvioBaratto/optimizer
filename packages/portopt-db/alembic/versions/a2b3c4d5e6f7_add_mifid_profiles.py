"""Add MiFID suitability-profile table (mifid_profiles).

Fase 5 of the ``fund`` bridge: the profiler's *step 0* system of record. A
client questionnaire maps to a persisted ``ConstraintSet`` stored **append-only**
with a per-portfolio ``version`` — an amended assessment never overwrites the
prior one, so a ``UNIQUE(portfolio_id, version)`` guards the versioning. Each
row keeps the raw questionnaire snapshot, the derived constraint set, and the
structured suitability assessment for MiFID record-keeping. Model lives in
``portopt_db``; repo behavior in ``fund`` — the ``agent_runs`` / ``paper_orders``
split.

Revision ID: a2b3c4d5e6f7
Revises: f0a1b2c3d4e5
Create Date: 2026-09-17
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

revision: str = "a2b3c4d5e6f7"
down_revision: str | Sequence[str] | None = "f0a1b2c3d4e5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Dialect-aware JSON: JSONB on PostgreSQL, plain JSON elsewhere (SQLite in tests).
_JSON = sa.JSON().with_variant(JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "mifid_profiles",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        # No FK: the `portfolios` table was dropped in the ingestion strip.
        sa.Column("portfolio_id", UUID(as_uuid=True), nullable=False),
        sa.Column("version", sa.Integer, nullable=False, server_default=sa.text("1")),
        sa.Column("questionnaire", _JSON, nullable=False),
        sa.Column("constraint_set", _JSON, nullable=False),
        sa.Column("suitability", _JSON, nullable=False),
        sa.Column("store_key", sa.String(100), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="active"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "portfolio_id", "version", name="uq_mifid_profile_portfolio_version"
        ),
    )
    op.create_index(
        "ix_mifid_profiles_portfolio_id", "mifid_profiles", ["portfolio_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_mifid_profiles_portfolio_id", table_name="mifid_profiles")
    op.drop_table("mifid_profiles")
