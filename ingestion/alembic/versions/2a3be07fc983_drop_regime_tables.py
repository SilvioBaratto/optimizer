"""drop_regime_tables

Revision ID: 2a3be07fc983
Revises: a7b8c9d0e1f2
Create Date: 2026-04-28 16:14:41.117873

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "2a3be07fc983"
down_revision: str | Sequence[str] | None = "a7b8c9d0e1f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Drop child table first (FK → regime_states.id)
    op.drop_index(
        "ix_regime_state_probs_state_id", table_name="regime_state_probabilities"
    )
    op.drop_constraint(
        "uq_regime_state_prob_regime", "regime_state_probabilities", type_="unique"
    )
    op.drop_table("regime_state_probabilities")

    # Drop parent table
    op.drop_constraint("uq_regime_state_date_model", "regime_states", type_="unique")
    op.drop_table("regime_states")


def downgrade() -> None:
    # Recreate parent
    op.create_table(
        "regime_states",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("state_date", sa.Date(), nullable=False),
        sa.Column("regime", sa.String(), nullable=False),
        sa.Column("model_type", sa.String(), nullable=False),
        sa.Column("since", sa.Date(), nullable=True),
        sa.Column("n_states", sa.Integer(), nullable=True),
        sa.Column("last_fitted", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "state_date", "model_type", name="uq_regime_state_date_model"
        ),
    )

    # Recreate child
    op.create_table(
        "regime_state_probabilities",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("regime_state_id", sa.UUID(), nullable=False),
        sa.Column("regime", sa.String(), nullable=False),
        sa.Column("probability", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["regime_state_id"], ["regime_states.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "regime_state_id", "regime", name="uq_regime_state_prob_regime"
        ),
    )
    op.create_index(
        "ix_regime_state_probs_state_id",
        "regime_state_probabilities",
        ["regime_state_id"],
    )
