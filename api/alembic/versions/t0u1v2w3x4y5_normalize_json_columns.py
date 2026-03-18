"""Normalize JSON columns into child tables.

Revision ID: t0u1v2w3x4y5
Revises: s9t0u1v2w3x4
Create Date: 2026-03-18
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

# revision identifiers, used by Alembic.
revision = "t0u1v2w3x4y5"
down_revision = "s9t0u1v2w3x4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # 1. Create 8 new child tables
    # ------------------------------------------------------------------
    op.create_table(
        "snapshot_weights",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("snapshot_id", UUID(as_uuid=True), sa.ForeignKey("portfolio_snapshots.id", ondelete="CASCADE"), nullable=False),
        sa.Column("ticker", sa.String(100), nullable=False),
        sa.Column("weight", sa.Float, nullable=False),
        sa.UniqueConstraint("snapshot_id", "ticker", name="uq_snapshot_weight_ticker"),
    )
    op.create_index("ix_snapshot_weights_snapshot_id", "snapshot_weights", ["snapshot_id"])

    op.create_table(
        "snapshot_sector_mappings",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("snapshot_id", UUID(as_uuid=True), sa.ForeignKey("portfolio_snapshots.id", ondelete="CASCADE"), nullable=False),
        sa.Column("ticker", sa.String(100), nullable=False),
        sa.Column("sector", sa.String(200), nullable=False),
        sa.UniqueConstraint("snapshot_id", "ticker", name="uq_snapshot_sector_mapping_ticker"),
    )
    op.create_index("ix_snapshot_sector_mappings_snapshot_id", "snapshot_sector_mappings", ["snapshot_id"])

    op.create_table(
        "snapshot_summary_entries",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("snapshot_id", UUID(as_uuid=True), sa.ForeignKey("portfolio_snapshots.id", ondelete="CASCADE"), nullable=False),
        sa.Column("key", sa.String(200), nullable=False),
        sa.Column("value_text", sa.Text, nullable=False),
        sa.UniqueConstraint("snapshot_id", "key", name="uq_snapshot_summary_key"),
    )
    op.create_index("ix_snapshot_summary_entries_snapshot_id", "snapshot_summary_entries", ["snapshot_id"])

    op.create_table(
        "snapshot_optimizer_params",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("snapshot_id", UUID(as_uuid=True), sa.ForeignKey("portfolio_snapshots.id", ondelete="CASCADE"), nullable=False),
        sa.Column("key", sa.String(200), nullable=False),
        sa.Column("value_text", sa.Text, nullable=False),
        sa.UniqueConstraint("snapshot_id", "key", name="uq_snapshot_optimizer_param_key"),
    )
    op.create_index("ix_snapshot_optimizer_params_snapshot_id", "snapshot_optimizer_params", ["snapshot_id"])

    op.create_table(
        "activity_event_metadata",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("event_id", UUID(as_uuid=True), sa.ForeignKey("activity_events.id", ondelete="CASCADE"), nullable=False),
        sa.Column("key", sa.String(200), nullable=False),
        sa.Column("value_text", sa.Text, nullable=False),
        sa.UniqueConstraint("event_id", "key", name="uq_activity_event_detail_key"),
    )
    op.create_index("ix_activity_event_metadata_event_id", "activity_event_metadata", ["event_id"])

    op.create_table(
        "regime_state_probabilities",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("regime_state_id", UUID(as_uuid=True), sa.ForeignKey("regime_states.id", ondelete="CASCADE"), nullable=False),
        sa.Column("regime", sa.String(20), nullable=False),
        sa.Column("probability", sa.Float, nullable=False),
        sa.UniqueConstraint("regime_state_id", "regime", name="uq_regime_state_prob_regime"),
    )
    op.create_index("ix_regime_state_probs_state_id", "regime_state_probabilities", ["regime_state_id"])

    op.create_table(
        "background_job_errors",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("job_id", UUID(as_uuid=True), sa.ForeignKey("background_jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("error_index", sa.Integer, nullable=False),
        sa.Column("message", sa.Text, nullable=False),
        sa.UniqueConstraint("job_id", "error_index", name="uq_bg_job_error_index"),
    )
    op.create_index("ix_background_job_errors_job_id", "background_job_errors", ["job_id"])

    op.create_table(
        "macro_news_themes",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("news_id", UUID(as_uuid=True), sa.ForeignKey("macro_news.id", ondelete="CASCADE"), nullable=False),
        sa.Column("theme", sa.String(50), nullable=False),
        sa.UniqueConstraint("news_id", "theme", name="uq_macro_news_theme"),
    )
    op.create_index("ix_macro_news_themes_news_id", "macro_news_themes", ["news_id"])
    op.create_index("ix_macro_news_themes_theme", "macro_news_themes", ["theme"])

    # ------------------------------------------------------------------
    # 2. Add 3 scalar columns to regime_states
    # ------------------------------------------------------------------
    op.add_column("regime_states", sa.Column("since", sa.Date, nullable=True))
    op.add_column("regime_states", sa.Column("n_states", sa.Integer, nullable=True))
    op.add_column("regime_states", sa.Column("last_fitted", sa.DateTime(timezone=True), nullable=True))

    # ------------------------------------------------------------------
    # 3. Data migration (PostgreSQL-specific)
    # ------------------------------------------------------------------

    # snapshot_weights: expand weights JSONB → child rows
    op.execute("""
        INSERT INTO snapshot_weights (id, created_at, updated_at, snapshot_id, ticker, weight)
        SELECT gen_random_uuid(), NOW(), NOW(), ps.id, kv.key, (kv.value)::float
        FROM portfolio_snapshots ps, jsonb_each_text(ps.weights) AS kv(key, value)
        WHERE ps.weights IS NOT NULL AND jsonb_typeof(ps.weights) = 'object'
    """)

    # snapshot_sector_mappings: expand sector_mapping JSONB
    op.execute("""
        INSERT INTO snapshot_sector_mappings (id, created_at, updated_at, snapshot_id, ticker, sector)
        SELECT gen_random_uuid(), NOW(), NOW(), ps.id, kv.key, kv.value
        FROM portfolio_snapshots ps, jsonb_each_text(ps.sector_mapping) AS kv(key, value)
        WHERE ps.sector_mapping IS NOT NULL AND jsonb_typeof(ps.sector_mapping) = 'object'
    """)

    # snapshot_summary_entries: expand summary JSONB (store values as JSON text)
    op.execute("""
        INSERT INTO snapshot_summary_entries (id, created_at, updated_at, snapshot_id, key, value_text)
        SELECT gen_random_uuid(), NOW(), NOW(), ps.id, kv.key, kv.value::text
        FROM portfolio_snapshots ps, jsonb_each(ps.summary) AS kv(key, value)
        WHERE ps.summary IS NOT NULL AND jsonb_typeof(ps.summary) = 'object'
    """)

    # snapshot_optimizer_params: expand optimizer_config JSONB
    op.execute("""
        INSERT INTO snapshot_optimizer_params (id, created_at, updated_at, snapshot_id, key, value_text)
        SELECT gen_random_uuid(), NOW(), NOW(), ps.id, kv.key, kv.value::text
        FROM portfolio_snapshots ps, jsonb_each(ps.optimizer_config) AS kv(key, value)
        WHERE ps.optimizer_config IS NOT NULL AND jsonb_typeof(ps.optimizer_config) = 'object'
    """)

    # activity_event_metadata: expand metadata JSONB (skip JSON nulls)
    op.execute("""
        INSERT INTO activity_event_metadata (id, created_at, updated_at, event_id, key, value_text)
        SELECT gen_random_uuid(), NOW(), NOW(), ae.id, kv.key, kv.value::text
        FROM activity_events ae, jsonb_each(ae.metadata) AS kv(key, value)
        WHERE ae.metadata IS NOT NULL AND jsonb_typeof(ae.metadata) = 'object'
    """)

    # regime_state_probabilities: expand probabilities JSONB array
    op.execute("""
        INSERT INTO regime_state_probabilities (id, created_at, updated_at, regime_state_id, regime, probability)
        SELECT gen_random_uuid(), NOW(), NOW(), rs.id,
               elem->>'regime', (elem->>'probability')::float
        FROM regime_states rs, jsonb_array_elements(rs.probabilities) AS elem
        WHERE rs.probabilities IS NOT NULL AND jsonb_typeof(rs.probabilities) = 'array'
    """)

    # regime_states scalar columns from metadata JSONB
    op.execute("""
        UPDATE regime_states
        SET since = (metadata->>'since')::date,
            n_states = (metadata->>'n_states')::integer,
            last_fitted = (metadata->>'last_fitted')::timestamptz
        WHERE metadata IS NOT NULL AND jsonb_typeof(metadata) = 'object'
    """)

    # background_job_errors: expand errors JSONB array
    op.execute("""
        INSERT INTO background_job_errors (id, created_at, updated_at, job_id, error_index, message)
        SELECT gen_random_uuid(), NOW(), NOW(), bj.id,
               (row_number() OVER (PARTITION BY bj.id ORDER BY ordinality))::integer - 1,
               elem
        FROM background_jobs bj, jsonb_array_elements_text(bj.errors) WITH ORDINALITY AS t(elem, ordinality)
        WHERE bj.errors IS NOT NULL AND jsonb_typeof(bj.errors) = 'array'
    """)

    # macro_news_themes: expand comma-separated themes TEXT
    op.execute("""
        INSERT INTO macro_news_themes (id, created_at, updated_at, news_id, theme)
        SELECT gen_random_uuid(), NOW(), NOW(), mn.id, trim(t.theme)
        FROM macro_news mn, unnest(string_to_array(mn.themes, ',')) AS t(theme)
        WHERE mn.themes IS NOT NULL AND mn.themes != ''
          AND trim(t.theme) != ''
    """)

    # ------------------------------------------------------------------
    # 4. Drop old columns and indexes
    # ------------------------------------------------------------------
    op.drop_index("ix_macro_news_themes", table_name="macro_news")

    op.drop_column("portfolio_snapshots", "weights")
    op.drop_column("portfolio_snapshots", "sector_mapping")
    op.drop_column("portfolio_snapshots", "summary")
    op.drop_column("portfolio_snapshots", "optimizer_config")
    op.drop_column("activity_events", "metadata")
    op.drop_column("regime_states", "probabilities")
    op.drop_column("regime_states", "metadata")
    op.drop_column("background_jobs", "errors")
    op.drop_column("macro_news", "themes")


def downgrade() -> None:
    # ------------------------------------------------------------------
    # 1. Re-add dropped columns
    # ------------------------------------------------------------------
    op.add_column("portfolio_snapshots", sa.Column("weights", JSONB, nullable=True))
    op.add_column("portfolio_snapshots", sa.Column("sector_mapping", JSONB, nullable=True))
    op.add_column("portfolio_snapshots", sa.Column("summary", JSONB, nullable=True))
    op.add_column("portfolio_snapshots", sa.Column("optimizer_config", JSONB, nullable=True))
    op.add_column("activity_events", sa.Column("metadata", JSONB, nullable=True))
    op.add_column("regime_states", sa.Column("probabilities", JSONB, nullable=True))
    op.add_column("regime_states", sa.Column("metadata", JSONB, nullable=True))
    op.add_column("background_jobs", sa.Column("errors", JSONB, nullable=True))
    op.add_column("macro_news", sa.Column("themes", sa.Text, nullable=True))

    # ------------------------------------------------------------------
    # 2. Reverse data migration: reconstruct JSON from child tables
    # ------------------------------------------------------------------

    # Reconstruct weights
    op.execute("""
        UPDATE portfolio_snapshots ps
        SET weights = sub.w
        FROM (
            SELECT snapshot_id, jsonb_object_agg(ticker, weight) AS w
            FROM snapshot_weights
            GROUP BY snapshot_id
        ) sub
        WHERE ps.id = sub.snapshot_id
    """)

    # Reconstruct sector_mapping
    op.execute("""
        UPDATE portfolio_snapshots ps
        SET sector_mapping = sub.m
        FROM (
            SELECT snapshot_id, jsonb_object_agg(ticker, sector) AS m
            FROM snapshot_sector_mappings
            GROUP BY snapshot_id
        ) sub
        WHERE ps.id = sub.snapshot_id
    """)

    # Reconstruct summary
    op.execute("""
        UPDATE portfolio_snapshots ps
        SET summary = sub.s
        FROM (
            SELECT snapshot_id, jsonb_object_agg(key, value_text::jsonb) AS s
            FROM snapshot_summary_entries
            GROUP BY snapshot_id
        ) sub
        WHERE ps.id = sub.snapshot_id
    """)

    # Reconstruct optimizer_config
    op.execute("""
        UPDATE portfolio_snapshots ps
        SET optimizer_config = sub.c
        FROM (
            SELECT snapshot_id, jsonb_object_agg(key, value_text::jsonb) AS c
            FROM snapshot_optimizer_params
            GROUP BY snapshot_id
        ) sub
        WHERE ps.id = sub.snapshot_id
    """)

    # Reconstruct activity_events.metadata
    op.execute("""
        UPDATE activity_events ae
        SET metadata = sub.m
        FROM (
            SELECT event_id, jsonb_object_agg(key, value_text::jsonb) AS m
            FROM activity_event_metadata
            GROUP BY event_id
        ) sub
        WHERE ae.id = sub.event_id
    """)

    # Reconstruct regime_states.probabilities
    op.execute("""
        UPDATE regime_states rs
        SET probabilities = sub.p
        FROM (
            SELECT regime_state_id,
                   jsonb_agg(jsonb_build_object('regime', regime, 'probability', probability)) AS p
            FROM regime_state_probabilities
            GROUP BY regime_state_id
        ) sub
        WHERE rs.id = sub.regime_state_id
    """)

    # Reconstruct regime_states.metadata from scalar columns
    op.execute("""
        UPDATE regime_states
        SET metadata = jsonb_strip_nulls(jsonb_build_object(
            'since', since::text,
            'n_states', n_states,
            'last_fitted', last_fitted::text
        ))
        WHERE since IS NOT NULL OR n_states IS NOT NULL OR last_fitted IS NOT NULL
    """)

    # Reconstruct background_jobs.errors
    op.execute("""
        UPDATE background_jobs bj
        SET errors = sub.e
        FROM (
            SELECT job_id,
                   jsonb_agg(message ORDER BY error_index) AS e
            FROM background_job_errors
            GROUP BY job_id
        ) sub
        WHERE bj.id = sub.job_id
    """)

    # Reconstruct macro_news.themes
    op.execute("""
        UPDATE macro_news mn
        SET themes = sub.t
        FROM (
            SELECT news_id, string_agg(theme, ',' ORDER BY theme) AS t
            FROM macro_news_themes
            GROUP BY news_id
        ) sub
        WHERE mn.id = sub.news_id
    """)

    # Make weights NOT NULL again
    op.execute("UPDATE portfolio_snapshots SET weights = '{}' WHERE weights IS NULL")
    op.alter_column("portfolio_snapshots", "weights", nullable=False)

    # Reconstruct probabilities NOT NULL
    op.execute("UPDATE regime_states SET probabilities = '[]' WHERE probabilities IS NULL")
    op.alter_column("regime_states", "probabilities", nullable=False)

    # Re-create dropped index
    op.create_index("ix_macro_news_themes", "macro_news", ["themes"])

    # ------------------------------------------------------------------
    # 3. Drop child tables and scalar columns
    # ------------------------------------------------------------------
    op.drop_column("regime_states", "since")
    op.drop_column("regime_states", "n_states")
    op.drop_column("regime_states", "last_fitted")

    op.drop_table("macro_news_themes")
    op.drop_table("background_job_errors")
    op.drop_table("regime_state_probabilities")
    op.drop_table("activity_event_metadata")
    op.drop_table("snapshot_optimizer_params")
    op.drop_table("snapshot_summary_entries")
    op.drop_table("snapshot_sector_mappings")
    op.drop_table("snapshot_weights")
