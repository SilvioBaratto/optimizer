"""Add factor_scores, factor_validation_reports, optimization_runs,
backtest_runs, risk_limits, and rebalancing_policies tables.

Revision ID: y5z6a7b8c9d0
Revises: x4y5z6a7b8c9
Create Date: 2026-04-15
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

revision: str = "y5z6a7b8c9d0"
down_revision: str | Sequence[str] | None = "x4y5z6a7b8c9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Dialect-aware JSON: JSONB on PostgreSQL, plain JSON elsewhere (SQLite in tests).
_JSON = sa.JSON().with_variant(JSONB(), "postgresql")


def upgrade() -> None:
    # 1. factor_scores
    op.create_table(
        "factor_scores",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("ticker", sa.String(20), nullable=False),
        sa.Column("score_date", sa.Date, nullable=False),
        sa.Column("factor_type", sa.String(50), nullable=False),
        sa.Column("factor_group", sa.String(50), nullable=False),
        sa.Column("raw_score", sa.Float, nullable=False),
        sa.Column("standardized_score", sa.Float, nullable=True),
        sa.Column("composite_score", sa.Float, nullable=True),
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
            "ticker",
            "score_date",
            "factor_type",
            name="uq_factor_scores_ticker_date_type",
        ),
    )
    op.create_index("ix_factor_scores_ticker", "factor_scores", ["ticker"])
    op.create_index("ix_factor_scores_score_date", "factor_scores", ["score_date"])
    op.create_index("ix_factor_scores_factor_type", "factor_scores", ["factor_type"])

    # 2. factor_validation_reports
    op.create_table(
        "factor_validation_reports",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("report_date", sa.Date, nullable=False),
        sa.Column("factor_type", sa.String(50), nullable=True),
        sa.Column("validation_type", sa.String(20), nullable=False),
        sa.Column("ic_mean", sa.Float, nullable=True),
        sa.Column("ic_std", sa.Float, nullable=True),
        sa.Column("icir", sa.Float, nullable=True),
        sa.Column("t_stat", sa.Float, nullable=True),
        sa.Column("p_value", sa.Float, nullable=True),
        sa.Column("vif", sa.Float, nullable=True),
        sa.Column("details", _JSON, nullable=True),
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
            "report_date",
            "factor_type",
            "validation_type",
            name="uq_factor_validation_reports_date_type_validation",
        ),
    )
    op.create_index(
        "ix_factor_validation_reports_report_date",
        "factor_validation_reports",
        ["report_date"],
    )
    op.create_index(
        "ix_factor_validation_reports_factor_type",
        "factor_validation_reports",
        ["factor_type"],
    )

    # 3. optimization_runs
    op.create_table(
        "optimization_runs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "portfolio_id",
            UUID(as_uuid=True),
            sa.ForeignKey("portfolios.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "job_id",
            UUID(as_uuid=True),
            sa.ForeignKey("background_jobs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "status",
            sa.String(20),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("optimizer_type", sa.String(50), nullable=False),
        sa.Column("universe_tickers", _JSON, nullable=False),
        sa.Column("config", _JSON, nullable=False),
        sa.Column("weights", _JSON, nullable=False),
        sa.Column("metrics", _JSON, nullable=False),
        sa.Column("risk_contributions", _JSON, nullable=False),
        sa.Column("efficient_frontier", _JSON, nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("solver_log", sa.Text, nullable=True),
        sa.Column("duration_seconds", sa.Float, nullable=True),
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
    )
    op.create_index(
        "ix_optimization_runs_portfolio_id", "optimization_runs", ["portfolio_id"]
    )
    op.create_index("ix_optimization_runs_status", "optimization_runs", ["status"])
    op.create_index(
        "ix_optimization_runs_created_at", "optimization_runs", ["created_at"]
    )

    # 4. backtest_runs
    op.create_table(
        "backtest_runs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "portfolio_id",
            UUID(as_uuid=True),
            sa.ForeignKey("portfolios.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "job_id",
            UUID(as_uuid=True),
            sa.ForeignKey("background_jobs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "status",
            sa.String(20),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("config", _JSON, nullable=False),
        sa.Column("equity_curve", _JSON, nullable=False),
        sa.Column("drawdowns", _JSON, nullable=False),
        sa.Column("monthly_returns", _JSON, nullable=False),
        sa.Column("yearly_returns", _JSON, nullable=False),
        sa.Column("rolling_metrics", _JSON, nullable=False),
        sa.Column("turnover_history", _JSON, nullable=False),
        sa.Column("cv_fold_metrics", _JSON, nullable=True),
        sa.Column("summary_stats", _JSON, nullable=False),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("duration_seconds", sa.Float, nullable=True),
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
    )
    op.create_index("ix_backtest_runs_portfolio_id", "backtest_runs", ["portfolio_id"])
    op.create_index("ix_backtest_runs_status", "backtest_runs", ["status"])
    op.create_index("ix_backtest_runs_created_at", "backtest_runs", ["created_at"])

    # 5. risk_limits
    op.create_table(
        "risk_limits",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "portfolio_id",
            UUID(as_uuid=True),
            sa.ForeignKey("portfolios.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("metric", sa.String(100), nullable=False),
        sa.Column("limit_type", sa.String(20), nullable=False),
        sa.Column("threshold", sa.Float, nullable=False),
        sa.Column("current_value", sa.Float, nullable=True),
        sa.Column("is_breached", sa.Boolean, nullable=False, server_default="0"),
        sa.Column("last_checked_at", sa.DateTime(timezone=True), nullable=True),
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
            "portfolio_id",
            "metric",
            "limit_type",
            name="uq_risk_limits_portfolio_metric_limit_type",
        ),
    )
    op.create_index("ix_risk_limits_portfolio_id", "risk_limits", ["portfolio_id"])

    # 6. rebalancing_policies
    op.create_table(
        "rebalancing_policies",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "portfolio_id",
            UUID(as_uuid=True),
            sa.ForeignKey("portfolios.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("policy_type", sa.String(20), nullable=False),
        sa.Column("config", _JSON, nullable=False),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="0"),
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
            "portfolio_id",
            "name",
            name="uq_rebalancing_policies_portfolio_name",
        ),
    )
    op.create_index(
        "ix_rebalancing_policies_portfolio_id",
        "rebalancing_policies",
        ["portfolio_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_rebalancing_policies_portfolio_id", table_name="rebalancing_policies"
    )
    op.drop_table("rebalancing_policies")

    op.drop_index("ix_risk_limits_portfolio_id", table_name="risk_limits")
    op.drop_table("risk_limits")

    op.drop_index("ix_backtest_runs_created_at", table_name="backtest_runs")
    op.drop_index("ix_backtest_runs_status", table_name="backtest_runs")
    op.drop_index("ix_backtest_runs_portfolio_id", table_name="backtest_runs")
    op.drop_table("backtest_runs")

    op.drop_index("ix_optimization_runs_created_at", table_name="optimization_runs")
    op.drop_index("ix_optimization_runs_status", table_name="optimization_runs")
    op.drop_index("ix_optimization_runs_portfolio_id", table_name="optimization_runs")
    op.drop_table("optimization_runs")

    op.drop_index(
        "ix_factor_validation_reports_factor_type",
        table_name="factor_validation_reports",
    )
    op.drop_index(
        "ix_factor_validation_reports_report_date",
        table_name="factor_validation_reports",
    )
    op.drop_table("factor_validation_reports")

    op.drop_index("ix_factor_scores_factor_type", table_name="factor_scores")
    op.drop_index("ix_factor_scores_score_date", table_name="factor_scores")
    op.drop_index("ix_factor_scores_ticker", table_name="factor_scores")
    op.drop_table("factor_scores")
