"""Drop orphaned tables with no ORM models.

Tables dropped: signal_distributions, stock_signals, news_articles,
regime_transitions, country_regime_assessments, market_indicators,
macro_analysis_runs, portfolio_positions, portfolios.

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-02-18
"""

from typing import Sequence, Union

from alembic import op


revision: str = "b2c3d4e5f6a7"
down_revision: Union[str, Sequence[str], None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop in FK-safe order: children first, then parents.

    # Remove FK from country_regime_assessments → economic_indicators
    # (points to a table we're keeping)
    op.drop_constraint(
        "country_regime_assessments_economic_indicators_id_fkey",
        "country_regime_assessments",
        type_="foreignkey",
    )

    # Leaf tables (no children depend on them)
    op.drop_table("signal_distributions")
    op.drop_table("stock_signals")
    op.drop_table("portfolio_positions")
    op.drop_table("portfolios")

    # news_articles and regime_transitions depend on country_regime_assessments
    op.drop_table("news_articles")
    op.drop_table("regime_transitions")

    # market_indicators and country_regime_assessments depend on macro_analysis_runs
    op.drop_table("market_indicators")
    op.drop_table("country_regime_assessments")
    op.drop_table("macro_analysis_runs")


def downgrade() -> None:
    raise NotImplementedError(
        "Downgrade not supported — dropped tables contained data that cannot be restored."
    )
