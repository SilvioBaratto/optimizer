"""Drop orphaned tables with no ORM models.

Tables dropped: signal_distributions, stock_signals, news_articles,
regime_transitions, country_regime_assessments, market_indicators,
macro_analysis_runs, portfolio_positions, portfolios.

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-02-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b2c3d4e5f6a7"
down_revision: str | Sequence[str] | None = "a1b2c3d4e5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _drop_if_exists(table: str, existing: set[str]) -> None:
    if table in existing:
        op.drop_table(table)


def upgrade() -> None:
    conn = op.get_bind()
    existing = set(sa.inspect(conn).get_table_names())

    # Remove FK from country_regime_assessments → economic_indicators
    if "country_regime_assessments" in existing:
        try:
            op.drop_constraint(
                "country_regime_assessments_economic_indicators_id_fkey",
                "country_regime_assessments",
                type_="foreignkey",
            )
        except Exception:
            pass

    # Drop in FK-safe order: children first, then parents.
    for tbl in (
        "signal_distributions",
        "stock_signals",
        "portfolio_positions",
        "portfolios",
        "news_articles",
        "regime_transitions",
        "market_indicators",
        "country_regime_assessments",
        "macro_analysis_runs",
    ):
        _drop_if_exists(tbl, existing)


def downgrade() -> None:
    raise NotImplementedError(
        "Downgrade not supported — dropped tables contained data that cannot be restored."
    )
