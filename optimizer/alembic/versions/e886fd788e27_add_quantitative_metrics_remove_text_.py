"""add_quantitative_metrics_remove_text_fields

Revision ID: e886fd788e27
Revises: 274b2d5df671
Create Date: 2025-10-17 08:01:49.442897

CHANGES:
- Remove: analyst_score, data_gaps, primary_risks (not used in mathematical signals)
- Add: annualized_return, sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio,
       beta, alpha, r_squared, information_ratio, benchmark_return

Result: Complete quantitative metrics for Chapter 2 framework
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision: str = 'e886fd788e27'
down_revision: Union[str, Sequence[str], None] = '274b2d5df671'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add quantitative metrics and remove text fields."""

    # Step 1: Remove unused fields
    op.drop_column('stock_signals', 'analyst_score')
    op.drop_column('stock_signals', 'data_gaps')
    op.drop_column('stock_signals', 'primary_risks')

    # Step 2: Add comprehensive quantitative metrics
    op.add_column('stock_signals',
        sa.Column('annualized_return', sa.Float(), nullable=True,
                  comment='Annualized return over lookback period'))

    op.add_column('stock_signals',
        sa.Column('sharpe_ratio', sa.Float(), nullable=True,
                  comment='Sharpe ratio (risk-adjusted return)'))

    op.add_column('stock_signals',
        sa.Column('sortino_ratio', sa.Float(), nullable=True,
                  comment='Sortino ratio (downside risk-adjusted return)'))

    op.add_column('stock_signals',
        sa.Column('max_drawdown', sa.Float(), nullable=True,
                  comment='Maximum drawdown over lookback period'))

    op.add_column('stock_signals',
        sa.Column('calmar_ratio', sa.Float(), nullable=True,
                  comment='Calmar ratio (return / max drawdown)'))

    op.add_column('stock_signals',
        sa.Column('beta', sa.Float(), nullable=True,
                  comment='Market beta (sensitivity to benchmark)'))

    op.add_column('stock_signals',
        sa.Column('alpha', sa.Float(), nullable=True,
                  comment="Jensen's alpha (excess return vs. benchmark)"))

    op.add_column('stock_signals',
        sa.Column('r_squared', sa.Float(), nullable=True,
                  comment='R² (coefficient of determination vs. benchmark)'))

    op.add_column('stock_signals',
        sa.Column('information_ratio', sa.Float(), nullable=True,
                  comment='Information ratio (alpha / tracking error)'))

    op.add_column('stock_signals',
        sa.Column('benchmark_return', sa.Float(), nullable=True,
                  comment='Benchmark annualized return over same period'))


def downgrade() -> None:
    """Revert to previous schema (not recommended - data loss)."""

    # Remove quantitative metrics
    op.drop_column('stock_signals', 'benchmark_return')
    op.drop_column('stock_signals', 'information_ratio')
    op.drop_column('stock_signals', 'r_squared')
    op.drop_column('stock_signals', 'alpha')
    op.drop_column('stock_signals', 'beta')
    op.drop_column('stock_signals', 'calmar_ratio')
    op.drop_column('stock_signals', 'max_drawdown')
    op.drop_column('stock_signals', 'sortino_ratio')
    op.drop_column('stock_signals', 'sharpe_ratio')
    op.drop_column('stock_signals', 'annualized_return')

    # Re-add removed fields (data will be NULL)
    op.add_column('stock_signals',
        sa.Column('analyst_score', sa.Float(), nullable=True))
    op.add_column('stock_signals',
        sa.Column('data_gaps', JSONB, nullable=True))
    op.add_column('stock_signals',
        sa.Column('primary_risks', JSONB, nullable=True))
