"""optimize_stock_signals_enums_only

Revision ID: 274b2d5df671
Revises: 41f326f24e32
Create Date: 2025-10-16 22:35:30.610601

OPTIMIZATION:
- Converts risk level strings to enums (~70% storage reduction)
- Removes all summary/notes Text fields (keep only numerical scores)
- Keeps only: enums, numerical scores, JSONB arrays

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '274b2d5df671'
down_revision: Union[str, Sequence[str], None] = '41f326f24e32'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Optimize stock_signals table - enums only."""

    # Step 1: Create risk_level_enum type (if not exists)
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE risk_level_enum AS ENUM ('low', 'medium', 'high', 'unknown');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)

    # Step 2: Drop text summary columns (keeping only numerical scores)
    op.drop_column('stock_signals', 'analysis_notes')
    op.drop_column('stock_signals', 'valuation_summary')
    op.drop_column('stock_signals', 'momentum_summary')
    op.drop_column('stock_signals', 'quality_summary')
    op.drop_column('stock_signals', 'growth_summary')
    op.drop_column('stock_signals', 'technical_summary')
    op.drop_column('stock_signals', 'analyst_summary')

    # Step 3: Convert risk level columns from text/varchar to enum
    # volatility_level (currently varchar(50))
    op.drop_column('stock_signals', 'volatility_level')
    op.add_column('stock_signals',
        sa.Column('volatility_level', sa.Enum('low', 'medium', 'high', 'unknown', name='risk_level_enum'), nullable=True,
                  comment='Volatility assessment: low/medium/high'))

    # beta_risk (currently text)
    op.drop_column('stock_signals', 'beta_risk')
    op.add_column('stock_signals',
        sa.Column('beta_risk', sa.Enum('low', 'medium', 'high', 'unknown', name='risk_level_enum'), nullable=True,
                  comment='Market sensitivity assessment: low/medium/high'))

    # debt_risk (currently text)
    op.drop_column('stock_signals', 'debt_risk')
    op.add_column('stock_signals',
        sa.Column('debt_risk', sa.Enum('low', 'medium', 'high', 'unknown', name='risk_level_enum'), nullable=True,
                  comment='Debt level assessment: low/medium/high'))

    # liquidity_risk (currently text)
    op.drop_column('stock_signals', 'liquidity_risk')
    op.add_column('stock_signals',
        sa.Column('liquidity_risk', sa.Enum('low', 'medium', 'high', 'unknown', name='risk_level_enum'), nullable=True,
                  comment='Trading liquidity assessment: low/medium/high'))


def downgrade() -> None:
    """Revert optimizations (not recommended - data loss)."""

    # Revert risk level columns from enum to text
    op.drop_column('stock_signals', 'volatility_level')
    op.add_column('stock_signals',
        sa.Column('volatility_level', sa.String(50), nullable=True))

    op.drop_column('stock_signals', 'beta_risk')
    op.add_column('stock_signals',
        sa.Column('beta_risk', sa.Text(), nullable=True))

    op.drop_column('stock_signals', 'debt_risk')
    op.add_column('stock_signals',
        sa.Column('debt_risk', sa.Text(), nullable=True))

    op.drop_column('stock_signals', 'liquidity_risk')
    op.add_column('stock_signals',
        sa.Column('liquidity_risk', sa.Text(), nullable=True))

    # Restore text summary columns (will be NULL)
    op.add_column('stock_signals',
        sa.Column('analysis_notes', sa.Text(), nullable=True))
    op.add_column('stock_signals',
        sa.Column('valuation_summary', sa.Text(), nullable=True))
    op.add_column('stock_signals',
        sa.Column('momentum_summary', sa.Text(), nullable=True))
    op.add_column('stock_signals',
        sa.Column('quality_summary', sa.Text(), nullable=True))
    op.add_column('stock_signals',
        sa.Column('growth_summary', sa.Text(), nullable=True))
    op.add_column('stock_signals',
        sa.Column('technical_summary', sa.Text(), nullable=True))
    op.add_column('stock_signals',
        sa.Column('analyst_summary', sa.Text(), nullable=True))

    # Drop enum type
    op.execute("DROP TYPE risk_level_enum")
