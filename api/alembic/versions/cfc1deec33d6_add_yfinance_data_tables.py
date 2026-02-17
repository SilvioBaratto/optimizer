"""add yfinance data tables

Revision ID: cfc1deec33d6
Revises: 1e14baf18033
Create Date: 2026-01-30 16:10:23.296819

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'cfc1deec33d6'
down_revision: Union[str, Sequence[str], None] = '1e14baf18033'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create 11 yfinance data tables."""
    # ticker_profiles
    op.create_table(
        'ticker_profiles',
        sa.Column('instrument_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('instruments.id', ondelete='CASCADE'), nullable=False),
        sa.Column('symbol', sa.String(50), nullable=True),
        sa.Column('short_name', sa.String(500), nullable=True),
        sa.Column('long_name', sa.String(500), nullable=True),
        sa.Column('isin', sa.String(20), nullable=True),
        sa.Column('exchange', sa.String(50), nullable=True),
        sa.Column('quote_type', sa.String(50), nullable=True),
        sa.Column('currency', sa.String(10), nullable=True),
        sa.Column('sector', sa.String(200), nullable=True),
        sa.Column('industry', sa.String(200), nullable=True),
        sa.Column('country', sa.String(100), nullable=True),
        sa.Column('website', sa.String(500), nullable=True),
        sa.Column('long_business_summary', sa.Text(), nullable=True),
        sa.Column('market_cap', sa.BigInteger(), nullable=True),
        sa.Column('enterprise_value', sa.BigInteger(), nullable=True),
        sa.Column('shares_outstanding', sa.BigInteger(), nullable=True),
        sa.Column('float_shares', sa.BigInteger(), nullable=True),
        sa.Column('implied_shares_outstanding', sa.BigInteger(), nullable=True),
        sa.Column('current_price', sa.Float(), nullable=True),
        sa.Column('previous_close', sa.Float(), nullable=True),
        sa.Column('open_price', sa.Float(), nullable=True),
        sa.Column('day_low', sa.Float(), nullable=True),
        sa.Column('day_high', sa.Float(), nullable=True),
        sa.Column('fifty_two_week_low', sa.Float(), nullable=True),
        sa.Column('fifty_two_week_high', sa.Float(), nullable=True),
        sa.Column('fifty_day_average', sa.Float(), nullable=True),
        sa.Column('two_hundred_day_average', sa.Float(), nullable=True),
        sa.Column('average_volume', sa.BigInteger(), nullable=True),
        sa.Column('average_volume_10days', sa.BigInteger(), nullable=True),
        sa.Column('regular_market_volume', sa.BigInteger(), nullable=True),
        sa.Column('bid', sa.Float(), nullable=True),
        sa.Column('ask', sa.Float(), nullable=True),
        sa.Column('bid_size', sa.Integer(), nullable=True),
        sa.Column('ask_size', sa.Integer(), nullable=True),
        sa.Column('beta', sa.Float(), nullable=True),
        sa.Column('trailing_pe', sa.Float(), nullable=True),
        sa.Column('forward_pe', sa.Float(), nullable=True),
        sa.Column('trailing_eps', sa.Float(), nullable=True),
        sa.Column('forward_eps', sa.Float(), nullable=True),
        sa.Column('price_to_sales_trailing_12months', sa.Float(), nullable=True),
        sa.Column('price_to_book', sa.Float(), nullable=True),
        sa.Column('enterprise_to_revenue', sa.Float(), nullable=True),
        sa.Column('enterprise_to_ebitda', sa.Float(), nullable=True),
        sa.Column('peg_ratio', sa.Float(), nullable=True),
        sa.Column('book_value', sa.Float(), nullable=True),
        sa.Column('profit_margins', sa.Float(), nullable=True),
        sa.Column('operating_margins', sa.Float(), nullable=True),
        sa.Column('gross_margins', sa.Float(), nullable=True),
        sa.Column('ebitda_margins', sa.Float(), nullable=True),
        sa.Column('return_on_assets', sa.Float(), nullable=True),
        sa.Column('return_on_equity', sa.Float(), nullable=True),
        sa.Column('total_revenue', sa.BigInteger(), nullable=True),
        sa.Column('revenue_per_share', sa.Float(), nullable=True),
        sa.Column('revenue_growth', sa.Float(), nullable=True),
        sa.Column('earnings_growth', sa.Float(), nullable=True),
        sa.Column('earnings_quarterly_growth', sa.Float(), nullable=True),
        sa.Column('ebitda', sa.BigInteger(), nullable=True),
        sa.Column('gross_profits', sa.BigInteger(), nullable=True),
        sa.Column('free_cashflow', sa.BigInteger(), nullable=True),
        sa.Column('operating_cashflow', sa.BigInteger(), nullable=True),
        sa.Column('total_cash', sa.BigInteger(), nullable=True),
        sa.Column('total_cash_per_share', sa.Float(), nullable=True),
        sa.Column('total_debt', sa.BigInteger(), nullable=True),
        sa.Column('debt_to_equity', sa.Float(), nullable=True),
        sa.Column('current_ratio', sa.Float(), nullable=True),
        sa.Column('quick_ratio', sa.Float(), nullable=True),
        sa.Column('dividend_rate', sa.Float(), nullable=True),
        sa.Column('dividend_yield', sa.Float(), nullable=True),
        sa.Column('ex_dividend_date', sa.Date(), nullable=True),
        sa.Column('payout_ratio', sa.Float(), nullable=True),
        sa.Column('five_year_avg_dividend_yield', sa.Float(), nullable=True),
        sa.Column('trailing_annual_dividend_rate', sa.Float(), nullable=True),
        sa.Column('trailing_annual_dividend_yield', sa.Float(), nullable=True),
        sa.Column('last_dividend_value', sa.Float(), nullable=True),
        sa.Column('target_high_price', sa.Float(), nullable=True),
        sa.Column('target_low_price', sa.Float(), nullable=True),
        sa.Column('target_mean_price', sa.Float(), nullable=True),
        sa.Column('target_median_price', sa.Float(), nullable=True),
        sa.Column('number_of_analyst_opinions', sa.Integer(), nullable=True),
        sa.Column('recommendation_key', sa.String(50), nullable=True),
        sa.Column('recommendation_mean', sa.Float(), nullable=True),
        sa.Column('full_time_employees', sa.Integer(), nullable=True),
        sa.Column('raw_json', postgresql.JSONB(), nullable=True),
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint('instrument_id', name='uq_ticker_profile_instrument'),
    )

    # price_history
    op.create_table(
        'price_history',
        sa.Column('instrument_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('instruments.id', ondelete='CASCADE'), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('open', sa.Numeric(20, 6), nullable=True),
        sa.Column('high', sa.Numeric(20, 6), nullable=True),
        sa.Column('low', sa.Numeric(20, 6), nullable=True),
        sa.Column('close', sa.Numeric(20, 6), nullable=True),
        sa.Column('volume', sa.BigInteger(), nullable=True),
        sa.Column('dividends', sa.Numeric(20, 6), nullable=True),
        sa.Column('stock_splits', sa.Numeric(20, 6), nullable=True),
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint('instrument_id', 'date', name='uq_price_history_instrument_date'),
    )
    op.create_index('ix_price_history_instrument_id', 'price_history', ['instrument_id'])
    op.create_index('ix_price_history_date', 'price_history', ['date'])

    # financial_statements
    op.create_table(
        'financial_statements',
        sa.Column('instrument_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('instruments.id', ondelete='CASCADE'), nullable=False),
        sa.Column('statement_type', sa.String(50), nullable=False),
        sa.Column('period_type', sa.String(20), nullable=False),
        sa.Column('period_date', sa.Date(), nullable=False),
        sa.Column('line_item', sa.String(200), nullable=False),
        sa.Column('value', sa.Numeric(30, 6), nullable=True),
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint('instrument_id', 'statement_type', 'period_type', 'period_date', 'line_item', name='uq_financial_statement_row'),
    )
    op.create_index('ix_financial_statements_instrument_id', 'financial_statements', ['instrument_id'])
    op.create_index('ix_financial_statements_type_period', 'financial_statements', ['statement_type', 'period_type'])

    # dividends
    op.create_table(
        'dividends',
        sa.Column('instrument_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('instruments.id', ondelete='CASCADE'), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('amount', sa.Numeric(20, 6), nullable=False),
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint('instrument_id', 'date', name='uq_dividend_instrument_date'),
    )
    op.create_index('ix_dividends_instrument_id', 'dividends', ['instrument_id'])

    # stock_splits
    op.create_table(
        'stock_splits',
        sa.Column('instrument_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('instruments.id', ondelete='CASCADE'), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('ratio', sa.Numeric(20, 6), nullable=False),
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint('instrument_id', 'date', name='uq_stock_split_instrument_date'),
    )
    op.create_index('ix_stock_splits_instrument_id', 'stock_splits', ['instrument_id'])

    # analyst_recommendations
    op.create_table(
        'analyst_recommendations',
        sa.Column('instrument_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('instruments.id', ondelete='CASCADE'), nullable=False),
        sa.Column('period', sa.String(50), nullable=False),
        sa.Column('strong_buy', sa.Integer(), nullable=True),
        sa.Column('buy', sa.Integer(), nullable=True),
        sa.Column('hold', sa.Integer(), nullable=True),
        sa.Column('sell', sa.Integer(), nullable=True),
        sa.Column('strong_sell', sa.Integer(), nullable=True),
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint('instrument_id', 'period', name='uq_analyst_rec_instrument_period'),
    )
    op.create_index('ix_analyst_recommendations_instrument_id', 'analyst_recommendations', ['instrument_id'])

    # analyst_price_targets
    op.create_table(
        'analyst_price_targets',
        sa.Column('instrument_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('instruments.id', ondelete='CASCADE'), nullable=False),
        sa.Column('current', sa.Float(), nullable=True),
        sa.Column('low', sa.Float(), nullable=True),
        sa.Column('high', sa.Float(), nullable=True),
        sa.Column('mean', sa.Float(), nullable=True),
        sa.Column('median', sa.Float(), nullable=True),
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint('instrument_id', name='uq_analyst_pt_instrument'),
    )

    # institutional_holders
    op.create_table(
        'institutional_holders',
        sa.Column('instrument_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('instruments.id', ondelete='CASCADE'), nullable=False),
        sa.Column('holder_name', sa.String(500), nullable=False),
        sa.Column('date_reported', sa.Date(), nullable=True),
        sa.Column('shares', sa.BigInteger(), nullable=True),
        sa.Column('value', sa.BigInteger(), nullable=True),
        sa.Column('pct_held', sa.Float(), nullable=True),
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint('instrument_id', 'holder_name', name='uq_inst_holder_instrument_name'),
    )
    op.create_index('ix_institutional_holders_instrument_id', 'institutional_holders', ['instrument_id'])

    # mutualfund_holders
    op.create_table(
        'mutualfund_holders',
        sa.Column('instrument_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('instruments.id', ondelete='CASCADE'), nullable=False),
        sa.Column('holder_name', sa.String(500), nullable=False),
        sa.Column('date_reported', sa.Date(), nullable=True),
        sa.Column('shares', sa.BigInteger(), nullable=True),
        sa.Column('value', sa.BigInteger(), nullable=True),
        sa.Column('pct_held', sa.Float(), nullable=True),
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint('instrument_id', 'holder_name', name='uq_mf_holder_instrument_name'),
    )
    op.create_index('ix_mutualfund_holders_instrument_id', 'mutualfund_holders', ['instrument_id'])

    # insider_transactions
    op.create_table(
        'insider_transactions',
        sa.Column('instrument_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('instruments.id', ondelete='CASCADE'), nullable=False),
        sa.Column('insider_name', sa.String(500), nullable=False),
        sa.Column('position', sa.String(500), nullable=True),
        sa.Column('transaction_type', sa.String(200), nullable=False),
        sa.Column('shares', sa.BigInteger(), nullable=True),
        sa.Column('value', sa.BigInteger(), nullable=True),
        sa.Column('start_date', sa.Date(), nullable=True),
        sa.Column('ownership', sa.String(50), nullable=True),
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint('instrument_id', 'insider_name', 'start_date', 'transaction_type', name='uq_insider_tx_row'),
    )
    op.create_index('ix_insider_transactions_instrument_id', 'insider_transactions', ['instrument_id'])

    # ticker_news
    op.create_table(
        'ticker_news',
        sa.Column('instrument_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('instruments.id', ondelete='CASCADE'), nullable=False),
        sa.Column('news_uuid', sa.String(200), nullable=False),
        sa.Column('title', sa.Text(), nullable=True),
        sa.Column('publisher', sa.String(500), nullable=True),
        sa.Column('link', sa.Text(), nullable=True),
        sa.Column('publish_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('news_type', sa.String(100), nullable=True),
        sa.Column('related_tickers', postgresql.JSONB(), nullable=True),
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint('instrument_id', 'news_uuid', name='uq_ticker_news_instrument_uuid'),
    )
    op.create_index('ix_ticker_news_instrument_id', 'ticker_news', ['instrument_id'])
    op.create_index('ix_ticker_news_publish_time', 'ticker_news', ['publish_time'])


def downgrade() -> None:
    """Drop all 11 yfinance data tables."""
    op.drop_table('ticker_news')
    op.drop_table('insider_transactions')
    op.drop_table('mutualfund_holders')
    op.drop_table('institutional_holders')
    op.drop_table('analyst_price_targets')
    op.drop_table('analyst_recommendations')
    op.drop_table('stock_splits')
    op.drop_table('dividends')
    op.drop_table('financial_statements')
    op.drop_table('price_history')
    op.drop_table('ticker_profiles')
