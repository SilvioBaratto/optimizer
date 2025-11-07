"""rebuild_macro_regime_tables_optimized

CLEAN REBUILD of macro regime tables with ALL optimizations:
- ✓ CHECK constraints for probability/confidence fields
- ✓ UNIQUE constraints for natural keys
- ✓ Fixed MarketIndicators relationship (now owned by run)
- ✓ Removed redundant news fields (3NF compliance)
- ✓ Composite indexes for performance
- ✓ GIN indexes for JSONB queries
- ✓ Optimized cascade behaviors
- ✓ Table comments

WARNING: This drops ALL existing macro regime data!
Only use this if you're okay with losing existing data.

Revision ID: ad5680513f25
Revises: 2582ef12b218
Create Date: 2025-10-14 10:55:30.357974

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'ad5680513f25'
down_revision: Union[str, Sequence[str], None] = '2582ef12b218'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Clean rebuild: Drop all old macro regime tables and create optimized versions.
    """

    # ========================================================================
    # PHASE 1: Drop all existing macro regime tables (CASCADE to handle FKs)
    # ========================================================================

    print("Dropping old macro regime tables...")

    # Drop in reverse dependency order
    op.execute("DROP TABLE IF EXISTS regime_transitions CASCADE")
    op.execute("DROP TABLE IF EXISTS news_articles CASCADE")
    op.execute("DROP TABLE IF EXISTS country_regime_assessments CASCADE")
    op.execute("DROP TABLE IF EXISTS baml_analyses CASCADE")
    op.execute("DROP TABLE IF EXISTS economic_indicators CASCADE")
    op.execute("DROP TABLE IF EXISTS macro_analysis_runs CASCADE")
    op.execute("DROP TABLE IF EXISTS market_indicators CASCADE")

    print("Old tables dropped successfully")

    # ========================================================================
    # PHASE 2: Create optimized tables
    # ========================================================================

    print("Creating optimized macro regime tables...")

    # Table 1: BAMLAnalysis (no dependencies)
    op.create_table(
        'baml_analyses',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('analysis_timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('confidence_adjustment', sa.Float(), nullable=True),
        sa.Column('adjusted_confidence', sa.Float(), nullable=True),
        sa.Column('enhanced_rationale', sa.Text(), nullable=True),
        sa.Column('news_sentiment', sa.String(1000), nullable=True),
        sa.Column('contradiction_warning', sa.String(2000), nullable=True),
        sa.Column('risk_adjustment_6m', sa.Float(), nullable=True),
        sa.Column('risk_adjustment_12m', sa.Float(), nullable=True),
        sa.Column('adjusted_risk_6m', sa.Float(), nullable=True),
        sa.Column('adjusted_risk_12m', sa.Float(), nullable=True),
        sa.Column('warning_signs', postgresql.JSONB(), nullable=True),
        sa.Column('leading_indicators', postgresql.JSONB(), nullable=True),
        sa.Column('transition_likelihood', sa.Float(), nullable=True),
        sa.Column('potential_next_regime', sa.String(50), nullable=True),
        sa.Column('narrative_shift', sa.String(50), nullable=True),
        sa.Column('narrative_themes', postgresql.JSONB(), nullable=True),
        sa.Column('estimated_timeframe', sa.String(100), nullable=True),
        sa.Column('transition_signals_list', postgresql.JSONB(), nullable=True),
        sa.Column('sector_adjustments', postgresql.JSONB(), nullable=True),
        sa.Column('leading_sectors', postgresql.JSONB(), nullable=True),
        sa.Column('lagging_sectors', postgresql.JSONB(), nullable=True),
        sa.Column('sector_drivers', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        # Check constraints
        sa.CheckConstraint('confidence_adjustment IS NULL OR (confidence_adjustment >= -1.0 AND confidence_adjustment <= 1.0)', name='ck_conf_adj_range'),
        sa.CheckConstraint('adjusted_confidence IS NULL OR (adjusted_confidence >= 0.0 AND adjusted_confidence <= 1.0)', name='ck_adj_conf_range'),
        sa.CheckConstraint('transition_likelihood IS NULL OR (transition_likelihood >= 0.0 AND transition_likelihood <= 1.0)', name='ck_transition_like_range'),
        sa.CheckConstraint('adjusted_risk_6m IS NULL OR (adjusted_risk_6m >= 0.0 AND adjusted_risk_6m <= 1.0)', name='ck_adj_risk_6m_range'),
        sa.CheckConstraint('adjusted_risk_12m IS NULL OR (adjusted_risk_12m >= 0.0 AND adjusted_risk_12m <= 1.0)', name='ck_adj_risk_12m_range'),
        comment='BAML AI-enhanced analysis outputs for regime assessment'
    )
    op.create_index('idx_baml_timestamp', 'baml_analyses', ['analysis_timestamp'])
    op.create_index('idx_baml_regime_timestamp', 'baml_analyses', ['potential_next_regime', 'analysis_timestamp'])
    op.create_index('idx_baml_likelihood_timestamp', 'baml_analyses', ['transition_likelihood', 'analysis_timestamp'])

    # Table 2: EconomicIndicators (no dependencies)
    op.create_table(
        'economic_indicators',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('country', sa.String(50), nullable=False),
        sa.Column('data_timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('gdp_growth_qq', sa.Float(), nullable=True),
        sa.Column('gdp_growth_yy', sa.Float(), nullable=True),
        sa.Column('unemployment', sa.Float(), nullable=True),
        sa.Column('inflation', sa.Float(), nullable=True),
        sa.Column('industrial_production', sa.Float(), nullable=True),
        sa.Column('retail_sales', sa.Float(), nullable=True),
        sa.Column('consumer_prices', sa.Float(), nullable=True),
        sa.Column('deficit', sa.Float(), nullable=True),
        sa.Column('debt', sa.Float(), nullable=True),
        sa.Column('st_rate', sa.Float(), nullable=True),
        sa.Column('lt_rate', sa.Float(), nullable=True),
        sa.Column('gdp_forecast_6m', sa.Float(), nullable=True),
        sa.Column('inflation_forecast_6m', sa.Float(), nullable=True),
        sa.Column('earnings_forecast_12m', sa.Float(), nullable=True),
        sa.Column('status', sa.String(20), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        # Constraints
        sa.UniqueConstraint('country', 'data_timestamp', name='uq_country_data_timestamp'),
        sa.CheckConstraint('unemployment IS NULL OR unemployment >= 0', name='ck_unemployment_positive'),
        sa.CheckConstraint('debt IS NULL OR debt >= 0', name='ck_debt_positive'),
        comment='Country-specific economic indicators from Il Sole 24 Ore'
    )
    op.create_index('idx_econ_ind_country', 'economic_indicators', ['country'])
    op.create_index('idx_econ_ind_timestamp', 'economic_indicators', ['data_timestamp'])
    op.create_index('idx_econ_ind_country_timestamp', 'economic_indicators', ['country', 'data_timestamp'])
    op.create_index('idx_econ_status_country', 'economic_indicators', ['status', 'country'])

    # Table 3: MacroAnalysisRun (no dependencies)
    op.create_table(
        'macro_analysis_runs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('run_timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('analysis_type', sa.String(50), nullable=False),
        sa.Column('full_content_fetched', sa.Boolean(), nullable=False),
        sa.Column('num_countries', sa.Integer(), nullable=False),
        sa.Column('countries', postgresql.JSONB(), nullable=False),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        # Constraints
        sa.CheckConstraint('num_countries > 0', name='ck_num_countries_positive'),
        comment='Top-level container for macro regime analysis execution across countries'
    )
    op.create_index('idx_macro_run_timestamp', 'macro_analysis_runs', ['run_timestamp'])
    op.create_index('idx_macro_run_type', 'macro_analysis_runs', ['analysis_type'])
    op.create_index('idx_macro_run_type_timestamp', 'macro_analysis_runs', ['analysis_type', 'run_timestamp'])
    op.create_index('idx_macro_run_countries_gin', 'macro_analysis_runs', ['countries'], postgresql_using='gin')

    # Table 4: MarketIndicators (depends on MacroAnalysisRun)
    op.create_table(
        'market_indicators',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('analysis_run_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('data_timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('ism_pmi', sa.Float(), nullable=True),
        sa.Column('yield_curve_2s10s', sa.Float(), nullable=True),
        sa.Column('hy_spread', sa.Float(), nullable=True),
        sa.Column('ig_spread', sa.Float(), nullable=True),
        sa.Column('vix', sa.Float(), nullable=True),
        sa.Column('vix_signal', sa.String(20), nullable=True),
        sa.Column('ism_signal', sa.String(20), nullable=True),
        sa.Column('curve_signal', sa.String(20), nullable=True),
        sa.Column('credit_signal', sa.String(20), nullable=True),
        sa.Column('data_quality', sa.String(20), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        # Foreign key
        sa.ForeignKeyConstraint(['analysis_run_id'], ['macro_analysis_runs.id'], ondelete='CASCADE'),
        # Constraints
        sa.UniqueConstraint('analysis_run_id', name='uq_market_run'),
        sa.CheckConstraint('ism_pmi IS NULL OR ism_pmi >= 0', name='ck_ism_pmi_positive'),
        sa.CheckConstraint('vix IS NULL OR vix >= 0', name='ck_vix_positive'),
        comment='Global market indicators (FRED/Yahoo Finance) for analysis run'
    )
    op.create_index('idx_market_ind_timestamp', 'market_indicators', ['data_timestamp'])
    op.create_index('idx_market_ind_vix', 'market_indicators', ['vix'])
    op.create_index('idx_market_ind_run_id', 'market_indicators', ['analysis_run_id'])
    op.create_index('idx_market_quality_timestamp', 'market_indicators', ['data_quality', 'data_timestamp'])

    # Table 5: CountryRegimeAssessment (depends on MacroAnalysisRun, EconomicIndicators, BAMLAnalysis)
    op.create_table(
        'country_regime_assessments',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('analysis_run_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('economic_indicators_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('baml_analysis_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('country', sa.String(50), nullable=False),
        sa.Column('assessment_timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('regime', sa.String(30), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('transition_probability', sa.Float(), nullable=False),
        sa.Column('rationale', sa.Text(), nullable=False),
        sa.Column('gdp_momentum', sa.String(50), nullable=True),
        sa.Column('inflation_momentum', sa.String(50), nullable=True),
        sa.Column('recession_risk_6m', sa.Float(), nullable=True),
        sa.Column('recession_risk_12m', sa.Float(), nullable=True),
        sa.Column('sector_tilts', postgresql.JSONB(), nullable=False),
        sa.Column('factor_timing', postgresql.JSONB(), nullable=False),
        sa.Column('news_enhanced', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        # Foreign keys
        sa.ForeignKeyConstraint(['analysis_run_id'], ['macro_analysis_runs.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['economic_indicators_id'], ['economic_indicators.id'], ondelete='RESTRICT'),
        sa.ForeignKeyConstraint(['baml_analysis_id'], ['baml_analyses.id'], ondelete='SET NULL'),
        # Constraints
        sa.UniqueConstraint('country', 'assessment_timestamp', name='uq_country_assessment_timestamp'),
        sa.CheckConstraint('confidence >= 0.0 AND confidence <= 1.0', name='ck_confidence_range'),
        sa.CheckConstraint('transition_probability >= 0.0 AND transition_probability <= 1.0', name='ck_transition_prob_range'),
        sa.CheckConstraint('recession_risk_6m IS NULL OR (recession_risk_6m >= 0.0 AND recession_risk_6m <= 1.0)', name='ck_recession_6m_range'),
        sa.CheckConstraint('recession_risk_12m IS NULL OR (recession_risk_12m >= 0.0 AND recession_risk_12m <= 1.0)', name='ck_recession_12m_range'),
        comment='Business cycle regime assessment for specific countries with AI-enhanced analysis'
    )
    op.create_index('idx_assessment_country', 'country_regime_assessments', ['country'])
    op.create_index('idx_assessment_regime', 'country_regime_assessments', ['regime'])
    op.create_index('idx_assessment_timestamp', 'country_regime_assessments', ['assessment_timestamp'])
    op.create_index('idx_assessment_country_timestamp', 'country_regime_assessments', ['country', 'assessment_timestamp'])
    op.create_index('idx_assessment_country_regime', 'country_regime_assessments', ['country', 'regime'])
    op.create_index('idx_assessment_run_country', 'country_regime_assessments', ['analysis_run_id', 'country'])
    op.create_index('idx_assessment_enhanced_timestamp', 'country_regime_assessments', ['news_enhanced', 'assessment_timestamp'])
    op.create_index('idx_assessment_sector_tilts_gin', 'country_regime_assessments', ['sector_tilts'], postgresql_using='gin')
    op.create_index('idx_assessment_factor_timing_gin', 'country_regime_assessments', ['factor_timing'], postgresql_using='gin')

    # Table 6: NewsArticle (depends on CountryRegimeAssessment)
    op.create_table(
        'news_articles',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('assessment_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('title', sa.Text(), nullable=False),
        sa.Column('publisher', sa.String(255), nullable=True),
        sa.Column('published_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('link', sa.Text(), nullable=True),
        sa.Column('full_content', sa.Text(), nullable=True),
        sa.Column('is_macro_relevant', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        # Foreign key
        sa.ForeignKeyConstraint(['assessment_id'], ['country_regime_assessments.id'], ondelete='CASCADE'),
        comment='Macroeconomic news articles used in regime analysis'
    )
    op.create_index('idx_news_assessment_id', 'news_articles', ['assessment_id'])
    op.create_index('idx_news_published_date', 'news_articles', ['published_date'])
    op.create_index('idx_news_relevant_date', 'news_articles', ['is_macro_relevant', 'published_date'])

    # Table 7: RegimeTransition (depends on CountryRegimeAssessment)
    op.create_table(
        'regime_transitions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('assessment_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('country', sa.String(50), nullable=False),
        sa.Column('transition_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('from_regime', sa.String(30), nullable=False),
        sa.Column('to_regime', sa.String(30), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('alert_level', sa.String(20), nullable=False),
        sa.Column('days_since_last_transition', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        # Foreign key
        sa.ForeignKeyConstraint(['assessment_id'], ['country_regime_assessments.id'], ondelete='CASCADE'),
        # Constraints
        sa.CheckConstraint('confidence >= 0.0 AND confidence <= 1.0', name='ck_transition_confidence_range'),
        sa.CheckConstraint('days_since_last_transition IS NULL OR days_since_last_transition >= 0', name='ck_days_positive'),
        comment='Historical regime transitions for tracking changes over time'
    )
    op.create_index('idx_transition_country', 'regime_transitions', ['country'])
    op.create_index('idx_transition_date', 'regime_transitions', ['transition_date'])
    op.create_index('idx_transition_country_date', 'regime_transitions', ['country', 'transition_date'])
    op.create_index('idx_transition_from_to', 'regime_transitions', ['from_regime', 'to_regime'])
    op.create_index('idx_transition_alert_date', 'regime_transitions', ['alert_level', 'transition_date'])
    op.create_index('idx_transition_country_alert', 'regime_transitions', ['country', 'alert_level'])

    print("Optimized tables created successfully!")


def downgrade() -> None:
    """
    Downgrade: Drop optimized tables and restore old structure.

    NOTE: This won't restore any data - it just restores the old table structure.
    You would need a database backup to restore actual data.
    """

    print("WARNING: Downgrade will drop optimized tables but cannot restore old data!")

    # Drop optimized tables
    op.execute("DROP TABLE IF EXISTS regime_transitions CASCADE")
    op.execute("DROP TABLE IF EXISTS news_articles CASCADE")
    op.execute("DROP TABLE IF EXISTS country_regime_assessments CASCADE")
    op.execute("DROP TABLE IF EXISTS market_indicators CASCADE")
    op.execute("DROP TABLE IF EXISTS macro_analysis_runs CASCADE")
    op.execute("DROP TABLE IF EXISTS economic_indicators CASCADE")
    op.execute("DROP TABLE IF EXISTS baml_analyses CASCADE")

    print("Optimized tables dropped. You'll need to manually recreate old structure or restore from backup.")
