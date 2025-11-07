#!/usr/bin/env python3
"""
Database Saver for Macro Regime Analysis
==========================================
Saves macro regime analysis results to PostgreSQL database.

Aligned with simplified schema that stores BAML BusinessCycleClassification
directly in CountryRegimeAssessment.
"""

from datetime import datetime
from typing import Dict, List, Optional
import uuid

from app.database import DatabaseManager
from app.models.macro_regime import (
    MacroAnalysisRun,
    MarketIndicators,
    CountryRegimeAssessment,
    EconomicIndicators,
    RegimeTransition,
)
from app.models.trading_economics import (
    TradingEconomicsSnapshot,
    TradingEconomicsIndicator,
    TradingEconomicsBondYield
)
from app.models.news import NewsArticle

def _normalize_regime_value(regime: Optional[str]) -> str:
    """
    Normalize regime value to database format.

    Converts various formats to match RegimeEnum:
    - 'MID_CYCLE', 'mid_cycle', 'MID CYCLE' → 'MID_CYCLE'
    - 'EARLY_CYCLE', 'early_cycle', 'EARLY CYCLE' → 'EARLY_CYCLE'
    - etc.
    """
    if not regime:
        return 'UNCERTAIN'

    # Convert to uppercase and replace spaces/hyphens with underscores
    normalized = regime.upper().replace(' ', '_').replace('-', '_').strip()

    # Map common variations
    regime_map = {
        'MID_CYCLE': 'MID_CYCLE',
        'MIDCYCLE': 'MID_CYCLE',
        'MID': 'MID_CYCLE',
        'EARLY_CYCLE': 'EARLY_CYCLE',
        'EARLYCYCLE': 'EARLY_CYCLE',
        'EARLY': 'EARLY_CYCLE',
        'LATE_CYCLE': 'LATE_CYCLE',
        'LATECYCLE': 'LATE_CYCLE',
        'LATE': 'LATE_CYCLE',
        'RECESSION': 'RECESSION',
        'CONTRACTION': 'RECESSION',
        'UNCERTAIN': 'UNCERTAIN',
        'UNKNOWN': 'UNCERTAIN',
        'MIXED': 'UNCERTAIN'
    }

    return regime_map.get(normalized, 'UNCERTAIN')


def _normalize_signal_value(signal: Optional[str], signal_type: str) -> Optional[str]:
    """
    Normalize signal values to match enum formats.

    Parameters
    ----------
    signal : str or None
        Raw signal value
    signal_type : str
        Type of signal: 'ism', 'yield_curve', 'credit_spread'

    Returns
    -------
    str or None
        Normalized signal value
    """
    if not signal:
        return None

    normalized = signal.upper().replace(' ', '_').replace('-', '_').strip()

    if signal_type == 'ism':
        # ISMSignalEnum: STRONG_EXPANSION, MILD_EXPANSION, MILD_CONTRACTION, DEEP_CONTRACTION
        ism_map = {
            'STRONG_EXPANSION': 'STRONG_EXPANSION',
            'STRONG': 'STRONG_EXPANSION',
            'EXPANSION': 'STRONG_EXPANSION',
            'MILD_EXPANSION': 'MILD_EXPANSION',
            'MODERATE': 'MILD_EXPANSION',
            'MILD_CONTRACTION': 'MILD_CONTRACTION',
            'CONTRACTION': 'MILD_CONTRACTION',
            'WEAK': 'MILD_CONTRACTION',
            'DEEP_CONTRACTION': 'DEEP_CONTRACTION',
            'SEVERE': 'DEEP_CONTRACTION'
        }
        return ism_map.get(normalized, 'MILD_EXPANSION')

    elif signal_type == 'yield_curve':
        # YieldCurveSignalEnum: STEEP, NORMAL, FLAT, INVERTED
        curve_map = {
            'STEEP': 'STEEP',
            'NORMAL': 'NORMAL',
            'FLAT': 'FLAT',
            'INVERTED': 'INVERTED'
        }
        return curve_map.get(normalized, 'NORMAL')

    elif signal_type == 'credit_spread':
        # CreditSpreadSignalEnum: TIGHT, NEUTRAL, WIDENING, STRESS
        credit_map = {
            'TIGHT': 'TIGHT',
            'NEUTRAL': 'NEUTRAL',
            'WIDENING': 'WIDENING',
            'STRESS': 'STRESS',
            'DISTRESS': 'STRESS',
            'RISK_ON': 'TIGHT',
            'RISK_OFF': 'WIDENING'
        }
        return credit_map.get(normalized, 'NEUTRAL')

    return None


def _normalize_factor_exposure(factor: Optional[str]) -> str:
    """Normalize factor exposure to match FactorExposureEnum."""
    if not factor:
        return 'BALANCED'

    normalized = factor.upper().replace(' ', '_').replace('-', '_').strip()

    factor_map = {
        'GROWTH_MOMENTUM': 'GROWTH_MOMENTUM',
        'GROWTH': 'GROWTH_MOMENTUM',
        'MOMENTUM': 'GROWTH_MOMENTUM',
        'QUALITY_DEFENSIVE': 'QUALITY_DEFENSIVE',
        'QUALITY': 'QUALITY_DEFENSIVE',
        'DEFENSIVE': 'QUALITY_DEFENSIVE',
        'BALANCED': 'BALANCED',
        'NEUTRAL': 'BALANCED'
    }

    return factor_map.get(normalized, 'BALANCED')


class MacroRegimeDatabaseSaver:
    """Saves macro regime analysis results to database."""

    def __init__(self):
        """Initialize database manager."""
        self.db_manager = DatabaseManager()
        self.db_manager.initialize()
        self.analysis_run_id = None

    def save_analysis_run(
        self,
        all_results: Dict[str, Dict],
        fred_data: Dict,
        notes: Optional[str] = None
    ) -> uuid.UUID:
        """
        Save complete macro regime analysis run to database.

        Parameters
        ----------
        all_results : dict
            Results for each country (from classify_pure_llm)
        fred_data : dict
            Global market indicators from FRED
        notes : str, optional
            Optional notes

        Returns
        -------
        uuid.UUID
            Analysis run ID
        """
        run_timestamp = datetime.now()
        countries = list(all_results.keys())

        print(f"\n{'='*100}")
        print("SAVING MACRO REGIME ANALYSIS TO DATABASE")
        print(f"{'='*100}")
        print(f"Countries: {', '.join(countries)}")
        print(f"Timestamp: {run_timestamp}")

        with self.db_manager.get_session() as session:
            # Create analysis run
            print("\n[1/4] Creating analysis run record...")
            analysis_run = MacroAnalysisRun(
                run_timestamp=run_timestamp,
                num_countries=len(countries),
                countries=countries,
                notes=notes
            )
            session.add(analysis_run)
            session.flush()
            print(f"  ✓ Created (ID: {analysis_run.id})")

            # Save market indicators
            print("\n[2/4] Saving global market indicators...")
            market_indicators_obj = self._save_market_indicators(
                session, fred_data, run_timestamp, analysis_run.id
            )
            session.flush()
            print(f"  ✓ Saved (ID: {market_indicators_obj.id})")

            # Save each country
            print("\n[3/4] Saving country assessments...")
            for country, results in all_results.items():
                print(f"  • {country}...", end=" ")
                self._save_country_assessment(
                    session, analysis_run.id, country, results, run_timestamp
                )
                print("✓")

            # Commit
            print("\n[4/4] Committing to database...")
            session.commit()
            print("  ✓ All data saved successfully")
            print(f"\n{'='*100}")
            print(f"Analysis Run ID: {analysis_run.id}")
            print(f"{'='*100}\n")

            return analysis_run.id

    def initialize_analysis_run(
        self,
        fred_data: Dict,
        expected_countries: List[str],
        notes: Optional[str] = None
    ) -> uuid.UUID:
        """
        Initialize an analysis run and save market indicators.

        Call this ONCE before analyzing countries incrementally.
        """
        run_timestamp = datetime.now()

        print(f"\n{'='*100}")
        print("INITIALIZING MACRO REGIME ANALYSIS RUN")
        print(f"{'='*100}")
        print(f"Expected countries: {', '.join(expected_countries)}")
        print(f"Timestamp: {run_timestamp}")

        with self.db_manager.get_session() as session:
            # Create analysis run
            print("\n[1/2] Creating analysis run record...")
            analysis_run = MacroAnalysisRun(
                run_timestamp=run_timestamp,
                num_countries=len(expected_countries),
                countries=expected_countries,
                notes=notes
            )
            session.add(analysis_run)
            session.flush()
            self.analysis_run_id = analysis_run.id
            print(f"  ✓ Created (ID: {self.analysis_run_id})")

            # Save market indicators
            print("\n[2/2] Saving global market indicators...")
            market_indicators_obj = self._save_market_indicators(
                session, fred_data, run_timestamp, self.analysis_run_id
            )
            session.flush()
            print(f"  ✓ Saved (ID: {market_indicators_obj.id})")

            # Commit
            session.commit()
            print(f"\n✅ Analysis run initialized successfully")
            print(f"{'='*100}\n")

            return self.analysis_run_id

    def save_single_country(
        self,
        country: str,
        results: Dict,
        run_id: Optional[uuid.UUID] = None
    ) -> None:
        """
        Save a single country's assessment.

        Parameters
        ----------
        country : str
            Country code
        results : dict
            Results from classify_pure_llm for this country
        run_id : uuid.UUID, optional
            Analysis run ID
        """
        if run_id is None:
            run_id = self.analysis_run_id

        if run_id is None:
            raise ValueError("No analysis run initialized. Call initialize_analysis_run() first.")

        assessment_timestamp = datetime.now()
        print(f"\n💾 Saving {country} to database...", end=" ")

        with self.db_manager.get_session() as session:
            self._save_country_assessment(session, run_id, country, results, assessment_timestamp)
            session.commit()

        print("✅")

    def _save_market_indicators(
        self,
        session,
        fred_data: Dict,
        data_timestamp: datetime,
        analysis_run_id: uuid.UUID
    ) -> MarketIndicators:
        """Save global market indicators (essential FRED data only)."""
        market_indicators = MarketIndicators(
            analysis_run_id=analysis_run_id,
            data_timestamp=data_timestamp,
            ism_pmi=fred_data.get('ism_pmi'),
            yield_curve_2s10s=fred_data.get('yield_curve_2s10s'),
            hy_spread=fred_data.get('hy_spread'),
            vix=fred_data.get('vix')
        )
        session.add(market_indicators)
        return market_indicators

    def _save_country_assessment(
        self,
        session,
        analysis_run_id: uuid.UUID,
        country: str,
        results: Dict,
        assessment_timestamp: datetime
    ):
        """
        Save assessment for one country.

        Expects results to contain:
        - 'llm_classification': BusinessCycleClassification object from BAML
        - 'ilsole_data': Economic indicators
        - 'news_data': News articles (optional)
        - 'transition': Transition info (optional)
        """
        # Get BAML classification (BusinessCycleClassification object)
        llm_classification = results.get('llm_classification')
        if not llm_classification:
            # Fallback: check for 'assessment' key (legacy)
            llm_classification = results.get('assessment')

        if not llm_classification:
            raise ValueError(f"No BAML classification found for {country}")

        ilsole_data = results.get('ilsole_data', {})
        news_data = results.get('news_data', [])
        transition_info = results.get('transition')

        # Save economic indicators
        econ_indicators = self._save_economic_indicators(
            session, country, ilsole_data, assessment_timestamp
        )
        session.flush()

        # Convert recession risks from percentage (0-100) to probability (0.0-1.0) if needed
        recession_risk_6m = llm_classification.recession_risk_6m
        if recession_risk_6m is not None and recession_risk_6m > 1.0:
            recession_risk_6m = recession_risk_6m / 100.0

        recession_risk_12m = llm_classification.recession_risk_12m
        if recession_risk_12m is not None and recession_risk_12m > 1.0:
            recession_risk_12m = recession_risk_12m / 100.0

        # Convert confidence to Python float (handle numpy types)
        confidence = llm_classification.confidence
        if confidence is not None:
            confidence = float(confidence)

        # Create CountryRegimeAssessment with BAML BusinessCycleClassification fields
        assessment_obj = CountryRegimeAssessment(
            analysis_run_id=analysis_run_id,
            economic_indicators_id=econ_indicators.id,
            country=country,
            assessment_timestamp=assessment_timestamp,
            # Core classification
            regime=_normalize_regime_value(llm_classification.regime),
            confidence=confidence,
            rationale=llm_classification.rationale,
            # Key indicator signals
            ism_signal=_normalize_signal_value(llm_classification.ism_signal, 'ism'),
            yield_curve_signal=_normalize_signal_value(llm_classification.yield_curve_signal, 'yield_curve'),
            credit_spread_signal=_normalize_signal_value(llm_classification.credit_spread_signal, 'credit_spread'),
            # Recession risk
            recession_risk_6m=recession_risk_6m,
            recession_risk_12m=recession_risk_12m,
            recession_drivers=llm_classification.recession_drivers if llm_classification.recession_drivers else [],
            # Portfolio positioning
            sector_tilts=llm_classification.sector_tilts if llm_classification.sector_tilts else {},
            recommended_overweights=llm_classification.recommended_overweights if llm_classification.recommended_overweights else [],
            recommended_underweights=llm_classification.recommended_underweights if llm_classification.recommended_underweights else [],
            factor_exposure=_normalize_factor_exposure(llm_classification.factor_exposure),
            # Risk monitoring
            primary_risks=llm_classification.primary_risks if llm_classification.primary_risks else [],
            conflicting_signals=llm_classification.conflicting_signals if llm_classification.conflicting_signals else []
        )
        session.add(assessment_obj)
        session.flush()

        # Save news articles
        if news_data:
            self._save_news_articles(session, assessment_obj.id, news_data)
            session.flush()

        # Save transition (if detected)
        if transition_info and transition_info.get('transition_detected'):
            self._save_regime_transition(session, assessment_obj.id, country, transition_info)
            session.flush()

    def _save_economic_indicators(
        self,
        session,
        country: str,
        ilsole_data: Dict,
        data_timestamp: datetime
    ) -> EconomicIndicators:
        """Save country economic indicators (essential Il Sole data only)."""
        real = ilsole_data.get('real', {})
        forecast = ilsole_data.get('forecast', {})

        econ_indicators = EconomicIndicators(
            country=country,
            data_timestamp=data_timestamp,
            gdp_growth_qq=real.get('gdp_growth_qq'),
            gdp_growth_yy=real.get('gdp_yy'),
            unemployment=real.get('unemployment'),
            inflation=real.get('consumer_prices') or real.get('inflation'),
            industrial_production=real.get('industrial_production'),
            gdp_forecast_6m=forecast.get('gdp_growth_6m'),
            inflation_forecast_6m=forecast.get('inflation_6m'),
            earnings_forecast_12m=forecast.get('earnings_12m')
        )
        session.add(econ_indicators)
        return econ_indicators

    def _save_news_articles(self, session, assessment_id: uuid.UUID, news_data: List[Dict]):
        """Save news articles (essential metadata only)."""
        for article in news_data[:50]:  # Limit to 50
            news_article = NewsArticle(
                assessment_id=assessment_id,
                title=article.get('title', ''),
                publisher=article.get('publisher'),
                published_date=self._parse_date(article.get('date')),
                link=article.get('link')
            )
            session.add(news_article)

    def _save_regime_transition(
        self,
        session,
        assessment_id: uuid.UUID,
        country: str,
        transition_info: Dict
    ):
        """Save regime transition."""
        # Convert numpy types to Python native types for PostgreSQL compatibility
        confidence = transition_info['confidence']
        if confidence is not None:
            confidence = float(confidence)  # Convert np.float64 to Python float

        days_since = transition_info.get('days_since_last_transition')
        if days_since is not None:
            days_since = int(days_since)  # Convert np.int64 to Python int if needed

        transition = RegimeTransition(
            assessment_id=assessment_id,
            country=country,
            transition_date=self._parse_date(transition_info.get('transition_date')),
            from_regime=_normalize_regime_value(transition_info['from_regime']),
            to_regime=_normalize_regime_value(transition_info['to_regime']),
            confidence=confidence,
            days_since_last_transition=days_since
        )
        session.add(transition)

    @staticmethod
    def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string."""
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except:
            try:
                return datetime.strptime(date_str, '%Y-%m-%d')
            except:
                return None

    def save_trading_economics_data(
        self,
        trading_economics_data: Dict[str, Dict],
        fetch_timestamp: Optional[datetime] = None
    ) -> List[uuid.UUID]:
        """
        Save Trading Economics data to database.

        Checks if data for today already exists before saving to avoid duplicates.

        Parameters
        ----------
        trading_economics_data : dict
            Trading Economics data for each country from get_all_portfolio_countries()
            Format: {
                'USA': {
                    'country': 'USA',
                    'status': 'success',
                    'indicators': {...},
                    'bond_yields': {...},
                    'industrial_production': {...},
                    'capacity_utilization': {...}
                },
                ...
            }
        fetch_timestamp : datetime, optional
            When data was fetched (defaults to now)

        Returns
        -------
        list of uuid.UUID
            Snapshot IDs for each country saved (or existing IDs if already saved today)
        """
        if fetch_timestamp is None:
            fetch_timestamp = datetime.now()

        snapshot_ids = []

        print(f"\n{'='*100}")
        print("SAVING TRADING ECONOMICS DATA TO DATABASE")
        print(f"{'='*100}")
        print(f"Countries: {', '.join(trading_economics_data.keys())}")
        print(f"Timestamp: {fetch_timestamp}")

        with self.db_manager.get_session() as session:
            # Import here to avoid circular dependency
            from sqlalchemy import select, cast, Date

            for country, data in trading_economics_data.items():
                if data.get('status') != 'success':
                    print(f"  ❌ Skipping {country} (status: {data.get('status')})")
                    continue

                # Check if data for this country already exists today
                today_date = fetch_timestamp.date()
                existing_snapshot = session.execute(
                    select(TradingEconomicsSnapshot)
                    .where(TradingEconomicsSnapshot.country == country)
                    .where(cast(TradingEconomicsSnapshot.fetch_timestamp, Date) == today_date)
                    .order_by(TradingEconomicsSnapshot.fetch_timestamp.desc())
                ).scalars().first()

                if existing_snapshot:
                    print(f"\n  [Trading Economics] {country}: Already saved today (ID: {existing_snapshot.id}) ⏭️")
                    snapshot_ids.append(existing_snapshot.id)
                    continue

                print(f"\n  [Trading Economics] Saving {country}...", end=" ")

                # Create snapshot
                snapshot = TradingEconomicsSnapshot(
                    country=country,
                    fetch_timestamp=fetch_timestamp,
                    source_url=data.get('source_url'),
                    num_indicators=data.get('num_indicators', 0),
                    num_bond_yields=data.get('num_bond_yields', 0)
                )

                # Add industrial production
                industrial_production = data.get('industrial_production')
                if industrial_production:
                    snapshot.industrial_production_value = industrial_production.get('value')
                    snapshot.industrial_production_previous = industrial_production.get('previous')
                    snapshot.industrial_production_reference = industrial_production.get('reference')
                    snapshot.industrial_production_unit = industrial_production.get('unit')

                # Add capacity utilization
                capacity_utilization = data.get('capacity_utilization')
                if capacity_utilization:
                    snapshot.capacity_utilization_value = capacity_utilization.get('value')
                    snapshot.capacity_utilization_previous = capacity_utilization.get('previous')
                    snapshot.capacity_utilization_reference = capacity_utilization.get('reference')
                    snapshot.capacity_utilization_unit = capacity_utilization.get('unit')

                session.add(snapshot)
                session.flush()

                # Save indicators
                indicators = data.get('indicators', {})
                for indicator_name, indicator_data in indicators.items():
                    indicator_obj = TradingEconomicsIndicator(
                        snapshot_id=snapshot.id,
                        indicator_name=indicator_name,
                        raw_name=indicator_data.get('raw_name', indicator_name),
                        value=indicator_data['value'],
                        previous=indicator_data.get('previous'),
                        unit=indicator_data.get('unit'),
                        reference=indicator_data.get('reference')
                    )
                    session.add(indicator_obj)

                # Save bond yields
                bond_yields = data.get('bond_yields', {})
                for maturity, yield_data in bond_yields.items():
                    bond_obj = TradingEconomicsBondYield(
                        snapshot_id=snapshot.id,
                        maturity=maturity,
                        raw_name=yield_data.get('raw_name', f"{country} {maturity}"),
                        yield_value=yield_data['yield'],
                        day_change=yield_data.get('day_change'),
                        month_change=yield_data.get('month_change'),
                        year_change=yield_data.get('year_change'),
                        date=yield_data.get('date')
                    )
                    session.add(bond_obj)

                session.flush()
                snapshot_ids.append(snapshot.id)
                print(f"✓ (ID: {snapshot.id})")

            # Commit all at once
            print(f"\n  Committing to database...")
            session.commit()
            print(f"  ✓ Trading Economics data processing complete")

            print(f"\n{'='*100}")
            print(f"Processed {len(snapshot_ids)} country snapshots")
            print(f"{'='*100}\n")

        return snapshot_ids


def save_analysis_to_database(
    all_results: Dict[str, Dict],
    fred_data: Dict,
    notes: Optional[str] = None
) -> uuid.UUID:
    """
    Save macro regime analysis to database.

    Parameters
    ----------
    all_results : dict
        Results from classify_pure_llm() for each country
    fred_data : dict
        Global market indicators from FRED
    notes : str, optional
        Optional notes

    Returns
    -------
    uuid.UUID
        Analysis run ID

    Example
    -------
    >>> from database_saver import save_analysis_to_database
    >>> run_id = save_analysis_to_database(all_results, fred_data)
    """
    saver = MacroRegimeDatabaseSaver()
    return saver.save_analysis_run(all_results, fred_data, notes)
