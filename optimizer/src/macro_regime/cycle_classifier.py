#!/usr/bin/env python3
"""
Business Cycle Classifier
==========================
Comprehensive business cycle detection using institutional methodology from guide.md

Combines:
1. Il Sole 24 Ore real indicators (GDP, unemployment, inflation)
2. Il Sole 24 Ore forecasts (GDP/inflation/earnings projections)
3. FRED real-time data (ISM PMI, yield curve, credit spreads)

Classification based on guide.md pages 69-90 (Business Cycle Positioning)

Returns:
- Regime: early_cycle, mid_cycle, late_cycle, recession
- Confidence: 0-1 probability
- Sector tilts: Overweight/underweight recommendations
- Factor timing: Value, Momentum, Quality, Growth adjustments
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, List, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime

# Lazy imports for BAML types - only imported when needed
if TYPE_CHECKING:
    from baml_client.types import (
        BusinessCycleClassification,
        NewsArticle,
        EconomicIndicators,
        MarketIndicators,
        MacroNewsSignals
    )

class BusinessCycleClassifier:
    """
    Institutional-grade business cycle classifier.

    Based on methodology from portfolio_guideline/guide.md
    """

    # Cycle phase definitions (guide.md pages 73-79)
    REGIME_DEFINITIONS = {
        'early_cycle': {
            'duration': '~1 year',
            'returns': '20%+ annualized',
            'description': 'Recovery from recession'
        },
        'mid_cycle': {
            'duration': '~4 years',
            'returns': '14% annualized',
            'description': 'Sustained expansion'
        },
        'late_cycle': {
            'duration': '~1.5 years',
            'returns': '5% annualized',
            'description': 'Peak and slowdown'
        },
        'recession': {
            'duration': '<1 year',
            'returns': 'Negative',
            'description': 'Economic contraction'
        }
    }

    # Macro-relevant keywords for news filtering
    MACRO_KEYWORDS = [
        'economy', 'gdp', 'inflation', 'recession', 'federal', 'central bank',
        'unemployment', 'jobs', 'growth', 'economic', 'fiscal', 'monetary',
        'earnings', 'layoff', 'rate', 'policy', 'market', 'financial'
    ]

    def __init__(self):
        """Initialize classifier."""
        self._baml_client = None

    def _classify_regime(self, real, forecast, gdp_mom, inf_mom,
                         ism, curve, hy) -> Tuple[str, float, str, Dict]:
        """
        Core classification logic from guide.md.

        Priority: Country-specific indicators (Il Sole) FIRST, then FRED global indicators.

        Returns: (regime, confidence, rationale, signals)
        """
        # Base signals dictionary that ALWAYS includes momentum (for database storage)
        base_signals = {
            'gdp_momentum': gdp_mom,
            'inflation_momentum': inf_mom
        }

        # =====================================================================
        # PRIORITY 1: COUNTRY-SPECIFIC RECESSION SIGNALS (Il Sole data)
        # =====================================================================
        # Strong recession signal from country data
        gdp_qq = real.get('gdp_growth_qq')
        gdp_fcst = forecast.get('gdp_growth_6m')

        if (gdp_qq is not None and gdp_qq < -0.5) or (gdp_fcst is not None and gdp_fcst < 0.5):
            if real.get('unemployment', 0) > 6.0 and forecast.get('earnings_12m', 0) < 0:
                return (
                    'recession',
                    0.95,
                    f'Country-specific: Negative GDP growth ({gdp_qq}%) + high unemployment + weak earnings',
                    {
                        **base_signals,
                        'gdp_qq': gdp_qq,
                        'gdp_forecast': gdp_fcst,
                        'unemployment': real.get('unemployment'),
                        'earnings_growth': forecast.get('earnings_12m'),
                        'source': 'country_specific'
                    }
                )

        # Moderate recession risk from country data
        if (gdp_fcst is not None and gdp_fcst < 0.5 and
            real.get('unemployment', 0) > 5.5):
            return (
                'recession',
                0.85,
                f'Country-specific: Weak growth forecast ({gdp_fcst}%) + elevated unemployment',
                {
                    **base_signals,
                    'gdp_forecast': gdp_fcst,
                    'unemployment': real.get('unemployment'),
                    'source': 'country_specific'
                }
            )

        # ONLY use FRED for severe recession signals (all 3 indicators extreme)
        if ism is not None and curve is not None and hy is not None:
            if ism < 45 and curve < 0 and hy > 500:
                return (
                    'recession',
                    0.90,
                    'Global indicators: All leading indicators (ISM, curve, spreads) signal severe recession',
                    {
                        **base_signals,
                        'ism_pmi': ism,
                        'yield_curve': curve,
                        'hy_spread': hy,
                        'source': 'global_markets',
                        'consensus': 'bearish'
                    }
                )

        # =====================================================================
        # EARLY CYCLE (guide.md page 73)
        # =====================================================================
        if gdp_mom is not None and gdp_mom > 1.0:
            if (forecast.get('gdp_growth_6m', 0) > 3.0 and
                forecast.get('earnings_12m', 0) > 10.0 and
                real.get('unemployment', 10) > 5.0):

                return (
                    'early_cycle',
                    0.90,
                    'Strong growth acceleration + robust earnings + policy support',
                    {
                        **base_signals,
                        'earnings_growth': f"{forecast.get('earnings_12m')}%",
                        'ism_pmi': ism,
                        'phase': 'recovery'
                    }
                )

        # Early cycle from ISM
        if ism is not None and ism > 55 and curve is not None and curve > 100:
            return (
                'early_cycle',
                0.85,
                'Strong ISM + steep curve signal early expansion',
                {
                    **base_signals,
                    'ism_pmi': ism,
                    'yield_curve': curve,
                    'cycle_momentum': 'accelerating'
                }
            )

        # =====================================================================
        # LATE CYCLE (guide.md page 77) - COUNTRY-SPECIFIC FIRST
        # =====================================================================
        # Country-specific late cycle: Decelerating growth + high inflation
        if gdp_mom is not None and gdp_mom < -0.5:
            if ((inf_mom is not None and inf_mom > 0.5) or
                forecast.get('inflation_6m', 0) > 3.5):

                return (
                    'late_cycle',
                    0.85,
                    f'Country-specific: Decelerating growth ({gdp_mom:.1f}%) + persistent inflation',
                    {
                        **base_signals,
                        'policy_stance': 'hawkish',
                        'phase': 'peak_slowdown',
                        'source': 'country_specific'
                    }
                )

        # Country-specific late cycle: Negative GDP growth (but not severe enough for recession)
        if gdp_qq is not None and -0.5 < gdp_qq < 0:
            return (
                'late_cycle',
                0.80,
                f'Country-specific: Marginal GDP contraction ({gdp_qq:.1f}%) signals late cycle',
                {
                    **base_signals,
                    'gdp_qq': gdp_qq,
                    'phase': 'slowdown',
                    'source': 'country_specific'
                }
            )

        # ONLY use FRED for late cycle if BOTH ISM is weak AND curve is inverted/flat
        # (require stricter conditions than before to avoid false positives)
        if ism is not None and curve is not None:
            if ism < 48 and curve < 25:  # Stricter: ISM < 48 (not 50) AND curve < 25 bps (not 0)
                return (
                    'late_cycle',
                    0.75,
                    f'Global indicators: Weak ISM ({ism:.1f}) + flat curve ({curve:.0f} bps) suggest late cycle',
                    {
                        **base_signals,
                        'ism_pmi': ism,
                        'yield_curve': curve,
                        'warning': 'recession_risk_elevated',
                        'source': 'global_markets'
                    }
                )

        # =====================================================================
        # MID CYCLE (guide.md page 76)
        # =====================================================================
        if (forecast.get('gdp_growth_6m', 0) >= 2.0 and
            forecast.get('gdp_growth_6m', 0) <= 3.5 and
            forecast.get('inflation_6m', 0) >= 1.5 and
            forecast.get('inflation_6m', 0) <= 3.0):

            if gdp_mom is not None and abs(gdp_mom) < 0.5:
                return (
                    'mid_cycle',
                    0.80,
                    'Goldilocks: stable growth + contained inflation + steady policy',
                    {
                        **base_signals,
                        'gdp_forecast': forecast.get('gdp_growth_6m'),
                        'inflation_forecast': forecast.get('inflation_6m'),
                        'stability': 'high',
                        'phase': 'sustained_expansion'
                    }
                )

        # =====================================================================
        # UNCERTAIN / TRANSITION
        # =====================================================================
        return (
            'uncertain',
            0.60,
            'Mixed signals - monitor closely for clarity',
            {
                **base_signals,
                'ism_pmi': ism,
                'recommendation': 'Maintain flexibility, bias toward defensives'
            }
        )

    def _get_sector_tilts(self, regime: str) -> Dict[str, float]:
        """
        Sector allocation recommendations by regime.
        From guide.md pages 73-79.

        Returns dict with sector: weight_adjustment (e.g., +0.05 = +5%)
        """
        tilts = {
            'early_cycle': {
                'Consumer Discretionary': +0.05,
                'Industrials': +0.04,
                'Technology': +0.04,
                'Financials': +0.03,
                'Real Estate': +0.03,
                'Materials': +0.02,
                'Utilities': -0.04,
                'Consumer Staples': -0.03,
                'Healthcare': -0.02
            },
            'mid_cycle': {
                # Minimal tilts - focus on stock selection (guide.md page 76)
                'Technology': +0.01,
                'Communication Services': +0.01,
                'Materials': -0.01,
                'Utilities': -0.01
            },
            'late_cycle': {
                'Energy': +0.05,
                'Materials': +0.03,
                'Consumer Staples': +0.03,
                'Healthcare': +0.02,
                'Utilities': +0.02,
                'Technology': -0.03,
                'Consumer Discretionary': -0.03,
                'Financials': -0.02
            },
            'recession': {
                'Consumer Staples': +0.06,
                'Healthcare': +0.05,
                'Utilities': +0.04,
                'Technology': -0.06,
                'Consumer Discretionary': -0.05,
                'Financials': -0.05,
                'Industrials': -0.04,
                'Real Estate': -0.03
            },
            'uncertain': {}
        }

        return tilts.get(regime, {})

    def _get_factor_timing(self, regime: str) -> Dict[str, str]:
        """
        Factor exposure recommendations by regime.
        Returns: 'overweight', 'neutral', 'underweight'
        """
        timing = {
            'early_cycle': {
                'Value': 'overweight',
                'Momentum': 'overweight',
                'Quality': 'neutral',
                'Growth': 'overweight',
                'Low Volatility': 'underweight'
            },
            'mid_cycle': {
                'Value': 'neutral',
                'Momentum': 'neutral',
                'Quality': 'overweight',
                'Growth': 'neutral',
                'Low Volatility': 'neutral'
            },
            'late_cycle': {
                'Value': 'overweight',
                'Momentum': 'underweight',
                'Quality': 'overweight',
                'Growth': 'underweight',
                'Low Volatility': 'overweight'
            },
            'recession': {
                'Value': 'neutral',
                'Momentum': 'underweight',
                'Quality': 'overweight',
                'Growth': 'underweight',
                'Low Volatility': 'overweight'
            },
            'uncertain': {
                'Value': 'neutral',
                'Momentum': 'neutral',
                'Quality': 'overweight',
                'Growth': 'neutral',
                'Low Volatility': 'overweight'
            }
        }

        return timing.get(regime, {})

    def _calculate_recession_risk(self, real, forecast, gdp_mom, ism, curve, hy) -> Dict:
        """
        Calculate probability of recession in 6 and 12 months.

        Per guide.md: Uses ISM PMI, yield curve, credit spreads as leading indicators.
        Il Sole 24 Ore data (GDP, unemployment, industrial production) is PRIMARY signal.
        When country shows actual contraction, heavily weight country-specific data.
        When country is healthy, global indicators matter more.
        """
        risk_6m = 0.05  # Base 5%
        risk_12m = 0.10  # Base 10%

        # =====================================================================
        # STEP 1: Determine if country is in actual trouble (PRIMARY from Il Sole)
        # =====================================================================
        gdp_qq = real.get('gdp_growth_qq')
        gdp_fcst = forecast.get('gdp_growth_6m')
        unemp = real.get('unemployment')
        ind_prod = real.get('industrial_production')

        # Assess CURRENT economic health from Il Sole data
        # Current contraction is PRIMARY signal - when GDP is negative AND production is falling,
        # this IS the recession regardless of optimistic forecasts
        country_in_trouble = False
        trouble_severity = 0.0  # 0 to 1 scale

        # GDP contraction is strongest signal - trust CURRENT data over forecasts
        if gdp_qq is not None and gdp_qq < 0:
            country_in_trouble = True
            if gdp_qq < -0.5:
                trouble_severity = 0.9  # Severe (increased from 0.8)
            elif gdp_qq < -0.3:  # Moderate threshold tightened
                trouble_severity = 0.7  # Moderate (increased from 0.5)
            elif gdp_qq < -0.1:  # Mild threshold tightened
                trouble_severity = 0.5  # Mild (increased from 0.3)
            else:
                trouble_severity = 0.3  # Very mild

        # Industrial production decline amplifies significantly
        # When production is collapsing while GDP contracts, recession is HERE
        if ind_prod is not None and ind_prod < -2.0:
            country_in_trouble = True
            production_severity = min(0.5, abs(ind_prod) / 4.0)  # Up to 0.5 additional
            trouble_severity += production_severity  # Amplify significantly

        # High unemployment adds risk
        if unemp is not None and unemp > 6.5:
            trouble_severity += 0.2 * min(1.0, (unemp - 6.5) / 2.0)  # Increased weight

        # Cap trouble severity at 1.0
        trouble_severity = min(1.0, trouble_severity)

        # =====================================================================
        # STEP 2: Apply country-specific risk (heavily weighted when in trouble)
        # =====================================================================
        if country_in_trouble:
            # Country showing actual contraction - this IS the recession signal!
            # Il Sole CURRENT data is PRIMARY - add substantial risk
            # When GDP contracts AND production collapses, forecasts are often too optimistic
            country_risk = 0.65 * trouble_severity  # Up to 65% risk from country data (increased from 50%)
            risk_6m += country_risk * 0.7  # 70% of full risk for 6M
            risk_12m += country_risk  # Full risk for 12M

        else:
            # Country healthy - smaller country-specific contribution
            # Weak forecast still adds some risk
            if gdp_fcst is not None and gdp_fcst < 1.0:
                risk_6m += 0.05
                risk_12m += 0.08

        # =====================================================================
        # STEP 3: Add global market indicators (SECONDARY - smaller weight)
        # =====================================================================
        # Global indicators matter more when country is healthy (they're leading indicators)
        # Matter less when country already in trouble (country data shows current state)
        global_weight = 0.8 if not country_in_trouble else 0.15  # Increased healthy weight, reduced troubled weight

        # ISM PMI (guide.md page 83: <48 signals defensive rotation)
        if ism is not None:
            ism_risk = 0
            if ism < 43:
                ism_risk = 0.30  # Severe
            elif ism < 46:
                ism_risk = 0.20  # Contraction
            elif ism < 49:
                ism_risk = 0.12  # Weak (increased for healthy countries)
            elif ism < 51:
                ism_risk = 0.06  # Near neutral (increased)

            risk_6m += ism_risk * global_weight * 0.8
            risk_12m += ism_risk * global_weight

        # Yield Curve (guide.md page 85: inversion = 6-18 month warning)
        if curve is not None:
            curve_risk = 0
            if curve < -25:
                curve_risk = 0.28  # Deeply inverted (increased)
            elif curve < 0:
                curve_risk = 0.20  # Inverted (increased)
            elif curve < 25:
                curve_risk = 0.10  # Very flat (increased)
            elif curve < 75:
                curve_risk = 0.04  # Flat (increased)

            risk_6m += curve_risk * global_weight * 0.7
            risk_12m += curve_risk * global_weight

        # Credit Spreads (guide.md page 87: >500 bps triggers risk-off)
        if hy is not None:
            spread_risk = 0
            if hy > 700:
                spread_risk = 0.22  # Severe stress (increased)
            elif hy > 500:
                spread_risk = 0.14  # Elevated stress (increased)
            elif hy > 400:
                spread_risk = 0.06  # Moderate stress (increased)

            risk_6m += spread_risk * global_weight
            risk_12m += spread_risk * global_weight

        # Cap at 95%
        return {
            '6month': min(0.95, risk_6m),
            '12month': min(0.95, risk_12m)
        }

    def _calculate_transition_probability(self, regime, gdp_mom, ism, curve) -> float:
        """Estimate probability of regime change in next 6 months."""
        if regime == 'uncertain':
            return 0.70  # High uncertainty = likely to resolve

        base_prob = 0.15  # Base 15% chance of transition

        # Increase if momentum diverging from current regime
        if regime == 'mid_cycle':
            if gdp_mom is not None and abs(gdp_mom) > 1.0:
                base_prob += 0.25
            if ism is not None and (ism < 48 or ism > 55):
                base_prob += 0.20

        elif regime == 'late_cycle':
            # High transition risk in late cycle
            base_prob = 0.45
            if curve is not None and curve < 0:
                base_prob += 0.20

        elif regime == 'early_cycle':
            # Low transition risk early in expansion
            base_prob = 0.10

        return min(0.95, base_prob)

    @staticmethod
    def _calculate_momentum(forecast_value, current_value) -> Optional[float]:
        """Calculate momentum as forecast - current."""
        if forecast_value is None or current_value is None:
            return None
        return forecast_value - current_value

    @staticmethod
    def _normalize_regime(regime: Optional[str]) -> str:
        """
        Normalize regime value to database format.

        Converts various regime formats to lowercase with underscores:
        - 'MID CYCLE' → 'mid_cycle'
        - 'EARLY CYCLE' → 'early_cycle'
        - 'LATE CYCLE' → 'late_cycle'
        - 'RECESSION' → 'recession'
        - 'UNCERTAIN' → 'uncertain'

        Parameters
        ----------
        regime : str or None
            Raw regime value from LLM or other source

        Returns
        -------
        str
            Normalized regime value matching database enum
        """
        if not regime:
            return 'uncertain'

        # Convert to lowercase and replace spaces with underscores
        normalized = regime.lower().replace(' ', '_').strip()

        # Map common variations to standard values
        regime_map = {
            'mid_cycle': 'mid_cycle',
            'midcycle': 'mid_cycle',
            'mid': 'mid_cycle',
            'early_cycle': 'early_cycle',
            'earlycycle': 'early_cycle',
            'early': 'early_cycle',
            'late_cycle': 'late_cycle',
            'latecycle': 'late_cycle',
            'late': 'late_cycle',
            'recession': 'recession',
            'contraction': 'recession',
            'uncertain': 'uncertain',
            'unknown': 'uncertain',
            'mixed': 'uncertain'
        }

        return regime_map.get(normalized, 'uncertain')

    def _filter_macro_news(self, news_data: List[Dict]) -> List[Dict]:
        """
        Filter news to macro-relevant articles only.

        Parameters
        ----------
        news_data : list
            Raw news articles

        Returns
        -------
        list
            Filtered news articles with macro keywords
        """
        if not news_data:
            return []

        filtered = []
        for article in news_data:
            title = article.get('title', '').lower()

            # Check if any macro keyword is in the title
            if any(keyword in title for keyword in self.MACRO_KEYWORDS):
                filtered.append(article)

        return filtered


    def classify_pure_llm(
        self,
        ilsole_data: Dict,
        fred_data: Dict,
        news_data: List[Dict],
        country: str = 'USA',
        country_trading_economics_data: Optional[Dict] = None
    ) -> BusinessCycleClassification:
        """
        Pure LLM-based classification without quantitative rules.

        This method passes all raw data directly to the LLM, which performs
        the entire classification using the institutional framework as context.

        Parameters
        ----------
        ilsole_data : dict
            Data from Il Sole 24 Ore (both real and forecast)
        fred_data : dict
            Data from FRED (VIX, HY Spread) + Trading Economics (PMI, Yield Curve)
        news_data : list
            List of news articles, each with keys: title, publisher, date, link, full_content
        country : str
            Country code for analysis
        country_trading_economics_data : dict, optional
            Complete Trading Economics data for this country including:
            - indicators: All 20 economic indicators with current/previous values
            - bond_yields: Government bond yields (2Y, 5Y, 10Y, 30Y) with changes
            - industrial_production: Current production levels
            - capacity_utilization: Capacity usage levels

        Returns
        -------
        BusinessCycleClassification
            Complete business cycle classification from LLM including:
            - regime: Business cycle phase (early_cycle, mid_cycle, late_cycle, recession, uncertain)
            - confidence: Classification confidence (0.0-1.0)
            - rationale: 2-3 sentence explanation
            - ism_signal, yield_curve_signal, credit_spread_signal: Classified market indicators
            - recession_risk_6m, recession_risk_12m: Recession probabilities
            - sector_tilts: Portfolio positioning recommendations
            - recommended_overweights, recommended_underweights: Sector recommendations
            - factor_exposure: Factor timing (growth_momentum, quality_defensive, balanced)
            - primary_risks: Key risks to monitor
            - conflicting_signals: Any contradictory indicators
        """
        # Import BAML client (lazy import)
        if self._baml_client is None:
            try:
                import sys
                from pathlib import Path

                # Add parent directories to path to find baml_client
                script_dir = Path(__file__).parent
                project_root = script_dir.parent.parent  # Go up to api_optimizer/

                if str(project_root) not in sys.path:
                    sys.path.insert(0, str(project_root))

                from baml_client import b
                self._baml_client = b
            except ImportError as e:
                raise ImportError(f"BAML client not available: {e}. Make sure baml_client/ exists in project root.")

        b = self._baml_client

        # Import BAML types at runtime (after path is set up)
        from baml_client.types import (
            NewsArticle,
            EconomicIndicators,
            MarketIndicators,
        )

        # Filter news to macro-relevant articles
        filtered_news = self._filter_macro_news(news_data)

        if not filtered_news or len(filtered_news) == 0:
            print(f"  [Pure LLM Classification] Warning: No relevant macro news found for {country}")
            print(f"  [Pure LLM Classification] Proceeding with classification using only quantitative data")
        else:
            # Limit to most recent 25 articles
            filtered_news = filtered_news[:25]
            print(f"  [Pure LLM Classification] Analyzing {len(filtered_news)} macro-relevant articles")

        # Prepare data structures for BAML
        real = ilsole_data.get('real', {})
        forecast = ilsole_data.get('forecast', {})

        # Extract Trading Economics indicators (if available)
        te_indicators = {}
        te_bond_yields = {}
        te_industrial_prod = None
        te_capacity_util = None

        if country_trading_economics_data and country_trading_economics_data.get('status') == 'success':
            te_indicators = country_trading_economics_data.get('indicators', {})
            te_bond_yields = country_trading_economics_data.get('bond_yields', {})
            te_industrial_prod = country_trading_economics_data.get('industrial_production', {})
            te_capacity_util = country_trading_economics_data.get('capacity_utilization', {})

        # Convert news to BAML NewsArticle objects (Pydantic)
        news_articles: List[NewsArticle] = [
            NewsArticle(
                title=article.get('title', ''),
                publisher=article.get('publisher', ''),
                date=article.get('date', ''),
                link=article.get('link'),
                full_content=article.get('full_content')
            )
            for article in filtered_news
        ]

        # Prepare economic indicators (Pydantic object) - merge Il Sole + Trading Economics
        economic_indicators = EconomicIndicators(
            # GDP (prioritize Il Sole, fallback to Trading Economics)
            gdp_growth_qq=real.get('gdp_growth_qq') or te_indicators.get('gdp_growth_rate', {}).get('value'),
            gdp_growth_yy=real.get('gdp_yy') or te_indicators.get('gdp_growth_yoy', {}).get('value'),
            gdp_forecast_6m=forecast.get('gdp_growth_6m'),

            # Labor Market
            unemployment=real.get('unemployment') or te_indicators.get('unemployment_rate', {}).get('value'),

            # Production & Capacity
            industrial_production=real.get('industrial_production') or (te_industrial_prod.get('value') if te_industrial_prod else None),
            capacity_utilization=te_capacity_util.get('value') if te_capacity_util else None,

            # Inflation & Prices
            inflation=real.get('inflation') or real.get('consumer_prices') or te_indicators.get('inflation_rate', {}).get('value'),
            inflation_mom=te_indicators.get('inflation_rate_mom', {}).get('value'),
            core_inflation=te_indicators.get('core_inflation', {}).get('value'),
            inflation_forecast=forecast.get('inflation_6m'),

            # Consumption & Retail
            retail_sales_mom=te_indicators.get('retail_sales_mom', {}).get('value'),

            # Corporate & Earnings
            earnings_growth_forecast=forecast.get('earnings_12m'),

            # Confidence Indicators
            business_confidence=te_indicators.get('business_confidence', {}).get('value'),
            consumer_confidence=te_indicators.get('consumer_confidence', {}).get('value'),

            # External Balance & Trade
            trade_balance=te_indicators.get('trade_balance', {}).get('value'),
            current_account=te_indicators.get('current_account', {}).get('value'),
            current_account_gdp=te_indicators.get('current_account_gdp', {}).get('value'),

            # Fiscal Position
            government_debt_gdp=te_indicators.get('government_debt_gdp', {}).get('value'),
            budget_balance_gdp=te_indicators.get('budget_balance_gdp', {}).get('value')
        )

        # Prepare market indicators (Pydantic object) - merge FRED + Trading Economics
        market_indicators = MarketIndicators(
            # PMI Indicators
            ism_pmi=fred_data.get('ism_pmi'),  # US ISM from FRED
            manufacturing_pmi=te_indicators.get('manufacturing_pmi', {}).get('value'),  # Country-specific
            services_pmi=te_indicators.get('services_pmi', {}).get('value'),
            composite_pmi=te_indicators.get('composite_pmi', {}).get('value'),

            # Interest Rates & Policy
            interest_rate=te_indicators.get('interest_rate', {}).get('value'),

            # Yield Curve & Bond Markets
            yield_curve_2s10s=fred_data.get('yield_curve_2s10s'),  # From Trading Economics via run_regime_analysis
            bond_yield_2y=te_bond_yields.get('2Y', {}).get('yield'),
            bond_yield_5y=te_bond_yields.get('5Y', {}).get('yield'),
            bond_yield_10y=te_bond_yields.get('10Y', {}).get('yield'),
            bond_yield_30y=te_bond_yields.get('30Y', {}).get('yield'),

            # Credit Markets & Risk
            hy_spread=fred_data.get('hy_spread'),

            # Volatility & Sentiment
            vix=fred_data.get('vix'),
            vix_signal=fred_data.get('vix_signal')
        )

        # Display info about Trading Economics data
        if country_trading_economics_data and country_trading_economics_data.get('status') == 'success':
            num_indicators = country_trading_economics_data.get('num_indicators', 0)
            num_bonds = country_trading_economics_data.get('num_bond_yields', 0)
            has_industrial_prod = country_trading_economics_data.get('industrial_production') is not None
            has_capacity_util = country_trading_economics_data.get('capacity_utilization') is not None
            print(f"  [Trading Economics] {num_indicators} indicators, {num_bonds} bond yields, "
                  f"{'✓' if has_industrial_prod else '✗'} industrial production, "
                  f"{'✓' if has_capacity_util else '✗'} capacity utilization")

        print(f"  [Pure LLM Classification] Running pure LLM classification for {country}...")

        try:
            # STEP 1: Summarize news articles (preprocessing to reduce context)
            print(f"  [Pure LLM Classification] Step 1/3: Summarizing {len(news_articles)} news articles...")
            news_signals = b.SummarizeNewsArticles(
                news=news_articles,
                country=country
            )
            print(f"  [Pure LLM Classification] News summarization complete")
            print(f"     News regime narrative: {news_signals.regime_narrative}")
            print(f"     Dominant themes: {', '.join(news_signals.dominant_themes[:3]) if news_signals.dominant_themes else 'none'}")

            # STEP 2: Classify business cycle with comprehensive structured data
            print(f"  [Pure LLM Classification] Step 2/3: Classifying business cycle regime with comprehensive data...")

            # Get today's date for temporal context
            today_str = datetime.now().strftime("%Y-%m-%d")

            llm_result = b.ClassifyBusinessCycleWithLLM(
                today=today_str,
                economic_indicators=economic_indicators,
                market_indicators=market_indicators,
                news_signals=news_signals,
                country=country
            )

            print(f"  [Pure LLM Classification] Classification complete")
            print(f"     Regime: {llm_result.regime}")
            print(f"     Confidence: {llm_result.confidence:.2f}")
            print(f"     Factor Exposure: {llm_result.factor_exposure}")

            return llm_result

        except Exception as e:
            print(f"  [Pure LLM Classification] Error during LLM classification: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Pure LLM classification failed for {country}: {e}")
        
if __name__ == "__main__":
    # Test classifier
    print("\n" + "="*80)
    print("BUSINESS CYCLE CLASSIFIER - TEST MODE")
    print("="*80)
    print("\nTo test the classifier, use run_regime_analysis.py:")
    print("  python run_regime_analysis.py --country USA")
    print("\nThis script requires live data from:")
    print("  - Il Sole 24 Ore scraper")
    print("  - FRED API")
    print("="*80)
