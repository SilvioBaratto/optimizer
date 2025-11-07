#!/usr/bin/env python3
"""
Business Cycle Regime Analysis - Main Script
=============================================
Comprehensive macro regime detection and portfolio action generation.

Usage:
    python run_regime_analysis.py

This script automatically:
1. Analyzes all PORTFOLIO_COUNTRIES (USA, Germany, France, UK, Japan)
2. Fetches Il Sole 24 Ore indicators (real + forecast) in batch mode
3. Fetches FRED real-time data (ISM, yield curve, spreads) once for all countries
4. Initializes database run and saves market indicators
5. Classifies business cycle regime for each country
6. Fetches and analyzes macroeconomic news (with full article content)
7. Tracks regime transitions from database history
8. Saves each country's analysis immediately to database (incremental save)

Configuration:
- Portfolio countries analyzed: USA, Germany, France, UK, Japan
- China and India excluded (not available in Trading212)
- News-enhanced analysis with full article content
- Automatic report saving to database
- Batch data fetching for efficiency
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Look for .env in project root (4 levels up from this script)
    # Script is at: optimizer/api_optimizer/universe/macro_regime/run_regime_analysis.py
    # .env is at:   optimizer/.env
    env_path = Path(__file__).parent.parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[DEBUG] Loaded .env from: {env_path}")
    else:
        print(f"[DEBUG] .env file not found at: {env_path}")
        print(f"[DEBUG] Checking if FRED_API_KEY is in system environment...")
except ImportError:
    print("[DEBUG] python-dotenv not installed, using system environment variables")
    pass

from src.macro_regime.ilsole_scraper import IlSoleScraper, PORTFOLIO_COUNTRIES
from src.macro_regime.fred_data import FREDDataFetcher
from src.macro_regime.cycle_classifier import BusinessCycleClassifier
from src.macro_regime.regime_tracker import RegimeTracker
from src.macro_regime.news_fetcher import fetch_news_for_country
from src.macro_regime.tradingeconomics_scraper import TradingEconomicsIndicatorsScraper


def extract_pmi_and_yield_curve(country_trading_economics_data: dict) -> dict:
    """
    Extract Manufacturing PMI and calculate yield curve from Trading Economics data.

    Parameters
    ----------
    country_trading_economics_data : dict
        Trading Economics data for a specific country

    Returns
    -------
    dict
        {
            'ism_pmi': float or None,
            'yield_curve_2s10s': float or None (in basis points)
        }
    """

    result = {
        'ism_pmi': None,
        'yield_curve_2s10s': None
    }

    # Extract Manufacturing PMI
    indicators = country_trading_economics_data.get('indicators', {})
    if 'manufacturing_pmi' in indicators:
        result['ism_pmi'] = indicators['manufacturing_pmi'].get('value')

    # Calculate yield curve (10Y - 2Y) in basis points
    bond_yields = country_trading_economics_data.get('bond_yields', {})
    if '10Y' in bond_yields and '2Y' in bond_yields:
        yield_10y = bond_yields['10Y'].get('yield')
        yield_2y = bond_yields['2Y'].get('yield')

        if yield_10y is not None and yield_2y is not None:
            # Trading Economics yields are in %, convert spread to basis points
            result['yield_curve_2s10s'] = (yield_10y - yield_2y) * 100

    return result


def run_regime_analysis_with_fred(country: str,
                                    fred_data: dict,
                                    ilsole_data: dict,
                                    country_trading_economics_data: dict,
                                    save_report: bool = True,
                                    fetch_full_content: bool = True) -> dict:
    """
    Run complete regime analysis workflow with pre-fetched data.

    Pure LLM classification is used - the LLM performs the entire business cycle
    classification using all available data (economic indicators, market indicators,
    and macroeconomic news) with institutional framework rules embedded in the prompt.

    Parameters
    ----------
    country : str
        Country code (USA, Germany, etc.)
    fred_data : dict
        Pre-fetched FRED data (VIX + HY Spread)
    ilsole_data : dict
        Pre-fetched Il Sole 24 Ore data (country-specific indicators)
    country_trading_economics_data : dict
        Pre-fetched Trading Economics data for this specific country
        (contains indicators, bond_yields, industrial_production, capacity_utilization)
    save_report : bool
        Save detailed report to file
    fetch_full_content : bool
        Fetch full article content from URLs (default True, recommended)

    Returns
    -------
    dict
        Complete analysis results

    Raises
    ------
    RuntimeError
        If news data cannot be fetched for the country
    """

    print("\n" + "="*100)
    print("BUSINESS CYCLE REGIME ANALYSIS")
    print("="*100)
    print(f"Country: {country}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*100)

    # =========================================================================
    # STEP 0: Merge Trading Economics Data into FRED Data
    # =========================================================================
    # Extract PMI and yield curve from Trading Economics (country-specific) and merge into fred_data
    te_indicators = extract_pmi_and_yield_curve(country_trading_economics_data)
    fred_data_merged = {**fred_data, **te_indicators}

    print(f"\n[Data Sources] Combined market indicators for {country}:")
    print(f"  From FRED/Yahoo: VIX={fred_data.get('vix', 'N/A')}, HY Spread={fred_data.get('hy_spread', 'N/A')} bps")
    print(f"  From Trading Economics: PMI={te_indicators.get('ism_pmi', 'N/A')}, Yield Curve={te_indicators.get('yield_curve_2s10s', 'N/A'):.0f} bps" if te_indicators.get('yield_curve_2s10s') else f"  From Trading Economics: PMI={te_indicators.get('ism_pmi', 'N/A')}, Yield Curve=N/A")

    # =========================================================================
    # STEP 1: Verify and Display Il Sole Data
    # =========================================================================
    print("\n[1/5] Verifying Il Sole 24 Ore data (country-specific)")
    print(f"  ✅ Il Sole data provided (status: {ilsole_data.get('status', 'unknown')})")

    # Display all real indicators (current economic state)
    real_data = ilsole_data.get('real', {})
    if real_data:
        print(f"\n  📊 Real Indicators (Current State):")
        print(f"     GDP Growth (Q/Q):        {real_data.get('gdp_growth_qq', 'N/A')}%")
        print(f"     Industrial Production:   {real_data.get('industrial_production', 'N/A')}%")
        print(f"     Unemployment:            {real_data.get('unemployment', 'N/A')}%")
        print(f"     Consumer Prices (CPI):   {real_data.get('consumer_prices', 'N/A')}%")
        print(f"     Budget Deficit/GDP:      {real_data.get('deficit', 'N/A')}%")
        print(f"     Public Debt/GDP:         {real_data.get('debt', 'N/A')}%")
        print(f"     Short-term Rate:         {real_data.get('st_rate', 'N/A')}%")
        print(f"     Long-term Rate:          {real_data.get('lt_rate', 'N/A')}%")

    # Display all forecast indicators (forward-looking expectations)
    forecast_data = ilsole_data.get('forecast', {})
    if forecast_data:
        print(f"\n  🔮 Forecast Indicators (6-12M Expectations):")
        print(f"     Last Inflation:          {forecast_data.get('last_inflation', 'N/A')}%")
        print(f"     Inflation (6M):          {forecast_data.get('inflation_6m', 'N/A')}%")
        print(f"     Inflation (10Y Avg):     {forecast_data.get('inflation_10y_avg', 'N/A')}%")
        print(f"     GDP Growth (6M):         {forecast_data.get('gdp_growth_6m', 'N/A')}%")
        print(f"     Earnings Growth (12M):   {forecast_data.get('earnings_12m', 'N/A')}%")
        print(f"     EPS Expected (12M):      {forecast_data.get('eps_expected_12m', 'N/A')}%")
        print(f"     PEG Ratio:               {forecast_data.get('peg_ratio', 'N/A')}")
        print(f"     ST Rate Forecast:        {forecast_data.get('st_rate_forecast', 'N/A')}")
        print(f"     LT Rate Forecast:        {forecast_data.get('lt_rate_forecast', 'N/A')}%")
        print(f"     Reference Date:          {forecast_data.get('reference_date', 'N/A')}")

    # =========================================================================
    # STEP 2: Verify FRED Data
    # =========================================================================
    print("\n[2/5] Verifying global market indicators (FRED data)")
    print(f"  ✅ FRED data provided (quality: {fred_data['data_quality']})")

    # =========================================================================
    # STEP 3: Fetch Macroeconomic News (MANDATORY for pure LLM classification)
    # =========================================================================
    print(f"\n[3/5] Fetching macroeconomic news for {country}...")
    content_mode = "with full content" if fetch_full_content else "headlines only"
    print(f"     Mode: {content_mode}")

    try:
        news_data = fetch_news_for_country(country, max_articles=50, fetch_full_content=fetch_full_content)
        print(f"  ✅ Fetched {len(news_data)} recent news articles")

        # Show content stats if full content was fetched
        if fetch_full_content and news_data:
            articles_with_content = sum(1 for article in news_data if article.get('full_content'))
            print(f"     Full content retrieved for {articles_with_content}/{len(news_data)} articles")

        # Validate news data
        if not news_data or len(news_data) == 0:
            print(f"  ❌ ERROR: No news articles found for {country}")
            print(f"  ❌ Pure LLM classification requires macroeconomic news for comprehensive analysis")
            raise RuntimeError(f"No news articles available for {country}")

    except Exception as e:
        print(f"  ❌ Failed to fetch news: {e}")
        print(f"  ❌ Cannot proceed without news data - pure LLM requires news for classification")
        raise

    # =========================================================================
    # STEP 4: Classify Business Cycle Regime
    # =========================================================================
    print("\n[4/5] Classifying business cycle regime...")
    print(f"  [Data Summary] Using {len(real_data)} real indicators + {len(forecast_data)} forecast indicators")

    # List which fields are actively used vs. available for future enhancements
    print(f"  [Core Indicators Used]:")
    print(f"     • GDP Growth Q/Q, Industrial Production, Unemployment")
    print(f"     • Consumer Prices (Inflation), GDP Forecast 6M, Inflation Forecast 6M")
    print(f"     • Earnings Growth 12M")
    print(f"  [Available for Enhancement]:")
    print(f"     • Interest rates (ST/LT + forecasts), Deficit, Debt, PEG Ratio, EPS")

    classifier = BusinessCycleClassifier()

    # Pure LLM classification is mandatory (best practice for institutional analysis)
    if not news_data or len(news_data) == 0:
        print(f"\n  ❌ ERROR: No news articles available for {country}")
        print(f"  ❌ Pure LLM analysis requires macroeconomic news for accuracy")
        print(f"  ❌ Skipping {country} - cannot proceed without news data")
        raise RuntimeError(f"News data required for {country} analysis")

    print(f"\n  [Pure LLM Classification] Analyzing with {len(news_data)} articles")
    assessment = classifier.classify_pure_llm(
        ilsole_data,
        fred_data_merged,
        news_data,
        country,
        country_trading_economics_data=country_trading_economics_data
    )

    # =========================================================================
    # STEP 5: Track Regime History and Complete Analysis
    # =========================================================================
    print("\n[5/5] Checking regime history from database...")

    tracker = RegimeTracker(country=country, use_database=True)
    tracker.add_assessment(assessment, ilsole_data, fred_data_merged)  # No-op in database mode

    transition_info = tracker.detect_transition()
    if transition_info and transition_info.get('transition_detected'):
        print(f"  🚨 TRANSITION DETECTED!")
        print(f"     {transition_info['from_regime']} → {transition_info['to_regime']}")
        print(f"     Alert Level: {transition_info['alert_level']}")
    else:
        print(f"  ✅ Regime stable (no transition)")

    summary = tracker.get_summary()
    if summary.get('regime_duration_days'):
        print(f"     Current regime duration: {summary['regime_duration_days']} days")

    print(f"  ✅ Analysis complete")

    # =========================================================================
    # FINAL OUTPUT
    # =========================================================================
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)
    
    if transition_info and transition_info.get('transition_detected'):
        print(f"\n🚨 REGIME TRANSITION ALERT 🚨")
        print(f"   {transition_info['from_regime']} → {transition_info['to_regime']}")

    print("\n" + "="*100)

    return {
        'assessment': assessment,
        'transition': transition_info,
        'tracker_summary': summary,
        'ilsole_data': ilsole_data,
        'fred_data': fred_data_merged,  # Return merged data with PMI and yield curve from Trading Economics
        'news_data': news_data
    }


def main():
    """
    Main entry point for portfolio countries macro regime analysis.

    Default configuration:
    - Analyzes all PORTFOLIO_COUNTRIES (USA, Germany, France, UK, Japan)
    - China and India excluded (not available in Trading212)
    - Enables news-enhanced analysis with full article content
    - Saves reports to database
    - Fetches both FRED global indicators and Il Sole country-specific data
    """
    print("\n" + "="*100)
    print("PORTFOLIO MACRO REGIME ANALYSIS - PURE LLM CLASSIFICATION")
    print("="*100)
    print("\nConfiguration:")
    print(f"  • Countries: {len(PORTFOLIO_COUNTRIES)} ({', '.join(PORTFOLIO_COUNTRIES)})")
    print(f"  • Classification Method: PURE LLM (no quantitative rules)")
    print(f"  • News Analysis: ENABLED (with full article content)")
    print(f"  • Report Saving: ENABLED")
    print(f"  • FRED Global Indicators: ENABLED")
    print("="*100)

    # Check FRED API key
    fred_key_from_env = os.getenv('FRED_API_KEY')
    print(f"\n[Environment Check] FRED API key: {'Found ✅' if fred_key_from_env else 'NOT FOUND ❌'}")

    if not fred_key_from_env:
        print("\n[ERROR] FRED_API_KEY not found in environment variables!")
        print("Make sure your .env file contains: FRED_API_KEY=your_key_here")
        return 1

    # Set default configuration
    countries = PORTFOLIO_COUNTRIES
    save_report = True
    fetch_full_content = True

    print(f"\n[Configuration] Analyzing {len(countries)} portfolio countries")
    print(f"                Classification: PURE LLM (institutional framework in prompt)")
    print(f"                Full article content: {'ENABLED' if fetch_full_content else 'DISABLED'}")
    print(f"                Save to database: {'ENABLED' if save_report else 'DISABLED'}")

    # Run analysis for each country
    try:
        all_results = {}

        # =====================================================================
        # FETCH FRED DATA ONCE (VIX + HY Spread only - same for all countries)
        # =====================================================================
        print(f"\n{'='*100}")
        print("FETCHING MARKET INDICATORS (VIX + CREDIT SPREADS)")
        print("="*100)
        print("Fetching VIX and HY credit spreads from FRED + Yahoo Finance...")
        print("(Manufacturing PMI and Treasury Yields will be fetched from Trading Economics)")

        fetcher = FREDDataFetcher()
        fred_data = fetcher.get_all_indicators()
        print(f"✅ Market indicators fetched (quality: {fred_data['data_quality']})")
        print(f"   HY Spread: {fred_data['hy_spread']:.0f} bps" if fred_data.get('hy_spread') else "   HY Spread: N/A")
        print(f"   VIX: {fred_data['vix']:.2f}" if fred_data.get('vix') else "   VIX: N/A")
        print(f"   VIX Regime: {fred_data['vix_signal']}")

        # =====================================================================
        # FETCH IL SOLE DATA FOR ALL COUNTRIES (Batch fetch)
        # =====================================================================
        print(f"\n{'='*100}")
        print(f"FETCHING IL SOLE 24 ORE DATA FOR ALL PORTFOLIO COUNTRIES")
        print("="*100)
        print(f"Fetching economic indicators for {len(countries)} countries: {', '.join(countries)}")
        print("(Using batch fetch for efficiency)")

        scraper = IlSoleScraper()

        # Fetch all countries at once using get_multiple_countries
        if len(countries) > 1:
            print(f"\n📊 Batch fetching {len(countries)} countries...")
            ilsole_raw_data = scraper.get_multiple_countries(countries)
            print(f"✅ Batch fetch complete!")

            # Convert format: get_multiple_countries returns 'real_indicators'/'forecasts'
            # but cycle classifier expects 'real'/'forecast'
            ilsole_all_data = {}
            for country, data in ilsole_raw_data.items():
                if data.get('status') == 'success':
                    ilsole_all_data[country] = {
                        'country': data.get('country'),
                        'real': data.get('real_indicators'),
                        'forecast': data.get('forecasts'),
                        'status': data.get('status'),
                        'timestamp': data.get('timestamp')
                    }
                else:
                    ilsole_all_data[country] = data  # Keep error status as-is
        else:
            # Single country - use get_all_data (already returns correct format)
            print(f"\n📊 Fetching single country: {countries[0]}...")
            single_country_data = scraper.get_all_data(countries[0])
            ilsole_all_data = {countries[0]: single_country_data}
            print(f"✅ Fetch complete!")

        # =====================================================================
        # FETCH TRADING ECONOMICS DATA FOR ALL COUNTRIES
        # =====================================================================
        print(f"\n{'='*100}")
        print(f"FETCHING TRADING ECONOMICS DATA FOR ALL PORTFOLIO COUNTRIES")
        print("="*100)
        print(f"Fetching indicators, bond yields, industrial production, capacity utilization")
        print(f"(Using batch fetch for efficiency)")

        te_scraper = TradingEconomicsIndicatorsScraper()
        trading_economics_data = te_scraper.get_all_portfolio_countries()

        # Display summary
        successful_countries = [c for c, d in trading_economics_data.items() if d.get('status') == 'success']
        print(f"\n✅ Trading Economics data fetched for {len(successful_countries)}/{len(countries)} countries")
        for country in successful_countries:
            data = trading_economics_data[country]
            print(f"   {country}: {data.get('num_indicators', 0)} indicators, "
                  f"{data.get('num_bond_yields', 0)} bond yields")

        # =====================================================================
        # INITIALIZE DATABASE RUN (if saving enabled)
        # =====================================================================
        saver = None
        run_id = None

        if save_report:
            try:
                from database_saver import MacroRegimeDatabaseSaver

                saver = MacroRegimeDatabaseSaver()

                run_id = saver.initialize_analysis_run(
                    fred_data=fred_data,
                    expected_countries=countries,
                    notes=f"Portfolio countries macro regime analysis (pure LLM) - incremental save"
                )

                # Save Trading Economics data to database (checks for duplicates)
                try:
                    snapshot_ids = saver.save_trading_economics_data(trading_economics_data)
                    print(f"✅ Trading Economics data ready ({len(snapshot_ids)} country snapshots)")
                except Exception as e:
                    print(f"\n⚠️  Failed to save Trading Economics data: {e}")
                    print("   Will continue with regime analysis")

            except Exception as e:
                print(f"\n⚠️  Failed to initialize database run: {e}")
                print("   Will continue analysis without database saving")
                saver = None

        # =====================================================================
        # ANALYZE EACH COUNTRY (using pre-fetched Il Sole data + shared FRED data)
        # =====================================================================
        for country in countries:
            print(f"\n{'='*100}")
            print(f"ANALYZING: {country}")
            print(f"{'='*100}\n")

            # Get pre-fetched data for this country
            country_data = ilsole_all_data.get(country)

            if not country_data or country_data.get('status') == 'error':
                print(f"❌ ERROR: Failed to fetch data for {country}")
                print(f"   Skipping {country} analysis...")
                continue

            try:
                # Get country-specific Trading Economics data
                country_te_data = trading_economics_data.get(country, {})

                results = run_regime_analysis_with_fred(
                    country=country,
                    fred_data=fred_data,  # Pass FRED data (VIX + HY Spread)
                    ilsole_data=country_data,  # Pass pre-fetched Il Sole data
                    country_trading_economics_data=country_te_data,  # Pass country-specific Trading Economics data
                    save_report=save_report,
                    fetch_full_content=fetch_full_content
                )
                all_results[country] = results

                # Save country immediately to database
                if saver is not None:
                    try:
                        saver.save_single_country(country, results, run_id)
                    except Exception as e:
                        print(f"\n⚠️  Failed to save {country} to database: {e}")
                        print(f"   Continuing with next country...")

            except RuntimeError as e:
                print(f"  ⚠️  Skipping {country} due to missing news data")
                continue

        # Print consolidated summary
        print("\n" + "="*100)
        print("CONSOLIDATED SUMMARY - ALL COUNTRIES")
        print("="*100)

        for country, results in all_results.items():
            assessment = results['assessment']

            print(f"\n{country}:")
            print(f"  Regime: {assessment.regime.upper()} ({assessment.confidence*100:.0f}% confidence)")
            print(f"  Recession Risk (12M): {assessment.recession_risk_12m*100:.0f}%" if assessment.recession_risk_12m else "  Recession Risk (12M): N/A")

        print("\n" + "="*100)

        # =====================================================================
        # DATABASE SAVE SUMMARY
        # =====================================================================
        if save_report and run_id:
            print(f"\n✅ All countries saved to database successfully!")
            print(f"   Analysis Run ID: {run_id}")
            print(f"   Countries saved: {len(all_results)}/{len(countries)}")
        elif save_report:
            print(f"\n⚠️  Database saving was not fully successful")
            print(f"   Check logs above for details")
        else:
            print(f"\n📝 Database saving was disabled")

        print("\n" + "="*100)

        return 0

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
