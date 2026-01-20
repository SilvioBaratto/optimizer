#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

from optimizer.src.macro_regime.ilsole_scraper import IlSoleScraper, PORTFOLIO_COUNTRIES
from optimizer.src.macro_regime.fred_data import FREDDataFetcher
from optimizer.src.macro_regime.cycle_classifier import BusinessCycleClassifier
from optimizer.src.macro_regime.regime_tracker import RegimeTracker
from optimizer.src.macro_regime.news_fetcher import fetch_news_for_country
from optimizer.src.macro_regime.tradingeconomics_scraper import TradingEconomicsIndicatorsScraper


def extract_pmi_and_yield_curve(country_trading_economics_data: dict) -> dict:
    """
    Extract Manufacturing PMI and calculate yield curve from Trading Economics data.
    """

    result = {"ism_pmi": None, "yield_curve_2s10s": None}

    # Extract Manufacturing PMI
    indicators = country_trading_economics_data.get("indicators", {})
    if "manufacturing_pmi" in indicators:
        result["ism_pmi"] = indicators["manufacturing_pmi"].get("value")

    # Calculate yield curve (10Y - 2Y) in basis points
    bond_yields = country_trading_economics_data.get("bond_yields", {})
    if "10Y" in bond_yields and "2Y" in bond_yields:
        yield_10y = bond_yields["10Y"].get("yield")
        yield_2y = bond_yields["2Y"].get("yield")

        if yield_10y is not None and yield_2y is not None:
            result["yield_curve_2s10s"] = (yield_10y - yield_2y) * 100

    return result


def run_regime_analysis_with_fred(
    country: str,
    fred_data: dict,
    ilsole_data: dict,
    country_trading_economics_data: dict,
    fetch_full_content: bool = True,
) -> dict:
    """
    Run complete regime analysis workflow with pre-fetched data.
    """

    # Merge Trading Economics Data into FRED Data
    te_indicators = extract_pmi_and_yield_curve(country_trading_economics_data)
    fred_data_merged = {**fred_data, **te_indicators}

    # Fetch Macroeconomic News
    news_data = fetch_news_for_country(
        country, max_articles=50, fetch_full_content=fetch_full_content
    )

    if not news_data or len(news_data) == 0:
        raise RuntimeError(f"No news articles available for {country}")

    # Classify Business Cycle Regime
    classifier = BusinessCycleClassifier()

    assessment = classifier.classify_pure_llm(
        ilsole_data,
        fred_data_merged,
        news_data,
        country,
        country_trading_economics_data=country_trading_economics_data,
    )

    # Track Regime History
    tracker = RegimeTracker(country=country, use_database=True)
    tracker.add_assessment(assessment, ilsole_data, fred_data_merged)

    transition_info = tracker.detect_transition()
    summary = tracker.get_summary()

    return {
        "assessment": assessment,
        "transition": transition_info,
        "tracker_summary": summary,
        "ilsole_data": ilsole_data,
        "fred_data": fred_data_merged,
        "news_data": news_data,
    }


def main():
    """
    Main entry point for portfolio countries macro regime analysis.
    """
    # Check FRED API key
    fred_key_from_env = os.getenv("FRED_API_KEY")
    if not fred_key_from_env:
        return 1

    # Set default configuration
    countries = PORTFOLIO_COUNTRIES
    save_report = True
    fetch_full_content = True

    try:
        all_results = {}

        # Fetch FRED Data (VIX + HY Spread)
        fetcher = FREDDataFetcher()
        fred_data = fetcher.get_all_indicators()

        # Fetch Il Sole Data for all countries
        scraper = IlSoleScraper()

        if len(countries) > 1:
            ilsole_raw_data = scraper.get_multiple_countries(countries)

            ilsole_all_data = {}
            for country, data in ilsole_raw_data.items():
                if data.get("status") == "success":
                    ilsole_all_data[country] = {
                        "country": data.get("country"),
                        "real": data.get("real_indicators"),
                        "forecast": data.get("forecasts"),
                        "status": data.get("status"),
                        "timestamp": data.get("timestamp"),
                    }
                else:
                    ilsole_all_data[country] = data
        else:
            single_country_data = scraper.get_all_data(countries[0])
            ilsole_all_data = {countries[0]: single_country_data}

        # Fetch Trading Economics Data
        te_scraper = TradingEconomicsIndicatorsScraper()
        trading_economics_data = te_scraper.get_all_portfolio_countries()

        # Initialize Database Run
        saver = None
        run_id = None

        if save_report:
            try:
                from database_saver import MacroRegimeDatabaseSaver

                saver = MacroRegimeDatabaseSaver()

                run_id = saver.initialize_analysis_run(
                    fred_data=fred_data,
                    expected_countries=countries,
                    notes=f"Portfolio countries macro regime analysis (pure LLM) - incremental save",
                )

                try:
                    saver.save_trading_economics_data(trading_economics_data)
                except Exception:
                    pass

            except Exception:
                saver = None

        # Analyze Each Country
        for country in countries:
            country_data = ilsole_all_data.get(country)

            if not country_data or country_data.get("status") == "error":
                continue

            try:
                country_te_data = trading_economics_data.get(country, {})

                results = run_regime_analysis_with_fred(
                    country=country,
                    fred_data=fred_data,
                    ilsole_data=country_data,
                    country_trading_economics_data=country_te_data,
                    fetch_full_content=fetch_full_content,
                )
                all_results[country] = results

                if saver is not None:
                    try:
                        saver.save_single_country(country, results, run_id)
                    except Exception:
                        pass

            except RuntimeError:
                continue

        return 0

    except Exception:
        return 1


if __name__ == "__main__":
    sys.exit(main())
