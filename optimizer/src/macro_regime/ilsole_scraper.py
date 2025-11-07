#!/usr/bin/env python3
"""
Il Sole 24 Ore Economic Data Scraper
=====================================
Scrapes economic data from Il Sole 24 Ore:
- Real Indicators: https://mercati.ilsole24ore.com/dati-macroeconomici/paesi-a-confronto/indicatori-reali
- Forecasts: https://mercati.ilsole24ore.com/dati-macroeconomici/paesi-a-confronto/previsione-economica

Real Indicators (Current):
- GDP quarterly growth T/T (Quarter-over-Quarter)
- Industrial production growth
- Unemployment rate
- Consumer price inflation
- Fiscal deficit/surplus
- Public debt
- Short-term & long-term interest rates

Forecasts (Analyst Consensus):
- Expected inflation over 6 months
- Expected GDP growth over 6 months
- Earnings growth over 12 months
- Expected EPS over 12 months
- PEG ratio (Price/Earnings to Growth)
- Interest rate forecasts (12 months)

Usage:
    scraper = IlSoleScraper()

    # Single country (both real + forecast)
    data = scraper.get_country_data('USA')

    # Or individual tables
    real = scraper.get_real_indicators('USA')
    fcst = scraper.get_forecasts('USA')

    # Multiple countries
    g7_data = scraper.get_multiple_countries(G7_COUNTRIES)
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import Dict, Optional, List
import time
from datetime import datetime


class IlSoleScraper:
    """Scraper for Il Sole 24 Ore economic forecast data."""

    BASE_URL = "https://mercati.ilsole24ore.com/dati-macroeconomici/paesi-a-confronto"

    # Country name mapping for FORECAST table (previsione-economica)
    COUNTRY_MAP_FORECAST = {
        'Usa': 'USA',
        'Germania': 'Germany',
        'Francia': 'France',
        'Italia': 'Italy',
        'UK': 'UK',
        'Giappone': 'Japan',
        'Cina': 'China',
        'Canada': 'Canada',
        'Australia': 'Australia',
        'Spagna': 'Spain',
        'Brasile': 'Brazil',
        'India': 'India',
        'Russia': 'Russia',
        'Messico': 'Mexico',
        'Svizzera': 'Switzerland',
        'Olanda': 'Netherlands',
        'Svezia': 'Sweden',
        'Norvegia': 'Norway',
        'Danimarca': 'Denmark',
        'Austria': 'Austria',
        'Belgio': 'Belgium',
        'Finlandia': 'Finland',
        'Irlanda': 'Ireland',
        'Singapore': 'Singapore',
        'Corea': 'South Korea',
        'Hong K.': 'Hong Kong',
        'Taiwan': 'Taiwan',
        'Indonesia': 'Indonesia',
        'Malaysia': 'Malaysia',
        'Thailandia': 'Thailand',
        'Filippine': 'Philippines',
        'N. Zelanda': 'New Zealand',
        'Argentina': 'Argentina',
        'Cile': 'Chile',
        'Sudafrica': 'South Africa'
    }

    # Country name mapping for REAL INDICATORS table (indicatori-reali)
    COUNTRY_MAP_REAL = {
        'Stati Uniti': 'USA',
        'Germania': 'Germany',
        'Francia': 'France',
        'Italia': 'Italy',
        'G.Bretagna': 'UK',
        'Giappone': 'Japan',
        'Cina': 'China',
        'Canada': 'Canada',
        'Australia': 'Australia',
        'Spagna': 'Spain',
        'Brasile': 'Brazil',
        'India': 'India',
        'Russia': 'Russia',
        'Messico': 'Mexico',
        'Svizzera': 'Switzerland',
        'Olanda': 'Netherlands',
        'Svezia': 'Sweden',
        'Norvegia': 'Norway',
        'Danimarca': 'Denmark',
        'Austria': 'Austria',
        'Belgio': 'Belgium',
        'Finlandia': 'Finland',
        'Irlanda': 'Ireland',
        'Singapore': 'Singapore',
        'Corea': 'South Korea',
        'Hong Kong': 'Hong Kong',
        'Taiwan': 'Taiwan',
        'Indonesia': 'Indonesia',
        'Malaysia': 'Malaysia',
        'Thailandia': 'Thailand',
        'Filippine': 'Philippines',
        'N.Zelanda': 'New Zealand',
        'Argentina': 'Argentina',
        'Cile': 'Chile',
        'Sudafrica': 'South Africa'
    }

    def __init__(self, timeout: int = 10):
        """
        Initialize scraper.

        Parameters
        ----------
        timeout : int
            Request timeout in seconds
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })

    def _fetch_page(self, endpoint: str) -> Optional[BeautifulSoup]:
        """Fetch and parse HTML page."""
        try:
            url = f"{self.BASE_URL}/{endpoint}"
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            print(f"Error fetching {endpoint}: {e}")
            return None

    def get_real_indicators(self, country: str = 'USA') -> Optional[Dict]:
        """
        Scrape real economic indicators table.

        Parameters
        ----------
        country : str
            Country code (USA, Germany, China, etc.)

        Returns
        -------
        dict or None
            Dictionary with real indicators:
            - gdp_growth_qq: GDP quarterly growth T/T (Quarter-over-Quarter)
            - industrial_production: Industrial production growth
            - unemployment: Unemployment rate
            - consumer_prices: Consumer price inflation
            - deficit: Fiscal deficit/surplus
            - debt: Public debt
            - st_rate: Short-term interest rate (discount rate)
            - lt_rate: Long-term interest rate
        """
        soup = self._fetch_page('indicatori-reali')
        if soup is None:
            return None

        try:
            # Find the table
            table = soup.find('table', {'class': 'mainTable'})
            if not table:
                print("Could not find real indicators table")
                return None

            # Find country row (use REAL indicators mapping)
            country_italian = [k for k, v in self.COUNTRY_MAP_REAL.items() if v == country]
            if not country_italian:
                print(f"Country {country} not found in real indicators mapping")
                return None

            country_name = country_italian[0]

            # Find table body
            tbody = table.find('tbody')
            if not tbody:
                print("Could not find table body")
                return None

            # Search for country row (indicatori-reali uses 'name' attribute)
            country_row = None
            for row in tbody.find_all('tr'):
                paese_cell = row.find('td', {'name': 'Paese'})
                if paese_cell and country_name.lower() in paese_cell.text.lower():
                    country_row = row
                    break

            if not country_row:
                print(f"Could not find data for {country_name}")
                return None

            # Extract data from cells with name attributes
            cells = {}
            for cell in country_row.find_all('td'):
                cell_name = cell.get('name')
                if cell_name:
                    cells[cell_name] = cell.text.strip()

            # Map to output format - using exact HTML attribute names from table
            # Based on actual cell names: TassoSconto, TassoInteresse, Pil_TT, ProdIndustriale_AA, etc.
            data = {
                'gdp_growth_qq': self._safe_float(cells.get('Pil_TT')),  # T/T only (Quarter-over-Quarter)
                'industrial_production': self._safe_float(cells.get('ProdIndustriale_AA')),
                'unemployment': self._safe_float(cells.get('Disoccupazione_AA')),
                'consumer_prices': self._safe_float(cells.get('PrezziConsumo_AA')),
                'deficit': self._safe_float(cells.get('Deficit_AA')),
                'debt': self._safe_float(cells.get('Debito_AA')),
                'st_rate': self._safe_float(cells.get('TassoSconto')),
                'lt_rate': self._safe_float(cells.get('TassoInteresse')),
                'timestamp': datetime.now().isoformat()
            }

            return data

        except Exception as e:
            print(f"Error parsing real indicators: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_forecasts(self, country: str = 'USA') -> Optional[Dict]:
        """
        Scrape economic forecast table.

        Parameters
        ----------
        country : str
            Country code (USA, Germany, China, etc.)

        Returns
        -------
        dict or None
            Dictionary with economic forecasts:
            - last_inflation: Current annual inflation
            - inflation_6m: Expected inflation over 6 months
            - inflation_10y_avg: 10-year average inflation
            - gdp_growth_6m: Expected GDP growth over 6 months
            - earnings_12m: Earnings growth over 12 months
            - eps_expected_12m: Expected EPS over 12 months
            - peg_ratio: Price/Earnings to Growth ratio
            - st_rate_forecast: Short-term interest rate forecast (12M)
            - lt_rate_forecast: Long-term interest rate forecast (12M)
            - reference_date: Data reference date
        """
        soup = self._fetch_page('previsione-economica')
        if soup is None:
            return None

        try:
            # Find the table
            table = soup.find('table', {'class': 'mainTable'})
            if not table:
                print("Could not find forecast table")
                return None

            # Find country row (use FORECAST mapping)
            country_italian = [k for k, v in self.COUNTRY_MAP_FORECAST.items() if v == country]
            if not country_italian:
                print(f"Country {country} not found in forecast mapping")
                return None

            country_name = country_italian[0]

            # Find table body
            tbody = table.find('tbody')
            if not tbody:
                print("Could not find table body")
                return None

            # Search for country row (previsione-economica uses 'id' for Paese)
            country_row = None
            for row in tbody.find_all('tr'):
                paese_cell = row.find('td', {'id': 'Paese'})
                if paese_cell and country_name.lower() in paese_cell.text.lower():
                    country_row = row
                    break

            if not country_row:
                print(f"Could not find data for {country_name}")
                return None

            # Extract data from cells with id attributes
            cells = {}
            for cell in country_row.find_all('td'):
                cell_id = cell.get('id')
                if cell_id:
                    cells[cell_id] = cell.text.strip()

            # Map to output format
            data = {
                'last_inflation': self._safe_float(cells.get('UltimaInflazione')),
                'inflation_6m': self._safe_float(cells.get('ConsensoInflazione')),
                'inflation_10y_avg': self._safe_float(cells.get('MediaInfl10Anni')),
                'gdp_growth_6m': self._safe_float(cells.get('ConsensoPil')),
                'earnings_12m': self._safe_float(cells.get('ConsensoUtili')),
                'eps_expected_12m': self._safe_float(cells.get('ConsensoEps')),
                'peg_ratio': self._safe_float(cells.get('ConsensoPeg')),
                'st_rate_forecast': self._safe_float(cells.get('ConsensoTassiBreve')),
                'lt_rate_forecast': self._safe_float(cells.get('ConsensoTassiLungo')),
                'reference_date': cells.get('DataRiferimento'),
                'timestamp': datetime.now().isoformat()
            }

            return data

        except Exception as e:
            print(f"Error parsing forecasts: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_country_data(self, country: str = 'USA') -> Dict:
        """
        Get both real indicators and forecast data for a country.

        Parameters
        ----------
        country : str
            Country code (USA, Germany, etc.)

        Returns
        -------
        dict
            Economic data with status, including both real indicators and forecasts
        """
        real_data = self.get_real_indicators(country)
        forecast_data = self.get_forecasts(country)

        if real_data is None and forecast_data is None:
            return {'status': 'error', 'country': country}

        return {
            'country': country,
            'real_indicators': real_data,
            'forecasts': forecast_data,
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        }

    def get_all_data(self, country: str = 'USA') -> Dict:
        """
        Get complete data for cycle classifier (compatible with run_regime_analysis.py).

        This method returns data in the format expected by BusinessCycleClassifier:
        - 'real' key instead of 'real_indicators'
        - 'forecast' key instead of 'forecasts'

        Parameters
        ----------
        country : str
            Country code (USA, Germany, etc.)

        Returns
        -------
        dict
            Economic data formatted for cycle classifier
        """
        result = self.get_country_data(country)

        if result['status'] == 'error':
            return result

        # Rename keys to match classifier expectations
        return {
            'country': result['country'],
            'real': result.get('real_indicators'),
            'forecast': result.get('forecasts'),
            'status': result['status'],
            'timestamp': result['timestamp']
        }

    def get_multiple_countries(self, countries: List[str]) -> Dict:
        """
        Get forecast data for multiple countries.

        Parameters
        ----------
        countries : list
            List of country codes (e.g., ['USA', 'Germany', 'China'])

        Returns
        -------
        dict
            Dictionary with country codes as keys
        """
        results = {}

        for country in countries:
            data = self.get_country_data(country)
            results[country] = data
            time.sleep(0.5)  # Be polite to server

        return results

    @staticmethod
    def _safe_float(value) -> Optional[float]:
        """Safely convert value to float."""
        if value is None or pd.isna(value):
            return None
        try:
            if isinstance(value, str):
                # Strip whitespace and work with cleaned value
                value = value.strip()

                # Check if string contains only dashes/hyphens (various unicode variants)
                # Remove all spaces and check if what remains is only dash-like characters
                value_no_spaces = value.replace(' ', '')
                if value_no_spaces and all(c in '-–—−' for c in value_no_spaces):
                    # Only dashes (ASCII hyphen, en dash, em dash, minus sign)
                    return None

                # Check for other missing data markers
                if value in ['N/A', 'n/a', '']:
                    return None

                # Handle range format like "0-4,25%" - take the upper bound
                # But NOT if it's a negative number
                if '-' in value and not value.startswith('-'):
                    value = value.split('-')[-1]

                # Remove % signs and convert comma to dot
                value = value.replace('%', '').replace(',', '.').strip()

                # Final check for empty string
                if not value:
                    return None

            return float(value)
        except (ValueError, TypeError):
            # If conversion fails, return None instead of raising
            return None


# =========================================================================
# Predefined country groups
# =========================================================================

# Portfolio allocation countries (based on portfolio guideline document pages 91-105)
# Allocation strategy: USA (55-65%), Europe (15-20%), Japan (8-12%)
# Note: China and India excluded (not available in Trading212)
PORTFOLIO_COUNTRIES = [
    'USA',       # 55-65% - AI infrastructure leadership, profit margin superiority
    'Germany',   # Europe's largest economy (part of 15-20% Europe allocation)
    'France',    # Major European economy
    'UK',        # Major European economy
    'Japan',     # 8-12% - Corporate governance reforms, BoJ normalization
]

# G7 countries excluding Italy (legacy - for backward compatibility)
G7_COUNTRIES = ['USA', 'Germany', 'Japan', 'UK', 'France', 'Canada']

G10_EXTENDED = [
    'USA', 'Germany', 'Japan', 'UK', 'France', 'Italy', 'Canada',
    'China', 'Australia', 'South Korea'
]

MAJOR_ECONOMIES = [
    'USA', 'China', 'Japan', 'Germany', 'UK', 'France', 'India',
    'Italy', 'Brazil', 'Canada', 'South Korea', 'Australia'
]


if __name__ == "__main__":
    """
    Test scraper with portfolio countries.

    Scrapes economic data for all countries in the portfolio allocation strategy:
    - USA (55-65% allocation)
    - Major Europe: Germany, France, UK (15-20% allocation)
    - Japan (8-12% allocation)

    Note: China and India excluded (not available in Trading212)
    """
    scraper = IlSoleScraper()

    print("\n" + "="*100)
    print("IL SOLE 24 ORE SCRAPER - PORTFOLIO COUNTRIES TEST")
    print("="*100)
    print("\nFetching economic data for portfolio allocation countries:")
    print("  • USA (55-65% target allocation)")
    print("  • Germany, France, UK (Europe: 15-20% target allocation)")
    print("  • Japan (8-12% target allocation)")
    print("\nNote: China and India excluded (not available in Trading212)")
    print("="*100)

    # Fetch data for all portfolio countries
    portfolio_data = scraper.get_multiple_countries(PORTFOLIO_COUNTRIES)

    # Enhanced summary table with real indicators, forecasts, and interest rates
    print(f"\n{'Country':<10} {'GDP':<7} {'Ind':<7} {'Unemp':<6} {'GDP':<7} {'Infl':<6} {'Earn':<7} {'ST':<6} {'LT':<6} {'Cycle':<12}")
    print(f"{'':10} {'Q/Q':7} {'Prod':7} {'':6} {'Fcst':7} {'':6} {'12M':7} {'Rate':6} {'Rate':6} {'Phase':12}")
    print("-"*85)

    success_count = 0
    failed_countries = []

    for country, result in portfolio_data.items():
        if result['status'] == 'success':
            success_count += 1
            real = result.get('real_indicators', {}) or {}
            fcst = result.get('forecasts', {}) or {}

            # Extract all indicators
            gdp_qq = real.get('gdp_growth_qq')
            ind_prod = real.get('industrial_production')
            unemp = real.get('unemployment')
            gdp_fcst = fcst.get('gdp_growth_6m')
            inflation = fcst.get('last_inflation')
            earnings = fcst.get('earnings_12m')
            st_rate = real.get('st_rate')
            lt_rate = real.get('lt_rate')
            peg = fcst.get('peg_ratio')

            # Simple cycle classification based on real + forecast data
            if (gdp_fcst is not None and gdp_fcst < 0.5) or (gdp_qq is not None and gdp_qq < 0):
                cycle = "Recession"
            elif (gdp_fcst is not None and gdp_fcst > 2.5) and (earnings is not None and earnings > 15):
                cycle = "Early Cycle"
            elif (gdp_fcst is not None and gdp_fcst < 1.5) and (peg is not None and peg > 2.0):
                cycle = "Late Cycle"
            elif gdp_fcst is not None and 1.5 <= gdp_fcst <= 2.5:
                cycle = "Mid Cycle"
            else:
                cycle = "Transition"

            # Format output with N/A for missing values
            gdp_str = f"{gdp_qq:>5.1f}%" if gdp_qq is not None else "  N/A"
            ind_str = f"{ind_prod:>5.1f}%" if ind_prod is not None else "  N/A"
            unemp_str = f"{unemp:>4.1f}%" if unemp is not None else " N/A"
            fcst_str = f"{gdp_fcst:>5.1f}%" if gdp_fcst is not None else "  N/A"
            infl_str = f"{inflation:>4.1f}%" if inflation is not None else " N/A"
            earn_str = f"{earnings:>5.1f}%" if earnings is not None else "  N/A"
            st_str = f"{st_rate:>4.2f}%" if st_rate is not None else " N/A"
            lt_str = f"{lt_rate:>4.2f}%" if lt_rate is not None else " N/A"

            print(f"{country:<10} {gdp_str:>7} {ind_str:>7} {unemp_str:>6} {fcst_str:>7} {infl_str:>6} {earn_str:>7} {st_str:>6} {lt_str:>6} {cycle:<12}")
        else:
            failed_countries.append(country)
            print(f"{country:<10} {'ERROR - Data unavailable':>74}")

    print("-"*85)

    # Detailed analysis section
    print("\n" + "="*100)
    print("KEY ECONOMIC INDICATORS ANALYSIS")
    print("="*100)

    print("\n📊 Industrial Production (Leading Recession Indicator):")
    for country, result in portfolio_data.items():
        if result['status'] == 'success':
            real = result.get('real_indicators', {}) or {}
            ind_prod = real.get('industrial_production')
            if ind_prod is not None:
                if ind_prod < -2.0:
                    status = "🔴 SEVERE CONTRACTION"
                elif ind_prod < 0:
                    status = "🟠 CONTRACTION"
                elif ind_prod < 2.0:
                    status = "🟡 WEAK GROWTH"
                else:
                    status = "🟢 EXPANSION"
                print(f"  {country:<10} {ind_prod:>6.1f}%  {status}")
            else:
                print(f"  {country:<10}    N/A")

    print("\n💰 Interest Rates (Monetary Policy Stance):")
    for country, result in portfolio_data.items():
        if result['status'] == 'success':
            real = result.get('real_indicators', {}) or {}
            st_rate = real.get('st_rate')
            lt_rate = real.get('lt_rate')

            if st_rate is not None and lt_rate is not None:
                spread = lt_rate - st_rate
                if spread > 100:
                    curve = "STEEP (Expansionary)"
                elif spread > 0:
                    curve = "NORMAL (Neutral)"
                elif spread > -50:
                    curve = "FLAT (Late Cycle)"
                else:
                    curve = "INVERTED (Recession Risk)"

                print(f"  {country:<10} ST: {st_rate:>4.2f}%  LT: {lt_rate:>4.2f}%  Spread: {spread:>5.0f}bps  ({curve})")
            elif st_rate is not None or lt_rate is not None:
                st_str = f"{st_rate:>4.2f}%" if st_rate is not None else " N/A"
                lt_str = f"{lt_rate:>4.2f}%" if lt_rate is not None else " N/A"
                print(f"  {country:<10} ST: {st_str}  LT: {lt_str}  (Partial data)")
            else:
                print(f"  {country:<10} N/A")

    print("\n" + "="*100)
    print("DATA QUALITY SUMMARY")
    print("="*100)
    print(f"\n✅ Successfully fetched: {success_count}/{len(PORTFOLIO_COUNTRIES)} countries")
    if failed_countries:
        print(f"❌ Failed: {', '.join(failed_countries)}")
    else:
        print(f"✅ All portfolio countries data retrieved successfully!")

    print("\n" + "="*100)
    print("GEOGRAPHIC ALLOCATION RECOMMENDATIONS")
    print("="*100)
    print("\nBased on portfolio guideline document (pages 91-105):")
    print("\n📊 Target Allocations (Trading212 Universe):")
    print("  • USA:              55-65%  (AI leadership, profit margins)")
    print("  • Europe Total:     15-20%  (Germany, France, UK)")
    print("  • Japan:            8-12%   (Corporate reforms, BoJ normalization)")
    print("\nNote: China and India excluded (not available in Trading212)")
    print("\n💡 Key Insights:")
    print("  • Overweight USA for technology & AI infrastructure leadership")
    print("  • Modest overweight Japan for governance reforms")
    print("  • Europe diversification across Germany, France, UK")
    print("  • Currency hedging: 50-100% for DM when USD strength expected")
    print("\n" + "="*100)
