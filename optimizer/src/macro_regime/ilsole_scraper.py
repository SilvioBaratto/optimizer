#!/usr/bin/env python3
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
        "Usa": "USA",
        "Germania": "Germany",
        "Francia": "France",
        "Italia": "Italy",
        "UK": "UK",
        "Giappone": "Japan",
        "Cina": "China",
        "Canada": "Canada",
        "Australia": "Australia",
        "Spagna": "Spain",
        "Brasile": "Brazil",
        "India": "India",
        "Russia": "Russia",
        "Messico": "Mexico",
        "Svizzera": "Switzerland",
        "Olanda": "Netherlands",
        "Svezia": "Sweden",
        "Norvegia": "Norway",
        "Danimarca": "Denmark",
        "Austria": "Austria",
        "Belgio": "Belgium",
        "Finlandia": "Finland",
        "Irlanda": "Ireland",
        "Singapore": "Singapore",
        "Corea": "South Korea",
        "Hong K.": "Hong Kong",
        "Taiwan": "Taiwan",
        "Indonesia": "Indonesia",
        "Malaysia": "Malaysia",
        "Thailandia": "Thailand",
        "Filippine": "Philippines",
        "N. Zelanda": "New Zealand",
        "Argentina": "Argentina",
        "Cile": "Chile",
        "Sudafrica": "South Africa",
    }

    # Country name mapping for REAL INDICATORS table (indicatori-reali)
    COUNTRY_MAP_REAL = {
        "Stati Uniti": "USA",
        "Germania": "Germany",
        "Francia": "France",
        "Italia": "Italy",
        "G.Bretagna": "UK",
        "Giappone": "Japan",
        "Cina": "China",
        "Canada": "Canada",
        "Australia": "Australia",
        "Spagna": "Spain",
        "Brasile": "Brazil",
        "India": "India",
        "Russia": "Russia",
        "Messico": "Mexico",
        "Svizzera": "Switzerland",
        "Olanda": "Netherlands",
        "Svezia": "Sweden",
        "Norvegia": "Norway",
        "Danimarca": "Denmark",
        "Austria": "Austria",
        "Belgio": "Belgium",
        "Finlandia": "Finland",
        "Irlanda": "Ireland",
        "Singapore": "Singapore",
        "Corea": "South Korea",
        "Hong Kong": "Hong Kong",
        "Taiwan": "Taiwan",
        "Indonesia": "Indonesia",
        "Malaysia": "Malaysia",
        "Thailandia": "Thailand",
        "Filippine": "Philippines",
        "N.Zelanda": "New Zealand",
        "Argentina": "Argentina",
        "Cile": "Chile",
        "Sudafrica": "South Africa",
    }

    def __init__(self, timeout: int = 10):
        """
        Initialize scraper.
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        )

    def _fetch_page(self, endpoint: str) -> Optional[BeautifulSoup]:
        """Fetch and parse HTML page."""
        try:
            url = f"{self.BASE_URL}/{endpoint}"
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return BeautifulSoup(response.content, "html.parser")
        except Exception:
            return None

    def get_real_indicators(self, country: str = "USA") -> Optional[Dict]:
        """
        Scrape real economic indicators table.
        """
        soup = self._fetch_page("indicatori-reali")
        if soup is None:
            return None

        try:
            # Find the table
            table = soup.find("table", {"class": "mainTable"})
            if not table:
                return None

            # Find country row (use REAL indicators mapping)
            country_italian = [k for k, v in self.COUNTRY_MAP_REAL.items() if v == country]
            if not country_italian:
                return None

            country_name = country_italian[0]

            # Find table body
            tbody = table.find("tbody")
            if not tbody:
                return None

            # Search for country row (indicatori-reali uses 'name' attribute)
            country_row = None
            for row in tbody.find_all("tr"):
                paese_cell = row.find("td", {"name": "Paese"})
                if paese_cell and country_name.lower() in paese_cell.text.lower():
                    country_row = row
                    break

            if not country_row:
                return None

            # Extract data from cells with name attributes
            cells = {}
            for cell in country_row.find_all("td"):
                cell_name = cell.get("name")
                if cell_name:
                    cells[cell_name] = cell.text.strip()

            # Map to output format - using exact HTML attribute names from table
            # Based on actual cell names: TassoSconto, TassoInteresse, Pil_TT, ProdIndustriale_AA, etc.
            data = {
                "gdp_growth_qq": self._safe_float(
                    cells.get("Pil_TT")
                ),  # T/T only (Quarter-over-Quarter)
                "industrial_production": self._safe_float(cells.get("ProdIndustriale_AA")),
                "unemployment": self._safe_float(cells.get("Disoccupazione_AA")),
                "consumer_prices": self._safe_float(cells.get("PrezziConsumo_AA")),
                "deficit": self._safe_float(cells.get("Deficit_AA")),
                "debt": self._safe_float(cells.get("Debito_AA")),
                "st_rate": self._safe_float(cells.get("TassoSconto")),
                "lt_rate": self._safe_float(cells.get("TassoInteresse")),
                "timestamp": datetime.now().isoformat(),
            }

            return data

        except Exception:
            return None

    def get_forecasts(self, country: str = "USA") -> Optional[Dict]:
        """
        Scrape economic forecast table.
        """
        soup = self._fetch_page("previsione-economica")
        if soup is None:
            return None

        try:
            # Find the table
            table = soup.find("table", {"class": "mainTable"})
            if not table:
                return None

            # Find country row (use FORECAST mapping)
            country_italian = [k for k, v in self.COUNTRY_MAP_FORECAST.items() if v == country]
            if not country_italian:
                return None

            country_name = country_italian[0]

            # Find table body
            tbody = table.find("tbody")
            if not tbody:
                return None

            # Search for country row (previsione-economica uses 'id' for Paese)
            country_row = None
            for row in tbody.find_all("tr"):
                paese_cell = row.find("td", {"id": "Paese"})
                if paese_cell and country_name.lower() in paese_cell.text.lower():
                    country_row = row
                    break

            if not country_row:
                return None

            # Extract data from cells with id attributes
            cells = {}
            for cell in country_row.find_all("td"):
                cell_id = cell.get("id")
                if cell_id:
                    cells[cell_id] = cell.text.strip()

            # Map to output format
            data = {
                "last_inflation": self._safe_float(cells.get("UltimaInflazione")),
                "inflation_6m": self._safe_float(cells.get("ConsensoInflazione")),
                "inflation_10y_avg": self._safe_float(cells.get("MediaInfl10Anni")),
                "gdp_growth_6m": self._safe_float(cells.get("ConsensoPil")),
                "earnings_12m": self._safe_float(cells.get("ConsensoUtili")),
                "eps_expected_12m": self._safe_float(cells.get("ConsensoEps")),
                "peg_ratio": self._safe_float(cells.get("ConsensoPeg")),
                "st_rate_forecast": self._safe_float(cells.get("ConsensoTassiBreve")),
                "lt_rate_forecast": self._safe_float(cells.get("ConsensoTassiLungo")),
                "reference_date": cells.get("DataRiferimento"),
                "timestamp": datetime.now().isoformat(),
            }

            return data

        except Exception:
            return None

    def get_country_data(self, country: str = "USA") -> Dict:
        """
        Get both real indicators and forecast data for a country.
        """
        real_data = self.get_real_indicators(country)
        forecast_data = self.get_forecasts(country)

        if real_data is None and forecast_data is None:
            return {"status": "error", "country": country}

        return {
            "country": country,
            "real_indicators": real_data,
            "forecasts": forecast_data,
            "status": "success",
            "timestamp": datetime.now().isoformat(),
        }

    def get_all_data(self, country: str = "USA") -> Dict:
        """
        Get complete data for cycle classifier (compatible with run_regime_analysis.py).
        """
        result = self.get_country_data(country)

        if result["status"] == "error":
            return result

        # Rename keys to match classifier expectations
        return {
            "country": result["country"],
            "real": result.get("real_indicators"),
            "forecast": result.get("forecasts"),
            "status": result["status"],
            "timestamp": result["timestamp"],
        }

    def get_multiple_countries(self, countries: List[str]) -> Dict:
        """
        Get forecast data for multiple countries.
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
                value_no_spaces = value.replace(" ", "")
                if value_no_spaces and all(c in "-–—−" for c in value_no_spaces):
                    # Only dashes (ASCII hyphen, en dash, em dash, minus sign)
                    return None

                # Check for other missing data markers
                if value in ["N/A", "n/a", ""]:
                    return None

                # Handle range format like "0-4,25%" - take the upper bound
                # But NOT if it's a negative number
                if "-" in value and not value.startswith("-"):
                    value = value.split("-")[-1]

                # Remove % signs and convert comma to dot
                value = value.replace("%", "").replace(",", ".").strip()

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
    "USA",  # 55-65% - AI infrastructure leadership, profit margin superiority
    "Germany",  # Europe's largest economy (part of 15-20% Europe allocation)
    "France",  # Major European economy
    "UK",  # Major European economy
    "Japan",  # 8-12% - Corporate governance reforms, BoJ normalization
]

# G7 countries excluding Italy (legacy - for backward compatibility)
G7_COUNTRIES = ["USA", "Germany", "Japan", "UK", "France", "Canada"]

G10_EXTENDED = [
    "USA",
    "Germany",
    "Japan",
    "UK",
    "France",
    "Italy",
    "Canada",
    "China",
    "Australia",
    "South Korea",
]

MAJOR_ECONOMIES = [
    "USA",
    "China",
    "Japan",
    "Germany",
    "UK",
    "France",
    "India",
    "Italy",
    "Brazil",
    "Canada",
    "South Korea",
    "Australia",
]


if __name__ == "__main__":
    # Test scraper with portfolio countries
    scraper = IlSoleScraper()
    portfolio_data = scraper.get_multiple_countries(PORTFOLIO_COUNTRIES)
