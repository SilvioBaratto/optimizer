#!/usr/bin/env python3
"""IlSole24Ore macroeconomic data scraper.

Scrapes the ``mercati.ilsole24ore.com`` country-comparison tables for
economic forecasts (previsione-economica) and real indicators
(indicatori-reali). The two pages use different HTML attribute conventions
for country cells (``id`` vs ``name``), reflected in the two country maps.
"""

import logging
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

from app.services.infrastructure import (
    CircuitBreaker,
    RateLimiter,
    retry_with_backoff,
)
from app.services.infrastructure.retry import is_transient_network_error

logger = logging.getLogger(__name__)

_ilsole_circuit_breaker = CircuitBreaker(service_name="IlSole24Ore", max_attempts=5)
_ilsole_rate_limiter = RateLimiter(delay=0.5)


class IlSoleScraper:
    """Scraper for IlSole24Ore macroeconomic country-comparison tables."""

    BASE_URL = "https://mercati.ilsole24ore.com/dati-macroeconomici/paesi-a-confronto"

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
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                # Pin the encoding rather than inherit requests' default, which
                # silently becomes "gzip, deflate, br" as soon as brotli is
                # importable anywhere on sys.path. No brotli decoder ships with
                # this image, so a "br" response body would arrive undecodable,
                # every parse would match 0 rows, and the circuit breaker would
                # latch open — exactly the failure that froze the Trading
                # Economics table for 16 days (see tradingeconomics_scraper).
                # gzip/deflate are decoded by requests natively.
                "Accept-Encoding": "gzip, deflate",
            }
        )

    def _fetch_page(self, endpoint: str) -> BeautifulSoup | None:
        url = f"{self.BASE_URL}/{endpoint}"

        def _action() -> BeautifulSoup:
            _ilsole_circuit_breaker.check()
            _ilsole_rate_limiter.acquire(endpoint)
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return BeautifulSoup(response.content, "html.parser")

        return retry_with_backoff(
            _action,
            max_retries=3,
            is_rate_limit_error=is_transient_network_error,
            on_rate_limit=_ilsole_circuit_breaker.trigger,
            on_success=lambda _: _ilsole_circuit_breaker.reset(),
        )

    def get_real_indicators(self, country: str = "USA") -> dict | None:
        """Fetch real economic indicators for a country.

        Args:
            country: English country name (must be in ``COUNTRY_MAP_REAL``).

        Returns:
            Dict of indicator values, or ``None`` if the page is unreachable
            or the country is not found in the table.
        """
        soup = self._fetch_page("indicatori-reali")
        if soup is None:
            return None

        try:
            table = soup.find("table", {"class": "mainTable"})
            if not table:
                return None

            country_italian = [
                k for k, v in self.COUNTRY_MAP_REAL.items() if v == country
            ]
            if not country_italian:
                return None

            country_name = country_italian[0]

            tbody = table.find("tbody")
            if not tbody:
                return None

            # indicatori-reali uses 'name' attribute; previsione-economica uses 'id'
            country_row = None
            for row in tbody.find_all("tr"):
                paese_cell = row.find("td", {"name": "Paese"})
                if paese_cell and country_name.lower() in paese_cell.text.lower():
                    country_row = row
                    break

            if not country_row:
                return None

            cells = {}
            for cell in country_row.find_all("td"):
                cell_name = cell.get("name")
                if cell_name:
                    cells[cell_name] = cell.text.strip()

            data = {
                "gdp_growth_qq": self._safe_float(
                    cells.get("Pil_TT")
                ),  # Quarter-over-Quarter only
                "industrial_production": self._safe_float(
                    cells.get("ProdIndustriale_AA")
                ),
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
            # Whole-page parse failure (site structure changed): log with
            # traceback and degrade to None so the caller falls back to other
            # macro sources — one scraper must not abort the macro fetch.
            logger.warning(
                "IlSole real-indicators parse failed for %s", country, exc_info=True
            )
            return None

    def get_forecasts(self, country: str = "USA") -> dict | None:
        """Fetch consensus forecast data for a country.

        Args:
            country: English country name (must be in ``COUNTRY_MAP_FORECAST``).

        Returns:
            Dict of forecast values, or ``None`` if the page is unreachable
            or the country is not found.
        """
        soup = self._fetch_page("previsione-economica")
        if soup is None:
            return None

        try:
            table = soup.find("table", {"class": "mainTable"})
            if not table:
                return None

            country_italian = [
                k for k, v in self.COUNTRY_MAP_FORECAST.items() if v == country
            ]
            if not country_italian:
                return None

            country_name = country_italian[0]

            tbody = table.find("tbody")
            if not tbody:
                return None

            # previsione-economica uses 'id' for Paese; indicatori-reali uses 'name'
            country_row = None
            for row in tbody.find_all("tr"):
                paese_cell = row.find("td", {"id": "Paese"})
                if paese_cell and country_name.lower() in paese_cell.text.lower():
                    country_row = row
                    break

            if not country_row:
                return None

            cells = {}
            for cell in country_row.find_all("td"):
                cell_id = cell.get("id")
                if cell_id:
                    cells[cell_id] = cell.text.strip()

            data = {
                "last_inflation": self._safe_float(cells.get("UltimaInflazione")),
                "inflation_6m": self._safe_float(cells.get("ConsensoInflazione")),
                "inflation_10y_avg": self._safe_float(cells.get("MediaInfl10Anni")),
                "gdp_growth_6m": self._safe_float(cells.get("ConsensoPil")),
                "earnings_12m": self._safe_float(cells.get("ConsensoUtili")),
                "eps_expected_12m": self._safe_float(cells.get("ConsensoEps")),
                "peg_ratio": self._safe_float(cells.get("ConsensoPeg")),
                "lt_rate_forecast": self._safe_float(cells.get("ConsensoTassiLungo")),
                "reference_date": cells.get("DataRiferimento"),
                "timestamp": datetime.now().isoformat(),
            }

            return data

        except Exception:
            # Whole-page parse failure (site structure changed): log with
            # traceback and degrade to None; the caller merges whatever macro
            # sources succeeded rather than aborting.
            logger.warning(
                "IlSole forecasts parse failed for %s", country, exc_info=True
            )
            return None

    def get_country_data(self, country: str = "USA") -> dict:
        """Fetch both real indicators and forecasts for a country.

        Args:
            country: English country name.

        Returns:
            Dict with ``real_indicators``, ``forecasts``, ``status``, and
            ``timestamp``. ``status`` is ``"error"`` only when both sources
            return ``None``.
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

    def get_all_data(self, country: str = "USA") -> dict:
        """Fetch country data with keys remapped for downstream classifier compatibility.

        Args:
            country: English country name.

        Returns:
            Same shape as ``get_country_data`` but with ``real`` and ``forecast``
            keys instead of ``real_indicators`` / ``forecasts``.
        """
        result = self.get_country_data(country)

        if result["status"] == "error":
            return result

        return {
            "country": result["country"],
            "real": result.get("real_indicators"),
            "forecast": result.get("forecasts"),
            "status": result["status"],
            "timestamp": result["timestamp"],
        }

    def get_multiple_countries(self, countries: list[str]) -> dict:
        """Fetch country data for each country in the list.

        Args:
            countries: English country names. Unknown names produce
                ``status: "error"`` entries.

        Returns:
            Dict mapping country name → ``get_country_data`` result.
        """
        results = {}

        for country in countries:
            data = self.get_country_data(country)
            results[country] = data
            # Rate limiting is applied by _ilsole_rate_limiter inside _fetch_page

        return results

    @staticmethod
    def _safe_float(value) -> float | None:
        """Convert an IlSole24Ore cell value to float, returning None for missing data.

        Handles Italian conventions: comma as decimal separator, range format
        like ``"0-4,25%"`` (takes the upper bound), and various dash characters
        used as placeholders for missing values.
        """
        if value is None or pd.isna(value):
            return None
        try:
            if isinstance(value, str):
                value = value.strip()

                value_no_spaces = value.replace(" ", "")
                if value_no_spaces and all(c in "--—−" for c in value_no_spaces):
                    # ASCII hyphen, en dash, em dash, minus sign all mean "no data"
                    return None

                if value in ["N/A", "n/a", ""]:
                    return None

                # Range like "0-4,25%" — take upper bound, but guard against
                # negative numbers that also contain a leading "-"
                if "-" in value and not value.startswith("-"):
                    value = value.split("-")[-1]

                value = value.replace("%", "").replace(",", ".").strip()

                if not value:
                    return None

            return float(value)
        except (ValueError, TypeError):
            return None


# Portfolio allocation countries (based on portfolio guideline document pages 91-105)
# Allocation strategy: USA (55-65%), Europe (15-20%), Japan (8-12%)
# Note: China and India excluded (not available in Trading212)
PORTFOLIO_COUNTRIES = [
    "USA",  # 55-65% - AI infrastructure leadership, profit margin superiority
    "Germany",  # Europe's largest economy (part of 15-20% Europe allocation)
    "France",
    "UK",
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
    scraper = IlSoleScraper()
    portfolio_data = scraper.get_multiple_countries(PORTFOLIO_COUNTRIES)
