"""Web scrapers for macroeconomic data sources."""

from app.services.scrapers.exceptions import ParseStructureError
from app.services.scrapers.fred_scraper import FRED_SERIES, FredScraper
from app.services.scrapers.ilsole_scraper import PORTFOLIO_COUNTRIES, IlSoleScraper
from app.services.scrapers.tradingeconomics_scraper import (
    TradingEconomicsIndicatorsScraper,
)

__all__ = [
    "FRED_SERIES",
    "FredScraper",
    "IlSoleScraper",
    "ParseStructureError",
    "PORTFOLIO_COUNTRIES",
    "TradingEconomicsIndicatorsScraper",
]
