"""Web scrapers for macroeconomic data sources."""

from app.services.macro.scrapers.exceptions import ParseStructureError
from app.services.macro.scrapers.fred_scraper import FRED_SERIES, FredScraper
from app.services.macro.scrapers.ilsole_scraper import (
    PORTFOLIO_COUNTRIES,
    IlSoleScraper,
)
from app.services.macro.scrapers.tradingeconomics_scraper import (
    TradingEconomicsIndicatorsScraper,
)

__all__ = [
    "FRED_SERIES",
    "PORTFOLIO_COUNTRIES",
    "FredScraper",
    "IlSoleScraper",
    "ParseStructureError",
    "TradingEconomicsIndicatorsScraper",
]
