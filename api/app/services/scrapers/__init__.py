"""Web scrapers for macroeconomic data sources."""

from app.services.scrapers.ilsole_scraper import IlSoleScraper, PORTFOLIO_COUNTRIES
from app.services.scrapers.tradingeconomics_scraper import TradingEconomicsIndicatorsScraper

__all__ = [
    "IlSoleScraper",
    "PORTFOLIO_COUNTRIES",
    "TradingEconomicsIndicatorsScraper",
]
