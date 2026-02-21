"""Web scrapers for macroeconomic data sources."""

from app.services.scrapers.ilsole_scraper import PORTFOLIO_COUNTRIES, IlSoleScraper
from app.services.scrapers.tradingeconomics_scraper import (
    TradingEconomicsIndicatorsScraper,
)

__all__ = [
    "PORTFOLIO_COUNTRIES",
    "IlSoleScraper",
    "TradingEconomicsIndicatorsScraper",
]
