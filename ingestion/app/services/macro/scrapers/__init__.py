"""Clients for macroeconomic data sources."""

from app.services.macro.scrapers.fred_scraper import FRED_SERIES, FredScraper

__all__ = [
    "FRED_SERIES",
    "FredScraper",
]
