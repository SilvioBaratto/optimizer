"""Repositories package.

All database access goes through typed repositories. Each domain owns one
repository; ``_shared`` holds the generic base plus the admin repository used
by the orphan reaper and the schema-introspection helpers.
"""

from app.repositories._shared import BaseRepository, RepositoryBase
from app.repositories.jobs.background_job_repository import BackgroundJobRepository
from app.repositories.macro.macro_regime_repository import MacroRegimeRepository
from app.repositories.macro.sentiment_repository import SentimentRepository
from app.repositories.market_data.yfinance_repository import YFinanceRepository
from app.repositories.universe.universe_repository import UniverseRepository

__all__ = [
    "BackgroundJobRepository",
    "BaseRepository",
    "MacroRegimeRepository",
    "RepositoryBase",
    "SentimentRepository",
    "UniverseRepository",
    "YFinanceRepository",
]
