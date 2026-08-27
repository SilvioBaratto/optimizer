"""Repositories package.

All database access goes through typed repositories. Each domain owns one
repository; ``_shared`` holds the generic base plus the admin repository used
by the orphan reaper and the schema-introspection helpers.
"""

from portopt_db.repositories.macro.macro_regime_repository import MacroRegimeRepository
from portopt_db.repositories.macro.sentiment_repository import SentimentRepository
from portopt_db.repositories.market_data.yfinance_repository import YFinanceRepository
from portopt_db.repositories.universe.universe_repository import UniverseRepository

from app.repositories._shared import BaseRepository, RepositoryBase
from app.repositories.jobs.background_job_repository import BackgroundJobRepository

__all__ = [
    "BackgroundJobRepository",
    "BaseRepository",
    "MacroRegimeRepository",
    "RepositoryBase",
    "SentimentRepository",
    "UniverseRepository",
    "YFinanceRepository",
]
