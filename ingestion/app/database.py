"""Ingestion-side database wiring.

The ``DatabaseManager`` engine/session layer now lives in ``portopt_db.engine``
(portopt-db extraction, P1). This module keeps the daemon-specific pieces: it
builds a :class:`portopt_db.config.DbConfig` from ``app.config.settings``,
constructs the module-level ``database_manager`` singleton every service opens
sessions through, and owns the ``init_db`` / ``close_db`` lifecycle.

``from app.database import database_manager`` (and ``init_db`` / ``close_db`` /
``DatabaseManager``) is unchanged for every caller.
"""

import logging

from portopt_db.config import DbConfig
from portopt_db.engine import DatabaseManager

from app.config import settings

logger = logging.getLogger(__name__)


def _build_config() -> DbConfig:
    """Map ``app.config.settings`` onto the injected ``DbConfig``."""
    return DbConfig(
        url=settings.database_url,
        echo=settings.database_echo,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        pool_timeout=settings.database_pool_timeout,
        pool_recycle=settings.database_pool_recycle,
        pool_pre_ping=settings.database_pool_pre_ping,
        pool_reset_on_return=settings.database_pool_reset_on_return,
        application_name=f"app-api-{settings.environment}",
    )


# Global database manager instance.
database_manager = DatabaseManager(_build_config())


def init_db() -> None:
    """Initialize the connection and create tables. Safe to call repeatedly.

    In development a DB failure is logged and tolerated; in production it raises.
    """
    try:
        logger.info("Initializing database system...")
        database_manager.initialize()
        database_manager.create_all_tables()
        logger.info("Database system initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        if settings.is_development:
            logger.warning("Continuing in development mode despite database errors")
            return
        raise


def close_db() -> None:
    """Close database connections and cleanup resources. Safe to call repeatedly."""
    try:
        logger.info("Closing database connections...")
        database_manager.close()
        logger.info("Database connections closed successfully")
    except Exception as e:
        logger.error(f"Error closing database connections: {e}")


__all__ = [
    "DatabaseManager",
    "close_db",
    "database_manager",
    "init_db",
]
