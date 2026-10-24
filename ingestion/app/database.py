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
from sqlalchemy import inspect

from app.config import settings

logger = logging.getLogger(__name__)


def _build_config() -> DbConfig:
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


database_manager = DatabaseManager(_build_config())


def _require_migrated_schema() -> None:
    """Raise unless Alembic has stamped this database.

    Alembic (``packages/portopt-db``) is the single schema owner. Creating tables
    here from the models would leave them unversioned, so the next ``alembic
    upgrade head`` would fail with DuplicateTable and the migration-only objects
    (e.g. the ``human_audit_log`` append-only trigger) would never be installed.
    Only the presence of ``alembic_version`` is verified, not that it is at head:
    a database behind head fails on the missing tables with a plain SQL error and
    stays migratable.
    """
    engine = database_manager.engine
    if engine is None or not inspect(engine).has_table("alembic_version"):
        raise RuntimeError(
            "Database schema has not been migrated: run `alembic upgrade head` "
            "(from packages/portopt-db) before starting the daemon or CLI."
        )


def init_db() -> None:
    """Initialize the connection and verify the schema was migrated.

    Never creates tables; Alembic owns the schema. Safe to call repeatedly.
    In development a failure (unreachable or unmigrated database) is logged and
    tolerated; in production it raises.

    Raises:
        RuntimeError: The database has no ``alembic_version`` table and the
            environment is not development.
    """
    try:
        logger.info("Initializing database system...")
        database_manager.initialize()
        _require_migrated_schema()
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
