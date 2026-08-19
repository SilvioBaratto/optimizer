"""Synchronous database management for the ingestion daemon.

- SQLAlchemy engine with the psycopg2 driver and a QueuePool
- Health check with a 30-second result cache
- ``get_session`` context manager: rollback on error, invalidate on
  disconnect, always close
- Lazy init — the first ``get_session`` initializes if startup did not

The scheduler's job services and every service function open sessions through
``database_manager.get_session``; nothing here is request-scoped.
"""

import logging
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import DisconnectionError, OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from app.config import settings
from app.models._shared import Base

# Configure module logger
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Database manager for synchronous SQLAlchemy operations.

    Features:
    - Connection pooling with automatic recovery
    - Health checking with timeout protection
    - Thread-safe session management
    - Local PostgreSQL optimized configuration
    """

    def __init__(self):
        self._engine: Engine | None = None
        self._session_factory: sessionmaker | None = None
        self._is_initialized: bool = False
        self._lock = threading.RLock()
        self._last_health_check: float = 0
        self._last_health_check_result: bool = False
        self._health_check_interval: float = 30.0  # Cache health checks for 30 seconds

    def initialize(self) -> None:
        """
        Initialize the database engine and session factory.

        This method is thread-safe and can be called multiple times.
        Subsequent calls are no-ops if already initialized.
        """
        with self._lock:
            if self._is_initialized:
                logger.debug("Database already initialized")
                return

            try:
                logger.info("Initializing database connection...")
                self._create_engine()
                self._create_session_factory()
                self._test_connection()
                self._is_initialized = True
                logger.info("Database initialization successful")

            except Exception as e:
                logger.error(f"Database initialization failed: {e}")
                self._cleanup_resources()
                raise

    def _create_engine(self) -> None:
        """Create SQLAlchemy engine with standard settings."""

        # Build connection arguments for local PostgreSQL
        connect_args = self._build_connect_args()

        # Configure engine with standard settings
        engine_kwargs = {
            "url": settings.database_url,
            "poolclass": QueuePool,
            "echo": settings.database_echo,
            "connect_args": connect_args,
            "future": True,  # Use SQLAlchemy 2.0 style
            "pool_size": settings.database_pool_size,
            "max_overflow": settings.database_max_overflow,
            "pool_timeout": settings.database_pool_timeout,
            "pool_recycle": settings.database_pool_recycle,
            "pool_pre_ping": settings.database_pool_pre_ping,
            "pool_reset_on_return": settings.database_pool_reset_on_return,
        }

        self._engine = create_engine(**engine_kwargs)  # type: ignore[call-overload]

        logger.info("Database engine created:")
        logger.info("  - Pool class: QueuePool")
        logger.info(f"  - Pool size: {settings.database_pool_size}")
        logger.info(f"  - Max overflow: {settings.database_max_overflow}")
        logger.info(f"  - Pool recycle: {settings.database_pool_recycle}s")
        logger.info(f"  - Pre-ping enabled: {settings.database_pool_pre_ping}")

    def _build_connect_args(self) -> dict[str, Any]:
        """Build psycopg2-specific connection arguments."""
        connect_args = {
            "application_name": f"app-api-{settings.environment}",
            "connect_timeout": 10,  # Connection timeout in seconds
        }

        # Add keep-alive settings for stable connections
        connect_args.update(
            {
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 3,
            }
        )

        return connect_args

    def _create_session_factory(self) -> None:
        """Create thread-safe session factory."""
        if not self._engine:
            raise RuntimeError("Engine must be created before session factory")

        self._session_factory = sessionmaker(
            bind=self._engine,
            autoflush=False,  # Explicit flushing for better control
            autocommit=False,  # Explicit transaction management
            expire_on_commit=False,  # Keep objects usable after commit
        )

        logger.debug("Session factory created")

    def _test_connection(self) -> None:
        """Test database connectivity with timeout protection."""
        if not self._engine:
            raise RuntimeError("Engine not initialized")

        try:
            with self._engine.connect() as conn:
                # Simple connectivity test
                result = conn.execute(text("SELECT 1 as test_connection"))
                row = result.fetchone()

                if row is None or row[0] != 1:
                    raise RuntimeError("Database connection test failed")

                logger.debug("Database connection test passed")

        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            raise

    def health_check(self) -> bool:
        """
        Perform database health check with caching and timeout protection.

        Returns:
            bool: True if database is healthy, False otherwise
        """
        if not self._is_initialized:
            logger.warning("Health check failed: Database not initialized")
            return False

        current_time = time.time()

        # Use cached result if recent
        if (current_time - self._last_health_check) < self._health_check_interval:
            logger.debug("Using cached health check result")
            return self._last_health_check_result

        try:
            with self.get_session() as session:
                # Quick health check query
                result = session.execute(text("SELECT 1 as health_check"))
                row = result.fetchone()

                if row is not None and row[0] == 1:
                    self._last_health_check = current_time
                    self._last_health_check_result = True
                    logger.debug("Database health check passed")
                    return True
                else:
                    self._last_health_check = current_time
                    self._last_health_check_result = False
                    logger.warning("Database health check returned unexpected value")
                    return False

        except Exception as e:
            self._last_health_check = current_time
            self._last_health_check_result = False
            logger.error(f"Database health check failed: {e}")
            return False

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Get database session with automatic error handling and cleanup.

        This context manager provides:
        - Automatic session creation and cleanup
        - Connection error recovery
        - Transaction rollback on exceptions
        - Thread-safe operation

        Yields:
            Session: SQLAlchemy database session

        Raises:
            SQLAlchemyError: For database-related errors
        """
        if not self._is_initialized:
            self.initialize()

        session = self._session_factory()  # type: ignore[misc]

        try:
            yield session

        except (DisconnectionError, OperationalError) as e:
            logger.error(f"Database connection error: {e}")
            if hasattr(session, "rollback"):
                session.rollback()

            # Attempt to recover by invalidating the connection
            try:
                if hasattr(session, "connection"):
                    session.connection().invalidate()
            except Exception as recovery_error:
                logger.debug(f"Connection invalidation failed: {recovery_error}")

            raise

        except SQLAlchemyError as e:
            logger.error(f"Database error: {e}")
            if hasattr(session, "rollback"):
                session.rollback()
            raise

        except Exception as e:
            logger.error(f"Unexpected error in database session: {e}")
            if hasattr(session, "rollback"):
                session.rollback()
            raise

        finally:
            try:
                session.close()
            except Exception as e:
                logger.debug(f"Error closing session: {e}")

    def create_all_tables(self) -> None:
        """
        Create all database tables defined in models.

        This method is safe to call multiple times.
        """
        if not self._engine:
            raise RuntimeError("Database engine not initialized")

        try:
            logger.info("Creating database tables...")
            Base.metadata.create_all(bind=self._engine)
            logger.info("Database tables created successfully")

        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise

    def close(self) -> None:
        """
        Cleanup database resources.

        This method is thread-safe and can be called multiple times.
        """
        with self._lock:
            if not self._is_initialized:
                return

            try:
                self._cleanup_resources()
                logger.info("Database connection closed successfully")

            except Exception as e:
                logger.error(f"Error during database cleanup: {e}")

    def _cleanup_resources(self) -> None:
        """Internal method to cleanup database resources."""
        if self._engine:
            try:
                self._engine.dispose()
                logger.debug("Database engine disposed")
            except Exception as e:
                logger.debug(f"Error disposing engine: {e}")
            finally:
                self._engine = None

        self._session_factory = None
        self._is_initialized = False
        self._last_health_check = 0
        self._last_health_check_result = False

    @property
    def is_initialized(self) -> bool:
        """Check if database is initialized and ready for use."""
        return self._is_initialized

    @property
    def engine(self) -> Engine | None:
        """Get the SQLAlchemy engine (for advanced use cases)."""
        return self._engine


# Global database manager instance
database_manager = DatabaseManager()


def init_db() -> None:
    """
    Initialize database connection and create tables.

    This function is called during application startup.
    It's safe to call multiple times.
    """
    try:
        logger.info("Initializing database system...")

        # Initialize connection
        database_manager.initialize()

        # Create tables if they don't exist
        database_manager.create_all_tables()

        logger.info("Database system initialized successfully")

    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

        # In development mode, we want to continue even if DB fails
        if settings.is_development:
            logger.warning("Continuing in development mode despite database errors")
            return

        # In production, database failure should stop the application
        raise


def close_db() -> None:
    """
    Close database connections and cleanup resources.

    This function is called during application shutdown.
    It's safe to call multiple times.
    """
    try:
        logger.info("Closing database connections...")
        database_manager.close()
        logger.info("Database connections closed successfully")

    except Exception as e:
        logger.error(f"Error closing database connections: {e}")


# Export commonly used items
__all__ = [
    "DatabaseManager",
    "close_db",
    "database_manager",
    "init_db",
]
