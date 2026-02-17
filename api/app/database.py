"""
Synchronous database management for FastAPI with local PostgreSQL.

This module provides:
- Synchronous SQLAlchemy engine with psycopg2 driver
- Connection pooling for local PostgreSQL
- Health check functionality with timeout protection
- Session management utilities
- Error handling and automatic recovery
- Environment-based configuration
"""

import logging
import time
import threading
from contextlib import contextmanager
from typing import Optional, Generator, Dict, Any

from sqlalchemy import create_engine, Engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool, NullPool
from sqlalchemy.exc import SQLAlchemyError, OperationalError, DisconnectionError

from app.config import settings
from app.models.base import Base

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
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
        self._is_initialized: bool = False
        self._lock = threading.RLock()
        self._last_health_check: float = 0
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

        self._engine = create_engine(**engine_kwargs)

        logger.info(f"Database engine created:")
        logger.info(f"  - Pool class: QueuePool")
        logger.info(f"  - Pool size: {settings.database_pool_size}")
        logger.info(f"  - Max overflow: {settings.database_max_overflow}")
        logger.info(f"  - Pool recycle: {settings.database_pool_recycle}s")
        logger.info(f"  - Pre-ping enabled: {settings.database_pool_pre_ping}")

    def _build_connect_args(self) -> Dict[str, Any]:
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
        current_time = time.time()

        # Use cached result if recent
        if (current_time - self._last_health_check) < self._health_check_interval:
            logger.debug("Using cached health check result")
            return True

        if not self._is_initialized:
            logger.warning("Health check failed: Database not initialized")
            return False

        try:
            with self.get_session() as session:
                # Quick health check query
                result = session.execute(text("SELECT 1 as health_check"))
                row = result.fetchone()

                if row is not None and row[0] == 1:
                    self._last_health_check = current_time
                    logger.debug("Database health check passed")
                    return True
                else:
                    logger.warning("Database health check returned unexpected value")
                    return False

        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    def get_detailed_status(self) -> Dict[str, Any]:
        """
        Get detailed database status information for monitoring.

        Returns:
            Dict with connection pool status, health, and configuration info
        """
        status = {
            "initialized": self._is_initialized,
            "engine_created": self._engine is not None,
            "pool_class": (
                self._engine.pool.__class__.__name__ if self._engine else None
            ),
            "healthy": False,
            "last_health_check": self._last_health_check,
            "configuration": {
                "pool_size": settings.database_pool_size,
                "max_overflow": settings.database_max_overflow,
                "pool_timeout": settings.database_pool_timeout,
                "pool_recycle": settings.database_pool_recycle,
                "environment": settings.environment,
            },
        }

        if self._engine and hasattr(self._engine.pool, "status"):
            try:
                # pool.status() returns a formatted string
                pool_status_str = self._engine.pool.status()

                # Access pool attributes directly (they're properties, not methods)
                pool_info = {
                    "status_string": pool_status_str,
                }

                status["pool_status"] = pool_info
            except Exception as e:
                logger.debug(f"Could not get pool status: {e}")
                status["pool_status"] = "unavailable"

        # Perform health check
        status["healthy"] = self.health_check()

        return status

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
            RuntimeError: If database is not initialized
            SQLAlchemyError: For database-related errors
        """
        if not self._is_initialized or not self._session_factory:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        session = self._session_factory()

        try:
            yield session

        except (DisconnectionError, OperationalError) as e:
            logger.error(f"Database connection error: {e}")
            session.rollback()

            # Attempt to recover by invalidating the connection
            try:
                session.connection().invalidate()
            except Exception as recovery_error:
                logger.debug(f"Connection invalidation failed: {recovery_error}")

            raise

        except SQLAlchemyError as e:
            logger.error(f"Database error: {e}")
            session.rollback()
            raise

        except Exception as e:
            logger.error(f"Unexpected error in database session: {e}")
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

    @property
    def is_initialized(self) -> bool:
        """Check if database is initialized and ready for use."""
        return self._is_initialized

    @property
    def engine(self) -> Optional[Engine]:
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


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency for database sessions.

    This function provides database sessions to FastAPI route handlers
    with automatic cleanup and error handling.

    Usage:
        @app.get("/users")
        def get_users(db: Session = Depends(get_db)):
            return db.query(User).all()

    Yields:
        Session: SQLAlchemy database session
    """
    with database_manager.get_session() as session:
        yield session


def get_db_status() -> Dict[str, Any]:
    """
    Get comprehensive database status information.

    Returns:
        Dict containing database health, configuration, and pool status
    """
    return database_manager.get_detailed_status()


# Utility functions for common database operations


def execute_raw_sql(sql: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
    """
    Execute raw SQL with parameter binding.

    Args:
        sql: SQL query string
        parameters: Optional parameters for the query

    Returns:
        Query result

    Raises:
        RuntimeError: If database is not initialized
        SQLAlchemyError: For database errors
    """
    with database_manager.get_session() as session:
        if parameters:
            result = session.execute(text(sql), parameters)
        else:
            result = session.execute(text(sql))

        session.commit()
        return result


def test_database_connection() -> bool:
    """
    Test database connectivity.

    Returns:
        bool: True if connection is successful, False otherwise
    """
    try:
        return database_manager.health_check()
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False


# Context manager for manual transaction management
@contextmanager
def database_transaction() -> Generator[Session, None, None]:
    """
    Context manager for explicit transaction handling.

    Usage:
        with database_transaction() as session:
            user = User(name="John")
            session.add(user)
            # Transaction automatically committed on success
            # or rolled back on exception

    Yields:
        Session: Database session with explicit transaction control
    """
    with database_manager.get_session() as session:
        try:
            session.begin()
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise


# Export commonly used items
__all__ = [
    "database_manager",
    "init_db",
    "close_db",
    "get_db",
    "get_db_status",
    "execute_raw_sql",
    "test_database_connection",
    "database_transaction",
    "DatabaseManager",
]