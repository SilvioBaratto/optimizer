"""
Production-ready synchronous database management for FastAPI with Supabase PostgreSQL.

This module provides:
- Synchronous SQLAlchemy engine with psycopg2 driver
- Connection pooling optimized for Supabase and Fly.io
- Health check functionality with timeout protection
- Session management utilities
- Error handling and automatic recovery
- Environment-based configuration
"""

import time
import threading
from contextlib import contextmanager
from typing import Optional, Generator, Dict, Any

from rich.console import Console
from rich.table import Table
from sqlalchemy import create_engine, Engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool, NullPool
from sqlalchemy.exc import SQLAlchemyError, OperationalError, DisconnectionError

from optimizer.database.config import settings
from optimizer.database.models.base import Base

# Rich console for output
console = Console()


class DatabaseManager:
    """
    Production-ready database manager for synchronous SQLAlchemy operations.

    Features:
    - Connection pooling with automatic recovery
    - Health checking with timeout protection
    - Thread-safe session management
    - Supabase-optimized configuration
    - Fly.io deployment compatibility
    """

    def __init__(self):
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
        self._is_initialized: bool = False
        self._lock = threading.RLock()
        self._last_health_check: float = 0
        self._health_check_interval: float = 30.0  # Cache health checks for 30 seconds

    def initialize(self, verbose: bool = False) -> None:
        """
        Initialize the database engine and session factory.

        This method is thread-safe and can be called multiple times.
        Subsequent calls are no-ops if already initialized.

        Args:
            verbose: If True, print detailed connection info
        """
        with self._lock:
            if self._is_initialized:
                return

            try:
                self._create_engine(verbose=verbose)
                self._create_session_factory()
                self._test_connection()
                self._is_initialized = True

            except Exception as e:
                console.print(f"[red]Database initialization failed:[/red] {e}")
                self._cleanup_resources()
                raise

    def _create_engine(self, verbose: bool = False) -> None:
        """Create SQLAlchemy engine with production-optimized settings."""

        # Determine pool class based on environment and configuration
        pool_class = self._get_optimal_pool_class()

        # Build connection arguments optimized for Supabase and Fly.io
        connect_args = self._build_connect_args()

        # Configure engine with production settings
        engine_kwargs = {
            "url": settings.database_url,
            "poolclass": pool_class,
            "echo": settings.database_echo,
            "connect_args": connect_args,
            "future": True,  # Use SQLAlchemy 2.0 style
        }

        # Add pool configuration only if not using NullPool
        if pool_class != NullPool:
            engine_kwargs.update(
                {
                    "pool_size": settings.database_pool_size,
                    "max_overflow": settings.database_max_overflow,
                    "pool_timeout": settings.database_pool_timeout,
                    "pool_recycle": settings.database_pool_recycle,
                    "pool_pre_ping": settings.database_pool_pre_ping,
                    "pool_reset_on_return": settings.database_pool_reset_on_return,
                }
            )

        self._engine = create_engine(**engine_kwargs)

        # Display connection info in verbose mode
        if verbose:
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column(style="dim")
            table.add_column()
            table.add_row("Pool", pool_class.__name__)
            if pool_class != NullPool:
                table.add_row("Pool size", str(settings.database_pool_size))
                table.add_row("Max overflow", str(settings.database_max_overflow))
            table.add_row("Pool recycle", f"{settings.database_pool_recycle}s")
            table.add_row("Pre-ping", "Yes" if settings.database_pool_pre_ping else "No")
            console.print(table)

    def _get_optimal_pool_class(self) -> type:
        """
        Determine the optimal pool class based on deployment environment.

        Returns:
            NullPool for serverless/container environments with external pooling
            QueuePool for traditional deployments and local development
        """
        # Check if we're using Supabase's transaction pooler (port 6543)
        is_transaction_pooler = ":6543/" in settings.database_url

        # Use NullPool when using external pooling (Supabase Transaction Pooler)
        if is_transaction_pooler:
            return NullPool

        # Use QueuePool for local development and direct connections
        return QueuePool

    def _build_connect_args(self) -> Dict[str, Any]:
        """Build psycopg2-specific connection arguments."""
        connect_args = {
            "application_name": "optimizer",
            "connect_timeout": 10,
        }

        # SSL configuration based on database location
        if settings.is_local_database:
            connect_args["sslmode"] = "disable"
        else:
            connect_args["sslmode"] = "require"
            connect_args.update(
                {
                    "keepalives": 1,
                    "keepalives_idle": 30,
                    "keepalives_interval": 10,
                    "keepalives_count": 3,
                }
            )

        # Statement timeout for query safety
        if settings.database_command_timeout:
            connect_args["options"] = f"-c statement_timeout={settings.database_command_timeout * 1000}"

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

    def _test_connection(self) -> None:
        """Test database connectivity with timeout protection."""
        if not self._engine:
            raise RuntimeError("Engine not initialized")

        try:
            with self._engine.connect() as conn:
                # Simple connectivity test
                result = conn.execute(text("SELECT 1 as test_connection"))
                test_value = result.fetchone()[0]

                if test_value != 1:
                    raise RuntimeError("Database connection test failed")

        except Exception as e:
            console.print(f"[red]Database connection test failed:[/red] {e}")
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
            return True

        if not self._is_initialized:
            return False

        try:
            with self.get_session() as session:
                # Quick health check query
                result = session.execute(text("SELECT 1 as health_check"))
                health_value = result.fetchone()[0]

                if health_value == 1:
                    self._last_health_check = current_time
                    return True
                else:
                    return False

        except Exception:
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
            },
        }

        if self._engine and hasattr(self._engine.pool, "status"):
            try:
                pool_status = self._engine.pool.status()
                status["pool_status"] = {
                    "pool_size": pool_status.get("pool_size", "N/A"),
                    "checked_in": pool_status.get("checked_in", "N/A"),
                    "checked_out": pool_status.get("checked_out", "N/A"),
                    "overflow": pool_status.get("overflow", "N/A"),
                }
            except Exception:
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

        except (DisconnectionError, OperationalError):
            session.rollback()

            # Attempt to recover by invalidating the connection
            try:
                session.connection().invalidate()
            except Exception:
                pass

            raise

        except SQLAlchemyError:
            session.rollback()
            raise

        except Exception:
            session.rollback()
            raise

        finally:
            try:
                session.close()
            except Exception:
                pass

    def create_all_tables(self) -> None:
        """
        Create all database tables defined in models.

        This method is safe to call multiple times.
        """
        if not self._engine:
            raise RuntimeError("Database engine not initialized")

        try:
            Base.metadata.create_all(bind=self._engine)

        except Exception as e:
            console.print(f"[red]Failed to create database tables:[/red] {e}")
            raise

    def close(self) -> None:
        """
        Cleanup database resources.

        This method is thread-safe and can be called multiple times.
        """
        with self._lock:
            if not self._is_initialized:
                return

            self._cleanup_resources()

    def _cleanup_resources(self) -> None:
        """Internal method to cleanup database resources."""
        if self._engine:
            try:
                self._engine.dispose()
            except Exception:
                pass
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


def init_db(verbose: bool = False) -> None:
    """
    Initialize database connection and create tables.

    This function is called during application startup.
    It's safe to call multiple times.

    Args:
        verbose: If True, print detailed connection info
    """
    try:
        # Initialize connection
        database_manager.initialize(verbose=verbose)

        # Create tables if they don't exist
        database_manager.create_all_tables()

    except Exception as e:
        console.print(f"[red]Database initialization failed:[/red] {e}")
        raise


def close_db() -> None:
    """
    Close database connections and cleanup resources.

    This function is called during application shutdown.
    It's safe to call multiple times.
    """
    database_manager.close()


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
    except Exception:
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
