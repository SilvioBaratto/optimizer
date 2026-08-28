"""Synchronous SQLAlchemy engine/session manager (config-injected).

Moved out of the ingestion daemon (``app.database``) so any consumer can own a
connection. Construct with a :class:`portopt_db.config.DbConfig`; the daemon
builds that from its ``settings`` and keeps the module-level singleton +
``init_db``/``close_db`` lifecycle on its side.
"""

import logging
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import DisconnectionError, OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from portopt_db.base import Base
from portopt_db.config import DbConfig

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Connection pooling + health check + session context manager.

    Thread-safe; lazy-initializes on first ``get_session`` if ``initialize``
    was not called at startup.
    """

    def __init__(self, config: DbConfig):
        self.config = config
        self._engine: Engine | None = None
        self._session_factory: sessionmaker[Session] | None = None
        self._is_initialized: bool = False
        self._lock = threading.RLock()
        self._last_health_check: float = 0
        self._last_health_check_result: bool = False
        self._health_check_interval: float = 30.0

    def initialize(self) -> None:
        """Create the engine + session factory + test connectivity (idempotent)."""
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
        cfg = self.config
        self._engine = create_engine(
            cfg.url,
            poolclass=QueuePool,
            echo=cfg.echo,
            connect_args=cfg.connect_args(),
            future=True,
            pool_size=cfg.pool_size,
            max_overflow=cfg.max_overflow,
            pool_timeout=cfg.pool_timeout,
            pool_recycle=cfg.pool_recycle,
            pool_pre_ping=cfg.pool_pre_ping,
            pool_reset_on_return=cfg.pool_reset_on_return,
        )
        logger.info(
            "Database engine created (QueuePool size=%s overflow=%s recycle=%ss "
            "pre_ping=%s)",
            cfg.pool_size,
            cfg.max_overflow,
            cfg.pool_recycle,
            cfg.pool_pre_ping,
        )

    def _create_session_factory(self) -> None:
        if not self._engine:
            raise RuntimeError("Engine must be created before session factory")
        self._session_factory = sessionmaker(
            bind=self._engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )
        logger.debug("Session factory created")

    def _test_connection(self) -> None:
        if not self._engine:
            raise RuntimeError("Engine not initialized")
        with self._engine.connect() as conn:
            row = conn.execute(text("SELECT 1 as test_connection")).fetchone()
            if row is None or row[0] != 1:
                raise RuntimeError("Database connection test failed")
        logger.debug("Database connection test passed")

    def health_check(self) -> bool:
        """Cached (30s) connectivity check; False if uninitialized or erroring."""
        if not self._is_initialized:
            logger.warning("Health check failed: Database not initialized")
            return False

        current_time = time.time()
        if (current_time - self._last_health_check) < self._health_check_interval:
            return self._last_health_check_result

        try:
            with self.get_session() as session:
                row = session.execute(text("SELECT 1 as health_check")).fetchone()
                ok = row is not None and row[0] == 1
                self._last_health_check = current_time
                self._last_health_check_result = ok
                if not ok:
                    logger.warning("Database health check returned unexpected value")
                return ok
        except Exception as e:
            self._last_health_check = current_time
            self._last_health_check_result = False
            logger.error(f"Database health check failed: {e}")
            return False

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Session with rollback-on-error, disconnect-invalidate, always-close."""
        if not self._is_initialized:
            self.initialize()

        session = self._session_factory()  # type: ignore[misc]
        try:
            yield session
        except (DisconnectionError, OperationalError) as e:
            logger.error(f"Database connection error: {e}")
            if hasattr(session, "rollback"):
                session.rollback()
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
        """Create every table registered on ``Base.metadata`` (idempotent)."""
        if not self._engine:
            raise RuntimeError("Database engine not initialized")
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=self._engine)
        logger.info("Database tables created successfully")

    def close(self) -> None:
        """Dispose the engine + reset state (idempotent, thread-safe)."""
        with self._lock:
            if not self._is_initialized:
                return
            try:
                self._cleanup_resources()
                logger.info("Database connection closed successfully")
            except Exception as e:
                logger.error(f"Error during database cleanup: {e}")

    def _cleanup_resources(self) -> None:
        if self._engine:
            try:
                self._engine.dispose()
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
        return self._is_initialized

    @property
    def engine(self) -> Engine | None:
        return self._engine


__all__ = ["DatabaseManager"]
