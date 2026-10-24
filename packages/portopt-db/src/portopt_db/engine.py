"""Synchronous SQLAlchemy engine and session manager, configuration-injected.

Construct with a `DbConfig`; the daemon builds that from its settings and keeps
the module-level singleton with `init_db`/`close_db` lifecycle on its side.
"""

import logging
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import (
    DBAPIError,
    DisconnectionError,
    OperationalError,
    SQLAlchemyError,
)
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from portopt_db.base import Base
from portopt_db.config import DbConfig

logger = logging.getLogger(__name__)

_MAX_LOGGED_ERROR_CHARS = 300


def summarize_db_error(exc: BaseException) -> str:
    """Return a short, single-line description of *exc* that is safe to log.

    ``str()`` of a SQLAlchemy ``DBAPIError`` embeds the full statement and its
    bound parameters; a 1,300-row multi-row upsert makes that ~300 KB per
    failure. For a driver error only the driver's own exception class and the
    first line of its message are kept (the part that names the cause).

    Args:
        exc: Any exception raised inside a database session.

    Returns:
        ``"<DriverError>: <first message line>"`` for a ``DBAPIError``, else
        ``str(exc)``, cut to a bounded length.
    """
    orig = exc.orig if isinstance(exc, DBAPIError) else None
    if orig is not None:
        lines = str(orig).strip().splitlines()
        message = f"{type(orig).__name__}: {lines[0] if lines else ''}"
    else:
        message = str(exc)
    if len(message) > _MAX_LOGGED_ERROR_CHARS:
        return message[:_MAX_LOGGED_ERROR_CHARS] + "..."
    return message


class DatabaseManager:
    """Connection pooling, health check, and session context manager.

    Thread-safe; lazy-initializes on first `get_session` if `initialize`
    was not called at startup. `close` is terminal for lazy initialization: a
    closed manager raises from `get_session` instead of quietly building a
    fresh engine nothing would dispose, and only an explicit `initialize`
    reopens it.
    """

    def __init__(self, config: DbConfig):
        """
        Args:
            config: Pool sizing, URL, echo flag, and per-dialect connect args
                for this manager instance.
        """
        self.config = config
        self._engine: Engine | None = None
        self._session_factory: sessionmaker[Session] | None = None
        self._is_initialized: bool = False
        self._closed: bool = False
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
                self._closed = False
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
        """Session with rollback-on-error and always-close.

        Dead connections are not invalidated here: SQLAlchemy discards them
        itself on a disconnect error, and `pool_pre_ping` screens the rest at
        checkout. After `rollback()` the session holds no connection, so a
        manual invalidate would only check out and throw away a healthy one.

        Raises:
            RuntimeError: The manager was closed and not re-initialized.
        """
        # Check, lazy init and session creation share one critical section so a
        # concurrent `close()` cannot null the factory between them.
        with self._lock:
            if self._closed:
                raise RuntimeError("Database manager is closed")
            if not self._is_initialized:
                self.initialize()
            factory = self._session_factory
            if factory is None:
                raise RuntimeError("Session factory not initialized")
            session = factory()
        try:
            yield session
        except (DisconnectionError, OperationalError) as e:
            logger.error(f"Database connection error: {summarize_db_error(e)}")
            if hasattr(session, "rollback"):
                session.rollback()
            raise
        except SQLAlchemyError as e:
            logger.error(f"Database error: {summarize_db_error(e)}")
            if hasattr(session, "rollback"):
                session.rollback()
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error in database session: {summarize_db_error(e)}"
            )
            if hasattr(session, "rollback"):
                session.rollback()
            raise
        finally:
            try:
                session.close()
            except Exception as e:
                logger.debug(f"Error closing session: {e}")

    def create_all_tables(self) -> None:
        """Create every table registered on `Base.metadata` (idempotent)."""
        if not self._engine:
            raise RuntimeError("Database engine not initialized")
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=self._engine)
        logger.info("Database tables created successfully")

    def close(self) -> None:
        """Dispose the engine + reset state (idempotent, thread-safe).

        Afterwards `get_session` raises until `initialize` is called again.
        """
        with self._lock:
            self._closed = True
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
