"""Fund-side database wiring (Phase 8, Task 7).

Mirrors ``ingestion/app/database.py``: builds a :class:`portopt_db.config.DbConfig`
from :data:`fund.config.settings` and constructs the module-level
``database_manager`` singleton every CLI/TUI command opens sessions through.

Construction is **lazy** — :class:`~portopt_db.engine.DatabaseManager` creates no
engine until the first :func:`get_session` (or :func:`init_db`), so
``import fund.database`` never fails when ``DATABASE_URL`` is unset (CI, a bare
import). Each command owns its transaction: it opens ``with get_session() as
session:`` and commits/rolls back at the block boundary; the repositories never
``commit`` themselves.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from portopt_db.config import DbConfig
from portopt_db.engine import DatabaseManager
from sqlalchemy.orm import Session

from fund.config import settings


def _build_config() -> DbConfig:
    """Map :data:`fund.config.settings` onto the injected :class:`DbConfig`.

    ``database_url`` may be ``None`` at import (CI/dev without a DB); an empty URL
    is harmless because the engine is built lazily and only a real ``get_session``
    would surface the misconfiguration.
    """
    return DbConfig(
        url=settings.database_url or "",
        application_name="fund-cli",
    )


# Module-level manager: cheap to construct (no engine until first use).
database_manager = DatabaseManager(_build_config())


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Open one sync session via the module-level manager (lazy-initialises).

    The caller owns the transaction boundary — commit on success, otherwise the
    manager rolls back and always closes the session.
    """
    with database_manager.get_session() as session:
        yield session


def init_db() -> None:
    """Eagerly initialise the engine + test connectivity (optional; idempotent).

    ``get_session`` lazy-initialises on first use, so this is only needed to fail
    fast at startup.
    """
    database_manager.initialize()


def close_db() -> None:
    """Dispose the engine + connection pool (idempotent)."""
    database_manager.close()


__all__ = ["close_db", "database_manager", "get_session", "init_db"]
