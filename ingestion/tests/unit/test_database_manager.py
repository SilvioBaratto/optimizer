"""Tests for ``app/database.py`` — the daemon's init_db/close_db lifecycle.

The ``DatabaseManager`` engine/session class moved to ``portopt_db.engine``
(portopt-db extraction) — its behavior is tested in
``packages/portopt-db/tests/test_engine.py``. What stays here is the
ingestion-side wiring: the module-level ``database_manager`` singleton built from
``app.config.settings`` and the ``init_db`` / ``close_db`` helpers (incl. the
dev-mode tolerance).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest
from portopt_db.config import DbConfig
from portopt_db.engine import DatabaseManager
from sqlalchemy import inspect, text

from app.database import close_db, init_db

M = "app.database"


@dataclass(frozen=True)
class _SqliteConfig(DbConfig):
    """``DbConfig`` whose connect args SQLite accepts (psycopg2 keys are not)."""

    def connect_args(self) -> dict[str, object]:
        return {}


@pytest.fixture
def sqlite_manager(tmp_path: Path) -> Iterator[DatabaseManager]:
    """A real ``DatabaseManager`` on an empty SQLite file, patched in as the singleton."""
    manager = DatabaseManager(_SqliteConfig(url=f"sqlite:///{tmp_path / 'db.sqlite'}"))
    with patch(f"{M}.database_manager", manager):
        yield manager
    manager.close()


def _table_names(manager: DatabaseManager) -> list[str]:
    assert manager.engine is not None
    return sorted(inspect(manager.engine).get_table_names())


class TestModuleLevelHelpers:
    def test_init_db_refuses_unmigrated_database_and_creates_no_tables(
        self, sqlite_manager: DatabaseManager
    ) -> None:
        """Alembic is the only schema owner: an unmigrated DB must stay empty."""
        with patch(f"{M}.settings") as st:
            st.is_development = False
            with pytest.raises(RuntimeError, match="alembic upgrade head"):
                init_db()

        assert _table_names(sqlite_manager) == []

    def test_init_db_in_development_creates_no_tables_on_unmigrated_database(
        self, sqlite_manager: DatabaseManager
    ) -> None:
        with patch(f"{M}.settings") as st:
            st.is_development = True
            init_db()

        assert _table_names(sqlite_manager) == []

    def test_init_db_leaves_a_migrated_database_untouched(
        self, sqlite_manager: DatabaseManager
    ) -> None:
        sqlite_manager.initialize()
        with sqlite_manager.get_session() as session:
            session.execute(text("CREATE TABLE alembic_version (version_num TEXT)"))
            session.execute(text("INSERT INTO alembic_version VALUES ('abc123')"))
            session.commit()

        with patch(f"{M}.settings") as st:
            st.is_development = False
            init_db()

        assert _table_names(sqlite_manager) == ["alembic_version"]

    def test_init_db_swallows_failure_in_development(self) -> None:
        """A dev box without Postgres should still boot far enough to iterate."""
        with (
            patch(f"{M}.database_manager") as dm,
            patch(f"{M}.settings") as st,
        ):
            dm.initialize.side_effect = RuntimeError("no db")
            st.is_development = True

            init_db()  # must not raise

    def test_init_db_reraises_outside_development(self) -> None:
        with (
            patch(f"{M}.database_manager") as dm,
            patch(f"{M}.settings") as st,
        ):
            dm.initialize.side_effect = RuntimeError("no db")
            st.is_development = False

            with pytest.raises(RuntimeError, match="no db"):
                init_db()

    def test_close_db_swallows_failure(self) -> None:
        with patch(f"{M}.database_manager") as dm:
            dm.close.side_effect = RuntimeError("stuck")
            close_db()  # must not raise


class TestConfigMapping:
    def test_singleton_builds_dbconfig_from_settings(self) -> None:
        from app.config import settings
        from app.database import _build_config

        cfg = _build_config()
        assert cfg.url == settings.database_url
        assert cfg.pool_size == settings.database_pool_size
        assert cfg.application_name == f"app-api-{settings.environment}"
