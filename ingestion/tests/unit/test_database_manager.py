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

    def test_server_timeouts_come_from_settings(self) -> None:
        from app.database import _build_config

        with patch(f"{M}.settings") as st:
            st.database_url = "postgresql://u:p@h:5432/d"
            st.database_echo = False
            st.database_pool_size = 5
            st.database_max_overflow = 10
            st.database_pool_timeout = 30
            st.database_pool_recycle = 3600
            st.database_pool_pre_ping = True
            st.database_pool_reset_on_return = "rollback"
            st.environment = "test"
            st.database_lock_timeout_ms = 7_000
            st.database_statement_timeout_ms = 45_000
            cfg = _build_config()

        assert cfg.lock_timeout_ms == 7_000
        assert cfg.statement_timeout_ms == 45_000
        # Ticker writes are one transaction spanning network calls; an idle
        # timeout would discard them, so the daemon must never enable it.
        assert cfg.idle_in_transaction_timeout_ms == 0

    def test_lock_timeout_is_on_by_default_statement_timeout_is_not(self) -> None:
        # A default deployment must not let a writer wait forever behind another
        # transaction's row lock while holding a pool connection. The statement
        # timeout stays opt-in: it would cancel legitimate bulk writes.
        from app.config import Settings

        s = Settings(_env_file=None)
        assert s.database_lock_timeout_ms == 60_000
        assert s.database_statement_timeout_ms == 0

    def test_default_settings_reach_the_connection_as_lock_timeout_only(
        self,
    ) -> None:
        from app.config import Settings
        from app.database import _build_config

        with patch(f"{M}.settings", Settings(_env_file=None)):
            options = _build_config().connect_args()["options"]

        assert options == "-c lock_timeout=60000"

    def test_env_example_lock_timeout_matches_the_settings_default(self) -> None:
        # The documented value is what operators copy; it must not contradict
        # the default the daemon actually applies.
        import re
        from pathlib import Path

        from app.config import Settings

        env_example = Path(__file__).resolve().parents[3] / ".env.example"
        match = re.search(
            r"^#?\s*DATABASE_LOCK_TIMEOUT_MS=(\d+)", env_example.read_text(), re.M
        )
        assert match, "DATABASE_LOCK_TIMEOUT_MS missing from .env.example"
        assert int(match.group(1)) == Settings(_env_file=None).database_lock_timeout_ms

    def test_every_database_env_var_in_env_example_is_read_by_settings(self) -> None:
        # A documented-but-unread DATABASE_* knob tells operators a protection
        # exists (e.g. a command timeout) while nothing enforces it.
        import re
        from pathlib import Path

        from app.config import Settings

        env_example = Path(__file__).resolve().parents[3] / ".env.example"
        documented = set(
            re.findall(r"^#?\s*(DATABASE_[A-Z_]+)=", env_example.read_text(), re.M)
        )
        read = {
            (f.alias or name).upper() for name, f in Settings.model_fields.items()
        } | {name.upper() for name in Settings.model_fields}
        assert documented, "expected DATABASE_* entries in .env.example"
        assert documented <= read, f"never read by Settings: {documented - read}"
