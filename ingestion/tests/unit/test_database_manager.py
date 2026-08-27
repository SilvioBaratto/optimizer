"""Tests for ``app/database.py`` — the daemon's init_db/close_db lifecycle.

The ``DatabaseManager`` engine/session class moved to ``portopt_db.engine``
(portopt-db extraction) — its behavior is tested in
``packages/portopt-db/tests/test_engine.py``. What stays here is the
ingestion-side wiring: the module-level ``database_manager`` singleton built from
``app.config.settings`` and the ``init_db`` / ``close_db`` helpers (incl. the
dev-mode tolerance).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.database import close_db, init_db

M = "app.database"


class TestModuleLevelHelpers:
    def test_init_db_initializes_and_creates_tables(self) -> None:
        with (
            patch(f"{M}.database_manager") as dm,
            patch.object(type(dm), "__bool__", return_value=True, create=True),
        ):
            init_db()

        dm.initialize.assert_called_once()
        dm.create_all_tables.assert_called_once()

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
        """app.database maps settings onto the injected DbConfig."""
        from app.config import settings
        from app.database import _build_config

        cfg = _build_config()
        assert cfg.url == settings.database_url
        assert cfg.pool_size == settings.database_pool_size
        assert cfg.application_name == f"app-api-{settings.environment}"
