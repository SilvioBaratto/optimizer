"""Tests for ``app/database.py`` — the daemon's only path to Postgres.

Everything the scheduler and the CLI do goes through
``database_manager.get_session``. There is no request scope and no FastAPI
dependency to lean on, so the manager owns pooling, lazy init, rollback, and
disconnect recovery on its own.

``create_engine`` is patched throughout — no server is contacted.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.exc import DisconnectionError, OperationalError, SQLAlchemyError

from app.database import DatabaseManager, close_db, init_db

M = "app.database"


def _make_manager(*, row=(1,)) -> tuple[DatabaseManager, MagicMock]:
    """Return an initialized manager whose engine and sessions are mocks."""
    mgr = DatabaseManager()
    engine = MagicMock()
    conn = engine.connect.return_value.__enter__.return_value
    conn.execute.return_value.fetchone.return_value = row
    return mgr, engine


class TestInitialize:
    def test_creates_engine_session_factory_and_tests_connection(self) -> None:
        mgr, engine = _make_manager()

        with patch(f"{M}.create_engine", return_value=engine) as create:
            mgr.initialize()

        assert mgr.is_initialized is True
        assert mgr.engine is engine
        create.assert_called_once()
        engine.connect.assert_called_once()

    def test_pool_settings_come_from_config(self) -> None:
        from app.config import settings

        mgr, engine = _make_manager()

        with patch(f"{M}.create_engine", return_value=engine) as create:
            mgr.initialize()

        kwargs = create.call_args.kwargs
        assert kwargs["pool_size"] == settings.database_pool_size
        assert kwargs["max_overflow"] == settings.database_max_overflow
        assert kwargs["pool_pre_ping"] == settings.database_pool_pre_ping

    def test_is_idempotent(self) -> None:
        """Lazy init means several callers may race into initialize()."""
        mgr, engine = _make_manager()

        with patch(f"{M}.create_engine", return_value=engine) as create:
            mgr.initialize()
            mgr.initialize()

        create.assert_called_once()

    def test_when_connection_test_returns_wrong_row_then_raises_and_cleans_up(
        self,
    ) -> None:
        mgr, engine = _make_manager(row=(99,))

        with patch(f"{M}.create_engine", return_value=engine):
            with pytest.raises(RuntimeError, match="connection test failed"):
                mgr.initialize()

        # A half-built manager must not be left behind for the next caller.
        assert mgr.is_initialized is False
        assert mgr.engine is None

    def test_when_connection_test_returns_no_row_then_raises(self) -> None:
        mgr, engine = _make_manager(row=None)

        with patch(f"{M}.create_engine", return_value=engine):
            with pytest.raises(RuntimeError):
                mgr.initialize()

    def test_when_create_engine_raises_then_resources_are_cleaned_up(self) -> None:
        mgr = DatabaseManager()

        with patch(f"{M}.create_engine", side_effect=OSError("boom")):
            with pytest.raises(OSError, match="boom"):
                mgr.initialize()

        assert mgr.is_initialized is False


class TestHealthCheck:
    def test_when_not_initialized_then_false(self) -> None:
        assert DatabaseManager().health_check() is False

    def test_when_query_returns_one_then_true(self) -> None:
        mgr, engine = _make_manager()

        with patch(f"{M}.create_engine", return_value=engine):
            mgr.initialize()
            session = MagicMock()
            session.execute.return_value.fetchone.return_value = (1,)
            mgr._session_factory = MagicMock(return_value=session)

            assert mgr.health_check() is True

    def test_result_is_cached_within_the_interval(self) -> None:
        """A per-call round-trip would put the healthcheck on the hot path."""
        mgr, engine = _make_manager()

        with patch(f"{M}.create_engine", return_value=engine):
            mgr.initialize()
            session = MagicMock()
            session.execute.return_value.fetchone.return_value = (1,)
            mgr._session_factory = MagicMock(return_value=session)

            mgr.health_check()
            calls_after_first = session.execute.call_count
            mgr.health_check()

        assert session.execute.call_count == calls_after_first

    def test_when_query_returns_unexpected_value_then_false(self) -> None:
        mgr, engine = _make_manager()

        with patch(f"{M}.create_engine", return_value=engine):
            mgr.initialize()
            session = MagicMock()
            session.execute.return_value.fetchone.return_value = (0,)
            mgr._session_factory = MagicMock(return_value=session)

            assert mgr.health_check() is False

    def test_when_query_raises_then_false(self) -> None:
        mgr, engine = _make_manager()

        with patch(f"{M}.create_engine", return_value=engine):
            mgr.initialize()
            session = MagicMock()
            session.execute.side_effect = SQLAlchemyError("down")
            mgr._session_factory = MagicMock(return_value=session)

            assert mgr.health_check() is False


class TestGetSession:
    def _ready(self) -> tuple[DatabaseManager, MagicMock]:
        mgr, engine = _make_manager()
        with patch(f"{M}.create_engine", return_value=engine):
            mgr.initialize()
        session = MagicMock()
        mgr._session_factory = MagicMock(return_value=session)
        return mgr, session

    def test_yields_a_session_and_always_closes_it(self) -> None:
        mgr, session = self._ready()

        with mgr.get_session() as s:
            assert s is session

        session.close.assert_called_once()

    def test_lazily_initializes_when_startup_did_not(self) -> None:
        """CLI subcommands and worker threads may reach a session before init_db."""
        mgr, engine = _make_manager()

        with patch(f"{M}.create_engine", return_value=engine):
            mgr._session_factory = None
            with mgr.get_session():
                pass

        assert mgr.is_initialized is True

    @pytest.mark.parametrize(
        "exc",
        [
            DisconnectionError("dropped"),
            OperationalError("stmt", {}, Exception("dropped")),
        ],
    )
    def test_on_disconnect_rolls_back_invalidates_and_reraises(self, exc) -> None:
        """A pooled connection killed server-side must not be handed out again."""
        mgr, session = self._ready()

        with pytest.raises(type(exc)):
            with mgr.get_session():
                raise exc

        session.rollback.assert_called_once()
        session.connection.return_value.invalidate.assert_called_once()
        session.close.assert_called_once()

    def test_when_invalidate_fails_then_original_error_still_propagates(self) -> None:
        mgr, session = self._ready()
        session.connection.return_value.invalidate.side_effect = RuntimeError("nope")

        with pytest.raises(DisconnectionError):
            with mgr.get_session():
                raise DisconnectionError("dropped")

        session.close.assert_called_once()

    def test_on_sqlalchemy_error_rolls_back_and_reraises(self) -> None:
        mgr, session = self._ready()

        with pytest.raises(SQLAlchemyError):
            with mgr.get_session():
                raise SQLAlchemyError("bad sql")

        session.rollback.assert_called_once()
        session.close.assert_called_once()

    def test_on_unexpected_error_rolls_back_and_reraises(self) -> None:
        """A scraper blowing up mid-transaction must not leave rows half-written."""
        mgr, session = self._ready()

        with pytest.raises(ValueError):
            with mgr.get_session():
                raise ValueError("bug in a service")

        session.rollback.assert_called_once()
        session.close.assert_called_once()

    def test_close_failure_is_swallowed(self) -> None:
        mgr, session = self._ready()
        session.close.side_effect = RuntimeError("already closed")

        with mgr.get_session():
            pass  # must not raise


class TestClose:
    def test_disposes_the_engine_and_resets_state(self) -> None:
        mgr, engine = _make_manager()

        with patch(f"{M}.create_engine", return_value=engine):
            mgr.initialize()
        mgr.close()

        engine.dispose.assert_called_once()
        assert mgr.is_initialized is False
        assert mgr.engine is None

    def test_when_not_initialized_then_noop(self) -> None:
        DatabaseManager().close()  # must not raise

    def test_dispose_failure_is_swallowed(self) -> None:
        mgr, engine = _make_manager()
        engine.dispose.side_effect = RuntimeError("stuck")

        with patch(f"{M}.create_engine", return_value=engine):
            mgr.initialize()
        mgr.close()

        assert mgr.is_initialized is False


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


class TestCreateAllTables:
    def test_when_engine_missing_then_raises(self) -> None:
        with pytest.raises(RuntimeError, match="not initialized"):
            DatabaseManager().create_all_tables()

    def test_creates_metadata_against_the_engine(self) -> None:
        mgr, engine = _make_manager()

        with patch(f"{M}.create_engine", return_value=engine):
            mgr.initialize()
            with patch(f"{M}.Base") as base:
                mgr.create_all_tables()

        base.metadata.create_all.assert_called_once_with(bind=engine)

    def test_create_failure_propagates(self) -> None:
        mgr, engine = _make_manager()

        with patch(f"{M}.create_engine", return_value=engine):
            mgr.initialize()
            with patch(f"{M}.Base") as base:
                base.metadata.create_all.side_effect = RuntimeError("ddl failed")
                with pytest.raises(RuntimeError, match="ddl failed"):
                    mgr.create_all_tables()
