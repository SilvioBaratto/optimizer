"""Tests for portopt_db.engine.DatabaseManager.

Config-injected. Most tests patch ``create_engine``; the ones that take
``tmp_path`` build a real file-SQLite engine instead. No server is contacted.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import event, text
from sqlalchemy.exc import DisconnectionError, OperationalError, SQLAlchemyError

from portopt_db.config import DbConfig
from portopt_db.engine import DatabaseManager

M = "portopt_db.engine"

_URL = "postgresql://u:p@h:5432/db"


def _cfg(**kw) -> DbConfig:
    return DbConfig(url=_URL, **kw)


class _SqliteConfig(DbConfig):
    """File-SQLite config: the psycopg2 connect args would be rejected."""

    def connect_args(self) -> dict:
        return {}


def _sqlite_manager(tmp_path) -> tuple[DatabaseManager, list[str]]:
    """Return an initialized real-engine manager and its pool event log."""
    mgr = DatabaseManager(_SqliteConfig(url=f"sqlite:///{tmp_path / 'engine.db'}"))
    mgr.initialize()
    events: list[str] = []
    assert mgr.engine is not None
    for name in ("checkout", "invalidate"):
        event.listen(mgr.engine, name, lambda *_a, _n=name: events.append(_n))
    return mgr, events


def _make_manager(
    *, row=(1,), cfg: DbConfig | None = None
) -> tuple[DatabaseManager, MagicMock]:
    """Return a manager whose engine and sessions are mocks."""
    mgr = DatabaseManager(cfg or _cfg())
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
        mgr, engine = _make_manager(
            cfg=_cfg(pool_size=7, max_overflow=13, pool_pre_ping=True)
        )
        with patch(f"{M}.create_engine", return_value=engine) as create:
            mgr.initialize()
        kwargs = create.call_args.kwargs
        assert kwargs["pool_size"] == 7
        assert kwargs["max_overflow"] == 13
        assert kwargs["pool_pre_ping"] is True

    def test_server_timeouts_reach_the_driver_connect_args(self) -> None:
        mgr, engine = _make_manager(cfg=_cfg(lock_timeout_ms=4_000))
        with patch(f"{M}.create_engine", return_value=engine) as create:
            mgr.initialize()
        options = create.call_args.kwargs["connect_args"]["options"]
        assert options == "-c lock_timeout=4000"

    def test_is_idempotent(self) -> None:
        mgr, engine = _make_manager()
        with patch(f"{M}.create_engine", return_value=engine) as create:
            mgr.initialize()
            mgr.initialize()
        create.assert_called_once()

    def test_when_connection_test_returns_wrong_row_then_raises_and_cleans_up(
        self,
    ) -> None:
        mgr, engine = _make_manager(row=(99,))
        with (
            patch(f"{M}.create_engine", return_value=engine),
            pytest.raises(RuntimeError, match="connection test failed"),
        ):
            mgr.initialize()
        assert mgr.is_initialized is False
        assert mgr.engine is None

    def test_when_connection_test_returns_no_row_then_raises(self) -> None:
        mgr, engine = _make_manager(row=None)
        with (
            patch(f"{M}.create_engine", return_value=engine),
            pytest.raises(RuntimeError),
        ):
            mgr.initialize()

    def test_when_create_engine_raises_then_resources_are_cleaned_up(self) -> None:
        mgr = DatabaseManager(_cfg())
        with (
            patch(f"{M}.create_engine", side_effect=OSError("boom")),
            pytest.raises(OSError, match="boom"),
        ):
            mgr.initialize()
        assert mgr.is_initialized is False


class TestHealthCheck:
    def test_when_not_initialized_then_false(self) -> None:
        assert DatabaseManager(_cfg()).health_check() is False

    def test_when_query_returns_one_then_true(self) -> None:
        mgr, engine = _make_manager()
        with patch(f"{M}.create_engine", return_value=engine):
            mgr.initialize()
            session = MagicMock()
            session.execute.return_value.fetchone.return_value = (1,)
            mgr._session_factory = MagicMock(return_value=session)
            assert mgr.health_check() is True

    def test_result_is_cached_within_the_interval(self) -> None:
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
    def test_on_connection_error_rolls_back_and_reraises(self, exc) -> None:
        mgr, session = self._ready()
        with pytest.raises(type(exc)), mgr.get_session():
            raise exc
        session.rollback.assert_called_once()
        session.close.assert_called_once()

    def test_on_operational_error_does_not_check_out_another_connection(
        self, tmp_path
    ) -> None:
        mgr, events = _sqlite_manager(tmp_path)
        try:
            with pytest.raises(OperationalError), mgr.get_session() as session:
                session.execute(text("SELECT 1"))
                raise OperationalError("stmt", {}, Exception("QueryCanceled"))
            assert events == ["checkout"]
        finally:
            mgr.close()

    def test_on_sqlalchemy_error_rolls_back_and_reraises(self) -> None:
        mgr, session = self._ready()
        with pytest.raises(SQLAlchemyError), mgr.get_session():
            raise SQLAlchemyError("bad sql")
        session.rollback.assert_called_once()
        session.close.assert_called_once()

    def test_on_unexpected_error_rolls_back_and_reraises(self) -> None:
        mgr, session = self._ready()
        with pytest.raises(ValueError), mgr.get_session():
            raise ValueError("bug in a service")
        session.rollback.assert_called_once()
        session.close.assert_called_once()

    def test_close_failure_is_swallowed(self) -> None:
        mgr, session = self._ready()
        session.close.side_effect = RuntimeError("already closed")
        with mgr.get_session():
            pass  # must not raise

    def test_after_close_raises_instead_of_building_a_new_engine(
        self, tmp_path
    ) -> None:
        mgr, _ = _sqlite_manager(tmp_path)
        mgr.close()
        with pytest.raises(RuntimeError, match="closed"), mgr.get_session():
            pass
        assert mgr.engine is None
        assert mgr.is_initialized is False

    def test_after_close_on_a_never_initialized_manager_still_raises(self) -> None:
        mgr = DatabaseManager(_cfg())
        mgr.close()
        with (
            patch(f"{M}.create_engine") as create,
            pytest.raises(RuntimeError, match="closed"),
            mgr.get_session(),
        ):
            pass
        create.assert_not_called()

    def test_explicit_initialize_reopens_a_closed_manager(self, tmp_path) -> None:
        mgr, _ = _sqlite_manager(tmp_path)
        mgr.close()
        mgr.initialize()
        with mgr.get_session() as session:
            assert session.execute(text("SELECT 1")).scalar() == 1
        mgr.close()

    def test_close_racing_get_session_yields_closed_error_not_type_error(
        self,
    ) -> None:
        mgr, _ = self._ready()
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                with mgr.get_session():
                    pass
            except BaseException as exc:
                errors.append(exc)

        with mgr._lock:
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join(timeout=0.2)
            assert thread.is_alive(), "get_session did not wait for the manager lock"
            mgr.close()
        thread.join(timeout=5)

        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)
        assert "closed" in str(errors[0])


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
        DatabaseManager(_cfg()).close()  # must not raise

    def test_dispose_failure_is_swallowed(self) -> None:
        mgr, engine = _make_manager()
        engine.dispose.side_effect = RuntimeError("stuck")
        with patch(f"{M}.create_engine", return_value=engine):
            mgr.initialize()
        mgr.close()
        assert mgr.is_initialized is False


class TestCreateAllTables:
    def test_when_engine_missing_then_raises(self) -> None:
        with pytest.raises(RuntimeError, match="not initialized"):
            DatabaseManager(_cfg()).create_all_tables()

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
