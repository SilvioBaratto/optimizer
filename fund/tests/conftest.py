"""Pytest configuration for the fund test suite.

``fund`` is installed editable into the shared workspace venv (src-layout), so
``import fund`` resolves with no ``sys.path`` surgery. The hygiene guard is
source-blind (it scans tracked file text, imports no implementation module),
so it needs no fixtures.

Persistence and tool tests need a real (in-memory) database. The ``db_session``
fixture mirrors the portopt-db harness: StaticPool SQLite with a per-test
SAVEPOINT rollback, built off ``portopt_db.models`` so every table — including
the ``agent_runs`` / ``agent_decisions`` audit tables — is on ``Base.metadata``
before ``create_all``.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

# Importing the models package registers every model on Base.metadata.
from portopt_db.models import Base
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

TEST_DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture(autouse=True)
def _isolate_agent_fs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Root every agent run's staged workdir at a per-test tmp dir, not the shared one.

    ``fund.agents.backend.stage_theory`` regenerates its workdir (``rmtree`` +
    ``copytree``) on *every* agent run; when agent-driving tests cluster, concurrent
    staging of the single shared ``fund/.runtime/agent-fs`` races and fails on Windows
    (``WinError 5``). Rooting each test at an isolated ``tmp_path`` makes agent runs
    hermetic — no cross-test contention — for a trivial ~2.6 MB copy. Tests that never
    build an agent are unaffected (the patched global is simply unused).
    """
    import fund.agents.backend as backend_mod

    monkeypatch.setattr(backend_mod, "_AGENT_FS_ROOT", tmp_path / "agent-fs")


@pytest.fixture(scope="session")
def test_engine() -> Generator[Engine, None, None]:
    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def db_session(test_engine: Engine) -> Generator[Session, None, None]:
    """New session per test; app-code ``commit()`` only releases the SAVEPOINT,
    and the outer transaction is rolled back on teardown."""
    connection = test_engine.connect()
    transaction = connection.begin()
    testing_session_local = sessionmaker(
        autocommit=False, autoflush=False, bind=connection
    )
    session = testing_session_local()
    session.begin_nested()

    @event.listens_for(session, "after_transaction_end")
    def restart_savepoint(sess: Session, trans: object) -> None:
        if trans.nested and not trans._parent.nested:  # type: ignore[attr-defined]
            sess.begin_nested()

    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()
