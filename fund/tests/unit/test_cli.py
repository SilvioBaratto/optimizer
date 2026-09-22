"""Task 7 — ``fund.cli``: the headless human operating cycle under ``CliRunner``.

Exercises every command (mandate set|show, profile, run, approve, reject, status,
report) with the network-free fakes: ``ScriptedFundModel`` / ``ScriptedProfilerModel``
(monkeypatched onto ``build_primary``) and a ``MemorySaver`` + ``InMemoryStore``
shared via a monkeypatched ``setup_langgraph`` (no Postgres).

Unlike the flush-only ``db_session`` harness (built for repos that never commit),
these tests need real per-command commits to survive across ``CliRunner`` calls, so
they stand up their **own** function-scoped in-memory SQLite engine and monkeypatch
``get_session`` to hand each command a fresh session off it — faithfully mirroring
production (per-command sessions, DB-persisted cross-command state, per-session
rollbacks on reads). The whole cycle — mandate → run → approve/reject → status/report
— thus drives to terminal DB state with zero live LLM/network/PG.

``profile`` / ``run`` / ``approve`` / ``reject`` build the model via ``build_primary``;
a keyless ``settings`` makes that raise cleanly and the command exits non-zero.
``mandate`` / ``status`` / ``report`` build no model.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
import typer.main
from _fund_fakes import (
    ASOF,
    PORTFOLIO_ID,
    UNIVERSE,
    ScriptedFundModel,
    expected_weights,
    make_constraint_set,
    seed_panel,
)
from _profiler_fakes import make_answers
from _profiler_fakes import make_model as make_profiler_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from portopt_db.models import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from typer.testing import CliRunner

import fund.cli as cli
from fund.audit import AgentRunRepository, put_constraint_set
from fund.config import FundConfig, settings

_ASOF_STR = ASOF.isoformat()


# --- fakes for the injected persistence handles -----------------------------


@dataclass
class _FakePool:
    """Stand-in for the psycopg pool; records that the command closed it."""

    closed: int = 0

    def close(self) -> None:
        self.closed += 1


@dataclass
class _FakePersistence:
    """What the monkeypatched ``setup_langgraph`` returns (shared saver/store)."""

    saver: Any
    store: Any
    pool: _FakePool = field(default_factory=_FakePool)


@pytest.fixture
def harness(monkeypatch):
    """Wire the CLI onto an isolated in-memory DB + in-memory LangGraph fakes."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,  # one connection => one in-memory DB for the test
    )
    Base.metadata.create_all(engine)
    session_local = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    saver = MemorySaver()
    store = InMemoryStore()
    pool = _FakePool()

    @contextmanager
    def _get_session():
        session = session_local()
        try:
            yield session
        finally:
            session.close()

    monkeypatch.setattr(cli, "get_session", _get_session)
    monkeypatch.setattr(
        cli, "setup_langgraph", lambda *a, **k: _FakePersistence(saver, store, pool)
    )
    yield SimpleNamespace(
        runner=CliRunner(),
        read=_get_session,  # a fresh session context manager for assertions/seeding
        saver=saver,
        store=store,
        pool=pool,
    )
    engine.dispose()


# --- shared drivers ---------------------------------------------------------


def _set_mandate(harness) -> None:
    res = harness.runner.invoke(
        cli.app,
        [
            "mandate",
            "set",
            str(PORTFOLIO_ID),
            "--capital",
            "100000",
            "--base-currency",
            "EUR",
            "--drift",
            "0.1",
        ],
    )
    assert res.exit_code == 0, res.output


def _seed(harness) -> dict[str, float]:
    """Seed the price panel + constraint set; return the optimizer's weights."""
    with harness.read() as session:
        seed_panel(session)
        session.commit()
    put_constraint_set(
        harness.store,
        make_constraint_set(),
        store_key=settings.constraint_set_store_key,
    )
    with harness.read() as session:
        return expected_weights(session)


def _seed_and_run(harness, monkeypatch) -> tuple[Any, dict[str, float], Any]:
    """Seed, patch the fund model, persist a mandate, drive ``run`` to the gate."""
    expected = _seed(harness)
    monkeypatch.setattr(
        cli,
        "build_primary",
        lambda *a, **k: ScriptedFundModel(universe=UNIVERSE, weights=expected),
    )
    _set_mandate(harness)
    res = harness.runner.invoke(
        cli.app, ["run", str(PORTFOLIO_ID), "--asof", _ASOF_STR]
    )
    assert res.exit_code == 0, res.output
    with harness.read() as session:
        run_id = AgentRunRepository(session).list_paused_runs(PORTFOLIO_ID)[0].id
    return run_id, expected, res


# --- surface ----------------------------------------------------------------


def test_cli_exposes_the_seven_commands() -> None:
    click_app = typer.main.get_command(cli.app)
    names = set(click_app.commands)  # type: ignore[attr-defined]
    assert {"profile", "run", "approve", "reject", "status", "report"} <= names
    assert set(click_app.commands["mandate"].commands) >= {"set", "show"}  # type: ignore[attr-defined]


# --- mandate set|show -------------------------------------------------------


def test_mandate_set_then_show_round_trips(harness) -> None:
    _set_mandate(harness)
    res = harness.runner.invoke(cli.app, ["mandate", "show", str(PORTFOLIO_ID)])
    assert res.exit_code == 0, res.output
    assert "100000" in res.output
    assert "EUR" in res.output


def test_mandate_show_missing_errors(harness) -> None:
    res = harness.runner.invoke(cli.app, ["mandate", "show", str(PORTFOLIO_ID)])
    assert res.exit_code != 0
    assert "no mandate" in res.output.lower()


# --- run --------------------------------------------------------------------


def test_run_prints_run_id_and_detaches(harness, monkeypatch) -> None:
    run_id, _expected, res = _seed_and_run(harness, monkeypatch)
    assert str(run_id) in res.output
    with harness.read() as session:
        row = AgentRunRepository(session).get_run(run_id)
        assert row.status == "paused"
    assert harness.pool.closed >= 1  # pool released after the run detaches


def test_run_without_mandate_errors(harness, monkeypatch) -> None:
    _seed(harness)
    monkeypatch.setattr(
        cli, "build_primary", lambda *a, **k: ScriptedFundModel(universe=UNIVERSE)
    )
    res = harness.runner.invoke(
        cli.app, ["run", str(PORTFOLIO_ID), "--asof", _ASOF_STR]
    )
    assert res.exit_code != 0
    assert "mandate" in res.output.lower()


# --- approve / reject -------------------------------------------------------


def test_approve_finalizes_run(harness, monkeypatch) -> None:
    run_id, expected, _res = _seed_and_run(harness, monkeypatch)
    res = harness.runner.invoke(cli.app, ["approve", str(run_id)])
    assert res.exit_code == 0, res.output
    with harness.read() as session:
        row = AgentRunRepository(session).get_run(run_id)
        assert row.status == "completed"
        assert row.weights == expected
    assert harness.pool.closed >= 2  # run + approve each closed the pool


def test_reject_finalizes_run(harness, monkeypatch) -> None:
    run_id, _expected, _res = _seed_and_run(harness, monkeypatch)
    res = harness.runner.invoke(cli.app, ["reject", str(run_id)])
    assert res.exit_code == 0, res.output
    with harness.read() as session:
        row = AgentRunRepository(session).get_run(run_id)
        assert row.status == "rejected"
        assert row.weights == {}


def test_approve_unknown_run_errors(harness, monkeypatch) -> None:
    monkeypatch.setattr(
        cli, "build_primary", lambda *a, **k: ScriptedFundModel(universe=UNIVERSE)
    )
    put_constraint_set(
        harness.store,
        make_constraint_set(),
        store_key=settings.constraint_set_store_key,
    )
    res = harness.runner.invoke(cli.app, ["approve", str(uuid.uuid4())])
    assert res.exit_code != 0
    assert harness.pool.closed >= 1  # pool released even on failure


def test_approve_without_ollama_key_exits_nonzero(harness, monkeypatch) -> None:
    # A keyless settings makes the real build_primary raise a clear RuntimeError.
    monkeypatch.setattr(cli, "settings", FundConfig())
    res = harness.runner.invoke(cli.app, ["approve", str(uuid.uuid4())])
    assert res.exit_code != 0
    assert "OLLAMA_API_KEY" in res.output


# --- status / report --------------------------------------------------------


def test_status_renders_for_portfolio(harness, monkeypatch) -> None:
    run_id, _expected, _res = _seed_and_run(harness, monkeypatch)
    res = harness.runner.invoke(cli.app, ["status", str(PORTFOLIO_ID)])
    assert res.exit_code == 0, res.output
    assert str(run_id) in res.output
    assert "paused" in res.output


def test_status_without_portfolio_lists_paused(harness, monkeypatch) -> None:
    run_id, _expected, _res = _seed_and_run(harness, monkeypatch)
    res = harness.runner.invoke(cli.app, ["status"])
    assert res.exit_code == 0, res.output
    assert str(run_id) in res.output


def test_report_renders_after_approve(harness, monkeypatch) -> None:
    run_id, expected, _res = _seed_and_run(harness, monkeypatch)
    harness.runner.invoke(cli.app, ["approve", str(run_id)])
    res = harness.runner.invoke(cli.app, ["report", str(run_id)])
    assert res.exit_code == 0, res.output
    assert str(run_id) in res.output
    assert "completed" in res.output
    assert any(ticker in res.output for ticker in expected)


def test_report_unknown_run_errors(harness) -> None:
    res = harness.runner.invoke(cli.app, ["report", str(uuid.uuid4())])
    assert res.exit_code != 0
    assert "no run" in res.output.lower()


# --- profile ----------------------------------------------------------------


def test_profile_reaches_gate_and_detaches(harness, monkeypatch) -> None:
    answers = make_answers()
    monkeypatch.setattr(
        cli,
        "build_primary",
        lambda *a, **k: make_profiler_model(answers, portfolio_id=str(PORTFOLIO_ID)),
    )
    res = harness.runner.invoke(
        cli.app,
        ["profile", str(PORTFOLIO_ID), "--answers", "Growth over ten years, EUR."],
    )
    assert res.exit_code == 0, res.output
    with harness.read() as session:
        paused = AgentRunRepository(session).list_paused_runs(PORTFOLIO_ID)
        assert len(paused) == 1
        assert str(paused[0].id) in res.output
    assert harness.pool.closed >= 1
