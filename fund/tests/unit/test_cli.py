"""``fund.cli``: the headless human operating cycle under ``CliRunner``.

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

import getpass
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest
import typer.main
from _fund_fakes import (
    ASOF,
    PORTFOLIO_ID,
    UNIVERSE,
    PmFailsAfterOrderModel,
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
from portopt_db.models.market_data.yfinance_data import PriceHistory
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from typer.testing import CliRunner

import fund.cli as cli
from fund import observe
from fund.audit import AgentRunRepository, MandateRepository, put_constraint_set
from fund.audit.human_audit_repository import HumanAuditLogRepository
from fund.config import FundConfig, settings
from fund.schemas.mandate import PortfolioMandate, RunTriggers

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


def _rich_mandate(base_currency: str = "USD") -> PortfolioMandate:
    """A mandate whose every field differs from what a bare CLI rebuild would write."""
    return PortfolioMandate(
        portfolio_id=str(PORTFOLIO_ID),
        capital=Decimal("100000"),
        base_currency=base_currency,
        drift_l1_threshold=0.05,
        hitl_gates=("place_orders", "save_profile"),
        triggers=RunTriggers(cron=True, drift=False),
        benchmark="SPY",
        remembered_directives=("no EM", "cap tech 20%"),
    )


def _persist_mandate(harness, mandate: PortfolioMandate) -> None:
    with harness.read() as session:
        MandateRepository(session).upsert(mandate)
        session.commit()


def _stored_mandate(harness) -> PortfolioMandate:
    with harness.read() as session:
        mandate = observe.get_mandate(session, PORTFOLIO_ID)
        session.rollback()
    assert mandate is not None
    return mandate


def _audit_rows(harness, action: str) -> list[dict[str, Any]]:
    """The ``human_audit_log`` rows of one ``action`` as plain dicts, oldest first."""
    with harness.read() as session:
        rows = HumanAuditLogRepository(session).list_recent(action=action)
        return [
            {
                "run_id": r.run_id,
                "portfolio_id": r.portfolio_id,
                "actor": r.actor,
                "source": r.source,
                "idempotency_key": r.idempotency_key,
                "prev_value": r.prev_value,
                "new_value": r.new_value,
            }
            for r in reversed(rows)
        ]


@pytest.fixture
def cli_actor(monkeypatch) -> str:
    """Pin the OS user the CLI records as the audit actor."""
    monkeypatch.setattr(getpass, "getuser", lambda: "adviser-1")
    return "adviser-1"


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


def test_mandate_set_overrides_only_the_flags_given(harness) -> None:
    seeded = _rich_mandate()
    _persist_mandate(harness, seeded)
    res = harness.runner.invoke(
        cli.app, ["mandate", "set", str(PORTFOLIO_ID), "--capital", "250000"]
    )
    assert res.exit_code == 0, res.output
    assert _stored_mandate(harness) == seeded.model_copy(
        update={"capital": Decimal("250000")}
    )


def test_mandate_set_can_change_a_single_non_capital_field(harness) -> None:
    seeded = _rich_mandate()
    _persist_mandate(harness, seeded)
    res = harness.runner.invoke(
        cli.app, ["mandate", "set", str(PORTFOLIO_ID), "--drift", "0.2"]
    )
    assert res.exit_code == 0, res.output
    assert _stored_mandate(harness) == seeded.model_copy(
        update={"drift_l1_threshold": 0.2}
    )


def test_mandate_set_without_capital_needs_an_existing_mandate(harness) -> None:
    res = harness.runner.invoke(cli.app, ["mandate", "set", str(PORTFOLIO_ID)])
    assert res.exit_code != 0
    assert "--capital is required" in res.output


def test_mandate_set_audits_creation_then_edit(harness, cli_actor) -> None:
    _set_mandate(harness)
    (created,) = _audit_rows(harness, "new_portfolio")
    assert created["portfolio_id"] == PORTFOLIO_ID
    assert created["actor"] == cli_actor
    assert created["source"] == "cli"
    assert created["new_value"]["base_currency"] == "EUR"

    res = harness.runner.invoke(
        cli.app, ["mandate", "set", str(PORTFOLIO_ID), "--capital", "250000"]
    )
    assert res.exit_code == 0, res.output
    (edit,) = _audit_rows(harness, "mandate_edit")
    assert edit["portfolio_id"] == PORTFOLIO_ID
    assert edit["actor"] == cli_actor
    assert edit["source"] == "cli"
    assert Decimal(edit["prev_value"]["capital"]) == Decimal("100000")
    assert Decimal(edit["new_value"]["capital"]) == Decimal("250000")


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


def test_run_with_override_flags_keeps_remembered_directives(
    harness, monkeypatch
) -> None:
    expected = _seed(harness)
    monkeypatch.setattr(
        cli,
        "build_primary",
        lambda *a, **k: ScriptedFundModel(universe=UNIVERSE, weights=expected),
    )
    seeded = _rich_mandate(base_currency="EUR")
    _persist_mandate(harness, seeded)
    res = harness.runner.invoke(
        cli.app, ["run", str(PORTFOLIO_ID), "--asof", _ASOF_STR, "--drift", "0.07"]
    )
    assert res.exit_code == 0, res.output
    assert _stored_mandate(harness) == seeded.model_copy(
        update={"drift_l1_threshold": 0.07}
    )


def test_run_audits_the_run_and_any_mandate_override(
    harness, monkeypatch, cli_actor
) -> None:
    run_id, _expected, _res = _seed_and_run(harness, monkeypatch)
    (run_row,) = _audit_rows(harness, "run")
    assert run_row["run_id"] == run_id
    assert run_row["portfolio_id"] == PORTFOLIO_ID
    assert run_row["actor"] == cli_actor
    assert run_row["source"] == "cli"
    assert _audit_rows(harness, "mandate_edit") == []  # no flags, no mandate change

    res = harness.runner.invoke(
        cli.app, ["run", str(PORTFOLIO_ID), "--asof", _ASOF_STR, "--drift", "0.07"]
    )
    assert res.exit_code == 0, res.output
    (edit,) = _audit_rows(harness, "mandate_edit")
    assert edit["prev_value"]["drift_l1_threshold"] == 0.1
    assert edit["new_value"]["drift_l1_threshold"] == 0.07
    assert len(_audit_rows(harness, "run")) == 2


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


def test_approve_without_a_ticket_says_so_and_exits_nonzero(
    harness, monkeypatch
) -> None:
    """``place_orders`` finding no fill bar finalises the run incomplete with no
    ticket: the CLI must not echo a bare "approved" nor exit 0 for it."""
    run_id, _expected, _res = _seed_and_run(harness, monkeypatch)
    with harness.read() as session:
        session.query(PriceHistory).filter(PriceHistory.date > ASOF).delete()
        session.commit()

    res = harness.runner.invoke(cli.app, ["approve", str(run_id)])

    assert res.exit_code != 0
    assert "no paper ticket" in res.output
    with harness.read() as session:
        assert AgentRunRepository(session).get_run(run_id).status == "incomplete"


def test_approve_surfaces_a_failed_pm_follow_up(harness, monkeypatch) -> None:
    """A PM turn that dies after the gate was consumed leaves the decision standing
    (exit 0, run completed) but the operator must be told the model did not finish."""
    run_id, expected, _res = _seed_and_run(harness, monkeypatch)
    monkeypatch.setattr(
        cli,
        "build_primary",
        lambda *a, **k: PmFailsAfterOrderModel(universe=UNIVERSE, weights=expected),
    )

    res = harness.runner.invoke(cli.app, ["approve", str(run_id)])

    assert res.exit_code == 0, res.output
    assert "PM follow-up failed" in res.output
    with harness.read() as session:
        assert AgentRunRepository(session).get_run(run_id).status == "completed"


@pytest.mark.parametrize("verb", ["approve", "reject"])
def test_resolving_a_run_appends_a_cli_audit_row(
    harness, monkeypatch, cli_actor, verb
) -> None:
    run_id, _expected, _res = _seed_and_run(harness, monkeypatch)
    res = harness.runner.invoke(cli.app, [verb, str(run_id)])
    assert res.exit_code == 0, res.output
    (row,) = _audit_rows(harness, verb)
    assert row["run_id"] == run_id
    assert row["portfolio_id"] == PORTFOLIO_ID
    assert row["actor"] == cli_actor
    assert row["source"] == "cli"
    assert row["idempotency_key"] == f"{run_id}:{verb}"


def test_repeated_approve_replays_instead_of_resuming_again(
    harness, monkeypatch
) -> None:
    run_id, _expected, _res = _seed_and_run(harness, monkeypatch)
    first = harness.runner.invoke(cli.app, ["approve", str(run_id)])
    again = harness.runner.invoke(cli.app, ["approve", str(run_id)])
    assert first.exit_code == 0, first.output
    assert again.exit_code == 0, again.output
    assert "already applied" in again.output
    assert len(_audit_rows(harness, "approve")) == 1


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


def test_profile_appends_a_cli_audit_row(harness, monkeypatch, cli_actor) -> None:
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
        (paused,) = AgentRunRepository(session).list_paused_runs(PORTFOLIO_ID)
        run_id = paused.id
    (row,) = _audit_rows(harness, "profile")
    assert row["run_id"] == run_id
    assert row["portfolio_id"] == PORTFOLIO_ID
    assert row["actor"] == cli_actor
    assert row["source"] == "cli"
