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
import json
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest
import typer.main
from _fund_fakes import (
    ASOF,
    PORTFOLIO_ID,
    UNIVERSE,
    FreezeMidRunModel,
    PmFailsAfterOrderModel,
    ScriptedFundModel,
    expected_weights,
    make_constraint_set,
    seed_panel,
)
from _profiler_fakes import ScriptedProfilerModel, final_reply, make_answers
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
from fund.audit import (
    AgentRunRepository,
    AutonomyRepository,
    AutonomyScope,
    AutonomyState,
    MandateRepository,
    OrderRepository,
    put_constraint_set,
)
from fund.audit.human_audit_repository import HumanAuditLogRepository
from fund.config import FundConfig, settings
from fund.schemas.enums import GicsSector
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


def test_nav_mark_command_forwards_the_rebuild_flag(monkeypatch) -> None:
    import fund.scheduler as scheduler

    calls: list[dict[str, Any]] = []

    def _fake_run_nav_mark(**kwargs: Any) -> bool:
        calls.append(kwargs)
        return True

    monkeypatch.setattr(scheduler, "run_nav_mark", _fake_run_nav_mark)
    runner = CliRunner()

    assert runner.invoke(cli.app, ["nav-mark"]).exit_code == 0
    assert runner.invoke(cli.app, ["nav-mark", "--rebuild"]).exit_code == 0

    assert [c.get("full", False) for c in calls] == [False, True]


def test_nav_mark_command_exits_nonzero_when_the_slot_did_not_complete(
    monkeypatch,
) -> None:
    import fund.scheduler as scheduler

    monkeypatch.setattr(scheduler, "run_nav_mark", lambda **_: False)

    assert CliRunner().invoke(cli.app, ["nav-mark"]).exit_code == 1


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


@pytest.mark.parametrize("drift", ["0", "-0.5", "2", "2.5", "5", "nan", "inf"])
def test_mandate_set_rejects_an_unbreachable_or_non_positive_drift(
    harness, drift
) -> None:
    res = harness.runner.invoke(
        cli.app,
        ["mandate", "set", str(PORTFOLIO_ID), "--capital", "1000", "--drift", drift],
    )
    assert res.exit_code == 1, res.output
    assert "--drift" in res.output
    with harness.read() as session:
        assert observe.get_mandate(session, PORTFOLIO_ID) is None
        session.rollback()


@pytest.mark.parametrize("drift", ["0.0001", "0.05", "1.99"])
def test_mandate_set_accepts_drift_below_the_l1_maximum(harness, drift) -> None:
    res = harness.runner.invoke(
        cli.app,
        ["mandate", "set", str(PORTFOLIO_ID), "--capital", "1000", "--drift", drift],
    )
    assert res.exit_code == 0, res.output
    assert _stored_mandate(harness).drift_l1_threshold == float(drift)


def test_run_rejects_a_drift_that_could_never_be_breached(harness, monkeypatch) -> None:
    _seed(harness)
    _persist_mandate(harness, _rich_mandate())
    monkeypatch.setattr(
        cli, "build_primary", lambda *a, **k: ScriptedFundModel(universe=UNIVERSE)
    )
    res = harness.runner.invoke(
        cli.app, ["run", str(PORTFOLIO_ID), "--asof", _ASOF_STR, "--drift", "5"]
    )
    assert res.exit_code == 1, res.output
    assert "--drift" in res.output
    assert _stored_mandate(harness).drift_l1_threshold == 0.05  # untouched


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


def test_run_that_reached_no_gate_says_so_and_exits_nonzero(
    harness, monkeypatch
) -> None:
    expected = _seed(harness)
    monkeypatch.setattr(
        cli,
        "build_primary",
        lambda *a, **k: ScriptedFundModel(
            universe=UNIVERSE, weights=expected, risk_passes=False
        ),
    )
    _set_mandate(harness)

    res = harness.runner.invoke(
        cli.app, ["run", str(PORTFOLIO_ID), "--asof", _ASOF_STR]
    )

    assert res.exit_code == 1, res.output
    assert "incomplete" in res.output
    assert "HITL gate" not in res.output
    assert "awaiting approval" not in res.output
    with harness.read() as session:
        repo = AgentRunRepository(session)
        assert repo.list_paused_runs(PORTFOLIO_ID) == []
        (row,) = _audit_rows(harness, "run")
        assert row["run_id"] is not None
        assert str(row["run_id"]) in res.output
        session.rollback()


def test_run_frozen_mid_round_fails_without_claiming_a_gate(
    harness, monkeypatch
) -> None:
    """A freeze that lands while the round runs leaves no gate, so the run fails.

    Reporting success with a "(HITL gate)" suffix would send the operator to an
    approval queue the run never entered.
    """
    expected = _seed(harness)

    def freeze() -> None:
        with harness.read() as other:
            AutonomyRepository(other).set_state(
                AutonomyScope.FUND, None, AutonomyState.FROZEN, reason="killswitch"
            )
            other.commit()

    monkeypatch.setattr(
        cli,
        "build_primary",
        lambda *a, **k: FreezeMidRunModel(
            universe=UNIVERSE, weights=expected, on_first_call=freeze
        ),
    )
    _set_mandate(harness)

    res = harness.runner.invoke(
        cli.app, ["run", str(PORTFOLIO_ID), "--asof", _ASOF_STR]
    )

    assert res.exit_code == 1
    assert "HITL gate" not in res.output
    assert "rejected" in res.output
    with harness.read() as session:
        assert AgentRunRepository(session).list_paused_runs(PORTFOLIO_ID) == []


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

    resolved = harness.runner.invoke(cli.app, ["reject", str(run_id)])
    assert resolved.exit_code == 0, resolved.output
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


def test_run_refuses_while_a_paused_run_awaits_the_adviser(
    harness, monkeypatch
) -> None:
    run_id, _expected, _res = _seed_and_run(harness, monkeypatch)

    res = harness.runner.invoke(
        cli.app, ["run", str(PORTFOLIO_ID), "--asof", _ASOF_STR, "--drift", "0.07"]
    )

    assert res.exit_code == 1, res.output
    assert str(run_id) in res.output
    assert "approve" in res.output and "reject" in res.output
    with harness.read() as session:
        paused_ids = [
            r.id for r in AgentRunRepository(session).list_paused_runs(PORTFOLIO_ID)
        ]
        session.rollback()
    assert paused_ids == [run_id]  # no second run stacked
    assert len(_audit_rows(harness, "run")) == 1
    assert _stored_mandate(harness).drift_l1_threshold == 0.1  # flag not applied
    assert harness.pool.closed >= 2


def test_run_resumes_once_the_paused_run_is_resolved(harness, monkeypatch) -> None:
    run_id, _expected, _res = _seed_and_run(harness, monkeypatch)
    assert harness.runner.invoke(cli.app, ["reject", str(run_id)]).exit_code == 0

    res = harness.runner.invoke(
        cli.app, ["run", str(PORTFOLIO_ID), "--asof", _ASOF_STR]
    )

    assert res.exit_code == 0, res.output


def test_run_without_mandate_reports_the_error_exactly_once(
    harness, monkeypatch
) -> None:
    _seed(harness)
    monkeypatch.setattr(
        cli, "build_primary", lambda *a, **k: ScriptedFundModel(universe=UNIVERSE)
    )
    res = harness.runner.invoke(
        cli.app, ["run", str(PORTFOLIO_ID), "--asof", _ASOF_STR]
    )
    assert res.exit_code == 1
    error_lines = [ln for ln in res.output.splitlines() if ln.startswith("error:")]
    assert len(error_lines) == 1, res.output
    assert error_lines[0] != "error:" and "no mandate" in error_lines[0]
    assert harness.pool.closed >= 1  # pool still released on the early exit


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


def test_approve_refuses_a_run_older_than_a_completed_one(harness, monkeypatch) -> None:
    stale_id, expected, _res = _seed_and_run(harness, monkeypatch)
    with harness.read() as session:
        audit = AgentRunRepository(session)
        newer = audit.create_run(
            portfolio_id=PORTFOLIO_ID,
            asof=ASOF,
            seed=1,
            universe=[],
            optimizer_config={"step": "fund"},
        )
        audit.finalize_run(newer.id, weights=expected)
        # SQLite's CURRENT_TIMESTAMP has one-second resolution, so pin the ordering.
        audit.get_run(stale_id).started_at = datetime(2026, 10, 24, 6, 0)
        audit.get_run(newer.id).started_at = datetime(2026, 10, 26, 7, 0)
        session.commit()

    res = harness.runner.invoke(cli.app, ["approve", str(stale_id)])

    assert res.exit_code == 1, res.output
    assert "newer" in res.output
    with harness.read() as session:
        assert AgentRunRepository(session).get_run(stale_id).status == "paused"
        session.rollback()
    assert _audit_rows(harness, "approve") == []


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


def _ticket_lines(output: str) -> list[str]:
    return [ln for ln in output.splitlines() if ln.startswith("ticket ")]


def _placed_ticket_id(harness):
    with harness.read() as session:
        (order,) = OrderRepository(session).list_for_portfolio(PORTFOLIO_ID)
        session.rollback()
        return order.id


def test_report_shows_the_ticket_the_run_placed_not_the_portfolios_newest(
    harness, monkeypatch
) -> None:
    run_id, _expected, _res = _seed_and_run(harness, monkeypatch)
    assert harness.runner.invoke(cli.app, ["approve", str(run_id)]).exit_code == 0
    placed = _placed_ticket_id(harness)
    with harness.read() as session:
        # A later, unrelated ticket for the same portfolio (another run's approval).
        later = OrderRepository(session).create(
            portfolio_id=PORTFOLIO_ID,
            asof=ASOF + timedelta(days=30),
            weights_hash="later-run",
            weights={"AAA": 1.0},
            fill_date=ASOF + timedelta(days=31),
            lines=[],
            notional=777777.0,
            total_commission=0.0,
            total_slippage_cost=0.0,
        )
        # SQLite's CURRENT_TIMESTAMP has one-second resolution, so pin "newest".
        later.created_at = datetime(2099, 1, 1)
        session.commit()

    res = harness.runner.invoke(cli.app, ["report", str(run_id)])

    assert res.exit_code == 0, res.output
    (line,) = _ticket_lines(res.output)
    assert str(placed) in line
    assert "777777" not in res.output


def test_report_of_a_rejected_run_shows_no_ticket(harness, monkeypatch) -> None:
    first, _expected, _res = _seed_and_run(harness, monkeypatch)
    assert harness.runner.invoke(cli.app, ["approve", str(first)]).exit_code == 0
    assert _placed_ticket_id(harness)  # the portfolio now has a filled ticket
    again = harness.runner.invoke(
        cli.app, ["run", str(PORTFOLIO_ID), "--asof", _ASOF_STR]
    )
    assert again.exit_code == 0, again.output
    with harness.read() as session:
        (second,) = AgentRunRepository(session).list_paused_runs(PORTFOLIO_ID)
        second_id = second.id
    assert harness.runner.invoke(cli.app, ["reject", str(second_id)]).exit_code == 0

    res = harness.runner.invoke(cli.app, ["report", str(second_id)])

    assert res.exit_code == 0, res.output
    assert "rejected" in res.output
    assert _ticket_lines(res.output) == []
    assert "no ticket for this run" in res.output


def test_report_metrics_come_from_the_requested_run(harness, monkeypatch) -> None:
    first, expected, _res = _seed_and_run(harness, monkeypatch)
    with harness.read() as session:
        AgentRunRepository(session).append_decision(
            first,
            agent="allocator",
            step="optimize_portfolio",
            llm_response=json.dumps({"weights": expected, "metrics": {"sharpe": 1.5}}),
        )
        session.commit()
    assert harness.runner.invoke(cli.app, ["approve", str(first)]).exit_code == 0
    again = harness.runner.invoke(
        cli.app, ["run", str(PORTFOLIO_ID), "--asof", _ASOF_STR]
    )
    assert again.exit_code == 0, again.output
    with harness.read() as session:
        (second,) = AgentRunRepository(session).list_paused_runs(PORTFOLIO_ID)
        second_id = second.id

    own = harness.runner.invoke(cli.app, ["report", str(first)])
    other = harness.runner.invoke(cli.app, ["report", str(second_id)])

    assert "sharpe=1.5000" in own.output
    # The newest run reports the optimizer figures it logged itself, never the first
    # run's.
    assert "metrics:" in other.output
    assert "sharpe=1.5000" not in other.output


def test_report_labels_the_mandate_as_the_current_one(harness, monkeypatch) -> None:
    run_id, _expected, _res = _seed_and_run(harness, monkeypatch)

    res = harness.runner.invoke(cli.app, ["report", str(run_id)])

    assert "current mandate:" in res.output


def test_report_prints_the_optimizer_metrics_of_the_run(harness, monkeypatch) -> None:
    run_id, _expected, _res = _seed_and_run(harness, monkeypatch)
    harness.runner.invoke(cli.app, ["approve", str(run_id)])

    res = harness.runner.invoke(cli.app, ["report", str(run_id)])

    assert res.exit_code == 0, res.output
    assert "metrics:" in res.output
    assert "sharpe_ratio=" in res.output


def test_report_ignores_a_newer_runs_metrics(harness, monkeypatch) -> None:
    run_id, _expected, _res = _seed_and_run(harness, monkeypatch)
    harness.runner.invoke(cli.app, ["approve", str(run_id)])
    with harness.read() as session:
        repo = AgentRunRepository(session)
        newer = repo.create_run(
            portfolio_id=PORTFOLIO_ID,
            asof=ASOF,
            seed=2,
            universe=UNIVERSE,
            optimizer_config={},
            status="paused",
        )
        repo.append_decision(
            newer.id,
            agent="allocator",
            step="optimize_portfolio",
            llm_response=json.dumps(
                {"weights": {"AAA": 1.0}, "metrics": {"sharpe_ratio": 123.456}}
            ),
        )
        session.commit()

    res = harness.runner.invoke(cli.app, ["report", str(run_id)])

    assert res.exit_code == 0, res.output
    assert "123.4560" not in res.output


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


def test_profile_blocked_by_a_suitability_breach_is_recorded_and_exits_nonzero(
    harness, monkeypatch
) -> None:
    model = make_profiler_model(
        make_answers(exclusions=tuple(GicsSector)), portfolio_id=str(PORTFOLIO_ID)
    )
    monkeypatch.setattr(cli, "build_primary", lambda *a, **k: model)

    res = harness.runner.invoke(
        cli.app,
        ["profile", str(PORTFOLIO_ID), "--answers", "Growth over ten years, EUR."],
    )

    assert res.exit_code == 1
    assert "blocked" in res.output
    assert "awaiting approval" not in res.output
    with harness.read() as session:
        (run,) = AgentRunRepository(session).list_runs_for_portfolio(PORTFOLIO_ID)
        assert run.status == "blocked"
    (row,) = _audit_rows(harness, "profile")
    assert row["new_value"]["outcome"] == "blocked"


def test_profile_that_never_reaches_the_gate_fails(harness, monkeypatch) -> None:
    model = ScriptedProfilerModel(
        structured_outcomes=[make_answers()], chat_responses=[final_reply()]
    )
    monkeypatch.setattr(cli, "build_primary", lambda *a, **k: model)

    res = harness.runner.invoke(
        cli.app,
        ["profile", str(PORTFOLIO_ID), "--answers", "Growth over ten years, EUR."],
    )

    assert res.exit_code == 1
    assert "awaiting approval" not in res.output
    assert "approval gate" in res.output
