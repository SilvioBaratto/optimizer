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
import re
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
    MEAN_RISK,
    PORTFOLIO_ID,
    UNIVERSE,
    FreezeMidRunModel,
    PmFailsAfterOrderModel,
    ScriptedFundModel,
    audit_candidate_set,
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
from fund.audit.orders_repository import weights_hash
from fund.config import FundConfig, settings
from fund.schemas.candidates import CANDIDATE_IDS
from fund.schemas.enums import GicsSector
from fund.schemas.mandate import PortfolioMandate, RunTriggers

_ASOF_STR = ASOF.isoformat()
# A rebalance gate is approved by naming one candidate; the fixture's profile maps
# the old single optimizer onto ``mean_risk``.
_CANDIDATE = ("--candidate", MEAN_RISK)
_OTHER_THAN_MEAN_RISK = tuple(cid for cid in CANDIDATE_IDS if cid != MEAN_RISK)
_REPORT_CANDIDATES_HEADING = "candidates (listed by id; the adviser chooses):"
# The same words the fund-run language guard refuses: the report must not rank either.
_RECOMMENDATION = re.compile(
    r"recommend|best|prefer|suggest|optimal|ideal|winner|matching the (client|profile)",
    re.IGNORECASE,
)


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
        eligible_tickers=tuple(UNIVERSE),
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


def test_mandate_set_exposes_the_eligible_options() -> None:
    click_app = typer.main.get_command(cli.app)
    command = click_app.commands["mandate"].commands["set"]  # type: ignore[attr-defined]
    options = {opt for param in command.params for opt in param.opts}
    assert {"--eligible", "--clear-eligible"} <= options
    eligible = next(p for p in command.params if "--eligible" in p.opts)
    assert eligible.multiple is True


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


def test_mandate_set_eligible_stores_the_repeated_flags_in_order(harness) -> None:
    _set_mandate(harness)

    res = harness.runner.invoke(
        cli.app,
        [
            *["mandate", "set", str(PORTFOLIO_ID)],
            *["--eligible", "vwce.de", "--eligible", "IBGS.L"],
        ],
    )

    assert res.exit_code == 0, res.output
    assert _stored_mandate(harness).eligible_tickers == ("VWCE.DE", "IBGS.L")


def test_mandate_set_eligible_replaces_rather_than_appends(harness) -> None:
    _persist_mandate(harness, _rich_mandate())

    res = harness.runner.invoke(
        cli.app, ["mandate", "set", str(PORTFOLIO_ID), "--eligible", "CCC"]
    )

    assert res.exit_code == 0, res.output
    assert _stored_mandate(harness).eligible_tickers == ("CCC",)


def test_mandate_set_clear_eligible_empties_the_allowlist(harness) -> None:
    _persist_mandate(harness, _rich_mandate())

    res = harness.runner.invoke(
        cli.app, ["mandate", "set", str(PORTFOLIO_ID), "--clear-eligible"]
    )

    assert res.exit_code == 0, res.output
    assert _stored_mandate(harness).eligible_tickers == ()


def test_mandate_set_rejects_eligible_together_with_clear_eligible(harness) -> None:
    seeded = _rich_mandate()
    _persist_mandate(harness, seeded)

    res = harness.runner.invoke(
        cli.app,
        ["mandate", "set", str(PORTFOLIO_ID), "--eligible", "CCC", "--clear-eligible"],
    )

    assert res.exit_code != 0
    assert "--clear-eligible" in res.output
    assert _stored_mandate(harness) == seeded


def test_mandate_set_surfaces_an_invalid_eligible_ticker_cleanly(harness) -> None:
    _persist_mandate(harness, _rich_mandate())

    res = harness.runner.invoke(
        cli.app, ["mandate", "set", str(PORTFOLIO_ID), "--eligible", "X" * 33]
    )

    assert res.exit_code != 0
    assert "eligible_tickers" in res.output


def test_mandate_show_lists_the_eligible_tickers(harness) -> None:
    _persist_mandate(harness, _rich_mandate())

    res = harness.runner.invoke(cli.app, ["mandate", "show", str(PORTFOLIO_ID)])

    assert res.exit_code == 0, res.output
    assert "eligible:   AAA, BBB, CCC" in res.output


def test_mandate_show_prints_a_dash_for_an_unrestricted_mandate(harness) -> None:
    _set_mandate(harness)

    res = harness.runner.invoke(cli.app, ["mandate", "show", str(PORTFOLIO_ID)])

    assert res.exit_code == 0, res.output
    assert "eligible:   -" in res.output


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


def test_run_with_override_flags_keeps_remembered_directives_and_eligible_tickers(
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


@pytest.mark.criterion("REQ-FUND-053")
def test_approve_finalizes_run(harness, monkeypatch) -> None:
    """Validates fund.cli.approve."""
    run_id, expected, _res = _seed_and_run(harness, monkeypatch)
    res = harness.runner.invoke(cli.app, ["approve", str(run_id), *_CANDIDATE])
    assert res.exit_code == 0, res.output
    with harness.read() as session:
        row = AgentRunRepository(session).get_run(run_id)
        assert row.status == "completed"
        assert row.weights == expected
    assert harness.pool.closed >= 2  # run + approve each closed the pool


@pytest.mark.criterion("REQ-FUND-053")
def test_approve_without_a_fill_bar_is_refused_and_the_run_stays_paused(
    harness, monkeypatch
) -> None:
    """Validates fund.cli.approve.

    No close after ``asof`` yet: the approve fails before the graph resumes, so the
    run stays paused at its live gate and the operator can approve again later."""
    run_id, _expected, _res = _seed_and_run(harness, monkeypatch)
    with harness.read() as session:
        session.query(PriceHistory).filter(PriceHistory.date > ASOF).delete()
        session.commit()

    res = harness.runner.invoke(cli.app, ["approve", str(run_id), *_CANDIDATE])

    assert res.exit_code != 0
    assert "no paper ticket" in res.output
    with harness.read() as session:
        assert AgentRunRepository(session).get_run(run_id).status == "paused"


@pytest.mark.criterion("REQ-FUND-053")
def test_approve_without_a_ticket_says_so_and_exits_nonzero(
    harness, monkeypatch
) -> None:
    """Validates fund.cli.approve.

    ``place_orders`` finding no fill bar finalises the run incomplete with no
    ticket: the CLI must not echo a bare "approved" nor exit 0 for it."""
    from fund.agents import graph as graph_mod

    monkeypatch.setattr(graph_mod, "_require_fill_bars", lambda *_a, **_k: None)
    run_id, _expected, _res = _seed_and_run(harness, monkeypatch)
    with harness.read() as session:
        session.query(PriceHistory).filter(PriceHistory.date > ASOF).delete()
        session.commit()

    res = harness.runner.invoke(cli.app, ["approve", str(run_id), *_CANDIDATE])

    assert res.exit_code != 0
    assert "no paper ticket" in res.output
    with harness.read() as session:
        assert AgentRunRepository(session).get_run(run_id).status == "incomplete"


@pytest.mark.criterion("REQ-FUND-053")
def test_approve_surfaces_a_failed_pm_follow_up(harness, monkeypatch) -> None:
    """Validates fund.cli.approve.

    A PM turn that dies after the gate was consumed leaves the decision standing
    (exit 0, run completed) but the operator must be told the model did not finish."""
    run_id, expected, _res = _seed_and_run(harness, monkeypatch)
    monkeypatch.setattr(
        cli,
        "build_primary",
        lambda *a, **k: PmFailsAfterOrderModel(universe=UNIVERSE, weights=expected),
    )

    res = harness.runner.invoke(cli.app, ["approve", str(run_id), *_CANDIDATE])

    assert res.exit_code == 0, res.output
    assert "PM follow-up failed" in res.output
    with harness.read() as session:
        assert AgentRunRepository(session).get_run(run_id).status == "completed"


@pytest.mark.criterion("REQ-FUND-053")
@pytest.mark.parametrize("verb", ["approve", "reject"])
def test_resolving_a_run_appends_a_cli_audit_row(
    harness, monkeypatch, cli_actor, verb
) -> None:
    """Validates fund.cli.approve and fund.cli.reject."""
    run_id, _expected, _res = _seed_and_run(harness, monkeypatch)
    candidate = _CANDIDATE if verb == "approve" else ()
    res = harness.runner.invoke(cli.app, [verb, str(run_id), *candidate])
    assert res.exit_code == 0, res.output
    (row,) = _audit_rows(harness, verb)
    assert row["run_id"] == run_id
    assert row["portfolio_id"] == PORTFOLIO_ID
    assert row["actor"] == cli_actor
    assert row["source"] == "cli"
    if verb == "approve":
        assert row["idempotency_key"].startswith(f"{run_id}:approve:")
    else:
        assert row["idempotency_key"] == f"{run_id}:reject"


@pytest.mark.criterion("REQ-FUND-053")
def test_repeated_approve_replays_instead_of_resuming_again(
    harness, monkeypatch
) -> None:
    """Validates fund.cli.approve."""
    run_id, _expected, _res = _seed_and_run(harness, monkeypatch)
    first = harness.runner.invoke(cli.app, ["approve", str(run_id), *_CANDIDATE])
    again = harness.runner.invoke(cli.app, ["approve", str(run_id), *_CANDIDATE])
    assert first.exit_code == 0, first.output
    assert again.exit_code == 0, again.output
    assert "already applied" in again.output
    assert len(_audit_rows(harness, "approve")) == 1


@pytest.mark.criterion("REQ-FUND-053")
def test_second_approve_resumes_a_run_that_regated(harness, monkeypatch) -> None:
    """Validates fund.cli.approve.

    An approve whose placement is refused re-gates the run; approving that new gate
    must resume it, not replay the first approve's audit row."""
    from fund.agents import graph as graph_mod

    run_id, expected, _res = _seed_and_run(harness, monkeypatch)
    monkeypatch.setattr(graph_mod, "_require_fill_bars", lambda *_a, **_k: None)
    monkeypatch.setattr(
        cli,
        "build_primary",
        lambda *a, **k: ScriptedFundModel(
            universe=UNIVERSE, weights=expected, max_orders=3
        ),
    )
    with harness.read() as session:
        bars = [
            {col.name: getattr(row, col.name) for col in PriceHistory.__table__.columns}
            for row in session.query(PriceHistory).filter(PriceHistory.date > ASOF)
        ]
        session.query(PriceHistory).filter(PriceHistory.date > ASOF).delete()
        session.commit()

    first = harness.runner.invoke(cli.app, ["approve", str(run_id), *_CANDIDATE])
    assert first.exit_code != 0
    assert "no paper ticket" in first.output
    with harness.read() as session:
        assert AgentRunRepository(session).get_run(run_id).status == "responded"
        session.execute(PriceHistory.__table__.insert(), bars)
        session.commit()

    second = harness.runner.invoke(cli.app, ["approve", str(run_id), *_CANDIDATE])

    assert second.exit_code == 0, second.output
    assert "already applied" not in second.output
    with harness.read() as session:
        assert AgentRunRepository(session).get_run(run_id).status == "completed"
    keys = [row["idempotency_key"] for row in _audit_rows(harness, "approve")]
    assert len(set(keys)) == 2


@pytest.mark.criterion("REQ-FUND-053")
def test_approve_refuses_a_run_older_than_a_completed_one(harness, monkeypatch) -> None:
    """Validates fund.cli.approve."""
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

    res = harness.runner.invoke(cli.app, ["approve", str(stale_id), *_CANDIDATE])

    assert res.exit_code == 1, res.output
    assert "newer" in res.output
    with harness.read() as session:
        assert AgentRunRepository(session).get_run(stale_id).status == "paused"
        session.rollback()
    assert _audit_rows(harness, "approve") == []


@pytest.mark.criterion("REQ-FUND-053")
def test_approve_requires_the_candidate_flag_for_a_rebalance_gate(
    harness, monkeypatch
) -> None:
    """Validates fund.cli.approve: a rebalance gate approved without --candidate is
    refused with the ids to choose from, exits 1 and leaves the gate open."""
    run_id, _expected, _res = _seed_and_run(harness, monkeypatch)

    res = harness.runner.invoke(cli.app, ["approve", str(run_id)])

    assert res.exit_code == 1, res.output
    assert "requires a candidate id" in res.output
    for candidate_id in CANDIDATE_IDS:
        assert candidate_id in res.output
    with harness.read() as session:
        assert AgentRunRepository(session).get_run(run_id).status == "paused"
        assert OrderRepository(session).list_for_portfolio(PORTFOLIO_ID) == []
        session.rollback()
    assert _audit_rows(harness, "approve") == []


@pytest.mark.criterion("REQ-FUND-053")
def test_approve_with_an_unknown_candidate_is_refused(harness, monkeypatch) -> None:
    """Validates fund.cli.approve: an id outside the set is refused by name."""
    run_id, _expected, _res = _seed_and_run(harness, monkeypatch)

    res = harness.runner.invoke(
        cli.app, ["approve", str(run_id), "--candidate", "max_sharpe"]
    )

    assert res.exit_code == 1, res.output
    assert "'max_sharpe' is not in the run's candidate set" in res.output
    with harness.read() as session:
        assert AgentRunRepository(session).get_run(run_id).status == "paused"
        session.rollback()
    assert _audit_rows(harness, "approve") == []


@pytest.mark.criterion("REQ-FUND-053")
@pytest.mark.parametrize("candidate_id", _OTHER_THAN_MEAN_RISK)
def test_approve_with_candidate_places_that_candidate(
    harness, monkeypatch, candidate_id
) -> None:
    """Validates fund.cli.approve: the run, its ticket and the audit row carry the
    named candidate's weights, not the questionnaire's mean-risk ones."""
    run_id, _expected, _res = _seed_and_run(harness, monkeypatch)
    with harness.read() as session:
        named = observe.candidate_set(session, run_id).candidate(candidate_id)
        session.rollback()

    res = harness.runner.invoke(
        cli.app, ["approve", str(run_id), "--candidate", candidate_id]
    )

    assert res.exit_code == 0, res.output
    with harness.read() as session:
        run = AgentRunRepository(session).get_run(run_id)
        assert run.status == "completed"
        assert run.weights == named.weights
        (order,) = OrderRepository(session).list_for_portfolio(PORTFOLIO_ID)
        assert order.weights_hash == weights_hash(named.weights)
        session.rollback()
    (row,) = _audit_rows(harness, "approve")
    assert row["new_value"]["candidate_id"] == candidate_id


@pytest.mark.criterion("REQ-FUND-053")
def test_profile_gate_approve_takes_no_candidate(harness, monkeypatch) -> None:
    """Validates fund.cli.approve: --candidate on a profile gate is refused and the
    gate stays open; the same gate approves without one."""
    run_id = _profile_to_gate(harness, monkeypatch)

    named = harness.runner.invoke(
        cli.app, ["approve", str(run_id), "--candidate", MEAN_RISK]
    )

    assert named.exit_code == 1, named.output
    assert "profiler gate" in named.output
    with harness.read() as session:
        assert AgentRunRepository(session).get_run(run_id).status == "paused"
        session.rollback()

    bare = harness.runner.invoke(cli.app, ["approve", str(run_id)])

    assert bare.exit_code == 0, bare.output
    with harness.read() as session:
        assert AgentRunRepository(session).get_run(run_id).status == "completed"
        session.rollback()


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


def test_status_labels_responded_run_in_both_views(harness, monkeypatch) -> None:
    # A barge-in re-gate leaves the run "responded" yet still awaiting approval; the
    # global queue must name that status and the per-portfolio view must flag hitl.
    run_id, _expected, _res = _seed_and_run(harness, monkeypatch)
    with harness.read() as session:
        AgentRunRepository(session).mark_responded(run_id)
        session.commit()

    global_res = harness.runner.invoke(cli.app, ["status"])
    scoped_res = harness.runner.invoke(cli.app, ["status", str(PORTFOLIO_ID)])

    assert global_res.exit_code == 0, global_res.output
    (global_line,) = [ln for ln in global_res.output.splitlines() if str(run_id) in ln]
    assert global_line.split("\t")[1] == "responded"
    assert scoped_res.exit_code == 0, scoped_res.output
    (scoped_line,) = [ln for ln in scoped_res.output.splitlines() if str(run_id) in ln]
    assert scoped_line.split("\t")[1] == "responded"
    assert scoped_line.endswith("hitl=True")


@pytest.mark.criterion("REQ-FUND-049")
def test_report_renders_after_approve(harness, monkeypatch) -> None:
    """Validates fund.cli.report."""
    run_id, expected, _res = _seed_and_run(harness, monkeypatch)
    harness.runner.invoke(cli.app, ["approve", str(run_id), *_CANDIDATE])
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


@pytest.mark.criterion("REQ-FUND-049")
def test_report_shows_the_ticket_the_run_placed_not_the_portfolios_newest(
    harness, monkeypatch
) -> None:
    """Validates fund.cli.report."""
    run_id, _expected, _res = _seed_and_run(harness, monkeypatch)
    assert (
        harness.runner.invoke(cli.app, ["approve", str(run_id), *_CANDIDATE]).exit_code
        == 0
    )
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


@pytest.mark.criterion("REQ-FUND-049")
def test_report_of_a_rejected_run_shows_no_ticket(harness, monkeypatch) -> None:
    """Validates fund.cli.report."""
    first, _expected, _res = _seed_and_run(harness, monkeypatch)
    assert (
        harness.runner.invoke(cli.app, ["approve", str(first), *_CANDIDATE]).exit_code
        == 0
    )
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


@pytest.mark.criterion("REQ-FUND-049")
def test_report_metrics_come_from_the_requested_run(harness, monkeypatch) -> None:
    """Validates fund.cli.report with fund.observe.run_metrics: an approved run
    reports the measures of the candidate the adviser chose for it, and a newer run
    still awaiting its choice has no metrics line (its candidates are listed, each
    with its own measures)."""
    first, _expected, _res = _seed_and_run(harness, monkeypatch)
    assert (
        harness.runner.invoke(cli.app, ["approve", str(first), *_CANDIDATE]).exit_code
        == 0
    )
    with harness.read() as session:
        chosen = observe.candidate_set(session, first).candidate(MEAN_RISK)
        volatility = chosen.metrics.annualized_volatility
        session.rollback()
    again = harness.runner.invoke(
        cli.app, ["run", str(PORTFOLIO_ID), "--asof", _ASOF_STR]
    )
    assert again.exit_code == 0, again.output
    with harness.read() as session:
        (second,) = AgentRunRepository(session).list_paused_runs(PORTFOLIO_ID)
        second_id = second.id

    own = harness.runner.invoke(cli.app, ["report", str(first)])
    other = harness.runner.invoke(cli.app, ["report", str(second_id)])

    (own_metrics,) = [ln for ln in own.output.splitlines() if ln.startswith("metrics:")]
    assert f"annualized_volatility={volatility:.4f}" in own_metrics
    # The newer run has no adviser choice yet, so no figures are reported as its
    # metrics, never the first run's.
    assert not any(ln.startswith("metrics:") for ln in other.output.splitlines())


def test_report_labels_the_mandate_as_the_current_one(harness, monkeypatch) -> None:
    run_id, _expected, _res = _seed_and_run(harness, monkeypatch)

    res = harness.runner.invoke(cli.app, ["report", str(run_id)])

    assert "current mandate:" in res.output


@pytest.mark.criterion("REQ-FUND-049")
def test_report_prints_the_optimizer_metrics_of_the_run(harness, monkeypatch) -> None:
    """Validates fund.cli.report."""
    run_id, _expected, _res = _seed_and_run(harness, monkeypatch)
    harness.runner.invoke(cli.app, ["approve", str(run_id), *_CANDIDATE])

    res = harness.runner.invoke(cli.app, ["report", str(run_id)])

    assert res.exit_code == 0, res.output
    assert "metrics:" in res.output
    # The chosen candidate's neutral measures are the run's metrics.
    assert "annualized_volatility=" in res.output


@pytest.mark.criterion("REQ-FUND-049")
def test_report_ignores_a_newer_runs_metrics(harness, monkeypatch) -> None:
    """Validates fund.cli.report."""
    run_id, _expected, _res = _seed_and_run(harness, monkeypatch)
    harness.runner.invoke(cli.app, ["approve", str(run_id), *_CANDIDATE])
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


def _candidate_lines(output: str) -> list[str]:
    """The report's candidate rows, in the order printed."""
    return [
        line
        for line in output.splitlines()
        if any(line.startswith(f"  {cid}  ") for cid in CANDIDATE_IDS)
    ]


@pytest.mark.criterion("REQ-FUND-053")
def test_report_lists_candidates_in_id_order_with_metrics_and_the_choice(
    harness, monkeypatch
) -> None:
    """Validates fund.cli._report_candidate_lines: every candidate in id order with
    its label, feasibility, five measures and check, then the adviser's choice."""
    run_id, _expected, _res = _seed_and_run(harness, monkeypatch)
    chosen = "min_risk"
    approved = harness.runner.invoke(
        cli.app, ["approve", str(run_id), "--candidate", chosen]
    )
    assert approved.exit_code == 0, approved.output
    with harness.read() as session:
        candidate_set = observe.candidate_set(session, run_id)
        session.rollback()

    res = harness.runner.invoke(cli.app, ["report", str(run_id)])

    assert res.exit_code == 0, res.output
    assert _REPORT_CANDIDATES_HEADING in res.output
    rows = _candidate_lines(res.output)
    assert [row.split()[0] for row in rows] == list(CANDIDATE_IDS)
    for row, candidate in zip(rows, candidate_set.candidates, strict=True):
        metrics = candidate.metrics
        assert candidate.label in row
        assert "  feasible  " in row
        assert f"annualized_volatility={metrics.annualized_volatility:.4f}" in row
        assert f"cvar_95={metrics.cvar_95:.4f}" in row
        assert f"largest_position={metrics.largest_position:.4f}" in row
        assert f"n_positions={metrics.n_positions}" in row
        assert f"turnover={metrics.turnover:.4f}" in row
        assert row.endswith("check=passed")
    assert f"chosen by the adviser: {chosen}" in res.output
    assert res.output.index(rows[-1]) < res.output.index("chosen by the adviser")


@pytest.mark.criterion("REQ-FUND-053")
def test_report_of_a_paused_run_lists_the_candidates_and_no_choice(
    harness, monkeypatch
) -> None:
    """Validates fund.cli._report_candidate_lines: before the adviser chooses, the
    candidates are listed and no choice is claimed."""
    run_id, _expected, _res = _seed_and_run(harness, monkeypatch)

    res = harness.runner.invoke(cli.app, ["report", str(run_id)])

    assert res.exit_code == 0, res.output
    assert len(_candidate_lines(res.output)) == len(CANDIDATE_IDS)
    assert "chosen by the adviser" not in res.output


@pytest.mark.criterion("REQ-FUND-053")
def test_report_shows_an_infeasible_candidate_with_its_breach(
    harness, monkeypatch
) -> None:
    """Validates fund.cli._report_candidate_lines: an infeasible candidate keeps its
    row, names its breach and shows no measures."""
    run_id, _expected, _res = _seed_and_run(harness, monkeypatch)
    with harness.read() as session:
        audit_candidate_set(session, run_id, {"AAA": 0.5, "CCC": 0.5})
        session.commit()

    res = harness.runner.invoke(cli.app, ["report", str(run_id)])

    rows = _candidate_lines(res.output)
    (hrp,) = (row for row in rows if row.startswith("  hrp  "))
    assert "infeasible: portfolio (solver_failed: not fitted by this test)" in hrp
    assert "annualized_volatility=—" in hrp
    assert hrp.endswith("check=—")


@pytest.mark.criterion("REQ-FUND-053")
def test_report_of_a_run_without_a_candidate_set_lists_none(
    harness, monkeypatch
) -> None:
    """Validates fund.cli._report_candidate_lines: a profile run has no candidates."""
    run_id = _profile_to_gate(harness, monkeypatch)

    res = harness.runner.invoke(cli.app, ["report", str(run_id)])

    assert res.exit_code == 0, res.output
    assert _REPORT_CANDIDATES_HEADING not in res.output
    assert _candidate_lines(res.output) == []


@pytest.mark.criterion("REQ-FUND-053")
def test_report_text_has_no_recommendation_language(harness, monkeypatch) -> None:
    """Validates fund.cli.report: neither a paused nor an approved run's report
    ranks or recommends a candidate."""
    run_id, _expected, _res = _seed_and_run(harness, monkeypatch)
    paused = harness.runner.invoke(cli.app, ["report", str(run_id)])
    approve = harness.runner.invoke(cli.app, ["approve", str(run_id), *_CANDIDATE])
    assert approve.exit_code == 0, approve.output
    approved = harness.runner.invoke(cli.app, ["report", str(run_id)])

    assert f"chosen by the adviser: {MEAN_RISK}" in approved.output
    for output in (paused.output, approved.output):
        assert _REPORT_CANDIDATES_HEADING in output
        assert _RECOMMENDATION.search(output) is None, _RECOMMENDATION.search(output)


# --- profile ----------------------------------------------------------------


def _profile_to_gate(harness, monkeypatch) -> uuid.UUID:
    """Drive ``profile`` to its save_profile gate and make the resume model a plain
    closing turn; return the paused profiler run's id."""
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
    monkeypatch.setattr(
        cli,
        "build_primary",
        lambda *a, **k: ScriptedProfilerModel(
            chat_responses=[final_reply(), final_reply()]
        ),
    )
    with harness.read() as session:
        (paused,) = AgentRunRepository(session).list_paused_runs(PORTFOLIO_ID)
        session.rollback()
        return paused.id


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
    with harness.read() as session:
        assert AgentRunRepository(session).list_paused_runs(PORTFOLIO_ID) == []


def _seed_nav_for(harness) -> None:
    from datetime import date
    from decimal import Decimal

    from fund.audit.nav_history_repository import NavHistoryRepository

    with harness.read() as session:
        NavHistoryRepository(session).upsert_navs(
            [
                {
                    "portfolio_id": PORTFOLIO_ID,
                    "date": date(2026, 1, 2),
                    "nav": Decimal("100000"),
                    "benchmark": None,
                }
            ]
        )
        session.commit()


def test_mandate_set_refuses_to_rebase_a_marked_portfolio(harness) -> None:
    _set_mandate(harness)  # EUR
    _seed_nav_for(harness)

    res = harness.runner.invoke(
        cli.app,
        [
            "mandate",
            "set",
            str(PORTFOLIO_ID),
            "--capital",
            "100000",
            "--base-currency",
            "USD",
        ],
    )

    assert res.exit_code == 1
    assert "base currency" in res.output.lower()
    assert "Traceback" not in res.output
    with harness.read() as session:
        assert observe.get_mandate(session, PORTFOLIO_ID).base_currency == "EUR"


def test_run_refuses_to_rebase_a_marked_portfolio(harness, monkeypatch) -> None:
    monkeypatch.setattr(cli, "build_primary", lambda *a, **k: object())
    _set_mandate(harness)  # EUR
    _seed_nav_for(harness)

    res = harness.runner.invoke(
        cli.app,
        [
            "run",
            str(PORTFOLIO_ID),
            "--asof",
            _ASOF_STR,
            "--base-currency",
            "USD",
        ],
    )

    assert res.exit_code == 1
    assert "base currency" in res.output.lower()
    with harness.read() as session:
        assert observe.get_mandate(session, PORTFOLIO_ID).base_currency == "EUR"


@pytest.mark.parametrize("bad", ["5", "2.5", "0", "-1"])
def test_mandate_set_rejects_an_out_of_range_drift_cleanly(harness, bad) -> None:
    res = harness.runner.invoke(
        cli.app,
        ["mandate", "set", str(PORTFOLIO_ID), "--capital", "100000", "--drift", bad],
    )

    assert res.exit_code == 1
    assert "--drift" in res.output
    assert "Traceback" not in res.output
    with harness.read() as session:
        assert observe.get_mandate(session, PORTFOLIO_ID) is None


def test_mandate_set_drift_help_states_the_upper_bound() -> None:
    click_app = typer.main.get_command(cli.app)
    set_cmd = click_app.commands["mandate"].commands["set"]  # type: ignore[attr-defined]
    drift = next(p for p in set_cmd.params if p.name == "drift")

    assert "(0, 2)" in drift.help  # the bound is open: 2 can never be breached


def _mandate_set_args(benchmark: str) -> list[str]:
    return [
        "mandate",
        "set",
        str(PORTFOLIO_ID),
        "--capital",
        "1000",
        "--benchmark",
        benchmark,
    ]


def test_mandate_set_warns_about_a_benchmark_it_cannot_price(harness) -> None:
    _seed(harness)

    res = harness.runner.invoke(
        cli.app,
        _mandate_set_args("SPY"),
    )

    assert res.exit_code == 0, res.output
    assert "warning" in res.output.lower()
    assert "SPY" in res.output
    assert _stored_mandate(harness).benchmark == "SPY"


def test_mandate_set_warns_about_an_eligible_ticker_it_cannot_price(harness) -> None:
    _seed(harness)

    res = harness.runner.invoke(
        cli.app,
        [*_mandate_set_args("AAA"), "--eligible", "AAA", "--eligible", "NEWETF.L"],
    )

    assert res.exit_code == 0, res.output
    assert "warning" in res.output.lower()
    assert "NEWETF.L" in res.output
    assert _stored_mandate(harness).eligible_tickers == ("AAA", "NEWETF.L")


def test_mandate_set_stores_a_known_benchmark_in_its_canonical_case(harness) -> None:
    _seed(harness)

    res = harness.runner.invoke(
        cli.app,
        _mandate_set_args("aaa"),
    )

    assert res.exit_code == 0, res.output
    assert "warning" not in res.output.lower()
    assert _stored_mandate(harness).benchmark == "AAA"


# --- profile answers: keep client data out of argv ---------------------------


def _profile_model(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "build_primary",
        lambda *a, **k: make_profiler_model(
            make_answers(), portfolio_id=str(PORTFOLIO_ID)
        ),
    )


def _spy_questionnaire(monkeypatch) -> list[Any]:
    """Record the ``questionnaire`` each ``apply_action`` call gets, then delegate."""
    seen: list[Any] = []
    real = cli.apply_action

    def spy(action, **kwargs):
        seen.append(kwargs.get("questionnaire"))
        return real(action, **kwargs)

    monkeypatch.setattr(cli, "apply_action", spy)
    return seen


def test_profile_reads_the_answers_from_a_file(harness, monkeypatch, tmp_path) -> None:
    _profile_model(monkeypatch)
    seen = _spy_questionnaire(monkeypatch)
    answers = tmp_path / "answers.txt"
    answers.write_text("Growth over ten years, EUR.\n", encoding="utf-8")

    res = harness.runner.invoke(
        cli.app, ["profile", str(PORTFOLIO_ID), "--answers-file", str(answers)]
    )

    assert res.exit_code == 0, res.output
    assert seen == ["Growth over ten years, EUR."]


def test_profile_reads_the_answers_from_stdin(harness, monkeypatch) -> None:
    _profile_model(monkeypatch)
    seen = _spy_questionnaire(monkeypatch)

    res = harness.runner.invoke(
        cli.app,
        ["profile", str(PORTFOLIO_ID), "--answers-file", "-"],
        input="Growth over ten years, EUR.\n",
    )

    assert res.exit_code == 0, res.output
    assert seen == ["Growth over ten years, EUR."]


def test_profile_needs_exactly_one_source_for_the_answers(
    harness, monkeypatch, tmp_path
) -> None:
    _profile_model(monkeypatch)
    answers = tmp_path / "answers.txt"
    answers.write_text("Growth.", encoding="utf-8")

    neither = harness.runner.invoke(cli.app, ["profile", str(PORTFOLIO_ID)])
    both = harness.runner.invoke(
        cli.app,
        [
            "profile",
            str(PORTFOLIO_ID),
            "--answers",
            "Growth.",
            "--answers-file",
            str(answers),
        ],
    )

    for res in (neither, both):
        assert res.exit_code != 0
        assert "--answers" in res.output
        assert "--answers-file" in res.output


def test_profile_rejects_an_unreadable_or_empty_answers_file(
    harness, monkeypatch, tmp_path
) -> None:
    _profile_model(monkeypatch)
    empty = tmp_path / "empty.txt"
    empty.write_text("  \n", encoding="utf-8")

    missing = harness.runner.invoke(
        cli.app,
        ["profile", str(PORTFOLIO_ID), "--answers-file", str(tmp_path / "nope.txt")],
    )
    blank = harness.runner.invoke(
        cli.app, ["profile", str(PORTFOLIO_ID), "--answers-file", str(empty)]
    )

    assert missing.exit_code != 0
    assert blank.exit_code != 0


# --- fallback model ---------------------------------------------------------


def _spy_apply_action(monkeypatch) -> dict[str, Any]:
    """Record the ``fallback`` each ``apply_action`` call receives, then delegate."""
    seen: dict[str, Any] = {}
    real = cli.apply_action

    def spy(action, **kwargs):
        seen[action] = kwargs.get("fallback")
        return real(action, **kwargs)

    monkeypatch.setattr(cli, "apply_action", spy)
    return seen


def test_run_builds_the_fallback_model_and_hands_it_to_the_round(
    harness, monkeypatch
) -> None:
    fallback = ScriptedFundModel(universe=UNIVERSE)
    monkeypatch.setattr(cli, "build_fallback", lambda *_a, **_k: fallback)
    seen = _spy_apply_action(monkeypatch)

    _seed_and_run(harness, monkeypatch)

    assert seen["run"] is fallback


@pytest.mark.criterion("REQ-FUND-053")
def test_approve_builds_the_fallback_model_and_hands_it_to_the_resume(
    harness, monkeypatch
) -> None:
    """Validates fund.cli.approve."""
    run_id, _expected, _res = _seed_and_run(harness, monkeypatch)
    fallback = ScriptedFundModel(universe=UNIVERSE)
    monkeypatch.setattr(cli, "build_fallback", lambda *_a, **_k: fallback)
    seen = _spy_apply_action(monkeypatch)

    harness.runner.invoke(cli.app, ["approve", str(run_id), *_CANDIDATE])

    assert seen["approve"] is fallback


def test_a_fallback_that_cannot_be_built_does_not_block_the_command(
    harness, monkeypatch
) -> None:
    def unbuildable(*_a, **_k):
        raise RuntimeError("no fallback key")

    monkeypatch.setattr(cli, "build_fallback", unbuildable)
    seen = _spy_apply_action(monkeypatch)

    _seed_and_run(harness, monkeypatch)

    assert seen["run"] is None


@pytest.mark.criterion("REQ-ING-029")
def test_profile_help_says_the_answers_go_to_the_llm_provider() -> None:
    """Validates ``fund.cli.profile``: ``--help`` says where the answers are sent."""
    result = CliRunner().invoke(cli.app, ["profile", "--help"])
    help_text = " ".join(result.output.split())
    assert "The answers are sent to the configured LLM provider." in help_text
