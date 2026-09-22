"""T6 — fund scheduler step functions, exercised headless (no started scheduler).

Contracts under test (SPEC §Phase-9C / plan Task 6):

* ``run_rebalance_sweep`` drives every ``triggers.cron`` mandate once, in
  deterministic ``portfolio_id`` order, skips non-cron mandates, survives one
  portfolio erroring, marks its slot completed, and **never approves/resumes**.
* ``run_drift_monitor`` auto-enqueues **only** breached (``drift_l1 > threshold``)
  ``triggers.drift`` portfolios, skips in-band ones, skips a portfolio already
  awaiting HITL, and logs a re-profiling marker when the profile is stale.
* ``_drive_rebalance`` resolves the mandate + active ``ConstraintSet`` (a missing
  set is a clean skip, not fatal) and **returns without resuming** the run.
* ``run_orphan_reaper`` fails lease-expired fund slots and commits.
* Every step claims a slot and heartbeats while it runs.

Because ``FundJobRepository`` owns its own transactions (each mutator ``commit``s
so the slot lands for a reaper in another process — and the heartbeat pulses from
a **separate thread + connection**), these tests use a private, function-scoped
**file-based** SQLite engine (mirrors the T2 ``job_session`` rationale) rather than
the shared SAVEPOINT ``db_session``. A file DB gives each session/thread its own
connection, so the cross-thread heartbeat is safe.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from _fund_fakes import ASOF, make_constraint_set
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from portopt_db.models import BackgroundJob, Base
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import fund.observe as observe
from fund.audit.fund_job_repository import FundJobRepository
from fund.audit.mandate_repository import MandateRepository
from fund.audit.mifid_repository import put_constraint_set
from fund.audit.repository import AgentRunRepository
from fund.config import settings
from fund.scheduler import (
    SchedulerRuntime,
    _drive_rebalance,
    _heartbeat,
    configure_runtime,
    run_drift_monitor,
    run_orphan_reaper,
    run_rebalance_sweep,
)
from fund.schemas.mandate import PortfolioMandate, RunTriggers

# Letter-bearing UUIDs (SQLite numeric-affinity trap) in a<b<c order so the
# deterministic ``list_active`` sweep is easy to assert.
PID_A = uuid.UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
PID_B = uuid.UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
PID_C = uuid.UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc")

SessionFactory = Callable[[], "contextmanager[Session]"]


# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def session_factory(tmp_path) -> Generator[SessionFactory, None, None]:
    """A function-scoped file-based SQLite engine + a session-context factory.

    File-based (not ``:memory:``/StaticPool) so the step's working session and the
    heartbeat thread's session each get their own connection — no cross-thread
    single-connection hazard — while committing to one shared DB. Discarded on
    teardown.
    """
    engine = create_engine(
        f"sqlite:///{tmp_path / 'sched.db'}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    Base.metadata.create_all(bind=engine)

    @contextmanager
    def _factory() -> Generator[Session, None, None]:
        session = Session(bind=engine)
        try:
            yield session
        finally:
            session.close()

    try:
        yield _factory
    finally:
        engine.dispose()


@pytest.fixture(autouse=True)
def _reset_module_runtime() -> Generator[None, None, None]:
    """Clear the module-level ``_runtime`` any test installs so it never leaks."""
    yield
    configure_runtime(None)


# --- seeding helpers --------------------------------------------------------


def _seed_mandate(
    factory: SessionFactory,
    pid: uuid.UUID,
    *,
    cron: bool,
    drift: bool,
    threshold: float = 0.1,
) -> None:
    mandate = PortfolioMandate(
        portfolio_id=str(pid),
        capital=Decimal("100000"),
        base_currency="EUR",
        drift_l1_threshold=threshold,
        triggers=RunTriggers(cron=cron, drift=drift),
    )
    with factory() as session:
        MandateRepository(session).upsert(mandate)
        session.commit()


def _seed_paused_run(factory: SessionFactory, pid: uuid.UUID) -> None:
    with factory() as session:
        repo = AgentRunRepository(session)
        run = repo.create_run(
            portfolio_id=pid, asof=ASOF, seed=1, universe=[], optimizer_config={}
        )
        repo.mark_paused(run.id)
        session.commit()


def _insert_stale_slot(factory: SessionFactory, job_type: str) -> None:
    with factory() as session:
        session.add(
            BackgroundJob(
                job_type=job_type,
                status="running",
                started_at=datetime.now(UTC),
                last_heartbeat_at=datetime.now(UTC) - timedelta(seconds=10_000),
            )
        )
        session.commit()


def _job_rows(factory: SessionFactory, job_type: str) -> list[BackgroundJob]:
    with factory() as session:
        return list(session.query(BackgroundJob).filter_by(job_type=job_type).all())


def _state(pid: uuid.UUID, drift_l1: float) -> observe.PortfolioState:
    return observe.PortfolioState(
        portfolio_id=pid, current={}, target={}, drift_l1=drift_l1, metrics={}
    )


def _not_due(session, pid, **_kwargs) -> observe.ReprofileStatus:
    return observe.ReprofileStatus(
        portfolio_id=pid,
        due=False,
        reasons=(),
        profiled_at=None,
        profile_age_days=None,
        reprofile_interval_days=365,
    )


# --- run_rebalance_sweep ----------------------------------------------------


def test_sweep_drives_each_cron_mandate_in_order(session_factory, monkeypatch):
    _seed_mandate(session_factory, PID_A, cron=True, drift=False)
    _seed_mandate(session_factory, PID_B, cron=True, drift=False)
    _seed_mandate(session_factory, PID_C, cron=False, drift=True)  # not cron → skip

    driven: list[uuid.UUID] = []
    monkeypatch.setattr(
        "fund.scheduler._drive_rebalance",
        lambda pid, asof, **_kw: driven.append(pid),
    )

    ok = run_rebalance_sweep(session_factory=session_factory, asof=ASOF)

    assert ok is True
    assert driven == [PID_A, PID_B]  # ordered by portfolio_id; C skipped
    rows = _job_rows(session_factory, "fund_rebalance_sweep")
    assert len(rows) == 1
    assert rows[0].status == "completed"


def test_sweep_survives_one_portfolio_error(session_factory, monkeypatch):
    _seed_mandate(session_factory, PID_A, cron=True, drift=False)
    _seed_mandate(session_factory, PID_B, cron=True, drift=False)

    driven: list[uuid.UUID] = []

    def _spy(pid, asof, **_kw):
        if pid == PID_A:
            raise RuntimeError("boom")
        driven.append(pid)

    monkeypatch.setattr("fund.scheduler._drive_rebalance", _spy)

    ok = run_rebalance_sweep(session_factory=session_factory, asof=ASOF)

    assert ok is True  # one portfolio erroring never aborts the sweep
    assert driven == [PID_B]
    assert _job_rows(session_factory, "fund_rebalance_sweep")[0].status == "completed"


def test_sweep_skips_when_slot_busy(session_factory, monkeypatch):
    _seed_mandate(session_factory, PID_A, cron=True, drift=False)
    with session_factory() as session:
        FundJobRepository(session).claim_or_create("fund_rebalance_sweep")  # occupy

    driven: list[int] = []
    monkeypatch.setattr(
        "fund.scheduler._drive_rebalance", lambda *a, **k: driven.append(1)
    )

    ok = run_rebalance_sweep(session_factory=session_factory, asof=ASOF)

    assert ok is False
    assert driven == []


def test_sweep_claims_slot_and_heartbeats(session_factory, monkeypatch):
    _seed_mandate(session_factory, PID_A, cron=True, drift=False)

    pulses: list[uuid.UUID] = []
    original = FundJobRepository.update_heartbeat

    def _counting(self, job_id):
        pulses.append(job_id)
        return original(self, job_id)

    monkeypatch.setattr(FundJobRepository, "update_heartbeat", _counting)
    monkeypatch.setattr("fund.scheduler._drive_rebalance", lambda *a, **k: None)

    ok = run_rebalance_sweep(session_factory=session_factory, asof=ASOF)

    assert ok is True
    assert len(pulses) >= 1  # the heartbeat thread pulsed while the step ran


# --- run_drift_monitor ------------------------------------------------------


def test_drift_monitor_enqueues_only_breached(session_factory, monkeypatch):
    _seed_mandate(
        session_factory, PID_A, cron=False, drift=True, threshold=0.1
    )  # breach
    _seed_mandate(
        session_factory, PID_B, cron=False, drift=True, threshold=0.1
    )  # in-band
    _seed_mandate(session_factory, PID_C, cron=True, drift=False)  # not drift → skip

    drifts = {PID_A: 0.5, PID_B: 0.05}
    monkeypatch.setattr(
        observe, "portfolio_state", lambda s, pid: _state(pid, drifts[pid])
    )
    monkeypatch.setattr(observe, "reprofile_status", _not_due)

    driven: list[uuid.UUID] = []
    monkeypatch.setattr(
        "fund.scheduler._drive_rebalance",
        lambda pid, asof, **_kw: driven.append(pid),
    )

    ok = run_drift_monitor(session_factory=session_factory, asof=ASOF)

    assert ok is True
    assert driven == [PID_A]  # only the breached drift portfolio
    assert _job_rows(session_factory, "fund_drift_monitor")[0].status == "completed"


def test_drift_monitor_skips_already_paused(session_factory, monkeypatch):
    _seed_mandate(session_factory, PID_A, cron=False, drift=True, threshold=0.1)
    _seed_paused_run(session_factory, PID_A)  # already awaiting HITL

    monkeypatch.setattr(
        observe, "portfolio_state", lambda s, pid: _state(pid, 0.9)
    )  # breach
    monkeypatch.setattr(observe, "reprofile_status", _not_due)

    driven: list[uuid.UUID] = []
    monkeypatch.setattr(
        "fund.scheduler._drive_rebalance",
        lambda pid, asof, **_kw: driven.append(pid),
    )

    ok = run_drift_monitor(session_factory=session_factory, asof=ASOF)

    assert ok is True
    assert driven == []  # breached but already paused → skipped (no stacking)


def test_drift_monitor_logs_reprofile_marker_when_due(
    session_factory, monkeypatch, caplog
):
    _seed_mandate(session_factory, PID_A, cron=False, drift=True, threshold=0.1)

    monkeypatch.setattr(
        observe, "portfolio_state", lambda s, pid: _state(pid, 0.01)
    )  # in-band
    monkeypatch.setattr(
        observe,
        "reprofile_status",
        lambda s, pid, **_kw: observe.ReprofileStatus(
            portfolio_id=pid,
            due=True,
            reasons=("annual",),
            profiled_at=None,
            profile_age_days=None,
            reprofile_interval_days=365,
        ),
    )
    monkeypatch.setattr("fund.scheduler._drive_rebalance", lambda *a, **k: None)

    with caplog.at_level(logging.WARNING, logger="fund.scheduler"):
        run_drift_monitor(session_factory=session_factory, asof=ASOF)

    assert "re-profiling" in caplog.text


# --- _drive_rebalance -------------------------------------------------------


def test_drive_rebalance_skips_missing_constraint_set(session_factory, monkeypatch):
    _seed_mandate(session_factory, PID_A, cron=True, drift=False)
    persistence = SimpleNamespace(store=InMemoryStore(), saver=MemorySaver())  # empty

    called: list[int] = []
    monkeypatch.setattr("fund.agents.graph.run_fund", lambda *a, **k: called.append(1))

    with session_factory() as session:
        _drive_rebalance(
            PID_A,
            ASOF,
            session=session,
            persistence=persistence,
            model=object(),
            fallback=None,
        )

    assert called == []  # no active ConstraintSet → run_fund never invoked


def test_drive_rebalance_reaches_gate_but_never_resumes(session_factory, monkeypatch):
    _seed_mandate(session_factory, PID_A, cron=True, drift=False)
    store = InMemoryStore()
    put_constraint_set(
        store, make_constraint_set(PID_A), store_key=settings.constraint_set_store_key
    )
    persistence = SimpleNamespace(store=store, saver=MemorySaver())

    fake_run = MagicMock()
    fake_run.run_id = uuid.uuid4()
    fake_run.status = "paused"
    monkeypatch.setattr("fund.agents.graph.run_fund", lambda *a, **k: fake_run)

    with session_factory() as session:
        _drive_rebalance(
            PID_A,
            ASOF,
            session=session,
            persistence=persistence,
            model=object(),
            fallback=None,
        )

    fake_run.resume.assert_not_called()  # driver drives to the gate, never approves


def test_drive_rebalance_skips_already_paused(session_factory, monkeypatch):
    _seed_mandate(session_factory, PID_A, cron=True, drift=False)
    _seed_paused_run(session_factory, PID_A)
    store = InMemoryStore()
    put_constraint_set(
        store, make_constraint_set(PID_A), store_key=settings.constraint_set_store_key
    )
    persistence = SimpleNamespace(store=store, saver=MemorySaver())

    called: list[int] = []
    monkeypatch.setattr("fund.agents.graph.run_fund", lambda *a, **k: called.append(1))

    with session_factory() as session:
        _drive_rebalance(
            PID_A,
            ASOF,
            session=session,
            persistence=persistence,
            model=object(),
            fallback=None,
        )

    assert called == []  # a paused run already awaits HITL → no second run


# --- run_orphan_reaper ------------------------------------------------------


def test_orphan_reaper_fails_stale_slots(session_factory):
    _insert_stale_slot(session_factory, "fund_rebalance_sweep")

    ok = run_orphan_reaper(session_factory=session_factory)

    assert ok is True
    rows = _job_rows(session_factory, "fund_rebalance_sweep")
    assert rows[0].status == "failed"


def test_orphan_reaper_spares_fresh_slots(session_factory):
    with session_factory() as session:
        FundJobRepository(session).claim_or_create("fund_drift_monitor")  # fresh

    ok = run_orphan_reaper(session_factory=session_factory)

    assert ok is True
    rows = _job_rows(session_factory, "fund_drift_monitor")
    assert rows[0].status == "pending"  # fresh lease untouched


def test_orphan_reaper_returns_false_on_error(session_factory, monkeypatch):
    def _boom(*_a, **_k):
        raise RuntimeError("db down")

    monkeypatch.setattr(FundJobRepository, "reap_orphans", _boom)

    ok = run_orphan_reaper(session_factory=session_factory)

    assert ok is False  # a reaper tick failure is non-fatal (logged), never raised


# --- _drive_rebalance additional guards -------------------------------------


def test_drive_rebalance_skips_missing_mandate(session_factory, monkeypatch):
    # No mandate seeded for PID_A → clean skip, run_fund never invoked.
    persistence = SimpleNamespace(store=InMemoryStore(), saver=MemorySaver())
    called: list[int] = []
    monkeypatch.setattr("fund.agents.graph.run_fund", lambda *a, **k: called.append(1))

    with session_factory() as session:
        _drive_rebalance(
            PID_A,
            ASOF,
            session=session,
            persistence=persistence,
            model=object(),
            fallback=None,
        )

    assert called == []


def test_drive_rebalance_skips_without_persistence(session_factory, monkeypatch):
    _seed_mandate(session_factory, PID_A, cron=True, drift=False)
    called: list[int] = []
    monkeypatch.setattr("fund.agents.graph.run_fund", lambda *a, **k: called.append(1))

    with session_factory() as session:
        _drive_rebalance(
            PID_A,
            ASOF,
            session=session,
            persistence=None,  # no persistence store available → skip
            model=object(),
            fallback=None,
        )

    assert called == []


# --- configure_runtime + infra-error / heartbeat branches -------------------


def test_configured_runtime_is_passed_to_the_driver(session_factory, monkeypatch):
    _seed_mandate(session_factory, PID_A, cron=True, drift=False)
    sentinel_model = object()
    sentinel_persistence = SimpleNamespace(store=InMemoryStore(), saver=MemorySaver())
    configure_runtime(
        SchedulerRuntime(
            model=sentinel_model, fallback=None, persistence=sentinel_persistence
        )
    )

    seen: dict[str, object] = {}
    monkeypatch.setattr(
        "fund.scheduler._drive_rebalance",
        lambda pid, asof, *, session, persistence, model, fallback: seen.update(
            model=model, persistence=persistence
        ),
    )

    ok = run_rebalance_sweep(session_factory=session_factory, asof=ASOF)  # runtime=None

    assert ok is True
    assert seen["model"] is sentinel_model
    assert seen["persistence"] is sentinel_persistence


def test_sweep_marks_slot_failed_on_infra_error(session_factory, monkeypatch):
    _seed_mandate(session_factory, PID_A, cron=True, drift=False)
    monkeypatch.setattr(
        MandateRepository,
        "list_active",
        lambda self: (_ for _ in ()).throw(RuntimeError("db exploded")),
    )

    ok = run_rebalance_sweep(session_factory=session_factory, asof=ASOF)

    assert ok is False
    assert _job_rows(session_factory, "fund_rebalance_sweep")[0].status == "failed"


def test_drift_monitor_skips_when_slot_busy(session_factory, monkeypatch):
    with session_factory() as session:
        FundJobRepository(session).claim_or_create("fund_drift_monitor")  # occupy

    monkeypatch.setattr("fund.scheduler._drive_rebalance", lambda *a, **k: None)

    ok = run_drift_monitor(session_factory=session_factory, asof=ASOF)

    assert ok is False


def test_drift_monitor_marks_slot_failed_on_infra_error(session_factory, monkeypatch):
    _seed_mandate(session_factory, PID_A, cron=False, drift=True)
    monkeypatch.setattr(
        MandateRepository,
        "list_active",
        lambda self: (_ for _ in ()).throw(RuntimeError("db exploded")),
    )

    ok = run_drift_monitor(session_factory=session_factory, asof=ASOF)

    assert ok is False
    assert _job_rows(session_factory, "fund_drift_monitor")[0].status == "failed"


def test_drift_monitor_survives_one_portfolio_error(session_factory, monkeypatch):
    _seed_mandate(session_factory, PID_A, cron=False, drift=True, threshold=0.1)
    _seed_mandate(session_factory, PID_B, cron=False, drift=True, threshold=0.1)

    def _state_or_boom(session, pid):
        if pid == PID_A:
            raise RuntimeError("state unavailable")
        return _state(pid, 0.01)  # B in-band

    monkeypatch.setattr(observe, "portfolio_state", _state_or_boom)
    monkeypatch.setattr(observe, "reprofile_status", _not_due)
    driven: list[uuid.UUID] = []
    monkeypatch.setattr(
        "fund.scheduler._drive_rebalance",
        lambda pid, asof, **_kw: driven.append(pid),
    )

    ok = run_drift_monitor(session_factory=session_factory, asof=ASOF)

    assert ok is True  # A erroring does not abort the monitor
    assert driven == []  # A errored (no enqueue); B in-band (no enqueue)


def test_heartbeat_thread_pulses_repeatedly(session_factory):
    with session_factory() as session:
        repo = FundJobRepository(session)
        job_id = repo.claim_or_create("fund_rebalance_sweep")
        repo.mark_running(job_id)

    pulses = 0
    original = FundJobRepository.update_heartbeat

    def _counting(self, jid):
        nonlocal pulses
        pulses += 1
        return original(self, jid)

    FundJobRepository.update_heartbeat = _counting  # type: ignore[method-assign]
    try:
        with _heartbeat(job_id, cadence=0.02, session_factory=session_factory):
            time.sleep(0.15)  # long enough for the immediate pulse + loop iterations
    finally:
        FundJobRepository.update_heartbeat = original  # type: ignore[method-assign]

    assert pulses >= 2  # immediate pulse plus at least one loop iteration
