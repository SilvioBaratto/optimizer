"""T5 — ``fund.observe.reprofile_status`` annual-only, model-free stale marker.

A derived re-profiling flag over the append-only ``mifid_profiles`` table: the
active (highest-version) profile's ``created_at`` is compared against ``now``; a
profile at least ``reprofile_interval_days`` old — or a portfolio never profiled —
is ``due`` with reason ``"annual"``. ``risk_drift`` is a *reserved* reason that
never fires today (no persisted drawdown/NAV to compute it from). The function
builds **no** LLM and drags in no agent stack; every read goes through the injected
sync ``Session`` (no ``commit``).

These tests drive it on the in-memory ``db_session`` harness with an injected
``now`` for determinism, plus the default-``now`` branch, asserting:

* an old (or exactly interval-aged) profile ⇒ ``due`` with ``("annual",)``;
* a fresh profile ⇒ not due, empty ``reasons``;
* a missing profile ⇒ due (``"annual"``), ``profiled_at``/``profile_age_days`` None;
* ``get_active`` picks the newest version's ``created_at``;
* ``"risk_drift"`` never appears in ``reasons``;
* the injected ``now`` is honoured and the default is tz-aware UTC;
* ``reprofile_status``/``ReprofileStatus`` are exported and import is agent-stack-free.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from portopt_db.models import MifidProfile

from fund import observe

# A letter-bearing UUID: ``MifidProfile.portfolio_id`` is a Postgres ``UUID`` column,
# which SQLite gives NUMERIC affinity — an all-decimal UUID could be REAL-coerced and
# crash the UUID result processor. Real portfolio ids are ``uuid4`` (mixed-hex).
_PID = uuid.UUID("bbbb2222-3333-4444-5555-666677778888")
_NOW = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)


def _seed_profile(
    session: Any,
    *,
    portfolio_id: uuid.UUID,
    created_at: datetime,
    version: int = 1,
    status: str = "active",
) -> MifidProfile:
    """Insert a ``mifid_profiles`` row with ``created_at`` set explicitly so the
    derived age is deterministic (SQLite's ``func.now()`` is coarse)."""
    profile = MifidProfile(
        portfolio_id=portfolio_id,
        version=version,
        questionnaire={},
        constraint_set={},
        suitability={},
        store_key="cs",
        status=status,
        created_at=created_at,
    )
    session.add(profile)
    session.flush()
    return profile


def test_reprofile_status_old_profile_is_due_annual(db_session):
    _seed_profile(db_session, portfolio_id=_PID, created_at=_NOW - timedelta(days=400))

    status = observe.reprofile_status(db_session, _PID, now=_NOW)

    assert status.due is True
    assert status.reasons == ("annual",)
    assert status.profile_age_days == 400
    assert status.profiled_at == _NOW - timedelta(days=400)
    assert status.reprofile_interval_days == 365


def test_reprofile_status_fresh_profile_not_due(db_session):
    _seed_profile(db_session, portfolio_id=_PID, created_at=_NOW - timedelta(days=10))

    status = observe.reprofile_status(db_session, _PID, now=_NOW)

    assert status.due is False
    assert status.reasons == ()
    assert status.profile_age_days == 10


def test_reprofile_status_missing_profile_is_due(db_session):
    status = observe.reprofile_status(db_session, uuid.uuid4(), now=_NOW)

    assert status.due is True
    assert status.reasons == ("annual",)
    assert status.profiled_at is None
    assert status.profile_age_days is None


def test_reprofile_status_at_interval_boundary_is_due(db_session):
    # Exactly ``reprofile_interval_days`` old counts as due (annual leg, inclusive).
    _seed_profile(db_session, portfolio_id=_PID, created_at=_NOW - timedelta(days=365))

    status = observe.reprofile_status(db_session, _PID, now=_NOW)

    assert status.due is True
    assert status.profile_age_days == 365


def test_reprofile_status_reads_newest_version(db_session):
    # get_active is latest-by-version: the fresh v2 must win over the stale v1.
    _seed_profile(
        db_session, portfolio_id=_PID, created_at=_NOW - timedelta(days=400), version=1
    )
    _seed_profile(
        db_session, portfolio_id=_PID, created_at=_NOW - timedelta(days=5), version=2
    )

    status = observe.reprofile_status(db_session, _PID, now=_NOW)

    assert status.due is False
    assert status.profile_age_days == 5


def test_reprofile_status_never_reports_risk_drift(db_session):
    # risk_drift is reserved but not wired (no persisted drawdown/NAV); it must not
    # appear even when the annual leg fires.
    _seed_profile(db_session, portfolio_id=_PID, created_at=_NOW - timedelta(days=400))

    due = observe.reprofile_status(db_session, _PID, now=_NOW)
    missing = observe.reprofile_status(db_session, uuid.uuid4(), now=_NOW)

    assert "risk_drift" not in due.reasons
    assert "risk_drift" not in missing.reasons


def test_reprofile_status_honours_custom_interval(db_session):
    # A 200-day-old profile is due under a 180-day interval but not under 365.
    _seed_profile(db_session, portfolio_id=_PID, created_at=_NOW - timedelta(days=200))

    short = observe.reprofile_status(
        db_session, _PID, reprofile_interval_days=180, now=_NOW
    )
    long = observe.reprofile_status(
        db_session, _PID, reprofile_interval_days=365, now=_NOW
    )

    assert short.due is True
    assert short.reprofile_interval_days == 180
    assert long.due is False


def test_reprofile_status_defaults_now_to_tz_aware_utc(db_session):
    # No injected ``now``: the default must be tz-aware UTC so subtracting the
    # tz-aware ``created_at`` never raises, and a just-made profile is not due.
    _seed_profile(
        db_session,
        portfolio_id=_PID,
        created_at=datetime.now(UTC) - timedelta(days=2),
    )

    status = observe.reprofile_status(db_session, _PID)

    assert status.due is False
    assert status.profiled_at is not None
    assert status.profile_age_days is not None
    assert status.profile_age_days < 365


def test_reprofile_status_is_exported_and_agent_stack_free():
    assert "reprofile_status" in observe.__all__
    assert "ReprofileStatus" in observe.__all__

    code = textwrap.dedent(
        """
        import sys
        from fund import observe
        observe.reprofile_status  # attribute exists at import time
        observe.ReprofileStatus
        forbidden = {
            "deepagents",
            "langchain",
            "langchain_core",
            "langchain_ollama",
            "langgraph",
            "app",
        }
        leaked = sorted({m.split(".")[0] for m in sys.modules} & forbidden)
        assert not leaked, leaked
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603
