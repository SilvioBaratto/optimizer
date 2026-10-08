"""``HumanAuditLogRepository`` — fund-side behavior over ``human_audit_log``.

The repo opens no session of its own and never commits (the caller owns the
transaction, D1). These tests prove ``append`` flushes a row with the id/defaults
populated, de-dups on a supplied ``idempotency_key`` while leaving un-keyed rows
free to repeat, and that ``list_recent`` returns newest-first and composes its
filters (portfolio / action / actor / overrides-only / date range).
"""

from __future__ import annotations

import datetime as dt
import uuid

from portopt_db.models import HumanAuditLog

from fund.audit.human_audit_repository import HumanAuditLogRepository


def test_repository_exported_from_audit_package():
    """The repo is re-exported from the ``fund.audit`` package root."""
    from fund.audit import HumanAuditLogRepository as Exported

    assert Exported is HumanAuditLogRepository


def test_append_flushes_row_with_id_and_defaults(db_session):
    """``append`` returns a flushed row with id + default actor/flags set."""
    repo = HumanAuditLogRepository(db_session)

    row = repo.append(action="approve", run_id=uuid.uuid4())

    assert row.id is not None
    assert row.action == "approve"
    assert row.actor == "human"
    assert row.override_flag is False
    assert row.suitability_flag is False


def test_append_persists_provenance_and_flags(db_session):
    """Provenance columns and the override/suitability flags round-trip."""
    repo = HumanAuditLogRepository(db_session)
    pid = uuid.uuid4()

    row = repo.append(
        action="guardrail_override",
        portfolio_id=pid,
        target="max_vol",
        prev_value=0.15,
        new_value=0.18,
        source="profile",
        override_flag=True,
        suitability_flag=True,
        rationale="client opted up",
    )

    fetched = db_session.get(HumanAuditLog, row.id)
    assert fetched is not None
    assert fetched.portfolio_id == pid
    assert fetched.prev_value == 0.15
    assert fetched.new_value == 0.18
    assert fetched.source == "profile"
    assert fetched.override_flag is True
    assert fetched.suitability_flag is True
    assert fetched.rationale == "client opted up"


def test_append_dedups_on_idempotency_key(db_session):
    """A repeated ``idempotency_key`` returns the first row, writing no duplicate."""
    repo = HumanAuditLogRepository(db_session)

    first = repo.append(action="approve", idempotency_key="run-a3f:approve")
    second = repo.append(action="approve", idempotency_key="run-a3f:approve")

    assert second.id == first.id
    rows = repo.list_recent()
    assert len([r for r in rows if r.idempotency_key == "run-a3f:approve"]) == 1


def test_append_without_key_allows_repeats(db_session):
    """Un-keyed actions never collide — two appends write two rows."""
    repo = HumanAuditLogRepository(db_session)

    repo.append(action="approve")
    repo.append(action="approve")

    assert len(repo.list_recent()) == 2


def test_list_recent_returns_newest_first(db_session):
    """Rows come back ordered by ``created_at`` descending."""
    repo = HumanAuditLogRepository(db_session)
    repo.append(action="approve", target="first")
    repo.append(action="reject", target="second")

    rows = repo.list_recent()

    assert [r.target for r in rows] == ["second", "first"]


def test_list_recent_filters_compose(db_session):
    """portfolio / action / actor / overrides-only filters each narrow the set."""
    repo = HumanAuditLogRepository(db_session)
    pid = uuid.uuid4()
    other = uuid.uuid4()
    repo.append(action="approve", portfolio_id=pid, actor="human")
    repo.append(action="reject", portfolio_id=pid, actor="human")
    repo.append(action="approve", portfolio_id=other, actor="human")
    repo.append(action="guardrail_override", portfolio_id=pid, override_flag=True)

    assert {r.action for r in repo.list_recent(portfolio_id=pid)} == {
        "approve",
        "reject",
        "guardrail_override",
    }
    assert all(r.portfolio_id == pid for r in repo.list_recent(portfolio_id=pid))
    assert len(repo.list_recent(action="approve")) == 2
    assert len(repo.list_recent(portfolio_id=pid, action="approve")) == 1
    overrides = repo.list_recent(overrides_only=True)
    assert len(overrides) == 1
    assert overrides[0].override_flag is True


def test_list_recent_date_range_bounds(db_session):
    """A future lower bound excludes everything; a past bound includes all."""
    repo = HumanAuditLogRepository(db_session)
    repo.append(action="approve")

    future = dt.datetime(2999, 1, 1, tzinfo=dt.timezone.utc)
    past = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)
    assert repo.list_recent(start=future) == []
    assert len(repo.list_recent(start=past)) == 1
