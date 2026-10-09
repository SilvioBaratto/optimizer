"""``human_audit_log`` ORM model on SQLite.

The append-only human-decision log lives in ``portopt_db`` (mirroring the
``agent_runs`` audit models: UUID PK, ``_JSON`` variant, indexes) so the shared
``Base.metadata`` builds it under SQLite in tests. These tests prove the
JSONB→JSON variant creates the table on SQLite, that the ``prev_value`` /
``new_value`` JSON columns round-trip both dicts and scalars, that the flag /
actor defaults apply, and that a duplicate ``idempotency_key`` is rejected while
rows without one may repeat.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import delete, inspect, select, update
from sqlalchemy.exc import DatabaseError, IntegrityError

from portopt_db.models import HumanAuditLog


def test_table_is_registered_on_metadata(test_engine):
    """The JSONB→JSON variant builds the table under SQLite ``create_all``."""
    assert "human_audit_log" in set(inspect(test_engine).get_table_names())


def test_defaults_apply_when_fields_omitted(db_session):
    """Actor defaults to ``human`` and both flags default to ``False``."""
    row = HumanAuditLog(action="approve")
    db_session.add(row)
    db_session.flush()

    fetched = db_session.execute(
        select(HumanAuditLog).where(HumanAuditLog.id == row.id)
    ).scalar_one()
    assert fetched.actor == "human"
    assert fetched.override_flag is False
    assert fetched.suitability_flag is False
    assert fetched.portfolio_id is None
    assert fetched.run_id is None
    assert fetched.created_at is not None


def test_prev_and_new_value_round_trip_dict_and_scalar(db_session):
    """``prev_value``/``new_value`` hold arbitrary JSON (scalar or dict)."""
    pid = uuid.uuid4()
    row = HumanAuditLog(
        portfolio_id=pid,
        action="guardrail_override",
        target="max_vol",
        prev_value=0.15,
        new_value={"value": 0.18, "unit": "annual"},
        source="profile",
        override_flag=True,
        suitability_flag=True,
    )
    db_session.add(row)
    db_session.flush()

    fetched = db_session.execute(
        select(HumanAuditLog).where(HumanAuditLog.id == row.id)
    ).scalar_one()
    assert fetched.portfolio_id == pid
    assert fetched.prev_value == 0.15
    assert fetched.new_value == {"value": 0.18, "unit": "annual"}
    assert fetched.override_flag is True
    assert fetched.suitability_flag is True
    assert fetched.source == "profile"


def test_duplicate_idempotency_key_is_rejected(db_session):
    """The UNIQUE ``idempotency_key`` guards against a duplicated action."""
    db_session.add(HumanAuditLog(action="approve", idempotency_key="k-1"))
    db_session.flush()
    db_session.add(HumanAuditLog(action="reject", idempotency_key="k-1"))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_rows_without_idempotency_key_may_repeat(db_session):
    """A NULL ``idempotency_key`` never collides — un-keyed rows may repeat."""
    db_session.add(HumanAuditLog(action="approve"))
    db_session.add(HumanAuditLog(action="approve"))
    db_session.flush()

    rows = db_session.execute(select(HumanAuditLog)).scalars().all()
    assert len(rows) == 2
    assert all(r.idempotency_key is None for r in rows)


def test_update_via_orm_is_blocked(db_session):
    """An ORM-instance UPDATE is rejected — the log is append-only."""
    row = HumanAuditLog(action="approve")
    db_session.add(row)
    db_session.flush()

    row.action = "tampered"
    with pytest.raises(DatabaseError, match="append-only"):
        db_session.flush()


def test_update_via_core_is_blocked(db_session):
    """A Core bulk UPDATE is rejected too — ORM mapper events would miss it."""
    row = HumanAuditLog(action="approve")
    db_session.add(row)
    db_session.flush()

    with pytest.raises(DatabaseError, match="append-only"):
        db_session.execute(
            update(HumanAuditLog)
            .where(HumanAuditLog.id == row.id)
            .values(action="tampered")
        )


def test_delete_is_blocked(db_session):
    """A DELETE is rejected — audit rows are never removed."""
    row = HumanAuditLog(action="approve")
    db_session.add(row)
    db_session.flush()

    with pytest.raises(DatabaseError, match="append-only"):
        db_session.execute(delete(HumanAuditLog).where(HumanAuditLog.id == row.id))
