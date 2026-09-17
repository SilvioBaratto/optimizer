"""T4 — ``mifid_profiles`` ORM model on SQLite.

The MiFID suitability-profile model lives in ``portopt_db`` (mirroring
``agent_run.py`` / ``paper_order.py``: UUID PK, ``_JSON`` variant, indexes) so
the shared ``Base.metadata`` builds it under SQLite in tests. These tests prove
the JSONB→JSON variant creates the table on SQLite, that the ``questionnaire`` /
``constraint_set`` / ``suitability`` JSON columns round-trip Python dicts, that
``version`` / ``status`` carry their server defaults, and that
``(portfolio_id, version)`` is unique — the append-only versioning key an
amended assessment relies on.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError

from portopt_db.models import MifidProfile


def test_mifid_profiles_table_is_registered_on_metadata(test_engine):
    tables = set(inspect(test_engine).get_table_names())
    assert "mifid_profiles" in tables


def test_mifid_profile_round_trips_json_columns(db_session):
    pid = uuid.uuid4()
    profile = MifidProfile(
        portfolio_id=pid,
        version=1,
        questionnaire={"knowledge": "basic", "likert": [4, 5, 3]},
        constraint_set={"a_gamma": 5.0, "objective": "max_sharpe"},
        suitability={"appetite_from_tolerance": 0.6, "inconsistency_flags": []},
        store_key="constraint_set",
    )
    db_session.add(profile)
    db_session.flush()

    fetched = db_session.execute(
        select(MifidProfile).where(MifidProfile.id == profile.id)
    ).scalar_one()
    assert fetched.portfolio_id == pid
    assert fetched.questionnaire == {"knowledge": "basic", "likert": [4, 5, 3]}
    assert fetched.constraint_set == {"a_gamma": 5.0, "objective": "max_sharpe"}
    assert fetched.suitability == {
        "appetite_from_tolerance": 0.6,
        "inconsistency_flags": [],
    }
    assert fetched.store_key == "constraint_set"
    assert fetched.created_at is not None
    assert fetched.updated_at is not None


def test_version_and_status_carry_their_defaults(db_session):
    profile = MifidProfile(
        portfolio_id=uuid.uuid4(),
        questionnaire={},
        constraint_set={},
        suitability={},
        store_key="constraint_set",
    )
    db_session.add(profile)
    db_session.flush()
    db_session.refresh(profile)

    assert profile.version == 1
    assert profile.status == "active"


def test_same_portfolio_and_version_twice_is_rejected(db_session):
    pid = uuid.uuid4()
    db_session.add(
        MifidProfile(
            portfolio_id=pid,
            version=1,
            questionnaire={},
            constraint_set={},
            suitability={},
            store_key="constraint_set",
        )
    )
    db_session.flush()

    db_session.add(
        MifidProfile(
            portfolio_id=pid,
            version=1,
            questionnaire={},
            constraint_set={},
            suitability={},
            store_key="constraint_set",
        )
    )
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_appending_a_new_version_for_the_same_portfolio_is_allowed(db_session):
    pid = uuid.uuid4()
    db_session.add(
        MifidProfile(
            portfolio_id=pid,
            version=1,
            questionnaire={},
            constraint_set={},
            suitability={},
            store_key="constraint_set",
        )
    )
    db_session.flush()

    db_session.add(
        MifidProfile(
            portfolio_id=pid,
            version=2,
            questionnaire={},
            constraint_set={},
            suitability={},
            store_key="constraint_set",
        )
    )
    db_session.flush()  # must NOT raise — distinct version

    rows = (
        db_session.execute(select(MifidProfile).where(MifidProfile.portfolio_id == pid))
        .scalars()
        .all()
    )
    assert {r.version for r in rows} == {1, 2}
