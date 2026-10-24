"""``MifidProfileRepository`` (append-only versioning) + Store helper.

The ``mifid_profiles`` *model* lives in ``portopt_db``; its *behavior* lives here
in ``fund`` (mirroring the ``agent_runs`` / ``paper_orders`` model/repo split).
The repo appends an immutable version per portfolio and reads the newest one
back. The Store helper writes the active ``ConstraintSet`` under namespace
``(portfolio_id,)`` / key = ``store_key`` so a ``ConstraintSetRef``
resolves — proven here against an in-memory LangGraph store (zero network, zero
Postgres, no live LLM).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from langgraph.store.memory import InMemoryStore
from portopt_db.models import MifidProfile
from sqlalchemy import text
from sqlalchemy.orm import Session

from fund.audit import (
    MifidProfileRepository,
    put_constraint_set,
    put_constraint_set_after_commit,
    resolve_constraint_set,
)
from fund.schemas import ConstraintSet, ConstraintSetRef
from fund.schemas.enums import Horizon, ObjectiveChoice, RiskMeasureChoice


def _constraint_set(portfolio_id: str = "pf-001", **overrides: object) -> ConstraintSet:
    kwargs: dict[str, object] = {
        "portfolio_id": portfolio_id,
        "base_currency": "EUR",
        "a_gamma": 2.5,
        "objective": ObjectiveChoice.GROWTH,
        "risk_measure": RiskMeasureChoice.CVAR,
        "beta": 0.95,
        "nu1": 0.05,
        "nu2": 0.10,
        "nu3": 0.20,
        "horizon": Horizon.LONG,
    }
    kwargs.update(overrides)
    return ConstraintSet(**kwargs)  # type: ignore[arg-type]


def _add(
    repo: MifidProfileRepository, portfolio_id: uuid.UUID, **overrides: Any
) -> MifidProfile:
    payload: dict[str, Any] = {
        "portfolio_id": portfolio_id,
        "questionnaire": {"knowledge": "basic"},
        "constraint_set": {"a_gamma": 2.5},
        "suitability": {"inconsistency_flags": []},
        "store_key": "constraint_set",
    }
    payload.update(overrides)
    return repo.add_version(**payload)


# --- repository: append-only versioning -------------------------------------


def test_add_version_persists_and_returns_row(db_session):
    repo = MifidProfileRepository(db_session)
    pid = uuid.uuid4()
    profile = _add(repo, pid)

    assert profile.id is not None
    assert profile.version == 1
    assert profile.status == "active"
    fetched = db_session.get(MifidProfile, profile.id)
    assert fetched is not None
    assert fetched.constraint_set == {"a_gamma": 2.5}
    assert fetched.store_key == "constraint_set"


def test_add_version_auto_increments_per_portfolio(db_session):
    repo = MifidProfileRepository(db_session)
    pid = uuid.uuid4()
    v1 = _add(repo, pid)
    v2 = _add(repo, pid)

    assert v1.version == 1
    assert v2.version == 2


def test_versions_are_scoped_per_portfolio(db_session):
    repo = MifidProfileRepository(db_session)
    pid_a, pid_b = uuid.uuid4(), uuid.uuid4()
    _add(repo, pid_a)
    b1 = _add(repo, pid_b)

    # b's first version is 1 even though a already carries a version 1.
    assert b1.version == 1


def test_get_active_returns_newest_version(db_session):
    repo = MifidProfileRepository(db_session)
    pid = uuid.uuid4()
    _add(repo, pid, suitability={"v": 1})
    _add(repo, pid, suitability={"v": 2})

    active = repo.get_active(pid)
    assert active is not None
    assert active.version == 2
    assert active.suitability == {"v": 2}


def test_get_active_returns_none_for_unknown_portfolio(db_session):
    repo = MifidProfileRepository(db_session)
    assert repo.get_active(uuid.uuid4()) is None


def test_list_portfolio_ids_distinct_across_versions(db_session):
    repo = MifidProfileRepository(db_session)
    pid_a, pid_b = uuid.uuid4(), uuid.uuid4()
    _add(repo, pid_a)
    _add(repo, pid_a)  # second version, same portfolio → one id
    _add(repo, pid_b)

    ids = repo.list_portfolio_ids()

    assert set(ids) == {pid_a, pid_b}
    assert len(ids) == 2


# --- Store helper: ConstraintSetRef round-trip ------------------------------


def test_put_constraint_set_returns_a_ref_that_resolves():
    store = InMemoryStore()
    cs = _constraint_set(portfolio_id="pf-42")

    ref = put_constraint_set(store, cs, store_key="constraint_set")

    assert isinstance(ref, ConstraintSetRef)
    assert ref.portfolio_id == "pf-42"
    assert ref.store_key == "constraint_set"
    resolved = resolve_constraint_set(store, ref)
    assert resolved == cs


def test_store_write_lands_under_portfolio_namespace_and_key():
    store = InMemoryStore()
    cs = _constraint_set(portfolio_id="pf-7")

    put_constraint_set(store, cs, store_key="constraint_set")

    item = store.get(("pf-7",), "constraint_set")
    assert item is not None
    assert item.value["a_gamma"] == cs.a_gamma


def test_resolve_returns_none_when_ref_is_unknown():
    store = InMemoryStore()
    ref = ConstraintSetRef(portfolio_id="pf-missing", store_key="constraint_set")

    assert resolve_constraint_set(store, ref) is None


# --- Store helper: write deferred until the session's transaction commits ----


def _cached(store: InMemoryStore, cs: ConstraintSet) -> bool:
    return store.get((cs.portfolio_id,), "constraint_set") is not None


def _touch(session: Session) -> None:
    """Open the transaction the way a repo write would."""
    session.execute(text("select 1"))


def test_deferred_put_waits_for_the_commit(committing_session):
    store, cs = InMemoryStore(), _constraint_set("pf-d1")
    _touch(committing_session)

    put_constraint_set_after_commit(
        committing_session, store, cs, store_key="constraint_set"
    )
    assert not _cached(store, cs)

    committing_session.commit()

    assert (
        resolve_constraint_set(
            store, ConstraintSetRef(portfolio_id="pf-d1", store_key="constraint_set")
        )
        == cs
    )


def test_deferred_put_is_dropped_by_a_rollback(committing_session):
    store, cs = InMemoryStore(), _constraint_set("pf-d2")
    _touch(committing_session)

    put_constraint_set_after_commit(
        committing_session, store, cs, store_key="constraint_set"
    )
    committing_session.rollback()
    committing_session.commit()

    assert not _cached(store, cs)


def test_deferred_put_is_dropped_when_the_session_closes_uncommitted(
    committing_session,
):
    store, cs = InMemoryStore(), _constraint_set("pf-d3")
    _touch(committing_session)

    put_constraint_set_after_commit(
        committing_session, store, cs, store_key="constraint_set"
    )
    committing_session.close()
    _touch(committing_session)
    committing_session.commit()

    assert not _cached(store, cs)


def test_a_released_savepoint_is_not_a_commit(committing_session):
    store, cs = InMemoryStore(), _constraint_set("pf-d4")
    _touch(committing_session)

    savepoint = committing_session.begin_nested()
    put_constraint_set_after_commit(
        committing_session, store, cs, store_key="constraint_set"
    )
    savepoint.commit()
    assert not _cached(store, cs)  # the outer transaction can still roll back

    committing_session.rollback()
    assert not _cached(store, cs)


def test_a_released_savepoint_publishes_with_the_outer_commit(committing_session):
    store, cs = InMemoryStore(), _constraint_set("pf-d5")
    _touch(committing_session)

    savepoint = committing_session.begin_nested()
    put_constraint_set_after_commit(
        committing_session, store, cs, store_key="constraint_set"
    )
    savepoint.commit()
    committing_session.commit()

    assert _cached(store, cs)


def test_a_rolled_back_savepoint_drops_only_its_own_puts(committing_session):
    store = InMemoryStore()
    outer, inner = _constraint_set("pf-outer"), _constraint_set("pf-inner")
    _touch(committing_session)

    put_constraint_set_after_commit(
        committing_session, store, outer, store_key="constraint_set"
    )
    savepoint = committing_session.begin_nested()
    put_constraint_set_after_commit(
        committing_session, store, inner, store_key="constraint_set"
    )
    savepoint.rollback()
    committing_session.commit()

    assert _cached(store, outer)
    assert not _cached(store, inner)


def test_a_missing_store_is_a_no_op(committing_session):
    _touch(committing_session)

    put_constraint_set_after_commit(
        committing_session, None, _constraint_set("pf-d6"), store_key="constraint_set"
    )

    committing_session.commit()  # nothing to publish, nothing to fail


def test_a_failing_store_never_breaks_the_commit(committing_session, caplog):
    class _BrokenStore:
        def put(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("store unreachable")

    _touch(committing_session)
    put_constraint_set_after_commit(
        committing_session, _BrokenStore(), _constraint_set("pf-d7"), store_key="k"
    )

    with caplog.at_level(logging.ERROR):
        committing_session.commit()  # the DB commit already happened; it must not raise

    assert "pf-d7" in caplog.text


def test_each_commit_publishes_only_what_was_queued_before_it(committing_session):
    store = InMemoryStore()
    first, second = _constraint_set("pf-first"), _constraint_set("pf-second")
    _touch(committing_session)
    put_constraint_set_after_commit(
        committing_session, store, first, store_key="constraint_set"
    )
    committing_session.commit()
    store.delete(("pf-first",), "constraint_set")

    _touch(committing_session)
    put_constraint_set_after_commit(
        committing_session, store, second, store_key="constraint_set"
    )
    committing_session.commit()

    assert _cached(store, second)
    assert not _cached(store, first)  # a published put is never replayed
