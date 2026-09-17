"""T5 — ``MifidProfileRepository`` (append-only versioning) + Store helper.

The ``mifid_profiles`` *model* lives in ``portopt_db``; its *behavior* lives here
in ``fund`` (mirroring the ``agent_runs`` / ``paper_orders`` model/repo split).
The repo appends an immutable version per portfolio and reads the newest one
back. The Store helper writes the active ``ConstraintSet`` under namespace
``(portfolio_id,)`` / key = ``store_key`` so a Phase-4 ``ConstraintSetRef``
resolves — proven here against an in-memory LangGraph store (zero network, zero
Postgres, no live LLM).
"""

from __future__ import annotations

import uuid
from typing import Any

from langgraph.store.memory import InMemoryStore
from portopt_db.models import MifidProfile

from fund.audit import (
    MifidProfileRepository,
    put_constraint_set,
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
