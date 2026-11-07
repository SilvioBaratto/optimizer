"""A cached ``ConstraintSet`` never outranks the profile row it caches.

The Store put that follows a re-profile commit is best effort and only logged when it
fails, so the cache can be left holding the previous, possibly looser, limits. The
``mifid_profiles`` row is the system of record and must win.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from langgraph.store.memory import InMemoryStore
from pydantic import ValidationError

from fund.audit import (
    MifidProfileRepository,
    put_constraint_set,
    resolve_constraint_set,
)
from fund.schemas import ConstraintSet, ConstraintSetRef
from fund.schemas.enums import Horizon, ObjectiveChoice, RiskMeasureChoice

_KEY = "constraint_set"


def _constraint_set(pid: uuid.UUID, **overrides: object) -> ConstraintSet:
    kwargs: dict[str, object] = {
        "portfolio_id": str(pid),
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


def _profile(session, pid: uuid.UUID, constraint_set: Any) -> None:
    payload = (
        constraint_set.model_dump(mode="json")
        if isinstance(constraint_set, ConstraintSet)
        else constraint_set
    )
    MifidProfileRepository(session).add_version(
        portfolio_id=pid,
        questionnaire={},
        constraint_set=payload,
        suitability={},
        store_key=_KEY,
    )
    session.commit()


def _ref(pid: uuid.UUID) -> ConstraintSetRef:
    return ConstraintSetRef(portfolio_id=str(pid), store_key=_KEY)


def test_a_stale_cache_entry_yields_to_the_newer_profile_row(committing_session):
    """A re-profile whose Store put was lost must not leave the old limits in force."""
    pid = uuid.uuid4()
    looser = _constraint_set(pid, nu2=0.30)
    tighter = _constraint_set(pid, nu2=0.10)
    _profile(committing_session, pid, looser)
    store = InMemoryStore()
    put_constraint_set(store, looser, store_key=_KEY)
    _profile(committing_session, pid, tighter)

    resolved = resolve_constraint_set(store, _ref(pid), session=committing_session)

    assert resolved == tighter


def test_the_stale_cache_entry_is_healed_once_the_session_commits(committing_session):
    """Reading the row refills the cache so later resolves agree with it."""
    pid = uuid.uuid4()
    looser = _constraint_set(pid, nu2=0.30)
    tighter = _constraint_set(pid, nu2=0.10)
    _profile(committing_session, pid, looser)
    store = InMemoryStore()
    put_constraint_set(store, looser, store_key=_KEY)
    _profile(committing_session, pid, tighter)

    resolve_constraint_set(store, _ref(pid), session=committing_session)
    committing_session.commit()

    item = store.get((str(pid),), _KEY)
    assert item is not None
    assert ConstraintSet.model_validate(item.value) == tighter


def test_a_cache_entry_matching_the_row_queues_no_refill(committing_session):
    """The common case stays free of Store writes."""
    pid = uuid.uuid4()
    current = _constraint_set(pid)
    _profile(committing_session, pid, current)
    store = InMemoryStore()
    put_constraint_set(store, current, store_key=_KEY)

    resolved = resolve_constraint_set(store, _ref(pid), session=committing_session)

    assert resolved == current
    publisher = committing_session.info.get("fund.constraint_set_publisher")
    assert publisher is None or publisher.pending == []


def test_a_cache_entry_is_kept_when_the_row_cannot_be_read(committing_session):
    """An unparseable row is no reason to discard a working cache entry."""
    pid = uuid.uuid4()
    cached = _constraint_set(pid)
    _profile(committing_session, pid, {"a_gamma": 2.5})
    store = InMemoryStore()
    put_constraint_set(store, cached, store_key=_KEY)

    resolved = resolve_constraint_set(store, _ref(pid), session=committing_session)

    assert resolved == cached


def test_a_cache_entry_without_a_profile_row_is_returned(committing_session):
    """No row means nothing to override the cache with."""
    pid = uuid.uuid4()
    cached = _constraint_set(pid)
    store = InMemoryStore()
    put_constraint_set(store, cached, store_key=_KEY)

    resolved = resolve_constraint_set(store, _ref(pid), session=committing_session)

    assert resolved == cached


def test_an_unreadable_row_with_nothing_cached_is_an_error_not_an_unprofiled_portfolio(
    committing_session,
):
    """Resolving to ``None`` would run the portfolio with no mandate screens at all."""
    pid = uuid.uuid4()
    _profile(committing_session, pid, {"a_gamma": 2.5})

    with pytest.raises(ValidationError):
        resolve_constraint_set(InMemoryStore(), _ref(pid), session=committing_session)
