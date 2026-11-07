"""A mandate allowlist too short to fill the book is flagged when it is saved.

With a per-name cap of ``c`` and a budget ``b`` the optimizer needs at least
``ceil(b / c)`` names. An allowlist shorter than that can never produce a valid
book, and the optimizer's own advice ("widen the universe") contradicts the
allowlist, so the adviser is told at save time instead of at the first failed round.
"""

from __future__ import annotations

import pytest
from _fund_fakes import PORTFOLIO_ID, make_constraint_set, make_mandate
from portopt_db.models import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from fund.actions import apply_action
from fund.audit import MifidProfileRepository
from fund.config import settings
from fund.schemas.constraint_set import Bounds, UniverseFilters


@pytest.fixture
def session():
    """A private in-memory session where commits do not leak between tests."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()


def _profile(session, **changes) -> None:
    constraint_set = make_constraint_set().model_copy(update=changes)
    MifidProfileRepository(session).add_version(
        portfolio_id=PORTFOLIO_ID,
        questionnaire={},
        constraint_set=constraint_set.model_dump(mode="json"),
        suitability={"band": "balanced"},
        store_key=settings.constraint_set_store_key,
    )


def _save(session, tickers: tuple[str, ...]):
    mandate = make_mandate().model_copy(update={"eligible_tickers": tickers})
    return apply_action(
        "mandate_edit", session=session, portfolio_id=PORTFOLIO_ID, mandate=mandate
    )


def _capacity_warnings(result) -> list[str]:
    warnings = (result.detail or {}).get("warnings", [])
    return [w for w in warnings if "at least" in w]


def test_an_allowlist_below_the_names_the_cap_needs_is_flagged(session) -> None:
    """A 25% cap needs four names; three can never reach a fully invested book."""
    _profile(session, bounds=Bounds(max_weights=0.25))

    result = _save(session, ("AAA", "BBB", "CCC"))

    (warning,) = _capacity_warnings(result)
    assert "3" in warning
    assert "4" in warning
    assert "warning:" in result.message


def test_the_position_cap_of_a_low_experience_client_counts(session) -> None:
    """The K&E cap tightens the ceiling, so it raises the number of names needed."""
    _profile(session, universe_filters=UniverseFilters(max_position_cap=0.2))

    result = _save(session, ("AAA", "BBB", "CCC", "DDD"))

    assert len(_capacity_warnings(result)) == 1


def test_an_allowlist_that_can_fill_the_book_is_not_flagged(session) -> None:
    """Exactly ``budget / cap`` names is enough."""
    _profile(session, bounds=Bounds(max_weights=0.25))

    result = _save(session, ("AAA", "BBB", "CCC", "DDD"))

    assert _capacity_warnings(result) == []


def test_no_allowlist_or_no_profile_means_no_capacity_warning(session) -> None:
    """Unrestricted mandates and unprofiled portfolios have nothing to compare."""
    assert _capacity_warnings(_save(session, ())) == []
    assert _capacity_warnings(_save(session, ("AAA",))) == []
