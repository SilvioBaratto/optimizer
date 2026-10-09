"""``fund_autonomy_state`` ORM model on SQLite.

The per-scope autonomy posture (ARMED/FROZEN + first-round gating) lives in
``portopt_db`` so the shared ``Base.metadata`` builds it under SQLite in tests.
These tests prove the table creates on SQLite, the ``armed`` / gated defaults
apply, the composite ``(scope, portfolio_id)`` uniqueness blocks a duplicate
portfolio row, and the partial unique index pins the fund row (NULL
portfolio_id) to a singleton while distinct portfolio rows and the fund row
coexist.
"""

from __future__ import annotations

import datetime
import uuid

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError

from portopt_db.models import FundAutonomyState


def test_table_is_registered_on_metadata(test_engine):
    """The model builds the table under SQLite ``create_all``."""
    assert "fund_autonomy_state" in set(inspect(test_engine).get_table_names())


def test_defaults_apply_when_fields_omitted(db_session):
    """A fresh row is ARMED, first-round gated, and un-frozen."""
    row = FundAutonomyState(scope="portfolio", portfolio_id=uuid.uuid4())
    db_session.add(row)
    db_session.flush()

    fetched = db_session.execute(
        select(FundAutonomyState).where(FundAutonomyState.id == row.id)
    ).scalar_one()
    assert fetched.state == "armed"
    assert fetched.first_round_gated is True
    assert fetched.frozen_reason is None
    assert fetched.frozen_at is None
    assert fetched.created_at is not None


def test_frozen_fields_round_trip(db_session):
    """``state`` / ``frozen_reason`` / ``frozen_at`` persist and read back."""
    frozen_at = datetime.datetime(2026, 10, 9, 12, 0, tzinfo=datetime.timezone.utc)
    row = FundAutonomyState(
        scope="fund",
        portfolio_id=None,
        state="frozen",
        first_round_gated=False,
        frozen_reason="killswitch",
        frozen_at=frozen_at,
    )
    db_session.add(row)
    db_session.flush()

    fetched = db_session.execute(
        select(FundAutonomyState).where(FundAutonomyState.id == row.id)
    ).scalar_one()
    assert fetched.state == "frozen"
    assert fetched.first_round_gated is False
    assert fetched.frozen_reason == "killswitch"
    assert fetched.frozen_at is not None


def test_composite_unique_blocks_duplicate_portfolio_rows(db_session):
    """Two rows sharing (scope, portfolio_id) violate the composite UNIQUE."""
    pid = uuid.uuid4()
    db_session.add(FundAutonomyState(scope="portfolio", portfolio_id=pid))
    db_session.flush()
    db_session.add(FundAutonomyState(scope="portfolio", portfolio_id=pid))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_only_one_fund_scope_row_allowed(db_session):
    """A plain UNIQUE can't stop two NULL-portfolio fund rows; the partial
    unique index does."""
    db_session.add(FundAutonomyState(scope="fund", portfolio_id=None))
    db_session.flush()
    db_session.add(FundAutonomyState(scope="fund", portfolio_id=None))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_fund_row_and_portfolio_rows_coexist(db_session):
    """The singleton fund row sits alongside distinct portfolio rows."""
    db_session.add(FundAutonomyState(scope="fund", portfolio_id=None))
    db_session.add(FundAutonomyState(scope="portfolio", portfolio_id=uuid.uuid4()))
    db_session.add(FundAutonomyState(scope="portfolio", portfolio_id=uuid.uuid4()))
    db_session.flush()

    rows = db_session.execute(select(FundAutonomyState)).scalars().all()
    assert len(rows) == 3
