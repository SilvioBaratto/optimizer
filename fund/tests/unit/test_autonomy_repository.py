"""``AutonomyRepository`` — fund-side behavior over ``fund_autonomy_state``.

The model lives in ``portopt_db``; the repo behavior here get-or-creates the
single fund row and per-portfolio rows (armed + first-round gated by default),
flips the ARMED/FROZEN posture (stamping/clearing ``frozen_at``), and consumes
the first-round flag exactly once. Injected session, no commit — these tests ride
the ``db_session`` SAVEPOINT harness.
"""

from __future__ import annotations

import uuid

from fund.audit.autonomy_repository import (
    AutonomyRepository,
    AutonomyScope,
    AutonomyState,
)


def test_enums_carry_the_wire_values():
    """The scope/state enums serialise to the varchar values the DB stores."""
    assert AutonomyScope.FUND.value == "fund"
    assert AutonomyScope.PORTFOLIO.value == "portfolio"
    assert AutonomyState.ARMED.value == "armed"
    assert AutonomyState.FROZEN.value == "frozen"


def test_fund_state_creates_armed_gated_singleton_when_absent(db_session):
    """``fund_state`` creates the single NULL-portfolio row, armed and gated."""
    repo = AutonomyRepository(db_session)
    row = repo.fund_state()
    assert row.scope == "fund"
    assert row.portfolio_id is None
    assert row.state == "armed"
    assert row.first_round_gated is True
    assert row.frozen_at is None
    assert row.frozen_reason is None


def test_fund_state_is_idempotent(db_session):
    """A second ``fund_state`` returns the same row, not a duplicate."""
    repo = AutonomyRepository(db_session)
    assert repo.fund_state().id == repo.fund_state().id


def test_portfolio_state_creates_armed_gated_row_when_absent(db_session):
    """``portfolio_state`` creates the portfolio's row, armed and gated."""
    repo = AutonomyRepository(db_session)
    pid = uuid.uuid4()
    row = repo.portfolio_state(pid)
    assert row.scope == "portfolio"
    assert row.portfolio_id == pid
    assert row.state == "armed"
    assert row.first_round_gated is True


def test_portfolio_state_is_idempotent(db_session):
    """A second ``portfolio_state`` for the same id returns the same row."""
    repo = AutonomyRepository(db_session)
    pid = uuid.uuid4()
    assert repo.portfolio_state(pid).id == repo.portfolio_state(pid).id


def test_set_state_frozen_stamps_reason_and_frozen_at(db_session):
    """Freezing sets the state, the reason, and the ``frozen_at`` timestamp."""
    repo = AutonomyRepository(db_session)
    pid = uuid.uuid4()
    row = repo.set_state(
        AutonomyScope.PORTFOLIO, pid, AutonomyState.FROZEN, reason="killswitch"
    )
    assert row.state == "frozen"
    assert row.frozen_reason == "killswitch"
    assert row.frozen_at is not None


def test_set_state_back_to_armed_clears_frozen_fields(db_session):
    """Re-arming clears both ``frozen_reason`` and ``frozen_at``."""
    repo = AutonomyRepository(db_session)
    pid = uuid.uuid4()
    repo.set_state(AutonomyScope.PORTFOLIO, pid, AutonomyState.FROZEN, reason="x")
    row = repo.set_state(AutonomyScope.PORTFOLIO, pid, AutonomyState.ARMED)
    assert row.state == "armed"
    assert row.frozen_reason is None
    assert row.frozen_at is None


def test_set_state_fund_scope_targets_the_singleton(db_session):
    """A fund-scope ``set_state`` writes the single NULL-portfolio row."""
    repo = AutonomyRepository(db_session)
    repo.set_state(AutonomyScope.FUND, None, AutonomyState.FROZEN, reason="halt")
    row = repo.fund_state()
    assert row.state == "frozen"
    assert row.frozen_reason == "halt"


def test_consume_first_round_returns_true_once_then_false(db_session):
    """The first consume returns True; every later consume returns False."""
    repo = AutonomyRepository(db_session)
    pid = uuid.uuid4()
    assert repo.consume_first_round(pid) is True
    assert repo.consume_first_round(pid) is False


def test_consume_first_round_flips_the_flag(db_session):
    """Consuming the first round leaves ``first_round_gated`` False in the row."""
    repo = AutonomyRepository(db_session)
    pid = uuid.uuid4()
    repo.consume_first_round(pid)
    assert repo.portfolio_state(pid).first_round_gated is False


def test_consume_first_round_is_portfolio_scoped(db_session):
    """Consuming one portfolio's first round leaves another's untouched."""
    repo = AutonomyRepository(db_session)
    pid_a, pid_b = uuid.uuid4(), uuid.uuid4()
    repo.consume_first_round(pid_a)
    assert repo.portfolio_state(pid_b).first_round_gated is True
