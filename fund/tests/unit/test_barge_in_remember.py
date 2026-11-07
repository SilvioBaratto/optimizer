"""A steer that also remembers its directive saves the directive on its own.

The respond consumes the gate on the checkpointer's own connection, so the audit row and
run status it writes must commit before the (separate) mandate write that remembers the
directive; otherwise a failing mandate write rolled the decision's audit trail back
after the PM had already acted on it.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from _fund_fakes import PORTFOLIO_ID, make_mandate
from _gate_env import make_gate_env

from fund import observe
from fund.audit import AgentRunRepository, HumanAuditLogRepository, MandateRepository
from fund.tui.app import ShellApp

_DIRECTIVE = "Always keep 5% cash."


@pytest.fixture
def env(tmp_path):
    """A run paused at its gate plus the persisted mandate the directive lands in."""
    harness = make_gate_env(tmp_path / "remember.db", max_orders=2)
    with harness.get_session() as session:
        MandateRepository(session).upsert(make_mandate())
        session.commit()
    yield harness
    harness.dispose()


def _shell(env) -> ShellApp:
    return ShellApp(
        PORTFOLIO_ID,
        session_factory=env.get_session,
        persistence=env.persistence,
        model_factory=env.model_factory,
        poll_interval=30.0,
    )


async def _steer(app, pilot, env, *, remember: bool) -> None:
    worker = app._barge_in_worker(env.run_id, PORTFOLIO_ID, _DIRECTIVE, remember)
    await worker.wait()
    await pilot.pause()


@pytest.mark.asyncio
async def test_a_failed_directive_save_keeps_the_respond_audited(
    env, monkeypatch
) -> None:
    """The decision's audit row and run status survive; the status line says why."""

    def refuse(self, mandate, **_kwargs):
        """Fail the way a lost pin race or a dropped connection would."""
        raise RuntimeError("mandate write failed")

    monkeypatch.setattr(MandateRepository, "upsert", refuse)
    app = _shell(env)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _steer(app, pilot, env, remember=True)

        assert "directive not saved" in app.last_status
        assert "mandate write failed" in app.last_status
    with env.get_session() as session:
        assert AgentRunRepository(session).get_run(env.run_id).status == "responded"
        rows = HumanAuditLogRepository(session).list_recent(action="respond")
        assert [r.run_id for r in rows] == [env.run_id]
        assert observe.get_mandate(session, PORTFOLIO_ID).remembered_directives == ()


@pytest.mark.asyncio
async def test_remembering_changes_only_the_directives(env) -> None:
    """The mandate keeps every other field and the audit row records the base."""
    with env.get_session() as session:
        MandateRepository(session).upsert(
            make_mandate().model_copy(update={"capital": Decimal("42000")})
        )
        session.commit()
    app = _shell(env)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _steer(app, pilot, env, remember=True)

    with env.get_session() as session:
        stored = observe.get_mandate(session, PORTFOLIO_ID)
        assert stored.remembered_directives == (_DIRECTIVE,)
        assert stored.capital == Decimal("42000")
        (row,) = HumanAuditLogRepository(session).list_recent(action="mandate_edit")
        assert row.prev_value["remembered_directives"] == []


@pytest.mark.asyncio
async def test_a_directive_already_remembered_is_not_added_twice(env) -> None:
    """Repeating a remembered steer neither duplicates nor re-audits the directive."""
    app = _shell(env)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app._remember_directive(PORTFOLIO_ID, _DIRECTIVE)
        app._remember_directive(PORTFOLIO_ID, _DIRECTIVE)

    with env.get_session() as session:
        stored = observe.get_mandate(session, PORTFOLIO_ID)
        assert stored.remembered_directives == (_DIRECTIVE,)
        assert (
            len(HumanAuditLogRepository(session).list_recent(action="mandate_edit"))
            == 1
        )
