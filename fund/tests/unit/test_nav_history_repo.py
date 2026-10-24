"""``NavHistoryRepository`` upsert + read round-trips on SQLite.

The ``nav_history`` *model* lives in ``portopt_db``; its *behavior* lives here,
mirroring the ``positions`` / ``agent_runs`` split. The repo opens no session of
its own (caller injects one) and does not commit. Its module-level imports stay
clear of the deep-agent framework so the scheduler can lazy-import it without
dragging ``deepagents`` into a bare ``import fund.scheduler``.
"""

from __future__ import annotations

import ast
import datetime as dt
import uuid
from decimal import Decimal
from pathlib import Path

from portopt_db.models import NavHistory

from fund.audit import nav_history_repository
from fund.audit.nav_history_repository import NavHistoryRepository

_PID = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
_PID2 = uuid.UUID("ffffffff-1111-2222-3333-444444444abc")


def _row(pid, d, nav, benchmark=None):
    return {
        "portfolio_id": pid,
        "date": d,
        "nav": Decimal(str(nav)),
        "benchmark": benchmark,
    }


def test_upsert_writes_rows_and_is_idempotent(db_session):
    """Upserting the same NAV payload twice converges to one row per date."""
    repo = NavHistoryRepository(db_session)
    rows = [
        _row(_PID, dt.date(2026, 10, 1), 100.0),
        _row(_PID, dt.date(2026, 10, 2), 101.0),
    ]
    assert repo.upsert_navs(rows) == 2
    db_session.flush()
    assert db_session.query(NavHistory).count() == 2
    assert repo.upsert_navs(rows) == 2
    db_session.flush()
    assert db_session.query(NavHistory).count() == 2


def test_upsert_updates_nav_on_conflict(db_session):
    """A later upsert for an existing (portfolio, date) overwrites nav/benchmark."""
    repo = NavHistoryRepository(db_session)
    d = dt.date(2026, 10, 1)
    repo.upsert_navs([_row(_PID, d, 100.0, Decimal("10"))])
    db_session.flush()
    repo.upsert_navs([_row(_PID, d, 105.0, Decimal("11"))])
    db_session.flush()

    got = db_session.query(NavHistory).one()
    assert Decimal(str(got.nav)) == Decimal("105")
    assert Decimal(str(got.benchmark)) == Decimal("11")


def test_upsert_empty_returns_zero(db_session):
    """An empty payload writes nothing and returns zero."""
    assert NavHistoryRepository(db_session).upsert_navs([]) == 0


def test_upsert_skips_incomplete_rows(db_session):
    """Rows missing portfolio/date/nav are dropped, not written."""
    repo = NavHistoryRepository(db_session)
    rows = [
        {"portfolio_id": _PID, "date": dt.date(2026, 10, 1)},  # no nav
        _row(_PID, dt.date(2026, 10, 2), 100.0),
    ]
    assert repo.upsert_navs(rows) == 1
    db_session.flush()
    assert db_session.query(NavHistory).count() == 1


def test_get_navs_ordered_and_range_filtered(db_session):
    """get_navs returns an ascending-by-date series within the window."""
    repo = NavHistoryRepository(db_session)
    repo.upsert_navs(
        [
            _row(_PID, dt.date(2026, 10, 3), 103.0),
            _row(_PID, dt.date(2026, 10, 1), 101.0),
            _row(_PID, dt.date(2026, 10, 2), 102.0),
            _row(_PID2, dt.date(2026, 10, 2), 999.0),
        ]
    )
    db_session.flush()

    got = repo.get_navs(_PID, dt.date(2026, 10, 1), dt.date(2026, 10, 2))
    assert [n.date for n in got] == [dt.date(2026, 10, 1), dt.date(2026, 10, 2)]
    assert all(n.portfolio_id == _PID for n in got)


def test_get_navs_without_range_returns_full_series(db_session):
    """Omitting the window returns every row for the portfolio, ordered."""
    repo = NavHistoryRepository(db_session)
    repo.upsert_navs(
        [
            _row(_PID, dt.date(2026, 10, 2), 102.0),
            _row(_PID, dt.date(2026, 10, 1), 101.0),
        ]
    )
    db_session.flush()
    got = repo.get_navs(_PID)
    assert [n.date for n in got] == [dt.date(2026, 10, 1), dt.date(2026, 10, 2)]


def test_last_nav_before_returns_newest_strictly_earlier_row(db_session):
    """The anchor row is the latest one before the bound, scoped to the portfolio."""
    repo = NavHistoryRepository(db_session)
    repo.upsert_navs(
        [
            _row(_PID, dt.date(2026, 10, 1), 101.0),
            _row(_PID, dt.date(2026, 10, 5), 105.0),
            _row(_PID, dt.date(2026, 10, 9), 109.0),
            _row(_PID2, dt.date(2026, 10, 7), 999.0),
        ]
    )
    db_session.flush()

    got = repo.last_nav_before(_PID, dt.date(2026, 10, 9))

    assert got is not None
    assert got.date == dt.date(2026, 10, 5)
    assert float(got.nav) == 105.0


def test_last_nav_before_is_none_without_an_earlier_row(db_session):
    repo = NavHistoryRepository(db_session)
    repo.upsert_navs([_row(_PID, dt.date(2026, 10, 5), 105.0)])
    db_session.flush()

    assert repo.last_nav_before(_PID, dt.date(2026, 10, 5)) is None
    assert repo.last_nav_before(_PID2, dt.date(2026, 10, 9)) is None


def test_module_declares_no_agent_stack_imports():
    """The repo module's own imports reference no agent stack.

    The scheduler stays agent-stack-free by lazy-importing this repo inside the
    step (``fund.audit`` loads its langgraph bootstrap on demand; see
    ``test_audit_submodule_imports``). What must hold here is that this module
    declares no ``deepagents`` / ``langgraph`` / ``fund.agents`` import of its own,
    so the lazy import adds no deep-agent dependency.
    """
    forbidden = ("deepagents", "langgraph", "fund.agents")
    source = Path(nav_history_repository.__file__).read_text(encoding="utf-8")
    names: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    offenders = [n for n in names if any(n.startswith(f) for f in forbidden)]
    assert offenders == [], f"agent-stack imports leaked into the module: {offenders}"
