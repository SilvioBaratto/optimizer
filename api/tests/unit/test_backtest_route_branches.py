"""Branch coverage for _fail_run in app.api.v1.backtest.backtest.

Pure-unit: no HTTP client, no real DB.
The 202/409/200/404 route matrix is already covered by
tests/unit/test_backtest_routes.py.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

_MODULE = "app.api.v1.backtest.backtest"
_DB_MGR = f"{_MODULE}.database_manager"
_EXEC_REPO = f"{_MODULE}.ExecutionRepository"


def _import_fail_run():
    from app.api.v1.backtest.backtest import _fail_run

    return _fail_run


def _make_db_mock(session: MagicMock | None = None) -> MagicMock:
    """Return a database_manager mock whose get_session context manager works."""
    dm = MagicMock()
    inner_session = session or MagicMock()
    dm.get_session.return_value.__enter__.return_value = inner_session
    dm.get_session.return_value.__exit__.return_value = False
    return dm


class TestFailRun:
    def test_success_calls_update_backtest_status_with_failed(self):
        run_id = str(uuid.uuid4())
        error_msg = "something broke"

        inner_session = MagicMock()
        dm = _make_db_mock(inner_session)
        repo_mock = MagicMock()

        with (
            patch(_DB_MGR, dm),
            patch(_EXEC_REPO, return_value=repo_mock),
        ):
            _import_fail_run()(run_id, error_msg)

        repo_mock.update_backtest_status.assert_called_once_with(
            uuid.UUID(run_id),
            "failed",
            {"error_message": error_msg},
        )
        inner_session.commit.assert_called_once()

    def test_success_commit_is_called(self):
        run_id = str(uuid.uuid4())
        inner_session = MagicMock()
        dm = _make_db_mock(inner_session)
        repo_mock = MagicMock()

        with (
            patch(_DB_MGR, dm),
            patch(_EXEC_REPO, return_value=repo_mock),
        ):
            _import_fail_run()(run_id, "err")

        inner_session.commit.assert_called_once()

    def test_exception_from_get_session_is_swallowed(self):
        run_id = str(uuid.uuid4())
        dm = MagicMock()
        dm.get_session.side_effect = Exception("db down")

        with patch(_DB_MGR, dm):
            _import_fail_run()(run_id, "err")  # must not raise

    def test_exception_inside_context_manager_is_swallowed(self):
        run_id = str(uuid.uuid4())
        inner_session = MagicMock()
        dm = _make_db_mock(inner_session)
        repo_mock = MagicMock()
        repo_mock.update_backtest_status.side_effect = Exception("repo boom")

        with (
            patch(_DB_MGR, dm),
            patch(_EXEC_REPO, return_value=repo_mock),
        ):
            _import_fail_run()(run_id, "err")  # must not raise
