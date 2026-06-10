"""Branch coverage for optimization route helpers and background workers.

Covers:
  - _run_tune_bg     (app.api.v1.optimization.tune)
  - _run_walk_forward_bg (app.api.v1.optimization.validate)
  - _validate_optimizer_type (app.api.v1.optimization.optimize)
  - _fail_run        (app.api.v1.optimization.optimize)
  - _execute_and_persist branches (app.api.v1.optimization.optimize)

Pure-unit: no HTTP client, no real DB.
The 202/409/200/404 route matrix is already covered by
tests/unit/test_tune_routes.py / test_validate_walk_forward.py /
test_optimize_routes.py.
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import CancelledError
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_TUNE_MODULE = "app.api.v1.optimization.tune"
_VALIDATE_MODULE = "app.api.v1.optimization.validate"
_OPTIMIZE_MODULE = "app.api.v1.optimization.optimize"


def _make_db_mock() -> MagicMock:
    dm = MagicMock()
    dm.get_session.return_value.__enter__.return_value = MagicMock()
    dm.get_session.return_value.__exit__.return_value = False
    return dm


def _statuses(mock_update) -> list[str]:
    return [
        c.kwargs.get("status") or (c.args[1] if len(c.args) > 1 else "")
        for c in mock_update.call_args_list
    ]


# ---------------------------------------------------------------------------
# Helper: generic wrapper-branch exerciser
# ---------------------------------------------------------------------------


def _exercise_wrapper(
    module: str,
    fn_name: str,
    service_patch: str,
    update_patch: str,
    service_return=None,
    service_side_effect=None,
    cancel_event: threading.Event | None = None,
    extra_kwargs: dict | None = None,
):
    """Import and call a background worker with controlled patches."""
    import importlib

    mod = importlib.import_module(module.replace(".", "/").replace("/", "."))
    worker = getattr(mod, fn_name)
    request = MagicMock()
    dm = _make_db_mock()

    with (
        patch(service_patch, return_value=service_return, side_effect=service_side_effect),
        patch(f"{module}.database_manager", dm),
        patch(update_patch) as mock_update,
    ):
        kwargs = {"cancel_event": cancel_event} if cancel_event is not None else {}
        if extra_kwargs:
            kwargs.update(extra_kwargs)
        worker("test-job-id", request, **kwargs)

    return mock_update


# ===========================================================================
# FILE-2a: Tune wrapper
# ===========================================================================


class TestTuneWrapper:
    _SVC = f"{_TUNE_MODULE}.run_tune"
    _UPD = f"{_TUNE_MODULE}._tune_job_service.update_job"

    def _run(self, svc_return=None, svc_side=None, event=None):
        result_mock = MagicMock()
        result_mock.model_dump.return_value = {}
        return _exercise_wrapper(
            _TUNE_MODULE,
            "_run_tune_bg",
            self._SVC,
            self._UPD,
            service_return=result_mock if svc_return is None else svc_return,
            service_side_effect=svc_side,
            cancel_event=event,
        )

    def test_success_sets_running_then_completed(self):
        mock_update = self._run()
        assert "running" in _statuses(mock_update)
        assert "completed" in _statuses(mock_update)

    def test_success_result_is_model_dump(self):
        mock_update = self._run()
        completed = next(
            c for c in mock_update.call_args_list
            if c.kwargs.get("status") == "completed"
        )
        assert "result" in completed.kwargs

    def test_cancelled_error_stops_without_completed_or_failed(self):
        mock_update = self._run(svc_side=CancelledError())
        assert "completed" not in _statuses(mock_update)
        assert "failed" not in _statuses(mock_update)

    def test_exception_sets_failed_with_error(self):
        mock_update = self._run(svc_side=Exception("tune-boom"))
        failed = [c for c in mock_update.call_args_list if c.kwargs.get("status") == "failed"]
        assert len(failed) == 1
        assert failed[0].kwargs.get("error") == "tune-boom"

    def test_explicit_cancel_event_covered(self):
        mock_update = self._run(event=threading.Event())
        assert "completed" in _statuses(mock_update)


# ===========================================================================
# FILE-2b: Validate wrapper
# ===========================================================================


class TestValidateWrapper:
    _SVC = f"{_VALIDATE_MODULE}.run_validation"
    _UPD = f"{_VALIDATE_MODULE}._job_service.update_job"

    def _run(self, svc_return=None, svc_side=None, event=None):
        return _exercise_wrapper(
            _VALIDATE_MODULE,
            "_run_walk_forward_bg",
            self._SVC,
            self._UPD,
            service_return=svc_return or {"folds": []},
            service_side_effect=svc_side,
            cancel_event=event,
        )

    def test_success_sets_running_then_completed(self):
        mock_update = self._run()
        assert "running" in _statuses(mock_update)
        assert "completed" in _statuses(mock_update)

    def test_success_result_passed_to_update(self):
        mock_update = self._run(svc_return={"folds": [1, 2]})
        completed = next(
            c for c in mock_update.call_args_list
            if c.kwargs.get("status") == "completed"
        )
        assert completed.kwargs.get("result") == {"folds": [1, 2]}

    def test_cancelled_error_stops_without_completed_or_failed(self):
        mock_update = self._run(svc_side=CancelledError())
        assert "completed" not in _statuses(mock_update)
        assert "failed" not in _statuses(mock_update)

    def test_exception_sets_failed_with_error(self):
        mock_update = self._run(svc_side=Exception("val-boom"))
        failed = [c for c in mock_update.call_args_list if c.kwargs.get("status") == "failed"]
        assert len(failed) == 1
        assert failed[0].kwargs.get("error") == "val-boom"

    def test_explicit_cancel_event_covered(self):
        mock_update = self._run(event=threading.Event())
        assert "completed" in _statuses(mock_update)


# ===========================================================================
# FILE-2c: optimize helpers
# ===========================================================================


class TestOptimizeValidateType:
    def test_unknown_optimizer_type_raises_422(self):
        from app.api.v1.optimization.optimize import _validate_optimizer_type

        with pytest.raises(HTTPException) as exc_info:
            _validate_optimizer_type("bogus_type")

        assert exc_info.value.status_code == 422


class TestOptimizeFailRun:
    def test_success_calls_update_and_commit(self):
        from app.api.v1.optimization.optimize import _fail_run

        session = MagicMock()
        run_id = uuid.uuid4()
        repo_mock = MagicMock()

        with patch(f"{_OPTIMIZE_MODULE}.ExecutionRepository", return_value=repo_mock):
            _fail_run(session, run_id, "something went wrong")

        repo_mock.update_optimization_status.assert_called_once_with(
            run_id, "failed", {"error_message": "something went wrong"}
        )
        session.commit.assert_called_once()

    def test_exception_in_update_is_swallowed(self):
        from app.api.v1.optimization.optimize import _fail_run

        session = MagicMock()
        run_id = uuid.uuid4()
        repo_mock = MagicMock()
        repo_mock.update_optimization_status.side_effect = Exception("db error")

        with patch(f"{_OPTIMIZE_MODULE}.ExecutionRepository", return_value=repo_mock):
            _fail_run(session, run_id, "error")  # must not raise


class TestExecuteAndPersist:
    """Branch coverage for _execute_and_persist."""

    _PIPELINE_PATCH = f"{_OPTIMIZE_MODULE}.build_pipeline"
    _EXTRACT_PATCH = f"{_OPTIMIZE_MODULE}.extract_results"
    _REPO_PATCH = f"{_OPTIMIZE_MODULE}.ExecutionRepository"

    _FAKE_RESULTS = {
        "weights": {},
        "metrics": {},
        "risk_contributions": {},
        "efficient_frontier": {},
    }

    def _make_pipeline_mock(self):
        pipeline = MagicMock()
        pipeline.named_steps = {"optimizer": MagicMock()}
        return pipeline, MagicMock()

    def _make_repo_mock(self):
        repo = MagicMock()
        repo.create_optimization_run.return_value = MagicMock(id=uuid.uuid4())
        repo.update_optimization_status.return_value = MagicMock(id=uuid.uuid4())
        return repo

    def test_run_id_provided_calls_update_optimization_status(self):
        from app.api.v1.optimization.optimize import _execute_and_persist
        from app.services.optimization.optimization_service import FRONTIER_TYPES

        frontier_type = next(iter(FRONTIER_TYPES))
        request = MagicMock()
        request.optimizer_type = frontier_type
        request.config = {"objective": "minimize_risk"}
        request.tickers = ["AAPL"]
        request.start_date = "2020-01-01"
        request.end_date = "2024-01-01"

        session = MagicMock()
        run_id = uuid.uuid4()
        repo_mock = self._make_repo_mock()
        pipeline_mock, returns_df_mock = self._make_pipeline_mock()

        with (
            patch(self._PIPELINE_PATCH, return_value=(pipeline_mock, returns_df_mock)),
            patch(self._EXTRACT_PATCH, return_value=self._FAKE_RESULTS),
            patch(self._REPO_PATCH, return_value=repo_mock),
        ):
            _execute_and_persist(request, session, run_id=run_id, job_id=None)

        repo_mock.update_optimization_status.assert_called_once()

    def test_run_id_none_calls_create_optimization_run(self):
        from app.api.v1.optimization.optimize import _execute_and_persist

        request = MagicMock()
        request.optimizer_type = "definitely_not_frontier"
        request.config = None
        request.tickers = ["AAPL"]
        request.start_date = "2020-01-01"
        request.end_date = "2024-01-01"

        session = MagicMock()
        repo_mock = self._make_repo_mock()
        pipeline_mock, returns_df_mock = self._make_pipeline_mock()

        with (
            patch(self._PIPELINE_PATCH, return_value=(pipeline_mock, returns_df_mock)),
            patch(self._EXTRACT_PATCH, return_value=self._FAKE_RESULTS),
            patch(self._REPO_PATCH, return_value=repo_mock),
        ):
            _execute_and_persist(request, session, run_id=None, job_id=None)

        repo_mock.create_optimization_run.assert_called_once()
        session.commit.assert_called_once()

    def test_frontier_non_minimize_risk_skips_set_params(self):
        from app.api.v1.optimization.optimize import _execute_and_persist
        from app.services.optimization.optimization_service import FRONTIER_TYPES

        frontier_type = next(iter(FRONTIER_TYPES))
        request = MagicMock()
        request.optimizer_type = frontier_type
        request.config = {"objective": "maximize_return"}
        request.tickers = ["AAPL"]
        request.start_date = "2020-01-01"
        request.end_date = "2024-01-01"

        session = MagicMock()
        repo_mock = self._make_repo_mock()
        pipeline_mock, returns_df_mock = self._make_pipeline_mock()
        optimizer_step = pipeline_mock.named_steps["optimizer"]

        with (
            patch(self._PIPELINE_PATCH, return_value=(pipeline_mock, returns_df_mock)),
            patch(self._EXTRACT_PATCH, return_value=self._FAKE_RESULTS),
            patch(self._REPO_PATCH, return_value=repo_mock),
        ):
            _execute_and_persist(request, session, run_id=None, job_id=None)

        # set_params should NOT be called because objective != "minimize_risk"
        optimizer_step.set_params.assert_not_called()
