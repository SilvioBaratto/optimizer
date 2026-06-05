"""Empty-data and exception-path tests for tuning_service (issue #825).

Existing tests in tests/unit/test_tuning_service.py already cover:
  - search_type='grid' dispatches to build_grid_search_cv
  - search_type='randomized' dispatches to build_randomized_search_cv
  - unknown optimizer_type propagated as ValueError

tuning_service has NO explicit raises of its own; it propagates from
build_pipeline and search_cv.fit.  Gap analysis identifies:
  - _build_search default branch (non-'randomized' key falls through to grid)
  - _extract_tune_results with a minimal/empty cv_results_ structure
  - on_progress callback is invoked (status="running" then "completed")
  - empty param_grid does not raise (grid search accepts empty)
"""

from __future__ import annotations

from datetime import date
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

_BUILD_PIPELINE = "app.services.optimization.tuning_service.build_pipeline"
_BUILD_GRID = "app.services.optimization.tuning_service.build_grid_search_cv"
_BUILD_RAND = "app.services.optimization.tuning_service.build_randomized_search_cv"


def _make_request(search_type: str = "grid", **overrides: Any):
    from app.schemas.optimization.tuning import TuneRequest

    data: dict[str, Any] = {
        "tickers": ["AAPL", "MSFT"],
        "start_date": date(2020, 1, 1),
        "end_date": date(2024, 1, 1),
        "optimizer_type": "mean_risk",
        "optimizer_config": {},
        "search_type": search_type,
        "param_grid": {"optimizer__risk_measure": ["variance"]},
        "n_iter": 1,
        "top_n": 1,
        "cv_config": None,
    }
    data.update(overrides)
    return TuneRequest(**data)


def _mock_search_cv(n_candidates: int = 1) -> MagicMock:
    search_cv = MagicMock()
    search_cv.best_params_ = {"optimizer__risk_measure": "variance"}
    search_cv.best_score_ = 0.75
    search_cv.cv_results_ = {
        "mean_test_score": [0.75] * n_candidates,
        "std_test_score": [0.02] * n_candidates,
        "params": [{"optimizer__risk_measure": "variance"}] * n_candidates,
        "rank_test_score": list(range(1, n_candidates + 1)),
    }
    return search_cv


# ---------------------------------------------------------------------------
# _extract_tune_results — empty cv_results_
# ---------------------------------------------------------------------------


class TestExtractTuneResultsMinimal:
    """_extract_tune_results with a single-candidate cv_results_ structure."""

    def test_when_single_candidate_then_top_n_contains_one_entry(self) -> None:
        from app.services.optimization.tuning_service import _extract_tune_results

        search_cv = _mock_search_cv(n_candidates=1)
        result = _extract_tune_results(search_cv, top_n=5)

        assert len(result.top_n) == 1
        assert result.top_n[0]["rank"] == 1

    def test_when_top_n_exceeds_candidates_then_clamped_to_available(self) -> None:
        """top_n=10 with 2 candidates → only 2 results returned, no IndexError."""
        from app.services.optimization.tuning_service import _extract_tune_results

        search_cv = _mock_search_cv(n_candidates=2)
        result = _extract_tune_results(search_cv, top_n=10)

        assert len(result.top_n) == 2

    def test_when_cv_results_present_then_best_score_is_float(self) -> None:
        from app.services.optimization.tuning_service import _extract_tune_results

        search_cv = _mock_search_cv()
        result = _extract_tune_results(search_cv, top_n=1)

        assert isinstance(result.best_score, float)

    def test_when_cv_results_present_then_summary_lists_are_lists(self) -> None:
        from app.services.optimization.tuning_service import _extract_tune_results

        search_cv = _mock_search_cv()
        result = _extract_tune_results(search_cv, top_n=1)

        assert isinstance(result.cv_results_summary["mean_test_score"], list)
        assert isinstance(result.cv_results_summary["std_test_score"], list)


# ---------------------------------------------------------------------------
# _build_search — default branch (grid when search_type != 'randomized')
# ---------------------------------------------------------------------------


class TestBuildSearchDefaultBranch:
    """Non-'randomized' search_type falls through to grid search factory."""

    def test_when_search_type_not_randomized_then_grid_search_called(self) -> None:
        from app.services.optimization.tuning_service import _build_search

        request = _make_request(search_type="grid")
        pipeline = MagicMock()

        with (
            patch(_BUILD_GRID, return_value=MagicMock()) as mock_grid,
            patch(_BUILD_RAND) as mock_rand,
        ):
            _build_search(request, pipeline)

        mock_grid.assert_called_once()
        mock_rand.assert_not_called()


# ---------------------------------------------------------------------------
# run_tune — on_progress callback invocation
# ---------------------------------------------------------------------------


class TestRunTuneProgressCallback:
    """on_progress is called with status='running' then 'completed'."""

    def test_when_on_progress_provided_then_called_with_running_and_completed(
        self,
    ) -> None:
        request = _make_request(search_type="grid")
        session = MagicMock()
        search_cv = _mock_search_cv()
        calls: list[dict] = []

        def progress_spy(**kwargs: Any) -> None:
            calls.append(dict(kwargs))

        with (
            patch(_BUILD_PIPELINE, return_value=(MagicMock(), MagicMock())),
            patch(_BUILD_GRID, return_value=search_cv),
        ):
            from app.services.optimization.tuning_service import run_tune

            run_tune(session, request, on_progress=progress_spy)

        statuses = [c.get("status") for c in calls]
        assert "running" in statuses
        assert "completed" in statuses

    def test_when_on_progress_none_then_no_exception(self) -> None:
        """Default noop callback must not raise."""
        request = _make_request(search_type="grid")
        session = MagicMock()
        search_cv = _mock_search_cv()

        with (
            patch(_BUILD_PIPELINE, return_value=(MagicMock(), MagicMock())),
            patch(_BUILD_GRID, return_value=search_cv),
        ):
            from app.services.optimization.tuning_service import run_tune

            result = run_tune(session, request, on_progress=None)

        assert result is not None


# ---------------------------------------------------------------------------
# run_tune — single-param grid (minimum valid input)
# ---------------------------------------------------------------------------


class TestRunTuneSingleParamGrid:
    """Single-entry param_grid produces a valid TuneResult without raising.

    Note: the TuneRequest schema enforces min_length=1 on param_grid, so an
    empty dict is rejected at the Pydantic boundary — tested here with the
    minimum valid case (one key/value pair).
    """

    def test_when_single_param_grid_entry_then_tune_result_returned(self) -> None:
        request = _make_request(
            search_type="grid",
            param_grid={"optimizer__risk_measure": ["variance"]},
        )
        session = MagicMock()
        search_cv = _mock_search_cv()

        with (
            patch(_BUILD_PIPELINE, return_value=(MagicMock(), MagicMock())),
            patch(_BUILD_GRID, return_value=search_cv),
        ):
            from app.services.optimization.tuning_service import run_tune

            result = run_tune(session, request)

        assert result.best_score == pytest.approx(0.75)
