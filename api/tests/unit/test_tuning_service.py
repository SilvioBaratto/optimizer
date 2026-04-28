"""Unit tests for app.services.tuning_service.

Covers:
  - search_type="grid"        → build_grid_search_cv is called
  - search_type="randomized"  → build_randomized_search_cv is called
  - unknown optimizer_type    → ValueError propagated
"""

from __future__ import annotations

from datetime import date
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

_BUILD_PIPELINE = "app.services.tuning_service.build_pipeline"
_BUILD_GRID = "app.services.tuning_service.build_grid_search_cv"
_BUILD_RAND = "app.services.tuning_service.build_randomized_search_cv"


def _make_request(search_type: str = "grid", **overrides: Any):
    """Return a TuneRequest-like object without hitting Pydantic validation."""
    from app.schemas.tuning import TuneRequest

    data: dict[str, Any] = {
        "tickers": ["AAPL", "MSFT"],
        "start_date": date(2020, 1, 1),
        "end_date": date(2024, 1, 1),
        "optimizer_type": "mean_risk",
        "optimizer_config": {},
        "search_type": search_type,
        "param_grid": {"optimizer__risk_measure": ["variance"]},
        "n_iter": 10,
        "top_n": 3,
        "cv_config": None,
    }
    data.update(overrides)
    return TuneRequest(**data)


def _mock_pipeline():
    """Return a mock pipeline and returns_df."""
    mock_pipe = MagicMock()
    mock_returns = MagicMock()
    return mock_pipe, mock_returns


def _mock_search_cv():
    """Return a mock search CV with minimal cv_results_ structure."""
    search_cv = MagicMock()
    search_cv.best_params_ = {"optimizer__risk_measure": "variance"}
    search_cv.best_score_ = 0.85
    search_cv.cv_results_ = {
        "mean_test_score": [0.85, 0.70],
        "std_test_score": [0.05, 0.03],
        "params": [
            {"optimizer__risk_measure": "variance"},
            {"optimizer__risk_measure": "cvar"},
        ],
        "rank_test_score": [1, 2],
    }
    return search_cv


class TestRunTuneGrid:
    """search_type='grid' delegates to build_grid_search_cv."""

    def test_run_tune_grid_calls_build_grid_search(self) -> None:
        request = _make_request(search_type="grid")
        session = MagicMock()
        mock_pipe, mock_returns = _mock_pipeline()
        search_cv = _mock_search_cv()

        with (
            patch(_BUILD_PIPELINE, return_value=(mock_pipe, mock_returns)),
            patch(_BUILD_GRID, return_value=search_cv) as mock_grid,
            patch(_BUILD_RAND) as mock_rand,
        ):
            from app.services.tuning_service import run_tune

            run_tune(session, request)

        mock_grid.assert_called_once()
        mock_rand.assert_not_called()


class TestRunTuneRandomized:
    """search_type='randomized' delegates to build_randomized_search_cv."""

    def test_run_tune_randomized_calls_build_randomized_search(self) -> None:
        request = _make_request(search_type="randomized")
        session = MagicMock()
        mock_pipe, mock_returns = _mock_pipeline()
        search_cv = _mock_search_cv()

        with (
            patch(_BUILD_PIPELINE, return_value=(mock_pipe, mock_returns)),
            patch(_BUILD_GRID) as mock_grid,
            patch(_BUILD_RAND, return_value=search_cv) as mock_rand,
        ):
            from app.services.tuning_service import run_tune

            run_tune(session, request)

        mock_rand.assert_called_once()
        mock_grid.assert_not_called()


class TestRunTuneUnknownOptimizer:
    """Unknown optimizer_type causes build_pipeline to raise ValueError."""

    def test_run_tune_unknown_optimizer_raises_value_error(self) -> None:
        request = _make_request(optimizer_type="not_real")
        session = MagicMock()

        with patch(_BUILD_PIPELINE, side_effect=ValueError("Unsupported optimizer_type")):
            from app.services.tuning_service import run_tune

            with pytest.raises(ValueError, match="Unsupported optimizer_type"):
                run_tune(session, request)
