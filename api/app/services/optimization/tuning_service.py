"""Tuning service: maps TuneRequest to optimizer library search calls.

Stateless functions following Single Responsibility Principle:
  - run_tune       — orchestrate search, return TuneResult
  - _build_search  — dispatch to grid or randomized search factory
  - _extract_tune_results — pull best_params, best_score, top_n from cv
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.orm import Session

from app.schemas.optimization.tuning import TuneRequest, TuneResult
from app.services._shared import ProgressCallback, _noop
from app.services.optimization.optimization_service import build_pipeline
from optimizer.tuning._factory import build_grid_search_cv, build_randomized_search_cv

logger = logging.getLogger(__name__)


def run_tune(
    session: Session,
    request: TuneRequest,
    on_progress: ProgressCallback | None = None,
) -> TuneResult:
    """Build estimator, run grid/randomized search, return TuneResult."""
    on_progress = on_progress or _noop
    on_progress(current=0, total=1, status="running")

    pipeline, returns_df = build_pipeline(
        request.tickers,
        request.start_date,
        request.end_date,
        request.optimizer_type,
        request.optimizer_config,
        session,
    )
    search_cv = _build_search(request, pipeline)
    search_cv.fit(returns_df)

    on_progress(current=1, total=1, status="completed")
    return _extract_tune_results(search_cv, request.top_n)


def _build_search(request: TuneRequest, pipeline: Any) -> Any:
    """Dispatch to grid or randomized search factory based on search_type."""
    if request.search_type == "randomized":
        return build_randomized_search_cv(pipeline, request.param_grid)
    return build_grid_search_cv(pipeline, request.param_grid)


def _to_list(value: Any) -> list:
    """Convert numpy array or list to plain list."""
    return value.tolist() if hasattr(value, "tolist") else list(value)


def _extract_tune_results(search_cv: Any, top_n: int) -> TuneResult:
    """Extract best_params, best_score, top_n candidates, and cv summary."""
    cv_results = search_cv.cv_results_
    ranked = sorted(
        zip(cv_results["rank_test_score"], cv_results["params"], strict=False),
        key=lambda x: x[0],
    )
    top_n_items = [{"rank": rank, "params": params} for rank, params in ranked[:top_n]]
    summary = {
        "mean_test_score": _to_list(cv_results["mean_test_score"]),
        "std_test_score": _to_list(cv_results["std_test_score"]),
    }
    return TuneResult(
        best_params=search_cv.best_params_,
        best_score=float(search_cv.best_score_),
        top_n=top_n_items,
        cv_results_summary=summary,
    )
