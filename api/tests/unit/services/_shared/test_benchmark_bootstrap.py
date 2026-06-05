"""Unit tests for ``_shared/_benchmark_bootstrap`` (issue #826).

NOTE on AC: the criterion 'mocked client/prices -> DataFrame of expected shape'
does not match this module — ``bootstrap_benchmarks`` returns a diagnostic
*dict* (``checked``/``seeded``/``up_to_date``/``errors``), not a DataFrame.
These tests assert the real summary contract.  ``DashboardRepository`` and
``seed_reference_indices`` are patched at the import site; the session factory
is a stub context manager.  No DB / yfinance.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.services._shared._benchmark_bootstrap import (
    _compute_delta,
    bootstrap_benchmarks,
)

_REPO = "app.services._shared._benchmark_bootstrap.DashboardRepository"
_SEED = "app.services._shared._benchmark_bootstrap.seed_reference_indices"
_TODAY = date(2024, 6, 10)


def _cov(count: int, latest: date | None) -> dict[str, tuple[int, date | None]]:
    return {"SPY": (count, latest)}


def _session_factory():
    @contextmanager
    def _factory():
        yield MagicMock()

    return _factory


def _seed_result(errors: list[str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        errors=errors or [],
        tickers_processed=1,
        instruments_created=1,
        instruments_already_present=0,
    )


# ---------------------------------------------------------------------------
# _compute_delta (pure)
# ---------------------------------------------------------------------------


class TestComputeDelta:
    def test_when_zero_count_then_stale(self) -> None:
        assert _compute_delta(_cov(0, _TODAY), _TODAY, stale_days=2) == ["SPY"]

    def test_when_latest_none_then_stale(self) -> None:
        assert _compute_delta(_cov(10, None), _TODAY, stale_days=2) == ["SPY"]

    def test_when_latest_older_than_cutoff_then_stale(self) -> None:
        old = date(2024, 6, 1)  # > 2 days before _TODAY
        assert _compute_delta(_cov(10, old), _TODAY, stale_days=2) == ["SPY"]

    def test_when_fresh_then_not_stale(self) -> None:
        assert _compute_delta(_cov(10, _TODAY), _TODAY, stale_days=2) == []


# ---------------------------------------------------------------------------
# bootstrap_benchmarks
# ---------------------------------------------------------------------------


class TestBootstrapBenchmarks:
    def test_when_no_tickers_then_skips_with_zero_checked(self) -> None:
        summary = bootstrap_benchmarks(
            _session_factory(), MagicMock(), tickers=[], today=_TODAY
        )
        assert summary == {
            "checked": 0,
            "seeded": [],
            "up_to_date": [],
            "errors": [],
        }

    def test_when_all_fresh_then_up_to_date_and_no_seed(self) -> None:
        with patch(_REPO) as MockRepo, patch(_SEED) as mock_seed:
            MockRepo.return_value.get_benchmark_coverage.return_value = {
                "SPY": (10, _TODAY)
            }
            summary = bootstrap_benchmarks(
                _session_factory(), MagicMock(), tickers=["SPY"], today=_TODAY
            )

        assert summary["up_to_date"] == ["SPY"]
        assert summary["seeded"] == []
        mock_seed.assert_not_called()

    def test_when_stale_then_seeds_delta(self) -> None:
        with patch(_REPO) as MockRepo, patch(_SEED) as mock_seed:
            MockRepo.return_value.get_benchmark_coverage.return_value = {
                "SPY": (0, None)
            }
            mock_seed.return_value = _seed_result()
            summary = bootstrap_benchmarks(
                _session_factory(), MagicMock(), tickers=["SPY"], today=_TODAY
            )

        mock_seed.assert_called_once()
        assert summary["seeded"] == ["SPY"]
        assert summary["errors"] == []

    def test_when_coverage_query_raises_then_error_captured(self) -> None:
        with patch(_REPO) as MockRepo:
            MockRepo.return_value.get_benchmark_coverage.side_effect = RuntimeError(
                "db down"
            )
            summary = bootstrap_benchmarks(
                _session_factory(), MagicMock(), tickers=["SPY"], today=_TODAY
            )

        assert summary["errors"] == ["coverage_query: db down"]

    def test_when_seed_raises_then_error_captured(self) -> None:
        with patch(_REPO) as MockRepo, patch(_SEED) as mock_seed:
            MockRepo.return_value.get_benchmark_coverage.return_value = {
                "SPY": (0, None)
            }
            mock_seed.side_effect = RuntimeError("yf rate limit")
            summary = bootstrap_benchmarks(
                _session_factory(), MagicMock(), tickers=["SPY"], today=_TODAY
            )

        assert summary["errors"] == ["seed: yf rate limit"]
