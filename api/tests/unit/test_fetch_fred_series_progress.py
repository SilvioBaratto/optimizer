"""TDD coverage for per-series on_progress in fetch_fred_series.

Asserts the service-layer surface only — wiring through run_bulk_fred_fetch
is covered by behavioural tests in issue #569.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

from app.services.macro.macro_regime_service import MacroRegimeService


def _build_service_with_fake_scraper() -> MacroRegimeService:
    repo = MagicMock()
    repo.get_fred_latest_date.return_value = None
    repo.upsert_fred_observations.return_value = 0

    scraper = MagicMock()
    scraper.get_series_observations.return_value = {
        "status": "success",
        "observations": [],
    }

    svc = MacroRegimeService(
        repo=repo,
        ilsole_scraper=MagicMock(),
        te_scraper=MagicMock(),
        fred_scraper=scraper,
    )
    return svc


class TestFetchFredSeriesAcceptsOnProgress:
    def test_when_signature_inspected_on_progress_param_present(self) -> None:
        sig = inspect.signature(MacroRegimeService.fetch_fred_series)
        assert "on_progress" in sig.parameters


class TestFetchFredSeriesEmitsPerSeriesProgress:
    def test_when_three_ids_fetched_callback_invoked_three_times(self) -> None:
        svc = _build_service_with_fake_scraper()
        cb = MagicMock()

        svc.fetch_fred_series(
            series_ids=["DGS2", "DGS10", "VIXCLS"],
            incremental=False,
            on_progress=cb,
        )

        assert cb.call_count == 3

    def test_when_iterating_kwargs_carry_idx_total_and_series_id(self) -> None:
        svc = _build_service_with_fake_scraper()
        cb = MagicMock()
        ids = ["DGS2", "DGS10", "VIXCLS"]

        svc.fetch_fred_series(series_ids=ids, incremental=False, on_progress=cb)

        kwargs_seen = [c.kwargs for c in cb.call_args_list]
        assert kwargs_seen[0] == {"current": 1, "total": 3, "current_country": "DGS2"}
        assert kwargs_seen[1] == {"current": 2, "total": 3, "current_country": "DGS10"}
        assert kwargs_seen[2] == {"current": 3, "total": 3, "current_country": "VIXCLS"}
