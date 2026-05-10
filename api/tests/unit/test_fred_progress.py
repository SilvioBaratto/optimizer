"""Lock per-series progress contract for fetch_fred_series (#569).

No DB. No network. Repo and scraper are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from app.services.macro_regime_service import MacroRegimeService


def _build_service() -> tuple[MacroRegimeService, MagicMock]:
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
    return svc, scraper


class TestFetchFredSeriesProgressContract:
    def test_when_two_series_fetched_spy_called_twice(self) -> None:
        svc, _ = _build_service()
        spy = MagicMock()

        svc.fetch_fred_series(series_ids=["DGS2", "DGS10"], on_progress=spy)

        assert spy.call_count == 2

    def test_when_iterating_first_call_carries_idx_total_and_first_series(
        self,
    ) -> None:
        svc, _ = _build_service()
        spy = MagicMock()

        svc.fetch_fred_series(series_ids=["DGS2", "DGS10"], on_progress=spy)

        assert spy.call_args_list[0].kwargs == {
            "current": 1,
            "total": 2,
            "current_country": "DGS2",
        }

    def test_when_iterating_second_call_carries_idx_total_and_second_series(
        self,
    ) -> None:
        svc, _ = _build_service()
        spy = MagicMock()

        svc.fetch_fred_series(series_ids=["DGS2", "DGS10"], on_progress=spy)

        assert spy.call_args_list[1].kwargs == {
            "current": 2,
            "total": 2,
            "current_country": "DGS10",
        }
