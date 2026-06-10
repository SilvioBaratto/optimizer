"""Branch-coverage tests for app/services/macro/macro_regime_service.py.

Maps term-missing ranges to concrete conditions not covered by:
  - tests/unit/services/macro/test_macro_regime_service.py
  - tests/unit/test_macro_regime_service_fred_key.py

Uncovered branches:
  Line  47->60  — fred_scraper property when _fred_scraper is already set
  Lines 77-106  — fetch_and_store: None countries (defaults), exception per-country
  Lines 170-172 — fetch_country: include_bonds=True but bond_yields empty dict
  Lines 195-196 — fetch_country: te success but indicators empty dict
  Lines 250-292 — fetch_fred_series: incremental=True with/without latest date,
                  result error status, fetch exception per-series
  Lines 304-370 — fetch_macro_news: yfinance happy path, scraper=None/not None,
                  publish_time parsing, full_content filter, upsert exception
  Line  375     — get_portfolio_countries returns list of PORTFOLIO_COUNTRIES
  Lines 425,429-432 — run_bulk_macro_fetch: exception path per-country (session.rollback)
  Lines 474-505 — run_bulk_fred_fetch happy path
  Lines 522-542 — run_macro_news_fetch happy path

Do NOT duplicate tests already in the two existing test files.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.services.macro.macro_regime_service import MacroRegimeService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_service(
    *,
    ilsole_forecasts=None,
    te_response=None,
    fred_scraper: object = None,
    repo: MagicMock | None = None,
) -> tuple[MacroRegimeService, MagicMock]:
    mock_repo = repo or MagicMock()
    mock_repo.upsert_economic_indicator.return_value = 1
    mock_repo.upsert_economic_indicator_observation.return_value = 1
    mock_repo.upsert_te_indicators.return_value = 3
    mock_repo.upsert_te_observations.return_value = 3
    mock_repo.upsert_bond_yields.return_value = 2
    mock_repo.upsert_bond_yield_observations.return_value = 2

    mock_ilsole = MagicMock()
    if isinstance(ilsole_forecasts, Exception):
        mock_ilsole.get_forecasts.side_effect = ilsole_forecasts
    else:
        mock_ilsole.get_forecasts.return_value = ilsole_forecasts

    mock_te = MagicMock()
    if isinstance(te_response, Exception):
        mock_te.get_country_indicators.side_effect = te_response
    else:
        mock_te.get_country_indicators.return_value = te_response

    service = MacroRegimeService(
        repo=mock_repo,
        ilsole_scraper=mock_ilsole,
        te_scraper=mock_te,
        fred_scraper=fred_scraper,  # type: ignore[arg-type]
    )
    return service, mock_repo


_TE_SUCCESS = {
    "status": "success",
    "indicators": {"inflation_rate": 3.1},
    "bond_yields": {"10y": 4.2},
}
_TE_SUCCESS_EMPTY_INDICATORS = {
    "status": "success",
    "indicators": {},
    "bond_yields": {},
}
_TE_SUCCESS_EMPTY_BONDS = {
    "status": "success",
    "indicators": {"inflation_rate": 3.1},
    "bond_yields": {},
}


# ---------------------------------------------------------------------------
# fred_scraper property — already-set branch (line 47->60)
# ---------------------------------------------------------------------------


class TestFredScraperPropertyAlreadySet:
    def test_when_fred_scraper_already_set_then_returned_directly(self) -> None:
        fake_scraper = MagicMock()
        service, _ = _build_service(fred_scraper=fake_scraper)
        assert service.fred_scraper is fake_scraper

    def test_when_fred_scraper_none_and_key_present_then_constructed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.config import settings
        from app.services.macro import macro_regime_service as svc_mod

        monkeypatch.setattr(settings, "fred_api_key", "test-key", raising=False)
        fake_cls = MagicMock(return_value=MagicMock())
        monkeypatch.setattr(svc_mod, "FredScraper", fake_cls)

        service, _ = _build_service(fred_scraper=None)
        result = service.fred_scraper
        assert result is not None
        fake_cls.assert_called_once_with(api_key="test-key")


# ---------------------------------------------------------------------------
# fetch_and_store — None countries defaults + per-country exception
# ---------------------------------------------------------------------------


class TestFetchAndStore:
    def test_when_countries_none_then_portfolio_countries_used(self) -> None:
        service, _ = _build_service(
            ilsole_forecasts={"gdp_growth_6m": 2.0},
            te_response=_TE_SUCCESS,
        )
        result = service.fetch_and_store(countries=None)
        # Should process PORTFOLIO_COUNTRIES — counts will be a non-empty dict
        assert "counts" in result
        assert "errors" in result

    def test_when_fetch_country_raises_then_error_accumulated_and_loop_continues(
        self,
    ) -> None:
        service, _ = _build_service(te_response=_TE_SUCCESS)
        # Patch fetch_country to raise on second call
        call_results = [
            {"counts": {"te_indicators": 1}, "errors": []},
        ]
        side_effects = [call_results[0], RuntimeError("network down")]
        service.fetch_country = MagicMock(side_effect=side_effects)

        result = service.fetch_and_store(countries=["USA", "Broken"])

        assert len(result["errors"]) == 1
        assert "Broken" in result["errors"][0]

    def test_when_all_countries_succeed_then_counts_accumulated(self) -> None:
        service, _ = _build_service()
        service.fetch_country = MagicMock(
            return_value={"counts": {"te_indicators": 2}, "errors": []}
        )

        result = service.fetch_and_store(countries=["USA", "Germany"])

        assert result["counts"]["te_indicators"] == 4  # 2 × 2

    def test_when_country_errors_then_accumulated_with_country_prefix(self) -> None:
        service, _ = _build_service()
        service.fetch_country = MagicMock(
            return_value={"counts": {}, "errors": ["ilsole_forecast: timeout"]}
        )

        result = service.fetch_and_store(countries=["Germany"])

        assert any("Germany" in e for e in result["errors"])


# ---------------------------------------------------------------------------
# fetch_country — te success with empty indicators / empty bonds
# ---------------------------------------------------------------------------


class TestFetchCountryEmptySubSets:
    def test_when_te_success_but_indicators_empty_then_te_counts_zeroed(self) -> None:
        service, _ = _build_service(
            ilsole_forecasts=None,
            te_response=_TE_SUCCESS_EMPTY_INDICATORS,
        )
        result = service.fetch_country("USA", include_bonds=True)
        assert result["counts"]["te_indicators"] == 0
        assert result["counts"]["te_observations"] == 0

    def test_when_te_success_but_bond_yields_empty_then_bond_counts_zeroed(
        self,
    ) -> None:
        service, _ = _build_service(
            ilsole_forecasts=None,
            te_response=_TE_SUCCESS_EMPTY_BONDS,
        )
        result = service.fetch_country("USA", include_bonds=True)
        assert result["counts"]["bond_yields"] == 0
        assert result["counts"]["bond_yield_observations"] == 0

    def test_when_include_bonds_false_then_bond_upsert_not_called(self) -> None:
        service, mock_repo = _build_service(
            ilsole_forecasts=None,
            te_response=_TE_SUCCESS,
        )
        service.fetch_country("USA", include_bonds=False)
        mock_repo.upsert_bond_yields.assert_not_called()
        mock_repo.upsert_bond_yield_observations.assert_not_called()


# ---------------------------------------------------------------------------
# fetch_fred_series — incremental with/without latest date, error + exception
# ---------------------------------------------------------------------------


class TestFetchFredSeries:
    def _make_fred_scraper(self, status: str = "success", n_obs: int = 3) -> MagicMock:
        scraper = MagicMock()
        scraper.get_series_observations.return_value = {
            "status": status,
            "observations": [MagicMock()] * n_obs,
            "error": "fetch failed" if status != "success" else None,
        }
        return scraper

    def test_when_incremental_and_latest_date_present_then_start_date_set(self) -> None:
        import datetime

        fake_scraper = self._make_fred_scraper()
        mock_repo = MagicMock()
        mock_repo.get_fred_latest_date.return_value = datetime.date(2025, 1, 1)
        mock_repo.upsert_fred_observations.return_value = 2
        service, _ = _build_service(fred_scraper=fake_scraper, repo=mock_repo)

        result = service.fetch_fred_series(series_ids=["GDP"], incremental=True)

        call_kwargs = fake_scraper.get_series_observations.call_args
        assert call_kwargs.kwargs.get("start_date") == "2025-01-02"
        assert result["counts"]["GDP"] == 2

    def test_when_incremental_and_no_latest_date_then_start_date_none(self) -> None:
        fake_scraper = self._make_fred_scraper()
        mock_repo = MagicMock()
        mock_repo.get_fred_latest_date.return_value = None
        mock_repo.upsert_fred_observations.return_value = 1
        service, _ = _build_service(fred_scraper=fake_scraper, repo=mock_repo)

        service.fetch_fred_series(series_ids=["GDP"], incremental=True)

        call_kwargs = fake_scraper.get_series_observations.call_args
        assert call_kwargs.kwargs.get("start_date") is None

    def test_when_not_incremental_then_repo_latest_date_not_called(self) -> None:
        fake_scraper = self._make_fred_scraper()
        mock_repo = MagicMock()
        mock_repo.upsert_fred_observations.return_value = 0
        service, _ = _build_service(fred_scraper=fake_scraper, repo=mock_repo)

        service.fetch_fred_series(series_ids=["GDP"], incremental=False)

        mock_repo.get_fred_latest_date.assert_not_called()

    def test_when_scraper_returns_error_status_then_error_accumulated(self) -> None:
        fake_scraper = self._make_fred_scraper(status="error")
        mock_repo = MagicMock()
        service, _ = _build_service(fred_scraper=fake_scraper, repo=mock_repo)

        result = service.fetch_fred_series(series_ids=["GDP"], incremental=False)

        assert result["counts"]["GDP"] == 0
        assert any("GDP" in e for e in result["errors"])

    def test_when_scraper_raises_then_exception_caught_and_error_accumulated(
        self,
    ) -> None:
        fake_scraper = MagicMock()
        fake_scraper.get_series_observations.side_effect = RuntimeError("timeout")
        mock_repo = MagicMock()
        service, _ = _build_service(fred_scraper=fake_scraper, repo=mock_repo)

        result = service.fetch_fred_series(series_ids=["GDP"], incremental=False)

        assert result["counts"]["GDP"] == 0
        assert any("GDP" in e for e in result["errors"])

    def test_when_series_ids_none_then_all_fred_series_processed(self) -> None:
        from app.services.macro.scrapers.fred_scraper import FRED_SERIES

        fake_scraper = MagicMock()
        fake_scraper.get_series_observations.return_value = {
            "status": "success",
            "observations": [],
        }
        mock_repo = MagicMock()
        mock_repo.upsert_fred_observations.return_value = 0
        service, _ = _build_service(fred_scraper=fake_scraper, repo=mock_repo)

        result = service.fetch_fred_series(series_ids=None, incremental=False)

        assert len(result["counts"]) == len(FRED_SERIES)

    def test_on_progress_called_for_each_series(self) -> None:
        fake_scraper = MagicMock()
        fake_scraper.get_series_observations.return_value = {
            "status": "success",
            "observations": [],
        }
        mock_repo = MagicMock()
        mock_repo.upsert_fred_observations.return_value = 0
        service, _ = _build_service(fred_scraper=fake_scraper, repo=mock_repo)

        on_progress = MagicMock()
        service.fetch_fred_series(
            series_ids=["A", "B"], incremental=False, on_progress=on_progress
        )

        assert on_progress.call_count == 2
        calls_kwargs = [c.kwargs for c in on_progress.call_args_list]
        assert any(kw.get("current_country") == "A" for kw in calls_kwargs)
        assert any(kw.get("current_country") == "B" for kw in calls_kwargs)


# ---------------------------------------------------------------------------
# fetch_macro_news — various branches
# ---------------------------------------------------------------------------


class TestFetchMacroNews:
    """fetch_macro_news imports get_yfinance_client and MacroNewsFetcher lazily
    inside the method body; patch target is the fully-qualified path used at call time."""

    # Both are imported lazily inside the method body; patch at their source module.
    _YF_CLIENT_PATCH = "app.services.market_data.yfinance.get_yfinance_client"
    _FETCHER_PATCH = "app.services.market_data.yfinance.news.macro_news.MacroNewsFetcher"

    def _mock_yf_client(self) -> MagicMock:
        client = MagicMock()
        client.search = MagicMock()
        return client

    def _make_article(
        self,
        news_id: str = "abc",
        publish_time: str | None = "2025-01-15T10:00:00",
        full_content: str | None = "Full article text.",
    ) -> dict:
        return {
            "news_id": news_id,
            "title": "Test Article",
            "publisher": "Reuters",
            "link": "https://example.com",
            "publish_time": publish_time,
            "source_ticker": "SPY",
            "source_query": "macro",
            "themes": ["economy"],
            "snippet": "Short snippet.",
            "full_content": full_content,
        }

    def test_when_fetcher_succeeds_then_count_returned(self) -> None:
        mock_repo = MagicMock()
        mock_repo.upsert_macro_news.return_value = 1
        service, _ = _build_service(repo=mock_repo)

        yf_client = self._mock_yf_client()
        fetcher_mock = MagicMock()
        fetcher_mock.fetch_all.return_value = [self._make_article()]

        with (
            patch(self._YF_CLIENT_PATCH, return_value=yf_client),
            patch(self._FETCHER_PATCH, return_value=fetcher_mock),
        ):
            result = service.fetch_macro_news()

        assert result["count"] == 1
        assert result["errors"] == []

    def test_when_article_has_no_full_content_then_filtered_out(self) -> None:
        mock_repo = MagicMock()
        mock_repo.upsert_macro_news.return_value = 0
        service, _ = _build_service(repo=mock_repo)

        yf_client = self._mock_yf_client()
        fetcher_mock = MagicMock()
        fetcher_mock.fetch_all.return_value = [self._make_article(full_content=None)]

        with (
            patch(self._YF_CLIENT_PATCH, return_value=yf_client),
            patch(self._FETCHER_PATCH, return_value=fetcher_mock),
        ):
            service.fetch_macro_news()

        # upsert called with empty list (all articles filtered)
        call_args = mock_repo.upsert_macro_news.call_args
        upserted_rows = (
            call_args.args[0] if call_args.args else call_args.kwargs.get("rows", [])
        )
        assert upserted_rows == []

    def test_when_article_has_no_publish_time_then_pub_time_is_none(self) -> None:
        mock_repo = MagicMock()
        mock_repo.upsert_macro_news.return_value = 1
        service, _ = _build_service(repo=mock_repo)

        yf_client = self._mock_yf_client()
        fetcher_mock = MagicMock()
        fetcher_mock.fetch_all.return_value = [self._make_article(publish_time=None)]

        with (
            patch(self._YF_CLIENT_PATCH, return_value=yf_client),
            patch(self._FETCHER_PATCH, return_value=fetcher_mock),
        ):
            service.fetch_macro_news()

        call_args = mock_repo.upsert_macro_news.call_args
        upserted_rows = (
            call_args.args[0] if call_args.args else call_args.kwargs.get("rows", [])
        )
        assert upserted_rows[0]["publish_time"] is None

    def test_when_fetcher_raises_then_error_returned(self) -> None:
        service, _ = _build_service()

        with patch(
            self._YF_CLIENT_PATCH,
            side_effect=RuntimeError("yfinance down"),
        ):
            result = service.fetch_macro_news()

        assert result["count"] == 0
        assert len(result["errors"]) == 1

    def test_when_upsert_raises_then_error_returned(self) -> None:
        mock_repo = MagicMock()
        mock_repo.upsert_macro_news.side_effect = RuntimeError("DB write failed")
        service, _ = _build_service(repo=mock_repo)

        yf_client = self._mock_yf_client()
        fetcher_mock = MagicMock()
        fetcher_mock.fetch_all.return_value = [self._make_article()]

        with (
            patch(self._YF_CLIENT_PATCH, return_value=yf_client),
            patch(self._FETCHER_PATCH, return_value=fetcher_mock),
        ):
            result = service.fetch_macro_news()

        assert result["count"] == 0
        assert len(result["errors"]) == 1

    def test_when_fetch_full_content_true_then_fetch_all_called_correctly(
        self,
    ) -> None:
        mock_repo = MagicMock()
        mock_repo.upsert_macro_news.return_value = 0
        service, _ = _build_service(repo=mock_repo)

        yf_client = self._mock_yf_client()
        fetcher_mock = MagicMock()
        fetcher_mock.fetch_all.return_value = []

        with (
            patch(self._YF_CLIENT_PATCH, return_value=yf_client),
            patch(self._FETCHER_PATCH, return_value=fetcher_mock),
            patch(
                "app.services.market_data.yfinance.news.scraper.ArticleScraper",
            ),
        ):
            service.fetch_macro_news(fetch_full_content=True)

        fetcher_mock.fetch_all.assert_called_once_with(
            max_articles=100, fetch_full_content=True
        )


# ---------------------------------------------------------------------------
# get_portfolio_countries (line 375)
# ---------------------------------------------------------------------------


class TestGetPortfolioCountries:
    def test_returns_list_of_strings(self) -> None:
        result = MacroRegimeService.get_portfolio_countries()
        assert isinstance(result, list)
        assert all(isinstance(c, str) for c in result)

    def test_returns_non_empty_list(self) -> None:
        assert len(MacroRegimeService.get_portfolio_countries()) > 0


# ---------------------------------------------------------------------------
# run_bulk_macro_fetch — exception path per-country (lines 429-432)
# ---------------------------------------------------------------------------


class TestRunBulkMacroFetchSuccessWithErrors:
    """Line 425: errors from fetch_country.result["errors"] are accumulated per-country."""

    def test_when_fetch_country_returns_sub_errors_then_accumulated_with_prefix(
        self,
    ) -> None:
        from app.services.macro.macro_regime_service import run_bulk_macro_fetch

        request = SimpleNamespace(countries=["Germany"], include_bonds=True)
        mock_session = MagicMock()
        mock_svc = MagicMock()
        # fetch_country succeeds but returns sub-level errors (line 424 loop)
        mock_svc.fetch_country.return_value = {
            "counts": {"te_indicators": 0},
            "errors": ["trading_economics: timeout"],
        }

        on_progress = MagicMock()

        with (
            patch("app.database.database_manager") as mock_db,
            patch("app.services.macro.macro_regime_service.MacroRegimeRepository"),
            patch(
                "app.services.macro.macro_regime_service.MacroRegimeService",
                return_value=mock_svc,
            ),
        ):
            mock_db.get_session.return_value.__enter__.return_value = mock_session
            mock_db.get_session.return_value.__exit__.return_value = False

            result = run_bulk_macro_fetch(request, on_progress=on_progress)

        # The error is surfaced in error_count even though fetch_country didn't raise
        assert result["error_count"] == 1
        # Final status is "failed" because errors were present
        final_kwargs = on_progress.call_args_list[-1].kwargs
        assert final_kwargs.get("status") == "failed"


class TestRunBulkMacroFetchExceptionPath:
    def test_when_fetch_country_raises_then_rollback_called_and_error_accumulated(
        self,
    ) -> None:
        from app.services.macro.macro_regime_service import run_bulk_macro_fetch

        request = SimpleNamespace(countries=["Broken"], include_bonds=True)
        mock_session = MagicMock()
        mock_svc = MagicMock()
        mock_svc.fetch_country.side_effect = RuntimeError("network error")

        with (
            patch("app.database.database_manager") as mock_db,
            patch("app.services.macro.macro_regime_service.MacroRegimeRepository"),
            patch(
                "app.services.macro.macro_regime_service.MacroRegimeService",
                return_value=mock_svc,
            ),
        ):
            mock_db.get_session.return_value.__enter__.return_value = mock_session
            mock_db.get_session.return_value.__exit__.return_value = False

            result = run_bulk_macro_fetch(request)

        mock_session.rollback.assert_called_once()
        assert result["error_count"] == 1

    def test_when_errors_present_then_final_status_is_failed(self) -> None:
        from app.services.macro.macro_regime_service import run_bulk_macro_fetch

        request = SimpleNamespace(countries=["Broken"], include_bonds=True)
        mock_session = MagicMock()
        mock_svc = MagicMock()
        mock_svc.fetch_country.side_effect = RuntimeError("network error")

        on_progress = MagicMock()

        with (
            patch("app.database.database_manager") as mock_db,
            patch("app.services.macro.macro_regime_service.MacroRegimeRepository"),
            patch(
                "app.services.macro.macro_regime_service.MacroRegimeService",
                return_value=mock_svc,
            ),
        ):
            mock_db.get_session.return_value.__enter__.return_value = mock_session
            mock_db.get_session.return_value.__exit__.return_value = False

            run_bulk_macro_fetch(request, on_progress=on_progress)

        final_kwargs = on_progress.call_args_list[-1].kwargs
        assert final_kwargs.get("status") == "failed"


# ---------------------------------------------------------------------------
# run_bulk_fred_fetch (lines 474-505)
# ---------------------------------------------------------------------------


class TestRunBulkFredFetch:
    def test_when_series_fetched_successfully_then_result_dict_returned(self) -> None:
        from app.services.macro.macro_regime_service import run_bulk_fred_fetch

        request = SimpleNamespace(series_ids=["GDP"], incremental=False)
        mock_session = MagicMock()
        mock_svc = MagicMock()
        mock_svc.fetch_fred_series.return_value = {
            "counts": {"GDP": 5},
            "errors": [],
        }

        with (
            patch("app.database.database_manager") as mock_db,
            patch("app.services.macro.macro_regime_service.MacroRegimeRepository"),
            patch(
                "app.services.macro.macro_regime_service.MacroRegimeService",
                return_value=mock_svc,
            ),
        ):
            mock_db.get_session.return_value.__enter__.return_value = mock_session
            mock_db.get_session.return_value.__exit__.return_value = False

            result = run_bulk_fred_fetch(request)

        assert result["series_fetched"] == 5
        assert result["error_count"] == 0

    def test_when_fetch_errors_then_error_count_accumulated(self) -> None:
        from app.services.macro.macro_regime_service import run_bulk_fred_fetch

        request = SimpleNamespace(series_ids=["BAD"], incremental=False)
        mock_session = MagicMock()
        mock_svc = MagicMock()
        mock_svc.fetch_fred_series.return_value = {
            "counts": {"BAD": 0},
            "errors": ["BAD: fetch failed"],
        }

        on_progress = MagicMock()

        with (
            patch("app.database.database_manager") as mock_db,
            patch("app.services.macro.macro_regime_service.MacroRegimeRepository"),
            patch(
                "app.services.macro.macro_regime_service.MacroRegimeService",
                return_value=mock_svc,
            ),
        ):
            mock_db.get_session.return_value.__enter__.return_value = mock_session
            mock_db.get_session.return_value.__exit__.return_value = False

            result = run_bulk_fred_fetch(request, on_progress=on_progress)

        assert result["error_count"] == 1
        final_kwargs = on_progress.call_args_list[-1].kwargs
        assert final_kwargs.get("status") == "completed"


# ---------------------------------------------------------------------------
# run_macro_news_fetch (lines 522-542)
# ---------------------------------------------------------------------------


class TestRunMacroNewsFetch:
    def test_when_news_fetch_succeeds_then_articles_stored_returned(self) -> None:
        from app.services.macro.macro_regime_service import run_macro_news_fetch

        request = SimpleNamespace(max_articles=10, fetch_full_content=False)
        mock_session = MagicMock()
        mock_svc = MagicMock()
        mock_svc.fetch_macro_news.return_value = {"count": 7, "errors": []}

        with (
            patch("app.database.database_manager") as mock_db,
            patch("app.services.macro.macro_regime_service.MacroRegimeRepository"),
            patch(
                "app.services.macro.macro_regime_service.MacroRegimeService",
                return_value=mock_svc,
            ),
        ):
            mock_db.get_session.return_value.__enter__.return_value = mock_session
            mock_db.get_session.return_value.__exit__.return_value = False

            result = run_macro_news_fetch(request)

        assert result["articles_stored"] == 7

    def test_on_progress_final_status_is_completed(self) -> None:
        from app.services.macro.macro_regime_service import run_macro_news_fetch

        request = SimpleNamespace(max_articles=5, fetch_full_content=False)
        mock_session = MagicMock()
        mock_svc = MagicMock()
        mock_svc.fetch_macro_news.return_value = {"count": 0, "errors": []}

        on_progress = MagicMock()

        with (
            patch("app.database.database_manager") as mock_db,
            patch("app.services.macro.macro_regime_service.MacroRegimeRepository"),
            patch(
                "app.services.macro.macro_regime_service.MacroRegimeService",
                return_value=mock_svc,
            ),
        ):
            mock_db.get_session.return_value.__enter__.return_value = mock_session
            mock_db.get_session.return_value.__exit__.return_value = False

            run_macro_news_fetch(request, on_progress=on_progress)

        final_kwargs = on_progress.call_args_list[-1].kwargs
        assert final_kwargs.get("status") == "completed"
