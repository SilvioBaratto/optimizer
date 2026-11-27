"""Branch-coverage tests for app/services/macro/macro_regime_service.py.

Maps term-missing ranges to concrete conditions not covered by:
  - tests/unit/services/macro/test_macro_regime_service.py
  - tests/unit/test_macro_regime_service_fred_key.py

Uncovered branches:
  fred_scraper property when _fred_scraper is already set
  fetch_fred_series: incremental=True with/without latest date,
                  result error status, fetch exception per-series
  fetch_macro_news: yfinance happy path, scraper=None/not None,
                  publish_time parsing, full_content filter, upsert exception
  run_bulk_fred_fetch happy path
  run_macro_news_fetch happy path

Do NOT duplicate tests already in the two existing test files.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import Session

from app.services._shared import _noop
from app.services.macro.macro_regime_service import (
    FRED_REVISION_LOOKBACK_DAYS,
    MacroRegimeService,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_service(
    *,
    fred_scraper: object = None,
    repo: MagicMock | None = None,
) -> tuple[MacroRegimeService, MagicMock]:
    mock_repo = repo or MagicMock()
    service = MacroRegimeService(
        repo=mock_repo,
        fred_scraper=fred_scraper,  # type: ignore[arg-type]
    )
    return service, mock_repo


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
        expected = datetime.date(2025, 1, 1) - datetime.timedelta(
            days=FRED_REVISION_LOOKBACK_DAYS
        )
        assert call_kwargs.kwargs.get("start_date") == expected.isoformat()
        assert result["counts"]["GDP"] == 2

    def test_when_incremental_then_start_date_reaches_back_a_full_quarter(
        self,
    ) -> None:
        import datetime

        fake_scraper = self._make_fred_scraper()
        mock_repo = MagicMock()
        latest = datetime.date(2026, 7, 1)
        mock_repo.get_fred_latest_date.return_value = latest
        mock_repo.upsert_fred_observations.return_value = 1
        service, _ = _build_service(fred_scraper=fake_scraper, repo=mock_repo)

        service.fetch_fred_series(series_ids=["A191RL1Q225SBEA"], incremental=True)

        call_kwargs = fake_scraper.get_series_observations.call_args
        start = datetime.date.fromisoformat(call_kwargs.kwargs["start_date"])
        assert latest - start >= datetime.timedelta(days=90)

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


class TestFetchFredSeriesDurability:
    """Each series is committed as it lands, not once after the whole sweep."""

    @pytest.fixture
    def fred_session(self):
        from portopt_db.models import Base
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy.pool import StaticPool

        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(engine)
        session = sessionmaker(bind=engine)()
        try:
            yield session
        finally:
            session.close()
            engine.dispose()

    def test_finished_series_survive_a_crash_during_a_later_series(
        self, fred_session
    ) -> None:
        """A forced exit mid-sweep keeps every series fetched before it.

        The scraper rolls the session back while fetching the second series, which
        is what a SIGTERM drain timeout does to uncommitted work.
        """
        import datetime

        from portopt_db.models.macro.macro_regime import FredObservation
        from portopt_db.repositories.macro.macro_regime_repository import (
            MacroRegimeRepository,
        )

        day = datetime.date(2026, 10, 1)

        def _observations(series_id, start_date=None):
            if series_id == "SECOND":
                fred_session.rollback()
                return {"status": "error", "error": "process died"}
            return {
                "status": "success",
                "observations": [{"date": day, "value": 1.5}],
            }

        scraper = MagicMock()
        scraper.get_series_observations.side_effect = _observations
        service = MacroRegimeService(
            repo=MacroRegimeRepository(fred_session),
            fred_scraper=scraper,
        )

        service.fetch_fred_series(series_ids=["FIRST", "SECOND"], incremental=False)

        stored = fred_session.query(FredObservation).filter_by(series_id="FIRST")
        assert stored.count() == 1

    def test_a_series_failing_mid_upsert_does_not_sink_the_later_series(
        self, fred_session
    ) -> None:
        """A failed write is rolled back so the next series can still commit.

        Postgres aborts the whole transaction on a failed statement and every
        later statement fails until a rollback; a failed flush leaves the SQLite
        session in the same rollback-only state. Without the per-series rollback
        the third series dies on the second one's leftover transaction.
        """
        import datetime

        from portopt_db.models.macro.macro_regime import FredObservation
        from portopt_db.repositories.macro.macro_regime_repository import (
            MacroRegimeRepository,
        )

        day = datetime.date(2026, 10, 1)

        class _FailsOnSecond(MacroRegimeRepository):
            def upsert_fred_observations(self, series_id, observations):
                if series_id == "SECOND":
                    self.session.add(FredObservation(series_id=None, date=day))
                    self.session.flush()
                return super().upsert_fred_observations(series_id, observations)

        scraper = MagicMock()
        scraper.get_series_observations.side_effect = lambda series_id, **kw: {
            "status": "success",
            "observations": [{"date": day, "value": 1.5}],
        }
        service = MacroRegimeService(
            repo=_FailsOnSecond(fred_session),
            fred_scraper=scraper,
        )

        result = service.fetch_fred_series(
            series_ids=["FIRST", "SECOND", "THIRD"], incremental=False
        )

        stored = {row.series_id for row in fred_session.query(FredObservation).all()}
        assert stored == {"FIRST", "THIRD"}
        assert [e.split(":")[0] for e in result["errors"]] == ["SECOND"]
        assert result["counts"] == {"FIRST": 1, "SECOND": 0, "THIRD": 1}


# ---------------------------------------------------------------------------
# fetch_macro_news — various branches
# ---------------------------------------------------------------------------


class TestFetchMacroNews:
    """fetch_macro_news imports get_yfinance_client and MacroNewsFetcher lazily
    inside the method body; patch target is the fully-qualified path used at call time."""

    _YF_CLIENT_PATCH = "app.services.market_data.yfinance.get_yfinance_client"
    _FETCHER_PATCH = (
        "app.services.market_data.yfinance.news.macro_news.MacroNewsFetcher"
    )

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

    def test_when_scrape_requested_and_article_has_no_full_content_then_filtered_out(
        self,
    ) -> None:
        mock_repo = MagicMock()
        mock_repo.upsert_macro_news.return_value = 0
        service, _ = _build_service(repo=mock_repo)

        yf_client = self._mock_yf_client()
        fetcher_mock = MagicMock()
        fetcher_mock.fetch_all.return_value = [self._make_article(full_content=None)]

        with (
            patch(self._YF_CLIENT_PATCH, return_value=yf_client),
            patch(self._FETCHER_PATCH, return_value=fetcher_mock),
            patch(
                "app.services.market_data.yfinance.news.scraper.ArticleScraper",
            ),
        ):
            service.fetch_macro_news(fetch_full_content=True)

        # upsert called with empty list (all articles filtered)
        call_args = mock_repo.upsert_macro_news.call_args
        upserted_rows = (
            call_args.args[0] if call_args.args else call_args.kwargs.get("rows", [])
        )
        assert upserted_rows == []

    def test_when_snippet_only_mode_then_articles_with_snippet_are_stored(
        self,
    ) -> None:
        mock_repo = MagicMock()
        mock_repo.upsert_macro_news.side_effect = len
        service, _ = _build_service(repo=mock_repo)

        yf_client = self._mock_yf_client()
        fetcher_mock = MagicMock()
        fetcher_mock.fetch_all.return_value = [self._make_article(full_content=None)]

        with (
            patch(self._YF_CLIENT_PATCH, return_value=yf_client),
            patch(self._FETCHER_PATCH, return_value=fetcher_mock),
        ):
            result = service.fetch_macro_news(fetch_full_content=False)

        assert result["count"] == 1
        upserted_rows = mock_repo.upsert_macro_news.call_args.args[0]
        assert upserted_rows[0]["snippet"] == "Short snippet."
        assert upserted_rows[0]["full_content"] is None

    def test_when_snippet_only_mode_then_articles_without_snippet_are_dropped(
        self,
    ) -> None:
        mock_repo = MagicMock()
        mock_repo.upsert_macro_news.side_effect = len
        service, _ = _build_service(repo=mock_repo)

        bare = self._make_article(full_content=None)
        bare["snippet"] = ""
        fetcher_mock = MagicMock()
        fetcher_mock.fetch_all.return_value = [bare]

        with (
            patch(self._YF_CLIENT_PATCH, return_value=self._mock_yf_client()),
            patch(self._FETCHER_PATCH, return_value=fetcher_mock),
        ):
            result = service.fetch_macro_news(fetch_full_content=False)

        assert result["count"] == 0

    def test_when_snippet_only_run_follows_scrape_run_then_scraped_body_survives(
        self, db_session: Session
    ) -> None:
        from portopt_db.models.macro.macro_regime import MacroNews
        from portopt_db.repositories.macro.macro_regime_repository import (
            MacroRegimeRepository,
        )

        service = MacroRegimeService(repo=MacroRegimeRepository(db_session))

        def run(article: dict, *, scrape: bool) -> int:
            fetcher_mock = MagicMock()
            fetcher_mock.fetch_all.return_value = [article]
            with (
                patch(self._YF_CLIENT_PATCH, return_value=self._mock_yf_client()),
                patch(self._FETCHER_PATCH, return_value=fetcher_mock),
                patch("app.services.market_data.yfinance.news.scraper.ArticleScraper"),
            ):
                result = service.fetch_macro_news(fetch_full_content=scrape)
            db_session.flush()
            return result["count"]

        assert run(self._make_article(full_content="Scraped body."), scrape=True) == 1
        assert run(self._make_article(full_content=None), scrape=False) == 1

        db_session.expire_all()
        stored = db_session.query(MacroNews).filter(MacroNews.news_id == "abc").one()
        assert stored.full_content == "Scraped body."

    def test_when_request_disables_scrape_then_run_stores_snippet_articles(
        self,
    ) -> None:
        from app.schemas.macro.macro_regime import MacroNewsFetchRequest
        from app.services.macro.macro_regime_service import run_macro_news_fetch

        mock_repo = MagicMock()
        mock_repo.upsert_macro_news.side_effect = len
        fetcher_mock = MagicMock()
        fetcher_mock.fetch_all.return_value = [self._make_article(full_content=None)]

        with (
            patch("app.database.database_manager") as mock_db,
            patch(
                "app.services.macro.macro_regime_service.MacroRegimeRepository",
                return_value=mock_repo,
            ),
            patch(self._YF_CLIENT_PATCH, return_value=self._mock_yf_client()),
            patch(self._FETCHER_PATCH, return_value=fetcher_mock),
        ):
            mock_db.get_session.return_value.__enter__.return_value = MagicMock()
            mock_db.get_session.return_value.__exit__.return_value = False

            result = run_macro_news_fetch(
                MacroNewsFetchRequest(fetch_full_content=False)
            )

        assert result["articles_stored"] == 1

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

    def test_when_every_news_source_fails_then_outage_reported_in_errors(
        self,
    ) -> None:
        mock_repo = MagicMock()
        service, _ = _build_service(repo=mock_repo)

        yf_client = self._mock_yf_client()
        yf_client.get_ticker.side_effect = RuntimeError("feed down")
        yf_client.search.search.return_value = None

        with (
            patch(self._YF_CLIENT_PATCH, return_value=yf_client),
            patch("app.services.infrastructure.retry.time"),
        ):
            result = service.fetch_macro_news()

        assert result["count"] == 0
        assert len(result["errors"]) == 1
        assert "no article" in result["errors"][0].lower()
        mock_repo.upsert_macro_news.assert_not_called()

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
            max_articles=100, fetch_full_content=True, on_progress=_noop
        )


# ---------------------------------------------------------------------------
# run_bulk_fred_fetch
# ---------------------------------------------------------------------------


class TestRunBulkFredFetch:
    @pytest.mark.criterion("REQ-ING-025")
    def test_when_series_fetched_successfully_then_result_dict_returned(self) -> None:
        """Validates ``run_bulk_fred_fetch`` and ``_final_progress``: a run that
        records no error returns its counts and reports a final status of
        ``completed``."""
        from app.services.macro.macro_regime_service import run_bulk_fred_fetch

        request = SimpleNamespace(series_ids=["GDP"], incremental=False)
        mock_session = MagicMock()
        mock_svc = MagicMock()
        mock_svc.fetch_fred_series.return_value = {
            "counts": {"GDP": 5},
            "errors": [],
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

        assert result["series_fetched"] == 1
        assert result["observations_upserted"] == 5
        assert result["error_count"] == 0
        final_kwargs = on_progress.call_args_list[-1].kwargs
        assert final_kwargs["status"] == "completed"
        assert final_kwargs["errors"] == []
        assert "error" not in final_kwargs

    def test_when_some_series_return_nothing_then_only_series_with_rows_counted(
        self,
    ) -> None:
        from app.services.macro.macro_regime_service import run_bulk_fred_fetch

        request = SimpleNamespace(series_ids=["A", "B", "C"], incremental=True)
        mock_svc = MagicMock()
        mock_svc.fetch_fred_series.return_value = {
            "counts": {"A": 100, "B": 0, "C": 40},
            "errors": ["B: fetch failed"],
        }

        with (
            patch("app.database.database_manager") as mock_db,
            patch("app.services.macro.macro_regime_service.MacroRegimeRepository"),
            patch(
                "app.services.macro.macro_regime_service.MacroRegimeService",
                return_value=mock_svc,
            ),
        ):
            mock_db.get_session.return_value.__enter__.return_value = MagicMock()
            mock_db.get_session.return_value.__exit__.return_value = False

            result = run_bulk_fred_fetch(request)

        assert result["series_fetched"] == 2
        assert result["observations_upserted"] == 140

    @pytest.mark.criterion("REQ-ING-025")
    def test_when_fetch_errors_then_final_status_is_failed(self) -> None:
        """Validates ``run_bulk_fred_fetch`` and ``_final_progress``: a run that
        records an error reports a final status of ``failed`` carrying the error
        text."""
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
        assert final_kwargs.get("status") == "failed"
        assert "BAD: fetch failed" in final_kwargs["error"]

    def test_when_api_key_missing_then_job_is_reported_failed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty key fetches nothing; reporting completed hides the data gap."""
        from app.config import settings as app_settings
        from app.services.macro.macro_regime_service import run_bulk_fred_fetch

        monkeypatch.setattr(app_settings, "fred_api_key", "", raising=False)
        request = SimpleNamespace(series_ids=None, incremental=True)
        on_progress = MagicMock()

        with (
            patch("app.database.database_manager") as mock_db,
            patch("app.services.macro.macro_regime_service.MacroRegimeRepository"),
        ):
            mock_db.get_session.return_value.__enter__.return_value = MagicMock()
            mock_db.get_session.return_value.__exit__.return_value = False

            run_bulk_fred_fetch(request, on_progress=on_progress)

        final_kwargs = on_progress.call_args_list[-1].kwargs
        assert final_kwargs.get("status") == "failed"
        assert "FRED_API_KEY missing" in final_kwargs["error"]


# ---------------------------------------------------------------------------
# run_macro_news_fetch
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

    def test_when_service_reports_errors_then_step_fails_instead_of_completing(
        self,
    ) -> None:
        from app.services.macro.macro_regime_service import run_macro_news_fetch

        request = SimpleNamespace(max_articles=5, fetch_full_content=False)
        mock_svc = MagicMock()
        mock_svc.fetch_macro_news.return_value = {
            "count": 0,
            "errors": ["no article from any macro news source"],
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
            mock_db.get_session.return_value.__enter__.return_value = MagicMock()
            mock_db.get_session.return_value.__exit__.return_value = False

            with pytest.raises(RuntimeError, match="no article"):
                run_macro_news_fetch(request, on_progress=on_progress)

        statuses = [c.kwargs.get("status") for c in on_progress.call_args_list]
        assert "completed" not in statuses
        errors_recorded = [
            c.kwargs["errors"]
            for c in on_progress.call_args_list
            if "errors" in c.kwargs
        ]
        assert errors_recorded == [["no article from any macro news source"]]
