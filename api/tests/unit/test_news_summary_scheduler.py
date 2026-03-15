"""Unit tests for the 30-minute news summary scheduler."""

from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

from app.models.macro_regime import MacroNews
from app.services.news_summary_scheduler import (
    MacroNewsSummaryScheduler,
    TickResult,
    _find_countries_with_new_articles,
    _is_morning_pipeline_complete,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 3, 15, 12, 0, tzinfo=timezone.utc)
_TODAY = date(2026, 3, 15)


def _make_article(
    source_ticker: str | None = None,
    source_query: str | None = None,
) -> MacroNews:
    article = MagicMock(spec=MacroNews)
    article.source_ticker = source_ticker
    article.source_query = source_query
    return article


def _mock_db_manager(mock_db_manager: MagicMock) -> MagicMock:
    """Wire up the context manager protocol on the mock and return the mock session."""
    mock_session = MagicMock()
    mock_db_manager.get_session.return_value.__enter__ = MagicMock(
        return_value=mock_session,
    )
    mock_db_manager.get_session.return_value.__exit__ = MagicMock(
        return_value=False,
    )
    return mock_session


# ---------------------------------------------------------------------------
# _find_countries_with_new_articles
# ---------------------------------------------------------------------------


class TestFindCountriesWithNewArticles:
    def test_empty_articles_returns_empty(self) -> None:
        repo = MagicMock()
        repo.get_macro_news.return_value = []
        assert _find_countries_with_new_articles(repo, _NOW) == []

    def test_ticker_articles_mapped_to_countries(self) -> None:
        repo = MagicMock()
        repo.get_macro_news.return_value = [
            _make_article(source_ticker="^GSPC"),
            _make_article(source_ticker="^FTSE"),
        ]
        result = _find_countries_with_new_articles(repo, _NOW)
        assert "UK" in result
        assert "USA" in result

    def test_global_tickers_excluded(self) -> None:
        repo = MagicMock()
        repo.get_macro_news.return_value = [
            _make_article(source_ticker="EEM"),
            _make_article(source_ticker="DX-Y.NYB"),
            _make_article(source_ticker="GC=F"),
        ]
        assert _find_countries_with_new_articles(repo, _NOW) == []

    def test_query_articles_mapped_to_countries(self) -> None:
        repo = MagicMock()
        repo.get_macro_news.return_value = [
            _make_article(source_query="ECB interest rate decision"),
        ]
        result = _find_countries_with_new_articles(repo, _NOW)
        assert "France" in result
        assert "Germany" in result

    def test_global_queries_excluded(self) -> None:
        repo = MagicMock()
        repo.get_macro_news.return_value = [
            _make_article(source_query="emerging markets capital flows"),
        ]
        assert _find_countries_with_new_articles(repo, _NOW) == []

    def test_results_sorted_alphabetically(self) -> None:
        repo = MagicMock()
        repo.get_macro_news.return_value = [
            _make_article(source_ticker="^FTSE"),
            _make_article(source_ticker="^GSPC"),
            _make_article(source_ticker="^GDAXI"),
        ]
        result = _find_countries_with_new_articles(repo, _NOW)
        assert result == sorted(result)

    def test_deduplicates_countries(self) -> None:
        repo = MagicMock()
        repo.get_macro_news.return_value = [
            _make_article(source_ticker="^GSPC"),
            _make_article(source_ticker="^TNX"),  # also USA
            _make_article(source_ticker="XLF"),  # also USA
        ]
        result = _find_countries_with_new_articles(repo, _NOW)
        assert result.count("USA") == 1

    def test_passes_since_as_start_date(self) -> None:
        repo = MagicMock()
        repo.get_macro_news.return_value = []
        since = datetime(2026, 3, 15, 11, 0, tzinfo=timezone.utc)
        _find_countries_with_new_articles(repo, since)
        repo.get_macro_news.assert_called_once_with(start_date=since, limit=500)


# ---------------------------------------------------------------------------
# _is_morning_pipeline_complete
# ---------------------------------------------------------------------------


class TestMorningPipelineGuard:
    def test_returns_false_when_no_summaries_today(self) -> None:
        repo = MagicMock()
        repo.get_all_news_summaries.return_value = []
        assert _is_morning_pipeline_complete(repo, _TODAY) is False

    def test_returns_true_when_summary_exists_today(self) -> None:
        repo = MagicMock()
        repo.get_all_news_summaries.return_value = [MagicMock()]
        assert _is_morning_pipeline_complete(repo, _TODAY) is True

    def test_passes_today_as_summary_date_filter(self) -> None:
        repo = MagicMock()
        repo.get_all_news_summaries.return_value = []
        _is_morning_pipeline_complete(repo, _TODAY)
        repo.get_all_news_summaries.assert_called_once_with(summary_date=_TODAY)


# ---------------------------------------------------------------------------
# MacroNewsSummaryScheduler lifecycle
# ---------------------------------------------------------------------------


class TestSchedulerLifecycle:
    def test_start_stop_does_not_raise(self) -> None:
        scheduler = MacroNewsSummaryScheduler(interval_seconds=60)
        scheduler.start()
        scheduler.stop()

    def test_double_start_is_safe(self) -> None:
        scheduler = MacroNewsSummaryScheduler(interval_seconds=60)
        scheduler.start()
        scheduler.start()  # should not raise
        scheduler.stop()

    def test_stop_without_start_does_not_raise(self) -> None:
        scheduler = MacroNewsSummaryScheduler(interval_seconds=60)
        scheduler.stop()


# ---------------------------------------------------------------------------
# Scheduler._tick() — morning guard
# ---------------------------------------------------------------------------


class TestTickWithMorningGuard:
    @patch("app.services.news_summary_scheduler.database_manager")
    def test_tick_skips_when_morning_pipeline_not_complete(
        self, mock_db_mgr: MagicMock,
    ) -> None:
        mock_session = _mock_db_manager(mock_db_mgr)

        mock_repo = MagicMock()
        mock_repo.get_all_news_summaries.return_value = []  # no summaries today

        with (
            patch(
                "app.services.news_summary_scheduler.MacroRegimeRepository",
                return_value=mock_repo,
            ),
            patch(
                "app.services.news_summary_scheduler.generate_country_summaries",
            ) as mock_gen,
        ):
            scheduler = MacroNewsSummaryScheduler(interval_seconds=60)
            result = scheduler._tick()

        mock_gen.assert_not_called()
        assert result.updated == []
        assert result.skipped == []
        assert result.errored == []

    @patch("app.services.news_summary_scheduler.database_manager")
    def test_tick_proceeds_when_morning_pipeline_complete(
        self, mock_db_mgr: MagicMock,
    ) -> None:
        mock_session = _mock_db_manager(mock_db_mgr)

        mock_repo = MagicMock()
        mock_repo.get_all_news_summaries.return_value = [MagicMock()]  # morning ran
        mock_repo.get_macro_news.return_value = [
            _make_article(source_ticker="^GSPC"),
        ]
        mock_repo.delete_old_news_summaries.return_value = 0

        with (
            patch(
                "app.services.news_summary_scheduler.MacroRegimeRepository",
                return_value=mock_repo,
            ),
            patch(
                "app.services.news_summary_scheduler.generate_country_summaries",
                return_value=[MagicMock()],
            ) as mock_gen,
        ):
            scheduler = MacroNewsSummaryScheduler(interval_seconds=60)
            result = scheduler._tick()

        mock_gen.assert_called_once()
        assert "USA" in result.updated


# ---------------------------------------------------------------------------
# Scheduler._tick() — core behaviour
# ---------------------------------------------------------------------------


class TestSchedulerTick:
    @patch("app.services.news_summary_scheduler.database_manager")
    def test_tick_skips_when_no_new_articles(self, mock_db_mgr: MagicMock) -> None:
        mock_session = _mock_db_manager(mock_db_mgr)

        mock_repo = MagicMock()
        mock_repo.get_all_news_summaries.return_value = [MagicMock()]  # morning done
        mock_repo.get_macro_news.return_value = []
        mock_repo.delete_old_news_summaries.return_value = 0

        with (
            patch(
                "app.services.news_summary_scheduler.MacroRegimeRepository",
                return_value=mock_repo,
            ),
            patch(
                "app.services.news_summary_scheduler.generate_country_summaries",
            ) as mock_gen,
        ):
            scheduler = MacroNewsSummaryScheduler(interval_seconds=60)
            scheduler._tick()

        mock_gen.assert_not_called()

    @patch("app.services.news_summary_scheduler.database_manager")
    def test_tick_calls_generate_when_new_articles(
        self, mock_db_mgr: MagicMock,
    ) -> None:
        mock_session = _mock_db_manager(mock_db_mgr)

        mock_repo = MagicMock()
        mock_repo.get_all_news_summaries.return_value = [MagicMock()]
        mock_repo.get_macro_news.return_value = [
            _make_article(source_ticker="^GSPC"),
        ]
        mock_repo.delete_old_news_summaries.return_value = 0

        with (
            patch(
                "app.services.news_summary_scheduler.MacroRegimeRepository",
                return_value=mock_repo,
            ),
            patch(
                "app.services.news_summary_scheduler.generate_country_summaries",
                return_value=[MagicMock()],
            ) as mock_gen,
        ):
            scheduler = MacroNewsSummaryScheduler(interval_seconds=60)
            scheduler._tick()

        mock_gen.assert_called_once_with(
            mock_session,
            force_refresh=True,
            countries=["USA"],
        )

    @patch("app.services.news_summary_scheduler.database_manager")
    def test_tick_always_calls_delete_old_summaries(
        self, mock_db_mgr: MagicMock,
    ) -> None:
        mock_session = _mock_db_manager(mock_db_mgr)

        mock_repo = MagicMock()
        mock_repo.get_all_news_summaries.return_value = [MagicMock()]
        mock_repo.get_macro_news.return_value = []
        mock_repo.delete_old_news_summaries.return_value = 0

        with patch(
            "app.services.news_summary_scheduler.MacroRegimeRepository",
            return_value=mock_repo,
        ):
            scheduler = MacroNewsSummaryScheduler(interval_seconds=60)
            scheduler._tick()

        mock_repo.delete_old_news_summaries.assert_called_once()

    @patch("app.services.news_summary_scheduler.database_manager")
    def test_tick_commits_session(self, mock_db_mgr: MagicMock) -> None:
        mock_session = _mock_db_manager(mock_db_mgr)

        mock_repo = MagicMock()
        mock_repo.get_all_news_summaries.return_value = [MagicMock()]
        mock_repo.get_macro_news.return_value = []
        mock_repo.delete_old_news_summaries.return_value = 0

        with patch(
            "app.services.news_summary_scheduler.MacroRegimeRepository",
            return_value=mock_repo,
        ):
            scheduler = MacroNewsSummaryScheduler(interval_seconds=60)
            scheduler._tick()

        mock_session.commit.assert_called_once()

    @patch("app.services.news_summary_scheduler.database_manager")
    def test_tick_nonfatal_on_generate_error(
        self, mock_db_mgr: MagicMock,
    ) -> None:
        mock_session = _mock_db_manager(mock_db_mgr)

        mock_repo = MagicMock()
        mock_repo.get_all_news_summaries.return_value = [MagicMock()]
        mock_repo.get_macro_news.return_value = [
            _make_article(source_ticker="^GSPC"),
        ]
        mock_repo.delete_old_news_summaries.return_value = 0

        with (
            patch(
                "app.services.news_summary_scheduler.MacroRegimeRepository",
                return_value=mock_repo,
            ),
            patch(
                "app.services.news_summary_scheduler.generate_country_summaries",
                side_effect=RuntimeError("LLM error"),
            ),
        ):
            scheduler = MacroNewsSummaryScheduler(interval_seconds=60)
            result = scheduler._tick()

        # Error is isolated — scheduler continues
        assert "USA" in result.errored

    @patch("app.services.news_summary_scheduler.database_manager")
    def test_tick_updates_last_run(self, mock_db_mgr: MagicMock) -> None:
        mock_session = _mock_db_manager(mock_db_mgr)

        mock_repo = MagicMock()
        mock_repo.get_all_news_summaries.return_value = [MagicMock()]
        mock_repo.get_macro_news.return_value = []
        mock_repo.delete_old_news_summaries.return_value = 0

        with patch(
            "app.services.news_summary_scheduler.MacroRegimeRepository",
            return_value=mock_repo,
        ):
            scheduler = MacroNewsSummaryScheduler(interval_seconds=60)
            before = scheduler._last_run
            scheduler._tick()
            assert scheduler._last_run > before


# ---------------------------------------------------------------------------
# TickResult
# ---------------------------------------------------------------------------


class TestTickResult:
    @patch("app.services.news_summary_scheduler.database_manager")
    def test_tick_returns_tick_result_dataclass(
        self, mock_db_mgr: MagicMock,
    ) -> None:
        mock_session = _mock_db_manager(mock_db_mgr)

        mock_repo = MagicMock()
        mock_repo.get_all_news_summaries.return_value = [MagicMock()]
        mock_repo.get_macro_news.return_value = []
        mock_repo.delete_old_news_summaries.return_value = 3

        with patch(
            "app.services.news_summary_scheduler.MacroRegimeRepository",
            return_value=mock_repo,
        ):
            scheduler = MacroNewsSummaryScheduler(interval_seconds=60)
            result = scheduler._tick()

        assert isinstance(result, TickResult)
        assert result.pruned == 3

    @patch("app.services.news_summary_scheduler.database_manager")
    def test_tick_error_country_isolated(
        self, mock_db_mgr: MagicMock,
    ) -> None:
        mock_session = _mock_db_manager(mock_db_mgr)

        mock_repo = MagicMock()
        mock_repo.get_all_news_summaries.return_value = [MagicMock()]
        mock_repo.get_macro_news.return_value = [
            _make_article(source_ticker="^GSPC"),  # USA
            _make_article(source_ticker="^FTSE"),  # UK
        ]
        mock_repo.delete_old_news_summaries.return_value = 0

        def side_effect(session, force_refresh, countries):
            if countries == ["USA"]:
                raise RuntimeError("LLM timeout")
            return [MagicMock()]

        with (
            patch(
                "app.services.news_summary_scheduler.MacroRegimeRepository",
                return_value=mock_repo,
            ),
            patch(
                "app.services.news_summary_scheduler.generate_country_summaries",
                side_effect=side_effect,
            ),
        ):
            scheduler = MacroNewsSummaryScheduler(interval_seconds=60)
            result = scheduler._tick()

        # USA failed but UK succeeded
        assert "USA" in result.errored
        assert "UK" in result.updated
