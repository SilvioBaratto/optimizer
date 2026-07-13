"""Unit tests for macro_news_summary service (issue #822).

Covers gaps NOT in test_macro_news_summary_service.py:
  - generate_country_summaries: full two-country pipeline, commit-per-country guard
  - _validate_llm_output: invalid sentiment, clamped scores
  - _summarize_country threshold gate (<3 articles → None)
  - run_news_summarize: database_manager patched, final progress status checked
NO network, NO real BAML calls, NO real DB.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.services.macro.macro_news_summary import (
    _validate_llm_output,
    generate_country_summaries,
    run_news_summarize,
)

# ---------------------------------------------------------------------------
# Article factory — avoids ORM construction
# ---------------------------------------------------------------------------


def _make_article(
    source_ticker: str | None = None,
    source_query: str | None = None,
    title: str = "Test Article",
    publisher: str = "Reuters",
    full_content: str = "Full article content for testing purposes.",
    snippet: str | None = "Snippet text",
) -> SimpleNamespace:
    from datetime import datetime, timezone

    return SimpleNamespace(
        source_ticker=source_ticker,
        source_query=source_query,
        title=title,
        publisher=publisher,
        publish_time=datetime(2026, 3, 15, 10, 0, tzinfo=timezone.utc),
        full_content=full_content,
        snippet=snippet,
    )


def _fake_llm_output(
    summary: str = "S",
    sentiment_value: str = "BULLISH",
    sentiment_score: float = 0.5,
) -> MagicMock:
    raw = MagicMock()
    raw.summary = summary
    raw.sentiment = MagicMock()
    raw.sentiment.value = sentiment_value
    raw.sentiment_score = sentiment_score
    return raw


# ---------------------------------------------------------------------------
# _validate_llm_output — edge cases
# ---------------------------------------------------------------------------


class TestValidateLlmOutputEdgeCases:
    def test_when_sentiment_not_in_valid_set_then_defaults_to_neutral(self) -> None:
        raw = _fake_llm_output(sentiment_value="WEIRD")
        result = _validate_llm_output(raw)
        assert result["sentiment"] == "NEUTRAL"

    def test_when_sentiment_score_above_max_then_clamped_to_one(self) -> None:
        raw = _fake_llm_output(sentiment_score=5.0)
        result = _validate_llm_output(raw)
        assert result["sentiment_score"] == pytest.approx(1.0)

    def test_when_sentiment_score_below_min_then_clamped_to_minus_one(self) -> None:
        raw = _fake_llm_output(sentiment_score=-5.0)
        result = _validate_llm_output(raw)
        assert result["sentiment_score"] == pytest.approx(-1.0)

    @pytest.mark.parametrize(
        "label",
        ["BULLISH", "BEARISH", "NEUTRAL", "MIXED"],
    )
    def test_when_valid_sentiment_label_then_passed_through(self, label: str) -> None:
        raw = _fake_llm_output(sentiment_value=label)
        result = _validate_llm_output(raw)
        assert result["sentiment"] == label

    def test_when_score_within_range_then_unchanged(self) -> None:
        raw = _fake_llm_output(sentiment_score=0.3)
        result = _validate_llm_output(raw)
        assert result["sentiment_score"] == pytest.approx(0.3)

    def test_when_summary_present_then_passed_through(self) -> None:
        raw = _fake_llm_output(summary="Market rallied on Fed pivot.")
        result = _validate_llm_output(raw)
        assert result["summary"] == "Market rallied on Fed pivot."


# ---------------------------------------------------------------------------
# generate_country_summaries — two-country pipeline + commit guard
# ---------------------------------------------------------------------------

_USA_TICKERS = [
    _make_article(source_ticker="^GSPC"),
    _make_article(source_ticker="^GSPC"),
    _make_article(source_ticker="^GSPC"),
]
_UK_TICKERS = [
    _make_article(source_ticker="^FTSE"),
    _make_article(source_ticker="^FTSE"),
    _make_article(source_ticker="^FTSE"),
]


class TestGenerateCountrySummariesTwoCountries:
    def _run(
        self,
        articles: list,
        force_refresh: bool = True,
    ) -> tuple[list, MagicMock, MagicMock, MagicMock]:
        mock_session = MagicMock()
        mock_repo = MagicMock()
        mock_repo.get_macro_news.return_value = articles
        mock_repo.get_macro_news_summary.return_value = None
        mock_repo.upsert_macro_news_summary = MagicMock()

        mock_b = MagicMock()
        mock_b.SummarizeCountryNews.return_value = _fake_llm_output()

        with (
            patch(
                "app.services.macro.macro_news_summary.MacroRegimeRepository",
                return_value=mock_repo,
            ),
            patch(
                "app.services.macro.macro_news_summary.b",
                mock_b,
            ),
        ):
            results = generate_country_summaries(
                mock_session,
                force_refresh=force_refresh,
                on_progress=MagicMock(),
            )

        return results, mock_session, mock_repo, mock_b

    def test_when_usa_and_uk_articles_then_two_results_returned(self) -> None:
        results, _, _, _ = self._run(_USA_TICKERS + _UK_TICKERS)
        assert len(results) == 2

    def test_when_usa_and_uk_articles_then_both_countries_in_results(self) -> None:
        results, _, _, _ = self._run(_USA_TICKERS + _UK_TICKERS)
        countries = {r.country for r in results}
        assert "USA" in countries
        assert "UK" in countries

    def test_commit_per_country_guard_pre_loop_plus_n_countries(self) -> None:
        """Regression guard: commit must fire 1 (pre-loop) + n_countries times.

        The pre-loop commit releases the read transaction's pooled connection
        before the LLM loop. Without it the pool was exhausted under the 07:00
        scheduler burst (pool size 5, each LLM call took ~30s).
        """
        results, mock_session, _, _ = self._run(_USA_TICKERS + _UK_TICKERS)
        n_countries = len(results)  # 2
        # 1 pre-loop commit + 1 per country
        expected_commits = 1 + n_countries
        assert mock_session.commit.call_count == expected_commits, (
            f"Expected {expected_commits} session.commit() calls "
            f"(1 pre-loop + {n_countries} per-country), "
            f"got {mock_session.commit.call_count}"
        )

    def test_when_force_refresh_true_then_llm_called_twice(self) -> None:
        _, _, _, mock_b = self._run(_USA_TICKERS + _UK_TICKERS, force_refresh=True)
        assert mock_b.SummarizeCountryNews.call_count == 2

    def test_when_on_progress_provided_then_fired_per_country(self) -> None:
        mock_session = MagicMock()
        mock_repo = MagicMock()
        mock_repo.get_macro_news.return_value = _USA_TICKERS + _UK_TICKERS
        mock_repo.get_macro_news_summary.return_value = None

        on_progress = MagicMock()

        with (
            patch(
                "app.services.macro.macro_news_summary.MacroRegimeRepository",
                return_value=mock_repo,
            ),
            patch(
                "app.services.macro.macro_news_summary.b.SummarizeCountryNews",
                return_value=_fake_llm_output(),
            ),
        ):
            generate_country_summaries(
                mock_session,
                force_refresh=True,
                on_progress=on_progress,
            )

        all_kwargs = [c.kwargs for c in on_progress.call_args_list]
        assert any("current_country" in kw for kw in all_kwargs)
        assert any("current" in kw for kw in all_kwargs)
        assert any("total" in kw for kw in all_kwargs)


# ---------------------------------------------------------------------------
# _summarize_country threshold gate
# ---------------------------------------------------------------------------


class TestSummarizeCountryThresholdGate:
    def test_when_fewer_than_3_articles_then_llm_not_called(self) -> None:
        """Country with <3 articles must be skipped (no LLM call, result=None)."""
        articles = [
            _make_article(source_ticker="^GSPC"),
            _make_article(source_ticker="^GSPC"),
        ]
        mock_session = MagicMock()
        mock_repo = MagicMock()
        mock_repo.get_macro_news.return_value = articles
        mock_repo.get_macro_news_summary.return_value = None

        mock_b = MagicMock()

        with (
            patch(
                "app.services.macro.macro_news_summary.MacroRegimeRepository",
                return_value=mock_repo,
            ),
            patch("app.services.macro.macro_news_summary.b", mock_b),
        ):
            results = generate_country_summaries(
                mock_session,
                force_refresh=True,
            )

        assert results == []
        mock_b.SummarizeCountryNews.assert_not_called()


# ---------------------------------------------------------------------------
# run_news_summarize
# ---------------------------------------------------------------------------


class TestRunNewsSummarize:
    def test_when_called_then_generate_country_summaries_invoked_with_session(
        self,
    ) -> None:
        from datetime import date

        from app.services.macro.macro_news_summary import CountrySummaryResult

        fake_results = [
            CountrySummaryResult(
                country="USA",
                summary_date=date(2026, 3, 15),
                summary="S1",
                sentiment="BULLISH",
                sentiment_score=0.5,
                article_count=3,
                news_summary="text1",
            ),
            CountrySummaryResult(
                country="UK",
                summary_date=date(2026, 3, 15),
                summary="S2",
                sentiment="NEUTRAL",
                sentiment_score=0.0,
                article_count=3,
                news_summary="text2",
            ),
        ]

        request = SimpleNamespace(force_refresh=True, countries=None)
        on_progress = MagicMock()
        mock_session = MagicMock()

        with (
            patch("app.database.database_manager") as mock_db,
            patch(
                "app.services.macro.macro_news_summary.generate_country_summaries",
                return_value=fake_results,
            ) as mock_gen,
        ):
            mock_db.get_session.return_value.__enter__.return_value = mock_session
            mock_db.get_session.return_value.__exit__.return_value = False

            run_news_summarize(request, on_progress=on_progress)

        mock_gen.assert_called_once()
        gen_call = mock_gen.call_args
        assert gen_call.args[0] is mock_session

    def test_when_called_then_session_commit_called_after_generate(self) -> None:
        from datetime import date

        from app.services.macro.macro_news_summary import CountrySummaryResult

        fake_results = [
            CountrySummaryResult(
                country="USA",
                summary_date=date(2026, 3, 15),
                summary="S",
                sentiment="BULLISH",
                sentiment_score=0.5,
                article_count=3,
                news_summary="t",
            )
        ]

        request = SimpleNamespace(force_refresh=False, countries=None)
        mock_session = MagicMock()

        with (
            patch("app.database.database_manager") as mock_db,
            patch(
                "app.services.macro.macro_news_summary.generate_country_summaries",
                return_value=fake_results,
            ),
        ):
            mock_db.get_session.return_value.__enter__.return_value = mock_session
            mock_db.get_session.return_value.__exit__.return_value = False

            run_news_summarize(request)

        mock_session.commit.assert_called()

    def test_when_two_results_then_final_progress_has_status_completed(
        self,
    ) -> None:
        from datetime import date

        from app.services.macro.macro_news_summary import CountrySummaryResult

        fake_results = [
            CountrySummaryResult(
                country="USA",
                summary_date=date(2026, 3, 15),
                summary="S1",
                sentiment="BULLISH",
                sentiment_score=0.5,
                article_count=3,
                news_summary="t",
            ),
            CountrySummaryResult(
                country="UK",
                summary_date=date(2026, 3, 15),
                summary="S2",
                sentiment="NEUTRAL",
                sentiment_score=0.0,
                article_count=3,
                news_summary="t",
            ),
        ]

        request = SimpleNamespace(force_refresh=True, countries=None)
        on_progress = MagicMock()
        mock_session = MagicMock()

        with (
            patch("app.database.database_manager") as mock_db,
            patch(
                "app.services.macro.macro_news_summary.generate_country_summaries",
                return_value=fake_results,
            ),
        ):
            mock_db.get_session.return_value.__enter__.return_value = mock_session
            mock_db.get_session.return_value.__exit__.return_value = False

            run_news_summarize(request, on_progress=on_progress)

        final_kwargs = on_progress.call_args_list[-1].kwargs
        assert final_kwargs.get("status") == "completed"

    def test_when_two_results_then_final_progress_result_contains_countries_summarized(
        self,
    ) -> None:
        from datetime import date

        from app.services.macro.macro_news_summary import CountrySummaryResult

        fake_results = [
            CountrySummaryResult(
                country="USA",
                summary_date=date(2026, 3, 15),
                summary="S1",
                sentiment="BULLISH",
                sentiment_score=0.5,
                article_count=3,
                news_summary="t",
            ),
            CountrySummaryResult(
                country="UK",
                summary_date=date(2026, 3, 15),
                summary="S2",
                sentiment="NEUTRAL",
                sentiment_score=0.0,
                article_count=3,
                news_summary="t",
            ),
        ]

        request = SimpleNamespace(force_refresh=True, countries=None)
        on_progress = MagicMock()
        mock_session = MagicMock()

        with (
            patch("app.database.database_manager") as mock_db,
            patch(
                "app.services.macro.macro_news_summary.generate_country_summaries",
                return_value=fake_results,
            ),
        ):
            mock_db.get_session.return_value.__enter__.return_value = mock_session
            mock_db.get_session.return_value.__exit__.return_value = False

            run_news_summarize(request, on_progress=on_progress)

        final_kwargs = on_progress.call_args_list[-1].kwargs
        result_payload = final_kwargs.get("result", {})
        assert result_payload.get("countries_summarized") == 2
