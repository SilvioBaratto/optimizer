"""Unit tests for macro news summary service."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from baml_client.types import CountryNewsSummary, SentimentLabel

from app.models.macro_regime import MacroNews
from app.services.macro_news_summary import (
    QUERY_COUNTRY_MAP,
    TICKER_COUNTRY_MAP,
    _clamp_sentiment_score,
    _format_articles,
    _get_countries_for_article,
    _validate_llm_output,
    generate_country_summaries,
)


# ---------------------------------------------------------------------------
# Helpers — mock objects
# ---------------------------------------------------------------------------

_DEFAULT_PUBLISH_TIME = datetime(2026, 3, 15, 10, 0, tzinfo=timezone.utc)
_SENTINEL = object()


def _make_article(
    title: str | None = "Test Article",
    publisher: str | None = "Reuters",
    publish_time: datetime | None | object = _SENTINEL,
    source_ticker: str | None = None,
    source_query: str | None = None,
    snippet: str | None = "Article snippet text.",
    full_content: str | None = None,
) -> MacroNews:
    article = MagicMock(spec=MacroNews)
    article.title = title
    article.publisher = publisher
    article.publish_time = _DEFAULT_PUBLISH_TIME if publish_time is _SENTINEL else publish_time
    article.source_ticker = source_ticker
    article.source_query = source_query
    article.snippet = snippet
    article.full_content = full_content
    return article


def _make_llm_output(
    sentiment: SentimentLabel = SentimentLabel.BULLISH,
    sentiment_score: float = 0.6,
    summary: str = "Test summary.",
) -> CountryNewsSummary:
    return CountryNewsSummary(
        summary=summary,
        sentiment=sentiment,
        sentiment_score=sentiment_score,
    )


# ---------------------------------------------------------------------------
# Country mapping
# ---------------------------------------------------------------------------


class TestTickerCountryMap:
    def test_has_20_entries(self) -> None:
        assert len(TICKER_COUNTRY_MAP) == 20

    def test_eem_maps_to_global(self) -> None:
        assert TICKER_COUNTRY_MAP["EEM"] == "_GLOBAL"

    def test_dx_y_nyb_maps_to_global(self) -> None:
        assert TICKER_COUNTRY_MAP["DX-Y.NYB"] == "_GLOBAL"

    def test_gspc_maps_to_usa(self) -> None:
        assert TICKER_COUNTRY_MAP["^GSPC"] == "USA"

    def test_ftse_maps_to_uk(self) -> None:
        assert TICKER_COUNTRY_MAP["^FTSE"] == "UK"

    def test_gdaxi_maps_to_germany(self) -> None:
        assert TICKER_COUNTRY_MAP["^GDAXI"] == "Germany"

    def test_fchi_maps_to_france(self) -> None:
        assert TICKER_COUNTRY_MAP["^FCHI"] == "France"


class TestQueryCountryMap:
    def test_has_12_entries(self) -> None:
        assert len(QUERY_COUNTRY_MAP) == 12

    def test_ecb_interest_rate_maps_to_germany_and_france(self) -> None:
        assert QUERY_COUNTRY_MAP["ECB interest rate decision"] == ["Germany", "France"]

    def test_eurozone_maps_to_germany_and_france(self) -> None:
        assert QUERY_COUNTRY_MAP["eurozone economy recession growth"] == [
            "Germany",
            "France",
        ]

    def test_fed_maps_to_usa(self) -> None:
        assert QUERY_COUNTRY_MAP["Federal Reserve interest rate decision"] == ["USA"]

    def test_boe_maps_to_uk(self) -> None:
        assert QUERY_COUNTRY_MAP["Bank of England rate decision"] == ["UK"]


# ---------------------------------------------------------------------------
# _get_countries_for_article
# ---------------------------------------------------------------------------


class TestGetCountriesForArticle:
    def test_ticker_usa(self) -> None:
        article = _make_article(source_ticker="^GSPC")
        assert _get_countries_for_article(article) == ["USA"]

    def test_ticker_uk(self) -> None:
        article = _make_article(source_ticker="^FTSE")
        assert _get_countries_for_article(article) == ["UK"]

    def test_ticker_global_returns_empty(self) -> None:
        article = _make_article(source_ticker="EEM")
        assert _get_countries_for_article(article) == []

    def test_ticker_dx_y_global_returns_empty(self) -> None:
        article = _make_article(source_ticker="DX-Y.NYB")
        assert _get_countries_for_article(article) == []

    def test_query_ecb_maps_to_both(self) -> None:
        article = _make_article(source_query="ECB interest rate decision")
        result = _get_countries_for_article(article)
        assert "Germany" in result
        assert "France" in result

    def test_query_fed_maps_to_usa(self) -> None:
        article = _make_article(source_query="Federal Reserve interest rate decision")
        assert _get_countries_for_article(article) == ["USA"]

    def test_unmapped_ticker_returns_empty(self) -> None:
        article = _make_article(source_ticker="UNKNOWN_TICKER")
        assert _get_countries_for_article(article) == []

    def test_unmapped_query_returns_empty(self) -> None:
        article = _make_article(source_query="random search query")
        assert _get_countries_for_article(article) == []

    def test_no_source_returns_empty(self) -> None:
        article = _make_article(source_ticker=None, source_query=None)
        assert _get_countries_for_article(article) == []

    def test_global_query_returns_empty(self) -> None:
        article = _make_article(source_query="emerging markets capital flows")
        assert _get_countries_for_article(article) == []


# ---------------------------------------------------------------------------
# _format_articles
# ---------------------------------------------------------------------------


class TestFormatArticles:
    def test_basic_formatting(self) -> None:
        article = _make_article(
            title="Fed holds rates",
            publisher="Reuters",
            publish_time=datetime(2026, 3, 15, 14, 30, tzinfo=timezone.utc),
        )
        result = _format_articles([article])
        assert "[1] Fed holds rates (Reuters, 2026-03-15 14:30 UTC)" in result

    def test_none_publish_time_excluded(self) -> None:
        article = _make_article(title="Test", publish_time=None, publisher="Reuters")
        result = _format_articles([article])
        assert "[1] Test (Reuters)" in result
        assert "UTC" not in result

    def test_none_publisher_excluded(self) -> None:
        article = _make_article(
            title="Test",
            publisher=None,
            publish_time=datetime(2026, 3, 15, 10, 0, tzinfo=timezone.utc),
        )
        result = _format_articles([article])
        assert "[1] Test (2026-03-15 10:00 UTC)" in result

    def test_no_meta_no_parens(self) -> None:
        article = _make_article(title="Test", publisher=None, publish_time=None)
        result = _format_articles([article])
        assert "[1] Test" in result
        assert "(" not in result

    def test_caps_at_15_articles(self) -> None:
        articles = [_make_article(title=f"Article {i}") for i in range(20)]
        result = _format_articles(articles)
        assert "[15]" in result
        assert "[16]" not in result

    def test_full_content_preferred_over_snippet(self) -> None:
        article = _make_article(
            snippet="Short snippet",
            full_content="Full article content here",
        )
        result = _format_articles([article])
        assert "Full article content here" in result
        assert "Short snippet" not in result

    def test_content_truncated_at_500_chars(self) -> None:
        article = _make_article(full_content="x" * 600)
        result = _format_articles([article])
        assert "..." in result

    def test_none_title_falls_back_to_untitled(self) -> None:
        article = _make_article(title=None, publisher="AP")
        result = _format_articles([article])
        assert "[1] Untitled" in result

    def test_exactly_15_articles_all_numbered(self) -> None:
        articles = [_make_article(title=f"Article {i}") for i in range(1, 16)]
        result = _format_articles(articles)
        for i in range(1, 16):
            assert f"[{i}]" in result
        assert "[16]" not in result

    def test_empty_list_returns_empty(self) -> None:
        assert _format_articles([]) == ""


# ---------------------------------------------------------------------------
# _clamp_sentiment_score
# ---------------------------------------------------------------------------


class TestClampSentimentScore:
    def test_within_range(self) -> None:
        assert _clamp_sentiment_score(0.5) == pytest.approx(0.5)

    def test_above_max(self) -> None:
        assert _clamp_sentiment_score(2.0) == pytest.approx(1.0)

    def test_below_min(self) -> None:
        assert _clamp_sentiment_score(-3.0) == pytest.approx(-1.0)

    def test_boundary_positive(self) -> None:
        assert _clamp_sentiment_score(1.0) == pytest.approx(1.0)

    def test_boundary_negative(self) -> None:
        assert _clamp_sentiment_score(-1.0) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# _validate_llm_output
# ---------------------------------------------------------------------------


class TestValidateLlmOutput:
    def test_valid_output_passes_through(self) -> None:
        raw = _make_llm_output()
        result = _validate_llm_output(raw)
        assert result["sentiment"] == "BULLISH"
        assert result["sentiment_score"] == pytest.approx(0.6)
        assert result["summary"] == "Test summary."

    def test_score_clamped_to_positive_bound(self) -> None:
        raw = _make_llm_output(sentiment_score=5.0)
        result = _validate_llm_output(raw)
        assert result["sentiment_score"] == pytest.approx(1.0)

    def test_score_clamped_to_negative_bound(self) -> None:
        raw = _make_llm_output(sentiment_score=-5.0)
        result = _validate_llm_output(raw)
        assert result["sentiment_score"] == pytest.approx(-1.0)

    def test_all_sentiment_labels_valid(self) -> None:
        for label in SentimentLabel:
            raw = _make_llm_output(sentiment=label)
            result = _validate_llm_output(raw)
            assert result["sentiment"] == label.value

    def test_invalid_sentiment_value_falls_back_to_neutral(self) -> None:
        raw = MagicMock()
        raw.summary = "Some summary."
        raw.sentiment_score = 0.1
        raw.sentiment = MagicMock()
        raw.sentiment.value = "SUPER_BULLISH"  # not in _VALID_SENTIMENTS

        result = _validate_llm_output(raw)

        assert result["sentiment"] == "NEUTRAL"
        assert result["summary"] == "Some summary."
        assert result["sentiment_score"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# _summarize_country (via generate_country_summaries)
# ---------------------------------------------------------------------------


class TestSummarizeCountry:
    def test_cached_result_returned_without_llm_call(self) -> None:
        mock_session = MagicMock()
        mock_repo = MagicMock()
        cached_row = MagicMock()
        cached_row.summary = "Cached summary"
        cached_row.sentiment = "NEUTRAL"
        cached_row.sentiment_score = 0.0
        cached_row.article_count = 5
        cached_row.news_summary = "Cached text"
        mock_repo.get_macro_news_summary.return_value = cached_row
        mock_repo.get_macro_news.return_value = [
            _make_article(source_ticker="^GSPC") for _ in range(5)
        ]

        with (
            patch(
                "app.services.macro_news_summary.MacroRegimeRepository",
                return_value=mock_repo,
            ),
            patch("app.services.macro_news_summary.b.SummarizeCountryNews") as mock_llm,
        ):
            results = generate_country_summaries(mock_session, force_refresh=False)

        mock_llm.assert_not_called()
        assert len(results) >= 1
        usa_results = [r for r in results if r.country == "USA"]
        assert len(usa_results) == 1
        assert usa_results[0].summary == "Cached summary"

    def test_below_threshold_skipped(self) -> None:
        mock_session = MagicMock()
        mock_repo = MagicMock()
        mock_repo.get_macro_news_summary.return_value = None
        mock_repo.get_macro_news.return_value = [
            _make_article(source_ticker="^GSPC") for _ in range(2)
        ]

        with patch(
            "app.services.macro_news_summary.MacroRegimeRepository",
            return_value=mock_repo,
        ):
            results = generate_country_summaries(mock_session, force_refresh=True)

        assert len(results) == 0

    def test_llm_called_on_force_refresh(self) -> None:
        mock_session = MagicMock()
        mock_repo = MagicMock()
        mock_repo.get_macro_news_summary.return_value = None
        mock_repo.get_macro_news.return_value = [
            _make_article(source_ticker="^GSPC") for _ in range(5)
        ]

        mock_llm_output = _make_llm_output()

        with (
            patch(
                "app.services.macro_news_summary.MacroRegimeRepository",
                return_value=mock_repo,
            ),
            patch(
                "app.services.macro_news_summary.b.SummarizeCountryNews",
                return_value=mock_llm_output,
            ) as mock_llm,
        ):
            results = generate_country_summaries(mock_session, force_refresh=True)

        mock_llm.assert_called_once()
        assert len(results) == 1
        assert results[0].country == "USA"
        assert results[0].sentiment == "BULLISH"

    def test_article_count_stored_in_result(self) -> None:
        mock_session = MagicMock()
        mock_repo = MagicMock()
        mock_repo.get_macro_news_summary.return_value = None
        mock_repo.get_macro_news.return_value = [
            _make_article(source_ticker="^GSPC") for _ in range(5)
        ]

        with (
            patch(
                "app.services.macro_news_summary.MacroRegimeRepository",
                return_value=mock_repo,
            ),
            patch(
                "app.services.macro_news_summary.b.SummarizeCountryNews",
                return_value=_make_llm_output(),
            ),
        ):
            results = generate_country_summaries(mock_session, force_refresh=True)

        assert len(results) == 1
        assert results[0].article_count == 5

    def test_news_summary_text_populated_after_llm_call(self) -> None:
        mock_session = MagicMock()
        mock_repo = MagicMock()
        mock_repo.get_macro_news_summary.return_value = None
        mock_repo.get_macro_news.return_value = [
            _make_article(source_ticker="^GSPC", title=f"News {i}") for i in range(3)
        ]

        with (
            patch(
                "app.services.macro_news_summary.MacroRegimeRepository",
                return_value=mock_repo,
            ),
            patch(
                "app.services.macro_news_summary.b.SummarizeCountryNews",
                return_value=_make_llm_output(),
            ),
        ):
            results = generate_country_summaries(mock_session, force_refresh=True)

        assert results[0].news_summary != ""
        assert "News 0" in results[0].news_summary

    def test_cache_lookup_uses_today_date(self) -> None:
        mock_session = MagicMock()
        mock_repo = MagicMock()
        cached_row = MagicMock()
        cached_row.summary = "Cached"
        cached_row.sentiment = "NEUTRAL"
        cached_row.sentiment_score = 0.0
        cached_row.article_count = 5
        cached_row.news_summary = "Cached text"
        mock_repo.get_macro_news_summary.return_value = cached_row
        mock_repo.get_macro_news.return_value = [
            _make_article(source_ticker="^GSPC") for _ in range(5)
        ]

        with patch(
            "app.services.macro_news_summary.MacroRegimeRepository",
            return_value=mock_repo,
        ):
            generate_country_summaries(mock_session, force_refresh=False)

        call_args = mock_repo.get_macro_news_summary.call_args
        passed_date = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get("summary_date")
        today = datetime.now(timezone.utc).date()
        assert passed_date == today

    def test_upsert_failure_nonfatal(self) -> None:
        mock_session = MagicMock()
        mock_repo = MagicMock()
        mock_repo.get_macro_news_summary.return_value = None
        mock_repo.upsert_macro_news_summary.side_effect = RuntimeError("DB error")
        mock_repo.get_macro_news.return_value = [
            _make_article(source_ticker="^GSPC") for _ in range(5)
        ]

        mock_llm_output = _make_llm_output()

        with (
            patch(
                "app.services.macro_news_summary.MacroRegimeRepository",
                return_value=mock_repo,
            ),
            patch(
                "app.services.macro_news_summary.b.SummarizeCountryNews",
                return_value=mock_llm_output,
            ),
        ):
            results = generate_country_summaries(mock_session, force_refresh=True)

        assert len(results) == 1


# ---------------------------------------------------------------------------
# generate_country_summaries — full pipeline
# ---------------------------------------------------------------------------


class TestGenerateCountrySummaries:
    def test_global_articles_excluded_from_results(self) -> None:
        mock_session = MagicMock()
        mock_repo = MagicMock()
        mock_repo.get_macro_news_summary.return_value = None
        mock_repo.get_macro_news.return_value = [
            _make_article(source_ticker="EEM") for _ in range(5)
        ]

        with patch(
            "app.services.macro_news_summary.MacroRegimeRepository",
            return_value=mock_repo,
        ):
            results = generate_country_summaries(mock_session, force_refresh=True)

        assert len(results) == 0

    def test_ecb_articles_appear_in_both_germany_and_france(self) -> None:
        mock_session = MagicMock()
        mock_repo = MagicMock()
        mock_repo.get_macro_news_summary.return_value = None
        mock_repo.get_macro_news.return_value = [
            _make_article(source_query="ECB interest rate decision")
            for _ in range(5)
        ]

        mock_llm_output = _make_llm_output()

        with (
            patch(
                "app.services.macro_news_summary.MacroRegimeRepository",
                return_value=mock_repo,
            ),
            patch(
                "app.services.macro_news_summary.b.SummarizeCountryNews",
                return_value=mock_llm_output,
            ),
        ):
            results = generate_country_summaries(mock_session, force_refresh=True)

        countries = {r.country for r in results}
        assert "Germany" in countries
        assert "France" in countries

    def test_empty_articles_returns_empty(self) -> None:
        mock_session = MagicMock()
        mock_repo = MagicMock()
        mock_repo.get_macro_news.return_value = []

        with patch(
            "app.services.macro_news_summary.MacroRegimeRepository",
            return_value=mock_repo,
        ):
            results = generate_country_summaries(mock_session)

        assert results == []

    def test_mixed_sources_grouped_correctly(self) -> None:
        mock_session = MagicMock()
        mock_repo = MagicMock()
        mock_repo.get_macro_news_summary.return_value = None
        mock_repo.get_macro_news.return_value = [
            _make_article(source_ticker="^GSPC"),
            _make_article(source_ticker="^GSPC"),
            _make_article(source_ticker="^GSPC"),
            _make_article(source_ticker="^FTSE"),
            _make_article(source_ticker="^FTSE"),
            _make_article(source_ticker="^FTSE"),
            _make_article(source_ticker="EEM"),  # global, excluded
        ]

        mock_llm_output = _make_llm_output()

        with (
            patch(
                "app.services.macro_news_summary.MacroRegimeRepository",
                return_value=mock_repo,
            ),
            patch(
                "app.services.macro_news_summary.b.SummarizeCountryNews",
                return_value=mock_llm_output,
            ),
        ):
            results = generate_country_summaries(mock_session, force_refresh=True)

        countries = {r.country for r in results}
        assert "USA" in countries
        assert "UK" in countries
        assert "_GLOBAL" not in countries

    def test_results_ordered_alphabetically_by_country(self) -> None:
        mock_session = MagicMock()
        mock_repo = MagicMock()
        mock_repo.get_macro_news_summary.return_value = None
        mock_repo.get_macro_news.return_value = [
            _make_article(source_ticker="^GSPC"),
            _make_article(source_ticker="^GSPC"),
            _make_article(source_ticker="^GSPC"),
            _make_article(source_ticker="^FTSE"),
            _make_article(source_ticker="^FTSE"),
            _make_article(source_ticker="^FTSE"),
            _make_article(source_ticker="^GDAXI"),
            _make_article(source_ticker="^GDAXI"),
            _make_article(source_ticker="^GDAXI"),
        ]

        with (
            patch(
                "app.services.macro_news_summary.MacroRegimeRepository",
                return_value=mock_repo,
            ),
            patch(
                "app.services.macro_news_summary.b.SummarizeCountryNews",
                return_value=_make_llm_output(),
            ),
        ):
            results = generate_country_summaries(mock_session, force_refresh=True)

        countries = [r.country for r in results]
        assert countries == sorted(countries)
