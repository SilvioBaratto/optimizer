"""Macro news summary service with country mapping and LLM summarization.

Workflow:
  1. Fetch recent MacroNews articles from the DB (last 24 hours).
  2. Map each article to one or more countries via ticker/query lookup.
  3. Group articles by country, skip _GLOBAL entries.
  4. For each country with >= 3 articles, call the BAML ``SummarizeCountryNews``
     function to generate a summary, key themes, and sentiment.
  5. Validate and clamp LLM output, persist to ``macro_news_summaries`` table.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Literal

from baml_client import b
from baml_client.types import CountryNewsSummary
from sqlalchemy.orm import Session

from app.models.macro_regime import MacroNews
from app.repositories.macro_regime_repository import MacroRegimeRepository
from app.services._progress import ProgressCallback, _noop

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Country mapping constants
# ---------------------------------------------------------------------------

# 20 tickers from MACRO_TICKERS mapped to countries.
# EEM and DX-Y.NYB map to _GLOBAL (not USA) per issue #213.
TICKER_COUNTRY_MAP: dict[str, str] = {
    # USA — rates
    "^TNX": "USA",
    "^TYX": "USA",
    "^IRX": "USA",
    # USA — volatility
    "^VIX": "USA",
    # Commodities — global
    "GC=F": "_GLOBAL",
    "CL=F": "_GLOBAL",
    # Currency / EM — global
    "DX-Y.NYB": "_GLOBAL",
    "EEM": "_GLOBAL",
    # USA — sector ETFs
    "XLF": "USA",
    "XLE": "USA",
    "XLK": "USA",
    "XLP": "USA",
    "XLU": "USA",
    # USA — broad market
    "^GSPC": "USA",
    # UK
    "^FTSE": "UK",
    "GBPUSD=X": "UK",
    # Germany
    "^GDAXI": "Germany",
    # France
    "^FCHI": "France",
    # Europe broad — global
    "^STOXX50E": "_GLOBAL",
    "EURUSD=X": "_GLOBAL",
}

# 12 queries from MACRO_SEARCH_QUERIES mapped to countries.
# ECB queries map to both Germany and France.
QUERY_COUNTRY_MAP: dict[str, list[str]] = {
    "Federal Reserve interest rate decision": ["USA"],
    "ISM manufacturing PMI economic": ["USA"],
    "treasury yield curve inversion": ["USA"],
    "high yield credit spread corporate bond": ["USA"],
    "sector rotation cyclical defensive": ["USA"],
    "CPI inflation consumer prices": ["USA"],
    "emerging markets capital flows": ["_GLOBAL"],
    "recession GDP employment nonfarm": ["USA"],
    "Bank of England rate decision": ["UK"],
    "UK inflation economy GDP": ["UK"],
    "ECB interest rate decision": ["Germany", "France"],
    "eurozone economy recession growth": ["Germany", "France"],
}

# ---------------------------------------------------------------------------
# Thresholds and bounds
# ---------------------------------------------------------------------------

_GLOBAL = "_GLOBAL"
_MIN_ARTICLES = 3
_MAX_ARTICLES = 15
_NEWS_LIMIT = 500
_CUTOFF_HOURS = 24
_CONTENT_TRUNCATE = 500
_SENTIMENT_SCORE_MIN = -1.0
_SENTIMENT_SCORE_MAX = 1.0
_VALID_SENTIMENTS = frozenset({"BULLISH", "BEARISH", "NEUTRAL", "MIXED"})


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CountrySummaryResult:
    """Result of a single country's news summarization."""

    country: str
    summary_date: date
    summary: str
    sentiment: str
    sentiment_score: float
    article_count: int
    news_summary: str  # raw text fed to the LLM


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_countries_for_article(article: MacroNews) -> list[str]:
    """Map an article to its target countries via ticker or query lookup.

    Returns an empty list for _GLOBAL or unmapped articles.
    """
    countries: list[str] = []

    if article.source_ticker and article.source_ticker in TICKER_COUNTRY_MAP:
        country = TICKER_COUNTRY_MAP[article.source_ticker]
        if country != _GLOBAL:
            countries.append(country)
        return countries

    if article.source_query and article.source_query in QUERY_COUNTRY_MAP:
        mapped = QUERY_COUNTRY_MAP[article.source_query]
        return [c for c in mapped if c != _GLOBAL]

    # Unmapped article
    source = article.source_ticker or article.source_query or "unknown"
    logger.warning("Unmapped article source: %s (title=%s)", source, article.title)
    return []


def _format_articles(articles: Sequence[MacroNews]) -> str:
    """Format articles into numbered text for the LLM, capped at _MAX_ARTICLES."""
    capped = list(articles)[:_MAX_ARTICLES]
    lines: list[str] = []

    for i, article in enumerate(capped, 1):
        # Build header: [N] TITLE (PUBLISHER, TIMESTAMP)
        parts = []
        if article.publisher:
            parts.append(article.publisher)
        if article.publish_time is not None:
            parts.append(article.publish_time.strftime("%Y-%m-%d %H:%M UTC"))
        meta = f" ({', '.join(parts)})" if parts else ""
        lines.append(f"[{i}] {article.title or 'Untitled'}{meta}")

        # Content: prefer full_content, fall back to snippet
        content = article.full_content or article.snippet or ""
        if len(content) > _CONTENT_TRUNCATE:
            content = content[: _CONTENT_TRUNCATE - 3] + "..."
        if content:
            lines.append(content)
        lines.append("")

    return "\n".join(lines).strip()


def _clamp_sentiment_score(value: float) -> float:
    """Clamp sentiment score to [-1.0, 1.0]."""
    return max(_SENTIMENT_SCORE_MIN, min(_SENTIMENT_SCORE_MAX, value))


def _validate_llm_output(raw: CountryNewsSummary) -> dict[str, Any]:
    """Validate and normalize LLM output into a dict for persistence."""
    sentiment_value = raw.sentiment.value
    if sentiment_value not in _VALID_SENTIMENTS:
        logger.warning(
            "Invalid sentiment '%s' from LLM, defaulting to NEUTRAL",
            sentiment_value,
        )
        sentiment_value = "NEUTRAL"

    score = _clamp_sentiment_score(raw.sentiment_score)
    if score != raw.sentiment_score:
        logger.warning(
            "Sentiment score %.4f out of range, clamped to %.4f",
            raw.sentiment_score,
            score,
        )

    return {
        "summary": raw.summary,
        "sentiment": sentiment_value,
        "sentiment_score": score,
    }


def _summarize_country(
    repo: MacroRegimeRepository,
    country: str,
    articles: list[MacroNews],
    today: date,
    force_refresh: bool,
) -> CountrySummaryResult | None:
    """Summarize news for a single country. Returns None if skipped."""
    # Cache gate
    if not force_refresh:
        cached = repo.get_macro_news_summary(country, today)
        if cached is not None:
            logger.info("Returning cached news summary for country=%s", country)
            return CountrySummaryResult(
                country=country,
                summary_date=today,
                summary=cached.summary or "",
                sentiment=cached.sentiment or "NEUTRAL",
                sentiment_score=cached.sentiment_score or 0.0,
                article_count=cached.article_count or 0,
                news_summary=cached.news_summary or "",
            )

    # Threshold gate
    if len(articles) < _MIN_ARTICLES:
        logger.warning(
            "Skipping %s: only %d articles (threshold=%d)",
            country,
            len(articles),
            _MIN_ARTICLES,
        )
        return None

    # Format and call LLM
    news_text = _format_articles(articles)
    raw: CountryNewsSummary = b.SummarizeCountryNews(
        country=country, news_text=news_text,
    )
    validated = _validate_llm_output(raw)

    result = CountrySummaryResult(
        country=country,
        summary_date=today,
        summary=validated["summary"],
        sentiment=validated["sentiment"],
        sentiment_score=validated["sentiment_score"],
        article_count=len(articles),
        news_summary=news_text,
    )

    # Persist (non-fatal)
    try:
        repo.upsert_macro_news_summary(
            country=country,
            summary_date=today,
            data={
                "summary": validated["summary"],
                "sentiment": validated["sentiment"],
                "sentiment_score": validated["sentiment_score"],
                "article_count": len(articles),
                "news_summary": news_text,
            },
        )
        logger.info("Persisted news summary for country=%s", country)
    except Exception:
        logger.exception(
            "Failed to persist news summary for country=%s (non-fatal)", country,
        )

    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_country_summaries(
    session: Session,
    force_refresh: bool = False,
    countries: list[str] | None = None,
) -> list[CountrySummaryResult]:
    """Generate daily news summaries for mapped countries.

    Args:
        session: Active SQLAlchemy session.
        force_refresh: When True, bypass the cache and re-invoke the LLM.
        countries: Restrict to these countries. None means all mapped countries.

    Returns:
        List of :class:`CountrySummaryResult` for each successfully summarized country.
    """
    repo = MacroRegimeRepository(session)

    # Fetch recent articles (last 24 hours, timezone-aware)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=_CUTOFF_HOURS)
    articles = repo.get_macro_news(start_date=cutoff, limit=_NEWS_LIMIT)

    # Group articles by country
    country_articles: dict[str, list[MacroNews]] = defaultdict(list)
    for article in articles:
        mapped = _get_countries_for_article(article)
        for country in mapped:
            country_articles[country].append(article)

    # Filter to requested countries
    if countries:
        allowed = frozenset(countries)
        country_articles = {k: v for k, v in country_articles.items() if k in allowed}

    today = datetime.now(timezone.utc).date()

    # Summarize per country
    results: list[CountrySummaryResult] = []
    for country, arts in sorted(country_articles.items()):
        result = _summarize_country(repo, country, arts, today, force_refresh)
        if result is not None:
            results.append(result)

    logger.info(
        "Generated %d country summaries from %d articles",
        len(results),
        len(articles),
    )
    return results


# ---------------------------------------------------------------------------
# Incremental refresh helpers (used by scheduler)
# ---------------------------------------------------------------------------

CountryOutcome = Literal["updated", "skipped", "error"]


def _find_countries_with_new_articles(
    repo: MacroRegimeRepository,
    since: datetime,
) -> list[str]:
    """Return country names that have at least one new article since *since*."""
    articles = repo.get_macro_news(start_date=since, limit=500)
    countries: set[str] = set()

    for article in articles:
        if article.source_ticker and article.source_ticker in TICKER_COUNTRY_MAP:
            country = TICKER_COUNTRY_MAP[article.source_ticker]
            if country != _GLOBAL:
                countries.add(country)
        elif article.source_query and article.source_query in QUERY_COUNTRY_MAP:
            for c in QUERY_COUNTRY_MAP[article.source_query]:
                if c != _GLOBAL:
                    countries.add(c)

    return sorted(countries)


def _is_morning_pipeline_complete(
    repo: MacroRegimeRepository,
    today: date,
) -> bool:
    """Return True if at least one MacroNewsSummary row exists for today."""
    summaries = repo.get_all_news_summaries(summary_date=today)
    return len(summaries) > 0


def _summarize_country_safe(
    session: Session,
    country: str,
) -> CountryOutcome:
    """Summarize a single country with error isolation.

    Returns "updated", "skipped", or "error".
    """
    try:
        results = generate_country_summaries(
            session,
            force_refresh=True,
            countries=[country],
        )
        session.commit()
        return "updated" if results else "skipped"
    except Exception:
        logger.exception(
            "Scheduler: summarization failed for country=%s (non-fatal)", country,
        )
        try:
            session.rollback()
        except Exception:
            logger.exception("Scheduler: rollback failed for country=%s", country)
        return "error"


# ---------------------------------------------------------------------------
# Standalone summarize (callable from routes and scheduler)
# ---------------------------------------------------------------------------


def run_news_summarize(
    request: Any,
    *,
    on_progress: ProgressCallback = _noop,
) -> list[CountrySummaryResult]:
    """Execute news summarization in the service layer.

    Args:
        request: ``MacroNewsSummarizeRequest`` with ``force_refresh`` and ``countries``.
        on_progress: Optional callback for progress updates.

    Returns:
        List of ``CountrySummaryResult`` for each summarized country.
    """
    from app.database import database_manager

    with database_manager.get_session() as session:
        results = generate_country_summaries(
            session,
            force_refresh=request.force_refresh,
            countries=request.countries,
        )
        session.commit()

        on_progress(
            status="completed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            current=len(results),
            total=len(results),
            result={
                "countries_summarized": len(results),
                "countries": [r.country for r in results],
            },
        )
        logger.info("News summarize completed: %d countries", len(results))

    return results
