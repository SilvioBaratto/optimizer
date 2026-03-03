"""Macro-themed news fetching from yfinance ticker feeds and search queries.

Aggregates articles from macro-relevant tickers (treasuries, VIX, commodities,
sector ETFs) and targeted search queries (Fed rate decisions, ISM PMI, etc.),
classifies them by ``MacroTheme``, and deduplicates by title.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..protocols import (
    ArticleScraperProtocol,
    SearchClientProtocol,
    YFinanceClientProtocol,
)
from .aggregator import _is_article_recent, _parse_article_date

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enum & static configuration
# ---------------------------------------------------------------------------

# Trusted, impartial publishers — wire services, major financial press,
# and established data providers.  Articles from unlisted publishers are
# silently dropped during processing.
TRUSTED_PUBLISHERS: set[str] = {
    # Wire services / agencies
    "Reuters",
    "Associated Press Finance",
    "AFP",
    # Major financial press
    "Bloomberg",
    "The Wall Street Journal",
    "Barrons.com",
    "MarketWatch",
    "Financial Times",
    "The Economist",
    "CNN Business",
    "CNBC",
    # Yahoo Finance (primary data source)
    "Yahoo Finance",
    "Yahoo Finance Video",
    "Yahoo Finance UK",
    # European press
    "The Guardian",
    "The Telegraph",
    "Euronews",
    "PA Media: Money",
    # Market data providers
    "Investing.com",
    "Barchart",
    "MT Newswires",
}


class MacroTheme(str, Enum):
    monetary_policy = "monetary_policy"
    growth_indicators = "growth_indicators"
    credit_conditions = "credit_conditions"
    yield_curve = "yield_curve"
    sector_rotation = "sector_rotation"
    volatility_risk = "volatility_risk"
    commodity_inflation = "commodity_inflation"
    geographic_allocation = "geographic_allocation"
    business_cycle = "business_cycle"


MACRO_TICKERS: dict[str, list[MacroTheme]] = {
    # --- USA ---
    # Fixed income / rates
    "^TNX": [MacroTheme.yield_curve, MacroTheme.monetary_policy],
    "^TYX": [MacroTheme.yield_curve, MacroTheme.monetary_policy],
    "^IRX": [MacroTheme.yield_curve, MacroTheme.monetary_policy],
    # Volatility
    "^VIX": [MacroTheme.volatility_risk],
    # Commodities
    "GC=F": [MacroTheme.commodity_inflation],
    "CL=F": [MacroTheme.commodity_inflation],
    # Currency / EM
    "DX-Y.NYB": [MacroTheme.geographic_allocation, MacroTheme.monetary_policy],
    "EEM": [MacroTheme.geographic_allocation],
    # Sector ETFs
    "XLF": [MacroTheme.sector_rotation, MacroTheme.credit_conditions],
    "XLE": [MacroTheme.sector_rotation, MacroTheme.commodity_inflation],
    "XLK": [MacroTheme.sector_rotation],
    "XLP": [MacroTheme.sector_rotation, MacroTheme.business_cycle],
    "XLU": [MacroTheme.sector_rotation, MacroTheme.business_cycle],
    # Broad market
    "^GSPC": [MacroTheme.business_cycle, MacroTheme.growth_indicators],
    # --- UK ---
    "^FTSE": [MacroTheme.business_cycle, MacroTheme.growth_indicators],
    "GBPUSD=X": [MacroTheme.geographic_allocation, MacroTheme.monetary_policy],
    # --- Germany ---
    "^GDAXI": [MacroTheme.business_cycle, MacroTheme.growth_indicators],
    # --- France ---
    "^FCHI": [MacroTheme.business_cycle, MacroTheme.growth_indicators],
    # --- Europe broad ---
    "^STOXX50E": [MacroTheme.business_cycle, MacroTheme.geographic_allocation],
    "EURUSD=X": [MacroTheme.geographic_allocation, MacroTheme.monetary_policy],
}

MACRO_SEARCH_QUERIES: dict[str, list[MacroTheme]] = {
    # --- USA ---
    "Federal Reserve interest rate decision": [
        MacroTheme.monetary_policy,
    ],
    "ISM manufacturing PMI economic": [
        MacroTheme.growth_indicators,
        MacroTheme.business_cycle,
    ],
    "treasury yield curve inversion": [
        MacroTheme.yield_curve,
    ],
    "high yield credit spread corporate bond": [
        MacroTheme.credit_conditions,
    ],
    "sector rotation cyclical defensive": [
        MacroTheme.sector_rotation,
    ],
    "CPI inflation consumer prices": [
        MacroTheme.commodity_inflation,
        MacroTheme.monetary_policy,
    ],
    "emerging markets capital flows": [
        MacroTheme.geographic_allocation,
    ],
    "recession GDP employment nonfarm": [
        MacroTheme.business_cycle,
        MacroTheme.growth_indicators,
    ],
    # --- UK ---
    "Bank of England rate decision": [
        MacroTheme.monetary_policy,
    ],
    "UK inflation economy GDP": [
        MacroTheme.growth_indicators,
        MacroTheme.commodity_inflation,
    ],
    # --- Europe / ECB ---
    "ECB interest rate decision": [
        MacroTheme.monetary_policy,
    ],
    "eurozone economy recession growth": [
        MacroTheme.business_cycle,
        MacroTheme.growth_indicators,
    ],
}

_THEME_KEYWORDS: dict[MacroTheme, list[str]] = {
    MacroTheme.monetary_policy: [
        r"(?i)\bfed(eral reserve)?\b",
        r"(?i)\binterest rate\b",
        r"(?i)\brate (cut|hike|decision|hold)\b",
        r"(?i)\bmonetary policy\b",
        r"(?i)\bfomc\b",
        r"(?i)\bpowell\b",
        r"(?i)\becb\b",
        r"(?i)\beuropean central bank\b",
        r"(?i)\bbank of england\b",
        r"(?i)\blagarde\b",
        r"(?i)\bbundesbank\b",
        r"(?i)\bbanque de france\b",
    ],
    MacroTheme.growth_indicators: [
        r"(?i)\bgdp\b",
        r"(?i)\bpmi\b",
        r"(?i)\bism\b",
        r"(?i)\bnonfarm payroll\b",
        r"(?i)\bjobs report\b",
        r"(?i)\beconomic growth\b",
        r"(?i)\bifo (index|survey|business)\b",
        r"(?i)\bzew (index|survey|indicator)\b",
    ],
    MacroTheme.credit_conditions: [
        r"(?i)\bcredit spread\b",
        r"(?i)\bhigh.?yield\b",
        r"(?i)\bcorporate bond\b",
        r"(?i)\bdefault rate\b",
        r"(?i)\blending\b",
    ],
    MacroTheme.yield_curve: [
        r"(?i)\byield curve\b",
        r"(?i)\binversion\b",
        r"(?i)\btreasury yield\b",
        r"(?i)\b(2|10|30).?year\b.*\byield\b",
        r"(?i)\bterm premium\b",
        r"(?i)\bbund yield\b",
        r"(?i)\bgilt yield\b",
        r"(?i)\boat yield\b",
    ],
    MacroTheme.sector_rotation: [
        r"(?i)\bsector rotation\b",
        r"(?i)\bcyclical\b",
        r"(?i)\bdefensive\b",
        r"(?i)\bsector (performance|allocation)\b",
    ],
    MacroTheme.volatility_risk: [
        r"(?i)\bvix\b",
        r"(?i)\bvolatility\b",
        r"(?i)\brisk.?off\b",
        r"(?i)\bfear (index|gauge)\b",
    ],
    MacroTheme.commodity_inflation: [
        r"(?i)\bcpi\b",
        r"(?i)\binflation\b",
        r"(?i)\bcommodit(y|ies)\b",
        r"(?i)\boil price\b",
        r"(?i)\bgold price\b",
        r"(?i)\bconsumer price\b",
    ],
    MacroTheme.geographic_allocation: [
        r"(?i)\bemerging market\b",
        r"(?i)\bdollar (index|strength|weakness)\b",
        r"(?i)\bcapital flow\b",
        r"(?i)\bglobal allocation\b",
    ],
    MacroTheme.business_cycle: [
        r"(?i)\brecession\b",
        r"(?i)\bexpansion\b",
        r"(?i)\bbusiness cycle\b",
        r"(?i)\bsoft landing\b",
        r"(?i)\bhard landing\b",
        r"(?i)\bslowdown\b",
    ],
}

# Pre-compile patterns for performance.
_COMPILED_KEYWORDS: dict[MacroTheme, list[re.Pattern[str]]] = {
    theme: [re.compile(pat) for pat in patterns]
    for theme, patterns in _THEME_KEYWORDS.items()
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _classify_themes(
    text: str,
    seed_themes: list[MacroTheme],
) -> list[MacroTheme]:
    """Return deduplicated themes from seed list + keyword matches."""
    themes = set(seed_themes)
    for theme, patterns in _COMPILED_KEYWORDS.items():
        for pat in patterns:
            if pat.search(text):
                themes.add(theme)
                break
    return sorted(themes, key=lambda t: t.value)


def _make_news_id(title: str, link: str) -> str:
    """Deterministic ID from title + link."""
    raw = f"{title}|{link}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class MacroNewsFetcher:
    """Fetches macro-themed news from yfinance tickers and search queries.

    Parameters
    ----------
    yf_client:
        YFinance client used for ticker-based news fetching.
    search_client:
        Search client used for query-based news fetching.
    scraper:
        Optional article scraper for full content retrieval.
    """

    yf_client: YFinanceClientProtocol
    search_client: SearchClientProtocol
    scraper: ArticleScraperProtocol | None = None
    _seen_titles: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        from .client import NewsClient

        self._news_client = NewsClient(
            yf_client=self.yf_client,
            scraper=self.scraper,
        )

    def fetch_all(
        self,
        max_articles: int = 30,
        fetch_full_content: bool = False,
        max_per_ticker: int = 3,
    ) -> list[dict[str, Any]]:
        """Fetch macro news from all configured tickers and search queries.

        Returns a list of article dicts sorted by publish time (newest first),
        deduplicated by title, classified by theme, and curated for signal
        density:

        * **Per-ticker cap** (``max_per_ticker``, default 3) prevents a
          single ticker from dominating the budget.
        * **Search-query priority** — search articles are included first
          because they target specific macro narratives.
        * **Theme diversity** — after search + capped ticker selection,
          remaining budget is filled by picking the most-recent article
          from any under-represented theme.
        """
        self._seen_titles.clear()
        # ticker -> list of articles from that ticker
        ticker_buckets: dict[str, list[dict[str, Any]]] = {}
        search_articles: list[dict[str, Any]] = []

        # 1. Ticker-based news (collected per-ticker for capping)
        for ticker, seed_themes in MACRO_TICKERS.items():
            try:
                news = self._news_client.fetch(ticker)
                if not news:
                    continue
                bucket: list[dict[str, Any]] = []
                for raw_article in news:
                    article = self._process_article(
                        raw_article,
                        seed_themes=seed_themes,
                        source_ticker=ticker,
                        source_query=None,
                        fetch_full_content=fetch_full_content,
                    )
                    if article is not None:
                        bucket.append(article)
                if bucket:
                    # Sort each bucket by recency, keep top N
                    bucket.sort(
                        key=lambda a: a.get("publish_time") or "",
                        reverse=True,
                    )
                    ticker_buckets[ticker] = bucket[:max_per_ticker]
            except Exception:
                logger.warning(
                    "Failed to fetch ticker news for %s", ticker, exc_info=True,
                )

        # 2. Search-query-based news
        for query, seed_themes in MACRO_SEARCH_QUERIES.items():
            try:
                result = self.search_client.search(query, max_results=8)
                if result is None:
                    continue
                for raw_article in result.get("news", []):
                    article = self._process_article(
                        raw_article,
                        seed_themes=seed_themes,
                        source_ticker=None,
                        source_query=query,
                        fetch_full_content=fetch_full_content,
                    )
                    if article is not None:
                        search_articles.append(article)
            except Exception:
                logger.warning("Failed search news for '%s'", query, exc_info=True)

        # 3. Merge: search-first, then capped ticker articles
        search_articles.sort(
            key=lambda a: a.get("publish_time") or "", reverse=True,
        )

        capped_ticker: list[dict[str, Any]] = []
        for bucket in ticker_buckets.values():
            capped_ticker.extend(bucket)
        capped_ticker.sort(
            key=lambda a: a.get("publish_time") or "", reverse=True,
        )

        # Give search articles priority (at least 40 % of budget)
        search_budget = max(max_articles * 4 // 10, min(len(search_articles), 10))
        ticker_budget = max_articles - min(search_budget, len(search_articles))

        selected = search_articles[:search_budget] + capped_ticker[:ticker_budget]

        # 4. Theme-diversity pass: if any theme is missing, pull its
        #    best article from the remaining pool.
        selected_ids = {a["news_id"] for a in selected}
        covered_themes: set[str] = set()
        for a in selected:
            for t in (a.get("themes") or "").split(","):
                if t:
                    covered_themes.add(t)

        all_themes = {t.value for t in MacroTheme}
        missing_themes = all_themes - covered_themes

        if missing_themes:
            remaining = [
                a for a in (capped_ticker + search_articles)
                if a["news_id"] not in selected_ids
            ]
            for article in remaining:
                article_themes = set(
                    (article.get("themes") or "").split(","),
                )
                if article_themes & missing_themes:
                    selected.append(article)
                    selected_ids.add(article["news_id"])
                    covered_themes |= article_themes
                    missing_themes -= article_themes
                if not missing_themes:
                    break

        selected.sort(
            key=lambda a: a.get("publish_time") or "", reverse=True,
        )
        return selected[:max_articles]

    # -- private helpers ----------------------------------------------------

    def _process_article(
        self,
        raw: dict[str, Any],
        *,
        seed_themes: list[MacroTheme],
        source_ticker: str | None,
        source_query: str | None,
        fetch_full_content: bool,
    ) -> dict[str, Any] | None:
        content = raw.get("content", raw)

        title = content.get("title", raw.get("title", ""))
        if not title:
            return None

        pub_time = content.get("pubDate", content.get("providerPublishTime", "N/A"))
        if not _is_article_recent(pub_time):
            return None

        if title in self._seen_titles:
            return None
        self._seen_titles.add(title)

        publisher = (
            content.get("provider", {}).get("displayName", "")
            if isinstance(content.get("provider"), dict)
            else content.get("publisher", "")
        )

        if publisher not in TRUSTED_PUBLISHERS:
            return None

        link = (
            content.get("canonicalUrl", {}).get("url", "")
            if isinstance(content.get("canonicalUrl"), dict)
            else content.get("link", "")
        )

        pub_date = _parse_article_date(pub_time)
        pub_time_iso = pub_date.isoformat() if pub_date else None

        # Snippet from summary or first 300 chars of content
        snippet = content.get("summary", "")
        if not snippet:
            snippet = content.get("description", "")
        if snippet and len(snippet) > 500:
            snippet = snippet[:497] + "..."

        # Classify themes using title + snippet
        classify_text = f"{title} {snippet}"
        themes = _classify_themes(classify_text, seed_themes)
        themes_str = ",".join(t.value for t in themes)

        news_id = _make_news_id(title, link)

        article: dict[str, Any] = {
            "news_id": news_id,
            "title": title,
            "publisher": publisher,
            "link": link,
            "publish_time": pub_time_iso,
            "source_ticker": source_ticker,
            "source_query": source_query,
            "themes": themes_str,
            "snippet": snippet,
        }

        if fetch_full_content and self.scraper is not None:
            if not link:
                return None
            content_result = self.scraper.fetch(link)
            if (
                content_result["success"]
                and content_result["content"]
                and len(content_result["content"]) >= 200
            ):
                article["full_content"] = content_result["content"]
            else:
                return None

        return article
