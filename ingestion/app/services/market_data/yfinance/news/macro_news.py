"""Macro-themed news fetching from yfinance ticker feeds and search queries.

Aggregates articles from macro-relevant tickers (treasuries, VIX, commodities,
sector ETFs) and targeted search queries (Fed rate decisions, ISM PMI, etc.),
classifies them by ``MacroTheme``, and deduplicates by title.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.services._shared import ProgressCallback, _noop

from ..protocols import (
    ArticleScraperProtocol,
    SearchClientProtocol,
    YFinanceClientProtocol,
)
from ._links import content_link, extract_article_link
from .aggregator import _is_article_recent, _parse_article_date

logger = logging.getLogger(__name__)

# Shorter scrapes are paywall or cookie-banner stubs, not article bodies.
_MIN_FULL_CONTENT_CHARS = 200


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

#
# NOTE: yfinance ``yf.Search`` uses strict entity/phrase matching, not
# natural-language concept search.  Multi-word descriptive queries like
# ``"eurozone economy recession growth"`` silently return zero results.
# Every query below has been validated against live yfinance to return
# at least one recent, trusted article.  Rewrites must be revalidated
# against live yfinance before shipping.
MACRO_SEARCH_QUERIES: dict[str, list[MacroTheme]] = {
    # --- USA ---
    "Federal Reserve interest rate decision": [
        MacroTheme.monetary_policy,
    ],
    "ISM manufacturing index": [
        MacroTheme.growth_indicators,
        MacroTheme.business_cycle,
    ],
    "10-year Treasury yield": [
        MacroTheme.yield_curve,
    ],
    "high yield bond market": [
        MacroTheme.credit_conditions,
    ],
    "S&P 500 sector rotation": [
        MacroTheme.sector_rotation,
    ],
    "CPI inflation report": [
        MacroTheme.commodity_inflation,
        MacroTheme.monetary_policy,
    ],
    "MSCI emerging markets": [
        MacroTheme.geographic_allocation,
    ],
    "US jobs report": [
        MacroTheme.business_cycle,
        MacroTheme.growth_indicators,
    ],
    # --- UK ---
    "Bank of England interest rate": [
        MacroTheme.monetary_policy,
    ],
    "Britain inflation": [
        MacroTheme.growth_indicators,
        MacroTheme.commodity_inflation,
    ],
    # --- Europe / ECB ---
    # NOTE: the only eurozone query that yfinance search actually returns
    # content for.  Every ``"eurozone ..."`` phrasing tested returned zero.
    "European Central Bank rate": [
        MacroTheme.monetary_policy,
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


_ALL_THEMES = frozenset(theme.value for theme in MacroTheme)

# Search queries return at most this many articles each.
_SEARCH_RESULTS_PER_QUERY = 8


def _publish_time(article: dict[str, Any]) -> str:
    """Sort key for newest-first order; an article without a time sorts last."""
    return article.get("publish_time") or ""


def _themes_of(article: dict[str, Any]) -> set[str]:
    """Return the theme values an article's ``themes`` field lists."""
    return set((article.get("themes") or "").split(","))


def _search_cap(max_articles: int) -> int:
    """Return how many search articles a budget of ``max_articles`` takes first."""
    # Search articles get priority: 40 % of the budget, or 10 if that is
    # more. Clamping to max_articles matters for a small budget, where an
    # unclamped take would leave a negative ticker budget and a negative
    # slice bound keeps nearly every ticker article instead of none.
    return min(max(max_articles * 4 // 10, 10), max_articles)


def _fill_missing_themes(
    selected: list[dict[str, Any]], pool: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Add one article for each theme the selection lacks, in pool order.

    Args:
        selected: The articles already chosen.
        pool: Every candidate, in the order the pass tries them; the chosen
            ones are skipped.

    Returns:
        ``selected`` followed by each pool article that covers a theme still
        missing when it is reached. The result can exceed the budget, so the
        caller cuts it again.
    """
    chosen = {article["news_id"] for article in selected}
    missing = set(_ALL_THEMES)
    for article in selected:
        missing -= _themes_of(article)
    filled = list(selected)
    for article in (a for a in pool if a["news_id"] not in chosen):
        if not missing:
            break
        themes = _themes_of(article)
        if themes & missing:
            filled.append(article)
            missing -= themes
    return filled


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class MacroNewsUnavailableError(RuntimeError):
    """Raised when a ``fetch_all`` run obtained no article from any source.

    An empty result is normal when every article is filtered out, but zero
    raw articles across all tickers and searches means the upstream feed is
    down, which callers must not record as a successful empty run.
    """


@dataclass
class _SourceTally:
    """How the sources of one ``fetch_all`` sweep answered.

    A source that delivered and then failed while its articles were processed
    counts as both delivering and failed.
    """

    delivering: int = 0
    failed: int = 0
    empty: int = 0

    def admit(self, articles: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        """Count one source's answer and return the articles left to process.

        Args:
            articles: The source's answer; ``None`` means the source failed.

        Returns:
            ``articles`` when it holds any, else ``[]``.
        """
        if articles is None:
            self.failed += 1
            return []
        if not articles:
            self.empty += 1
            return []
        self.delivering += 1
        return articles

    def raise_if_nothing_delivered(self) -> None:
        """Raise when no source answered with a single article.

        Raises:
            MacroNewsUnavailableError: Every source failed or was empty.
        """
        if self.delivering == 0:
            raise MacroNewsUnavailableError(
                "No article from any macro news source: "
                f"{self.failed} failed, {self.empty} empty "
                f"of {len(MACRO_TICKERS) + len(MACRO_SEARCH_QUERIES)} sources"
            )


@dataclass
class MacroNewsFetcher:
    """Fetches macro-themed news from yfinance tickers and search queries.

    Attributes:
        yf_client: YFinanceClientProtocol used for ticker-based news fetching.
        search_client: SearchClientProtocol used for query-based news fetching.
        scraper: Optional article scraper for full content retrieval.
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
        on_progress: ProgressCallback = _noop,
    ) -> list[dict[str, Any]]:
        """Fetch macro news from all configured tickers and search queries.

        Curates articles for signal density with three passes:

        * **Per-ticker cap** (``max_per_ticker``, default 3) prevents a
          single ticker from dominating the budget.
        * **Search-query priority** — search articles are included first
          because they target specific macro narratives.
        * **Theme diversity** — after search + capped ticker selection,
          remaining budget is filled by picking the most-recent article
          from any under-represented theme.

        Args:
            max_articles: Overall cap on the returned list.
            fetch_full_content: When ``True``, scrape the full article body
                via the configured scraper and drop articles whose body cannot
                be scraped, so caps and budgets only count scrapable articles.
            max_per_ticker: Per-ticker article cap to prevent domination.
            on_progress: Receives the number of sources up front and a report
                before each one is fetched, so the job's stall watch sees the
                sweep advance through a long rate-limited run.

        Returns:
            List of article dicts sorted by publish time (newest first),
            deduplicated by title and classified by macro theme.

        Raises:
            MacroNewsUnavailableError: When no ticker feed and no search query
                returned a single raw article (every source failed or was
                empty), as opposed to articles that were merely filtered out.
        """
        self._seen_titles.clear()
        tally = _SourceTally()
        on_progress(total=len(MACRO_TICKERS) + len(MACRO_SEARCH_QUERIES))
        ticker_articles = self._collect_ticker_news(
            tally,
            on_progress,
            max_per_ticker=max_per_ticker,
            fetch_full_content=fetch_full_content,
        )
        search_articles = self._collect_search_news(tally, on_progress)
        tally.raise_if_nothing_delivered()
        selected, pool = self._merge_within_budget(
            search_articles,
            ticker_articles,
            max_articles,
            fetch_full_content=fetch_full_content,
        )
        selected = _fill_missing_themes(selected, pool)
        selected.sort(key=_publish_time, reverse=True)
        return selected[:max_articles]

    # -- private helpers ----------------------------------------------------

    def _collect_ticker_news(
        self,
        tally: _SourceTally,
        on_progress: ProgressCallback,
        *,
        max_per_ticker: int,
        fetch_full_content: bool,
    ) -> list[dict[str, Any]]:
        """Collect each ticker feed's newest unique articles.

        Runs before the searches: the titles kept here are registered, so the
        search pass drops later copies of them.

        Args:
            tally: Records how each feed answered.
            on_progress: Told about each ticker before its feed is fetched.
            max_per_ticker: Most articles kept from one feed.
            fetch_full_content: Scrape each kept article's body and skip the
                articles that have none.

        Returns:
            The kept articles, feed by feed in ``MACRO_TICKERS`` order. A feed
            that fails while its articles are processed contributes none.
        """
        kept: list[dict[str, Any]] = []
        for position, (ticker, seed_themes) in enumerate(
            MACRO_TICKERS.items(), start=1
        ):
            on_progress(current=position, current_ticker=ticker)
            try:
                news = tally.admit(self._news_client.fetch(ticker))
                bucket = list(
                    self._processed(news, seed_themes, ticker=ticker, query=None)
                )
                bucket.sort(key=_publish_time, reverse=True)
                kept.extend(
                    self._keep_unique(
                        bucket, max_per_ticker, fetch_full_content=fetch_full_content
                    )
                )
            except Exception:
                tally.failed += 1
                logger.warning(
                    "Failed to fetch ticker news for %s", ticker, exc_info=True
                )
        return kept

    def _collect_search_news(
        self, tally: _SourceTally, on_progress: ProgressCallback
    ) -> list[dict[str, Any]]:
        """Collect the articles of every search query.

        Args:
            tally: Records how each query answered.
            on_progress: Told about each query before it is searched, numbered
                after the tickers.

        Returns:
            The processed articles, query by query in ``MACRO_SEARCH_QUERIES``
            order. A query that fails while its articles are processed keeps
            the ones processed before the failure.
        """
        found: list[dict[str, Any]] = []
        queries = MACRO_SEARCH_QUERIES.items()
        for position, (query, seed_themes) in enumerate(
            queries, start=len(MACRO_TICKERS) + 1
        ):
            on_progress(current=position, current_ticker=query)
            try:
                result = self.search_client.search(
                    query, max_results=_SEARCH_RESULTS_PER_QUERY
                )
                news = tally.admit(
                    None if result is None else (result.get("news") or [])
                )
                # Appended one by one so a later failing item leaves the earlier
                # ones in place.
                for article in self._processed(
                    news, seed_themes, ticker=None, query=query
                ):
                    found.append(article)
            except Exception:
                tally.failed += 1
                logger.warning("Failed search news for '%s'", query, exc_info=True)
        return found

    def _processed(
        self,
        raw_articles: list[dict[str, Any]],
        seed_themes: list[MacroTheme],
        *,
        ticker: str | None,
        query: str | None,
    ) -> Iterator[dict[str, Any]]:
        """Yield each raw article that passes ``_process_article``, in feed order."""
        for raw in raw_articles:
            article = self._process_article(
                raw, seed_themes=seed_themes, source_ticker=ticker, source_query=query
            )
            if article is not None:
                yield article

    def _merge_within_budget(
        self,
        search_articles: list[dict[str, Any]],
        ticker_articles: list[dict[str, Any]],
        max_articles: int,
        *,
        fetch_full_content: bool,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Take search articles first, then the newest ticker articles.

        Args:
            search_articles: Processed search articles.
            ticker_articles: The capped articles of every ticker feed.
            max_articles: The overall budget.
            fetch_full_content: Scrape the search articles' bodies, at most
                as many as the search share of the budget, and skip the
                articles that have none.

        Returns:
            The selection, at most ``max_articles`` long, and the pool the
            theme pass draws from: every ticker article, then every search
            article not shadowed by a title kept before, each newest first.
        """
        search_cap = _search_cap(max_articles)
        search_articles = sorted(search_articles, key=_publish_time, reverse=True)
        # Scraping is the cost, so stop at the cap; unscraped candidates are
        # free to keep, which leaves them to the theme-diversity pass.
        search_articles = self._keep_unique(
            search_articles,
            search_cap if fetch_full_content else None,
            fetch_full_content=fetch_full_content,
        )
        ticker_articles = sorted(ticker_articles, key=_publish_time, reverse=True)
        search_take = min(len(search_articles), search_cap)
        ticker_budget = max(0, max_articles - search_take)
        selected = search_articles[:search_take] + ticker_articles[:ticker_budget]
        return selected, ticker_articles + search_articles

    def _keep_unique(
        self,
        candidates: list[dict[str, Any]],
        limit: int | None,
        *,
        fetch_full_content: bool,
    ) -> list[dict[str, Any]]:
        """Keep up to ``limit`` candidates whose titles were not kept before.

        Titles are registered here and nowhere else, so an article rejected by
        a filter or cut by the per-ticker cap never shadows a later copy of the
        same story from another feed or from search.

        With ``fetch_full_content`` the body is scraped here, in recency order,
        and an article without a usable body is skipped. Scraping only until
        ``limit`` is reached keeps unscrapable articles from using up the cap
        and spares the fetches for candidates beyond it. That only saves work
        when ``limit`` is no larger than what the later selection can take.
        """
        kept: list[dict[str, Any]] = []
        for article in candidates:
            if limit is not None and len(kept) >= limit:
                break
            if article["title"] in self._seen_titles:
                continue
            if fetch_full_content and not self._attach_full_content(article):
                continue
            self._seen_titles.add(article["title"])
            kept.append(article)
        return kept

    def _attach_full_content(self, article: dict[str, Any]) -> bool:
        """Scrape ``article`` and store the body under ``full_content``.

        Returns:
            ``True`` when the article now carries a body of at least
            ``_MIN_FULL_CONTENT_CHARS`` characters, or when no scraper is
            configured and so no article could ever get one. ``False`` when
            the page could not be scraped or yielded only a stub.
        """
        if self.scraper is None:
            return True
        link = article.get("link")
        if not link:
            return False
        try:
            result = self.scraper.fetch(link)
        except Exception:
            logger.warning("Full-content scrape failed for %s", link, exc_info=True)
            return False
        text = result["content"] if result["success"] else None
        if not text or len(text) < _MIN_FULL_CONTENT_CHARS:
            return False
        article["full_content"] = text
        return True

    def _process_article(
        self,
        raw: dict[str, Any],
        *,
        seed_themes: list[MacroTheme],
        source_ticker: str | None,
        source_query: str | None,
    ) -> dict[str, Any] | None:
        content = raw.get("content", raw)

        title = content.get("title", raw.get("title", ""))
        if not title:
            return None

        pub_time = content.get("pubDate", content.get("providerPublishTime", "N/A"))
        if not _is_article_recent(pub_time):
            return None

        publisher = (
            content.get("provider", {}).get("displayName", "")
            if isinstance(content.get("provider"), dict)
            else content.get("publisher", "")
        )

        if publisher not in TRUSTED_PUBLISHERS:
            return None

        if title in self._seen_titles:
            return None

        link = extract_article_link(raw, content_link)

        pub_date = _parse_article_date(pub_time)
        pub_time_iso = pub_date.isoformat() if pub_date else None

        snippet = content.get("summary", "")
        if not snippet:
            snippet = content.get("description", "")
        if snippet and len(snippet) > 500:
            snippet = snippet[:497] + "..."

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

        return article
