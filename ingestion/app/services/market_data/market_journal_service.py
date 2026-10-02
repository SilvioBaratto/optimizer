"""Service building the global daily macro/market/news digest (``market_journal``).

Reads already-ingested FRED, market-summary, and macro-news rows and assembles a
single deterministic digest row per trading day.  No ``optimizer`` and no agent
stack; the narrative is a pure template (never model-generated), so every row is
reproducible for audit.  The caller owns nothing: the function opens its own
session via ``database_manager.get_session`` and commits, mirroring the other
bulk services.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

from portopt_db.repositories.macro.macro_regime_repository import MacroRegimeRepository
from portopt_db.repositories.market_data.market_journal_repository import (
    MarketJournalRepository,
)
from portopt_db.repositories.market_data.market_summary_repository import (
    MarketSummaryRepository,
)

from app.schemas.market_data.market_journal import MarketJournalBuildRequest
from app.services._shared import ProgressCallback, _noop
from app.services.macro.scrapers.fred_scraper import FRED_SERIES

logger = logging.getLogger(__name__)

# When a request leaves ``fred_series`` empty, summarise the full ingested FRED
# catalog; the digest skips any series without observations.
_DEFAULT_FRED_SERIES: tuple[str, ...] = tuple(FRED_SERIES)


def _f(value: Any) -> float | None:
    """Coerce a possibly-``Decimal`` numeric to ``float`` for JSON storage.

    ``market_summaries`` columns read back as ``Decimal``, which the SQLite JSON
    variant cannot serialise; coercing here keeps the sections portable.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_macro_deltas(
    repo: MacroRegimeRepository, series_ids: tuple[str, ...]
) -> dict[str, Any]:
    """Return latest-vs-prior deltas for each FRED series that has data."""
    out: dict[str, Any] = {}
    for sid in series_ids:
        obs = list(repo.get_fred_observations(sid, limit=2))
        if not obs:
            continue
        latest = obs[-1]
        prior = obs[-2] if len(obs) >= 2 else None
        latest_val = _f(latest.value)
        prior_val = _f(prior.value) if prior is not None else None
        delta = (
            round(latest_val - prior_val, 6)
            if latest_val is not None and prior_val is not None
            else None
        )
        out[sid] = {
            "latest": latest_val,
            "date": latest.date.isoformat(),
            "prior": prior_val,
            "delta": delta,
        }
    return out


def _build_market_moves(
    repo: MarketSummaryRepository, markets: tuple[str, ...], as_of: datetime.date
) -> tuple[dict[str, Any], int]:
    """Return per-market symbol moves plus the total summary-row count.

    Prefers the snapshot dated ``as_of``; when none exists it falls back to the
    market's freshest snapshot so a weekday digest still reflects the weekend
    sweep.
    """
    out: dict[str, Any] = {}
    total = 0
    for market in markets:
        rows = list(repo.get_summaries(market, as_of))
        if not rows:
            latest = repo.get_latest_as_of(market)
            if latest is not None:
                rows = list(repo.get_summaries(market, latest))
        if not rows:
            continue
        out[market] = {
            row.symbol: {
                "price": _f(row.price),
                "change": _f(row.change),
                "change_percent": _f(row.change_percent),
            }
            for row in rows
        }
        total += len(rows)
    return out, total


def _build_news_themes(
    repo: MacroRegimeRepository, as_of: datetime.date, limit: int
) -> tuple[dict[str, Any], int]:
    """Return aggregated themes + headlines for recent macro news.

    Only articles published on or before ``as_of`` are considered, so a
    backfilled digest never leaks future headlines.
    """
    end = datetime.datetime.combine(as_of, datetime.time.max)
    articles = list(repo.get_macro_news(end_date=end, limit=limit))
    if not articles:
        return {}, 0

    theme_counts: dict[str, int] = {}
    headlines: list[str] = []
    for article in articles:
        raw = article.themes
        if raw:
            for theme in raw.split(","):
                name = theme.strip()
                if name:
                    theme_counts[name] = theme_counts.get(name, 0) + 1
        if article.title:
            headlines.append(article.title)

    section: dict[str, Any] = {}
    if theme_counts:
        section["themes"] = dict(sorted(theme_counts.items()))
    if headlines:
        section["headlines"] = headlines
    return section, len(articles)


def _build_narrative(
    as_of: datetime.date,
    region: str,
    macro_deltas: dict[str, Any],
    market_moves: dict[str, Any],
    news_themes: dict[str, Any],
) -> str:
    """Assemble a deterministic one-line narrative from the JSON sections.

    Pure: the output depends only on the arguments, so re-running the builder on
    unchanged data reproduces the string byte-for-byte (keys are sorted to pin
    ordering independent of dict insertion order).
    """
    lines = [f"Market digest for {as_of.isoformat()} ({region})."]

    if macro_deltas:
        parts = []
        for sid in sorted(macro_deltas):
            entry = macro_deltas[sid]
            delta = entry.get("delta")
            latest = entry.get("latest")
            if delta is not None:
                parts.append(f"{sid} {latest} (delta {delta})")
            else:
                parts.append(f"{sid} {latest}")
        lines.append(f"Macro: {len(macro_deltas)} series — {', '.join(parts)}.")
    else:
        lines.append("Macro: no series available.")

    if market_moves:
        parts = [
            f"{market}: {', '.join(sorted(market_moves[market]))}"
            for market in sorted(market_moves)
        ]
        lines.append(f"Markets: {'; '.join(parts)}.")
    else:
        lines.append("Markets: no summaries available.")

    themes = news_themes.get("themes")
    headlines = news_themes.get("headlines")
    if themes:
        parts = [f"{name} ({count})" for name, count in sorted(themes.items())]
        lines.append(f"News themes: {', '.join(parts)}.")
    elif headlines:
        lines.append(f"News: {len(headlines)} recent headlines.")
    else:
        lines.append("News: no recent articles.")

    return " ".join(lines)


def run_build_market_journal(
    request: MarketJournalBuildRequest,
    *,
    on_progress: ProgressCallback = _noop,
) -> dict[str, Any]:
    """Build and upsert one ``market_journal`` row for the requested date.

    Opens its own session, reads the FRED / market-summary / macro-news sources,
    assembles the JSON sections plus a deterministic narrative, upserts the row,
    and commits.  Idempotent: re-running for the same ``as_of`` overwrites the
    row rather than duplicating it.

    Args:
        request: Build parameters (target date, region, series/markets, news
            limit).  A ``None`` ``as_of`` defaults to today.
        on_progress: Optional callback; invoked with ``total`` at start and
            ``status="completed"`` at the end.  A failure is *not* swallowed: the
            exception propagates so the scheduler's ``_run_step`` records the job
            as failed (it flips any non-``completed`` status to ``completed``, so
            a swallowed failure would be mislabelled a success).

    Returns:
        Result dict with ``as_of``, ``region``, ``source_counts``, and
        ``error_count`` keys.
    """
    from app.database import database_manager

    as_of = request.as_of or datetime.date.today()
    region = request.region or "US"
    series_ids = request.fred_series or _DEFAULT_FRED_SERIES

    on_progress(total=1)
    with database_manager.get_session() as session:
        macro_repo = MacroRegimeRepository(session)
        market_repo = MarketSummaryRepository(session)
        journal_repo = MarketJournalRepository(session)

        macro_deltas = _build_macro_deltas(macro_repo, series_ids)
        market_moves, market_rows = _build_market_moves(
            market_repo, request.markets, as_of
        )
        news_themes, news_articles = _build_news_themes(
            macro_repo, as_of, request.news_limit
        )
        source_counts = {
            "fred_series": len(macro_deltas),
            "market_rows": market_rows,
            "news_articles": news_articles,
        }
        narrative = _build_narrative(
            as_of, region, macro_deltas, market_moves, news_themes
        )
        journal_repo.upsert_journal(
            as_of,
            macro_deltas=macro_deltas,
            market_moves=market_moves,
            news_themes=news_themes,
            narrative=narrative,
            source_counts=source_counts,
            region=region,
        )
        session.commit()

    result_dict = {
        "as_of": as_of.isoformat(),
        "region": region,
        "source_counts": source_counts,
        "error_count": 0,
    }
    on_progress(
        status="completed",
        finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        errors=[],
        result=result_dict,
    )
    logger.info(
        "Market journal built for %s (region=%s, sources=%s)",
        as_of,
        region,
        source_counts,
    )
    return result_dict
