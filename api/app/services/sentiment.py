"""News sentiment pipeline for Idzorek alpha (view confidence) adjustment.

Architecture:
  1. Fetch stored news headlines from the DB for a given ticker.
  2. Score each headline via the BAML ``ScoreNewsSentiment`` LLM function.
  3. Compute an exponentially decayed sentiment signal (recent news weights more).
  4. Adjust Idzorek alpha_k: aligned sentiment boosts confidence; conflicting
     sentiment reduces it.  Result is always clamped to (0.01, 0.99).

Formula
-------
sentiment_signal = Σ_t  s_t · exp(-λ · (T - t))   where λ = ln(2) / half_life

α_k_adjusted = α_k_base · (1 + |sentiment_signal| · alignment_weight)
alignment_weight = +1 if sentiment and view direction agree, −1 if they conflict
Result clamped to (0.01, 0.99).
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone

import pandas as pd
from baml_client import b
from baml_client.types import NewsArticle
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.universe import Instrument
from app.models.yfinance_data import TickerNews

logger = logging.getLogger(__name__)

# Bounds on the adjusted alpha
_ALPHA_MIN: float = 0.01
_ALPHA_MAX: float = 0.99


# ---------------------------------------------------------------------------
# Core math helpers (pure, deterministic, no side effects)
# ---------------------------------------------------------------------------


def compute_sentiment_signal(
    sentiment_series: pd.Series,
    half_life_days: float = 5.0,
) -> float:
    """Exponentially decayed sum of sentiment scores.

    Args:
        sentiment_series: Sentiment scores indexed by publish datetime.
            Scores should be in [-1, 1].  Timezone-aware or naive datetimes
            are both accepted (naive treated as UTC).
        half_life_days: Half-life of the decay in calendar days.  News
            from ``half_life_days`` ago contributes half as much as today's
            news.  Must be positive.

    Returns:
        Scalar float.  Positive = net bullish, negative = net bearish,
        zero = no signal or empty series.
    """
    if sentiment_series.empty:
        return 0.0

    if half_life_days <= 0:
        raise ValueError(f"half_life_days must be positive, got {half_life_days}")

    lam = math.log(2.0) / half_life_days
    now = datetime.now(timezone.utc)

    total = 0.0
    for dt, score in sentiment_series.items():
        if isinstance(dt, datetime):
            ts = dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
        else:
            # Fallback for non-datetime index entries
            ts = now
        age_days = max(0.0, (now - ts).total_seconds() / 86_400.0)
        weight = math.exp(-lam * age_days)
        total += float(score) * weight

    return total


def adjust_idzorek_alpha(
    alpha_base: float,
    sentiment_signal: float,
    view_direction: int,
) -> float:
    """Adjust Idzorek alpha_k using a sentiment-view alignment signal.

    Formula::

        alignment_weight = +1 if signal direction matches view direction, -1 otherwise
        α_adjusted = α_base · (1 + |signal| · alignment_weight)
        result clamped to (0.01, 0.99)

    Args:
        alpha_base: Starting Idzorek confidence in (0, 1).
        sentiment_signal: Exponentially decayed sentiment sum (signed).
            Positive = net bullish news; negative = net bearish news.
            Zero → alpha returned unchanged (after clamping).
        view_direction: +1 (bullish view) or -1 (bearish view).

    Returns:
        Adjusted alpha clamped to (0.01, 0.99).
    """
    if sentiment_signal == 0.0:
        return max(_ALPHA_MIN, min(_ALPHA_MAX, alpha_base))

    signals_agree = (sentiment_signal > 0) == (view_direction > 0)
    alignment_weight = 1.0 if signals_agree else -1.0

    adjusted = alpha_base * (1.0 + abs(sentiment_signal) * alignment_weight)
    return max(_ALPHA_MIN, min(_ALPHA_MAX, adjusted))


# ---------------------------------------------------------------------------
# DB fetch + LLM scoring
# ---------------------------------------------------------------------------


def _get_instrument_id(session: Session, ticker: str) -> object | None:
    row = session.execute(
        select(Instrument.id).where(Instrument.ticker == ticker)
    ).scalar_one_or_none()
    return row


def fetch_news_sentiment(
    session: Session,
    ticker: str,
    lookback_days: int = 30,
) -> pd.Series:
    """Fetch news from the DB for *ticker*, score via BAML, return a dated Series.

    Args:
        session: Active SQLAlchemy session.
        ticker: Asset ticker (e.g. ``"AAPL"``).
        lookback_days: How many calendar days of news to include.

    Returns:
        ``pd.Series`` with timezone-aware UTC datetimes as index and
        BAML-scored sentiment values in [-1, 1] as values.  Returns an
        empty Series if the ticker is unknown or has no stored news.
    """
    instrument_id = _get_instrument_id(session, ticker)
    if instrument_id is None:
        logger.warning(
            "ticker %s not found in instruments table — no sentiment", ticker
        )
        return pd.Series(dtype=float)

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    rows = (
        session.execute(
            select(TickerNews)
            .where(
                TickerNews.instrument_id == instrument_id,
                TickerNews.title.isnot(None),
                TickerNews.publish_time >= cutoff,
            )
            .order_by(TickerNews.publish_time.asc())
        )
        .scalars()
        .all()
    )

    if not rows:
        logger.debug("No news found for %s in the last %d days", ticker, lookback_days)
        return pd.Series(dtype=float)

    articles = [NewsArticle(title=str(row.title)) for row in rows]

    try:
        result = b.ScoreNewsSentiment(ticker=ticker, articles=articles)
    except Exception as exc:
        logger.error("BAML ScoreNewsSentiment failed for %s: %s", ticker, exc)
        return pd.Series(dtype=float)

    scores = result.scores
    n_articles = len(rows)
    if len(scores) != n_articles:
        logger.warning(
            "BAML returned %d scores for %d articles (%s) — truncating/padding",
            len(scores),
            n_articles,
            ticker,
        )
        # Align lengths: pad with 0 or truncate
        if len(scores) < n_articles:
            scores = scores + [0.0] * (n_articles - len(scores))
        else:
            scores = scores[:n_articles]

    # Build timezone-aware datetimes
    dates = []
    for row in rows:
        dt = row.publish_time
        if dt is None:
            dt = datetime.now(timezone.utc)
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dates.append(dt)

    return pd.Series(scores, index=pd.DatetimeIndex(dates), dtype=float)
