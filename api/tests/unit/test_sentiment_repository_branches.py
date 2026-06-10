"""Branch-coverage tests for SentimentRepository.

Covers the missed lines reported at 47%:
  43      get_instrument_id_by_ticker — None branch
  47      get_instrument_id_by_ticker — found branch
  57      get_recent_news — empty branch
  87-150  get_macro_news_fallback — all three cascading strategies:
            A) direct source_ticker match
            B) sector-ETF match via TickerProfile.sector
            C) title/content ilike text-search fallback

Uses real db_session (SAVEPOINT) only — no mocks.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models.macro.macro_regime import MacroNews
from app.models.market_data.yfinance_data import TickerNews, TickerProfile
from app.repositories.macro.sentiment_repository import SentimentRepository
from tests._fixtures._helpers import add_and_flush
from tests._fixtures.universe import seed_universe

# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2024, 6, 10, 12, 0, 0, tzinfo=timezone.utc)
_CUTOFF = datetime(2024, 6, 9, 0, 0, 0, tzinfo=timezone.utc)   # 1 day before _NOW
_BEFORE_CUTOFF = datetime(2024, 6, 8, 0, 0, 0, tzinfo=timezone.utc)


def _seed_instrument(session: Session, ticker: str):
    seed = seed_universe(session, ticker=ticker, exchange_name=f"EX-{ticker}")
    return seed.instrument


def _seed_profile(
    session: Session, instrument, sector: str
) -> TickerProfile:
    return add_and_flush(
        session,
        TickerProfile(
            instrument_id=instrument.id,
            symbol=instrument.ticker,
            sector=sector,
            country="United States",
            currency="USD",
        ),
    )


def _seed_ticker_news(
    session: Session,
    instrument,
    title: str,
    publish_time: datetime,
    news_uuid: str,
) -> TickerNews:
    return add_and_flush(
        session,
        TickerNews(
            instrument_id=instrument.id,
            news_uuid=news_uuid,
            title=title,
            publish_time=publish_time,
        ),
    )


def _seed_macro_news(
    session: Session,
    news_id: str,
    title: str,
    publish_time: datetime,
    source_ticker: str | None = None,
    full_content: str | None = None,
) -> MacroNews:
    return add_and_flush(
        session,
        MacroNews(
            news_id=news_id,
            title=title,
            publish_time=publish_time,
            source_ticker=source_ticker,
            full_content=full_content,
        ),
    )


# ---------------------------------------------------------------------------
# get_instrument_id_by_ticker
# ---------------------------------------------------------------------------


class TestGetInstrumentIdByTicker:
    def test_when_ticker_absent_returns_none(self, db_session: Session) -> None:
        repo = SentimentRepository(db_session)
        result = repo.get_instrument_id_by_ticker("NOTHERE")
        assert result is None

    def test_when_ticker_present_returns_uuid(self, db_session: Session) -> None:
        inst = _seed_instrument(db_session, "AAPL")
        repo = SentimentRepository(db_session)
        result = repo.get_instrument_id_by_ticker("AAPL")
        assert result == inst.id


# ---------------------------------------------------------------------------
# get_recent_news
# ---------------------------------------------------------------------------


class TestGetRecentNews:
    def test_when_no_rows_after_cutoff_returns_empty(
        self, db_session: Session
    ) -> None:
        inst = _seed_instrument(db_session, "MSFT")
        _seed_ticker_news(
            db_session, inst,
            title="Old news",
            publish_time=_BEFORE_CUTOFF,
            news_uuid="old-uuid-1",
        )
        repo = SentimentRepository(db_session)
        result = repo.get_recent_news(inst.id, _CUTOFF)
        assert list(result) == []

    def test_when_rows_after_cutoff_returns_them_ascending(
        self, db_session: Session
    ) -> None:
        inst = _seed_instrument(db_session, "GOOG")
        early = _seed_ticker_news(
            db_session, inst,
            title="Early news",
            publish_time=datetime(2024, 6, 9, 6, 0, 0, tzinfo=timezone.utc),
            news_uuid="uuid-early",
        )
        late = _seed_ticker_news(
            db_session, inst,
            title="Late news",
            publish_time=datetime(2024, 6, 9, 18, 0, 0, tzinfo=timezone.utc),
            news_uuid="uuid-late",
        )
        repo = SentimentRepository(db_session)
        result = list(repo.get_recent_news(inst.id, _CUTOFF))
        assert len(result) == 2
        assert result[0].id == early.id
        assert result[1].id == late.id

    def test_when_title_is_null_row_excluded(self, db_session: Session) -> None:
        inst = _seed_instrument(db_session, "NVDA")
        add_and_flush(
            db_session,
            TickerNews(
                instrument_id=inst.id,
                news_uuid="uuid-notitle",
                title=None,
                publish_time=_NOW,
            ),
        )
        repo = SentimentRepository(db_session)
        result = repo.get_recent_news(inst.id, _CUTOFF)
        assert list(result) == []


# ---------------------------------------------------------------------------
# get_macro_news_fallback — Strategy A: direct source_ticker match
# ---------------------------------------------------------------------------


class TestMacroNewsFallbackStrategyA:
    def test_when_direct_source_ticker_matches_returns_rows(
        self, db_session: Session
    ) -> None:
        _seed_instrument(db_session, "TSLA")
        news = _seed_macro_news(
            db_session,
            news_id="macro-a-1",
            title="Tesla direct news",
            publish_time=_NOW,
            source_ticker="TSLA",
        )
        repo = SentimentRepository(db_session)
        result = list(repo.get_macro_news_fallback("TSLA", _CUTOFF))
        assert len(result) == 1
        assert result[0].id == news.id

    def test_when_direct_match_exists_strategy_b_not_reached(
        self, db_session: Session
    ) -> None:
        """Strategy A short-circuits — sector ETF row is not returned."""
        inst = _seed_instrument(db_session, "INTC")
        _seed_profile(db_session, inst, sector="Technology")

        # Direct match
        direct = _seed_macro_news(
            db_session,
            news_id="macro-direct-1",
            title="INTC direct",
            publish_time=_NOW,
            source_ticker="INTC",
        )
        # Sector ETF row that should NOT be returned
        _seed_macro_news(
            db_session,
            news_id="macro-xlk-1",
            title="Tech sector news",
            publish_time=_NOW,
            source_ticker="XLK",
        )
        repo = SentimentRepository(db_session)
        result = list(repo.get_macro_news_fallback("INTC", _CUTOFF))
        assert len(result) == 1
        assert result[0].id == direct.id

    def test_when_source_ticker_matches_but_before_cutoff_excluded(
        self, db_session: Session
    ) -> None:
        _seed_instrument(db_session, "AMD")
        _seed_macro_news(
            db_session,
            news_id="macro-old-1",
            title="Old AMD news",
            publish_time=_BEFORE_CUTOFF,
            source_ticker="AMD",
        )
        repo = SentimentRepository(db_session)
        result = list(repo.get_macro_news_fallback("AMD", _CUTOFF))
        assert result == []


# ---------------------------------------------------------------------------
# get_macro_news_fallback — Strategy B: sector ETF match
# ---------------------------------------------------------------------------


class TestMacroNewsFallbackStrategyB:
    def test_when_no_direct_match_sector_etf_used(
        self, db_session: Session
    ) -> None:
        # XLK maps to "Technology" in _ETF_TO_SECTOR
        inst = _seed_instrument(db_session, "CRM")
        _seed_profile(db_session, inst, sector="Technology")

        xlk_news = _seed_macro_news(
            db_session,
            news_id="macro-xlk-crm",
            title="Tech ETF news",
            publish_time=_NOW,
            source_ticker="XLK",
        )
        repo = SentimentRepository(db_session)
        result = list(repo.get_macro_news_fallback("CRM", _CUTOFF))
        assert len(result) == 1
        assert result[0].id == xlk_news.id

    def test_when_sector_not_in_etf_map_falls_through_to_strategy_c(
        self, db_session: Session
    ) -> None:
        """A sector with no ETF mapping skips B and falls to C."""
        inst = _seed_instrument(db_session, "BRKB")
        _seed_profile(db_session, inst, sector="UnknownSector")

        # Add a row that only strategy C (ilike) would find
        text_match = _seed_macro_news(
            db_session,
            news_id="macro-text-brkb",
            title="BRKB makes headlines",
            publish_time=_NOW,
            source_ticker=None,
        )
        repo = SentimentRepository(db_session)
        result = list(repo.get_macro_news_fallback("BRKB", _CUTOFF))
        assert len(result) == 1
        assert result[0].id == text_match.id

    def test_when_no_profile_falls_through_to_strategy_c(
        self, db_session: Session
    ) -> None:
        """No TickerProfile → sector query returns None → skip B."""
        _seed_instrument(db_session, "HOOD")
        # No profile seeded intentionally
        content_match = _seed_macro_news(
            db_session,
            news_id="macro-content-hood",
            title="Some unrelated title",
            publish_time=_NOW,
            source_ticker=None,
            full_content="HOOD reports strong earnings",
        )
        repo = SentimentRepository(db_session)
        result = list(repo.get_macro_news_fallback("HOOD", _CUTOFF))
        assert len(result) == 1
        assert result[0].id == content_match.id

    def test_when_sector_etf_news_before_cutoff_excluded(
        self, db_session: Session
    ) -> None:
        inst = _seed_instrument(db_session, "NOW")
        _seed_profile(db_session, inst, sector="Technology")
        _seed_macro_news(
            db_session,
            news_id="macro-xlk-old",
            title="Old tech news",
            publish_time=_BEFORE_CUTOFF,
            source_ticker="XLK",
        )
        repo = SentimentRepository(db_session)
        result = list(repo.get_macro_news_fallback("NOW", _CUTOFF))
        assert result == []


# ---------------------------------------------------------------------------
# get_macro_news_fallback — Strategy C: title/content text search
# ---------------------------------------------------------------------------


class TestMacroNewsFallbackStrategyC:
    def test_when_ticker_in_title_returns_row(self, db_session: Session) -> None:
        _seed_instrument(db_session, "PLTR")
        # No direct match, no profile → falls to C
        title_match = _seed_macro_news(
            db_session,
            news_id="macro-title-pltr",
            title="PLTR expands government contracts",
            publish_time=_NOW,
            source_ticker=None,
        )
        repo = SentimentRepository(db_session)
        result = list(repo.get_macro_news_fallback("PLTR", _CUTOFF))
        assert len(result) == 1
        assert result[0].id == title_match.id

    def test_when_ticker_in_full_content_returns_row(
        self, db_session: Session
    ) -> None:
        _seed_instrument(db_session, "RBLX")
        content_match = _seed_macro_news(
            db_session,
            news_id="macro-content-rblx",
            title="Gaming sector update",
            publish_time=_NOW,
            source_ticker=None,
            full_content="RBLX reports user growth acceleration",
        )
        repo = SentimentRepository(db_session)
        result = list(repo.get_macro_news_fallback("RBLX", _CUTOFF))
        assert len(result) == 1
        assert result[0].id == content_match.id

    def test_when_no_strategy_matches_returns_empty(
        self, db_session: Session
    ) -> None:
        _seed_instrument(db_session, "ZZZZ")
        # Seed an unrelated macro news row
        _seed_macro_news(
            db_session,
            news_id="macro-unrelated",
            title="Completely unrelated article",
            publish_time=_NOW,
            source_ticker="OTHER",
        )
        repo = SentimentRepository(db_session)
        result = list(repo.get_macro_news_fallback("ZZZZ", _CUTOFF))
        assert result == []

    def test_when_text_match_before_cutoff_excluded(
        self, db_session: Session
    ) -> None:
        _seed_instrument(db_session, "DDOG")
        _seed_macro_news(
            db_session,
            news_id="macro-old-ddog",
            title="DDOG old article",
            publish_time=_BEFORE_CUTOFF,
            source_ticker=None,
        )
        repo = SentimentRepository(db_session)
        result = list(repo.get_macro_news_fallback("DDOG", _CUTOFF))
        assert result == []
