"""Branch-coverage tests for SentimentRepository.

Tests the three cascading fallback strategies in get_macro_news_fallback:
  A) direct source_ticker match
  B) sector-ETF match via TickerProfile.sector
  C) title/content ilike text-search fallback

Uses real db_session (SAVEPOINT) only — no mocks.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from uuid import UUID

from _fixtures import add_and_flush, get_or_create_exchange, seed_universe
from sqlalchemy.orm import Session

from portopt_db.models.macro.macro_regime import MacroNews
from portopt_db.models.market_data.yfinance_data import TickerNews, TickerProfile
from portopt_db.models.universe.universe import Instrument
from portopt_db.repositories.macro.sentiment_repository import SentimentRepository

_NOW = datetime(2024, 6, 10, 12, 0, 0, tzinfo=timezone.utc)
_CUTOFF = datetime(2024, 6, 9, 0, 0, 0, tzinfo=timezone.utc)  # 1 day before _NOW
_BEFORE_CUTOFF = datetime(2024, 6, 8, 0, 0, 0, tzinfo=timezone.utc)


def _seed_instrument(session: Session, ticker: str):
    seed = seed_universe(session, ticker=ticker, exchange_name=f"EX-{ticker}")
    return seed.instrument


def _seed_profile(session: Session, instrument, sector: str) -> TickerProfile:
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


class TestGetInstrumentIdByTickerAfterVenueTransfer:
    def test_when_symbol_moved_venues_resolves_to_the_active_listing(
        self, db_session: Session
    ) -> None:
        old = seed_universe(db_session, ticker="XFER", exchange_name="XFER-NYSE")
        old.instrument.delisted_at = date(2024, 6, 1)
        new = seed_universe(db_session, ticker="XFER", exchange_name="XFER-NASDAQ")

        result = SentimentRepository(db_session).get_instrument_id_by_ticker("XFER")

        assert result == new.instrument.id

    def test_when_only_a_delisted_listing_exists_still_resolves_it(
        self, db_session: Session
    ) -> None:
        gone = seed_universe(db_session, ticker="GONE", exchange_name="GONE-NYSE")
        gone.instrument.delisted_at = date(2024, 6, 1)
        db_session.flush()

        result = SentimentRepository(db_session).get_instrument_id_by_ticker("GONE")

        assert result == gone.instrument.id


class TestMacroNewsFallbackAfterVenueTransfer:
    def test_when_both_listings_have_profiles_sector_comes_from_the_active_one(
        self, db_session: Session
    ) -> None:
        old = seed_universe(db_session, ticker="XFER", exchange_name="XFER-NYSE")
        old.instrument.delisted_at = date(2024, 6, 1)
        _seed_profile(db_session, old.instrument, sector="Utilities")
        new = seed_universe(db_session, ticker="XFER", exchange_name="XFER-NASDAQ")
        _seed_profile(db_session, new.instrument, sector="Technology")
        xlk_news = _seed_macro_news(
            db_session,
            news_id="macro-xlk-xfer",
            title="Tech ETF news",
            publish_time=_NOW,
            source_ticker="XLK",
        )
        _seed_macro_news(
            db_session,
            news_id="macro-xlu-xfer",
            title="Utilities ETF news",
            publish_time=_NOW,
            source_ticker="XLU",
        )

        result = list(
            SentimentRepository(db_session).get_macro_news_fallback("XFER", _CUTOFF)
        )

        assert [r.id for r in result] == [xlk_news.id]

    def test_when_only_a_delisted_listing_has_a_profile_its_sector_is_used(
        self, db_session: Session
    ) -> None:
        old = seed_universe(db_session, ticker="XFER", exchange_name="XFER-NYSE")
        old.instrument.delisted_at = date(2024, 6, 1)
        _seed_profile(db_session, old.instrument, sector="Utilities")
        seed_universe(db_session, ticker="XFER", exchange_name="XFER-NASDAQ")
        xlu_news = _seed_macro_news(
            db_session,
            news_id="macro-xlu-only",
            title="Utilities ETF news",
            publish_time=_NOW,
            source_ticker="XLU",
        )

        result = list(
            SentimentRepository(db_session).get_macro_news_fallback("XFER", _CUTOFF)
        )

        assert [r.id for r in result] == [xlu_news.id]


class TestDualListedTicker:
    """One ticker on two exchanges is legal; lookups must not raise."""

    # Letter-bearing ids so SQLite does not coerce them to numbers; fixed so the
    # first listing always sorts first by id, which makes a plain ORDER BY id
    # pick the delisted row and only the delisted_at ordering pick the live one.
    _FIRST_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
    _SECOND_ID = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")

    @classmethod
    def _seed_dual_listing(cls, session: Session, ticker: str):
        listings = []
        for instrument_id, exchange_name in (
            (cls._FIRST_ID, "EX-FIRST"),
            (cls._SECOND_ID, "EX-SECOND"),
        ):
            exchange = get_or_create_exchange(session, exchange_name)
            listings.append(
                add_and_flush(
                    session,
                    Instrument(
                        id=instrument_id,
                        ticker=ticker,
                        short_name=ticker,
                        exchange=exchange,
                    ),
                )
            )
        return listings[0], listings[1]

    def test_instrument_lookup_prefers_the_active_listing(
        self, db_session: Session
    ) -> None:
        delisted, active = self._seed_dual_listing(db_session, "DUAL")
        delisted.delisted_at = date(2023, 1, 1)
        db_session.flush()
        repo = SentimentRepository(db_session)
        assert repo.get_instrument_id_by_ticker("DUAL") == active.id

    def test_fallback_sector_lookup_tolerates_a_dual_listed_ticker(
        self, db_session: Session
    ) -> None:
        first, second = self._seed_dual_listing(db_session, "DUAL")
        _seed_profile(db_session, first, sector="Technology")
        _seed_profile(db_session, second, sector="Technology")
        _seed_macro_news(
            db_session,
            news_id="dual-1",
            title="Chip sector rallies",
            publish_time=_NOW,
            source_ticker="XLK",
        )
        repo = SentimentRepository(db_session)
        result = list(repo.get_macro_news_fallback("DUAL", _CUTOFF))
        assert [r.news_id for r in result] == ["dual-1"]


class TestGetRecentNews:
    def test_when_no_rows_after_cutoff_returns_empty(self, db_session: Session) -> None:
        inst = _seed_instrument(db_session, "MSFT")
        _seed_ticker_news(
            db_session,
            inst,
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
            db_session,
            inst,
            title="Early news",
            publish_time=datetime(2024, 6, 9, 6, 0, 0, tzinfo=timezone.utc),
            news_uuid="uuid-early",
        )
        late = _seed_ticker_news(
            db_session,
            inst,
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


class TestMacroNewsFallbackStrategyA:
    """Covers the direct source_ticker match path (Strategy A)."""

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


class TestMacroNewsFallbackStrategyB:
    """Covers the sector-ETF fallback when no direct source_ticker match exists (Strategy B)."""

    def test_when_no_direct_match_sector_etf_used(self, db_session: Session) -> None:
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


class TestMacroNewsFallbackStrategyC:
    """Covers the title/content ilike text-search fallback when A and B both miss (Strategy C)."""

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

    def test_when_ticker_in_full_content_returns_row(self, db_session: Session) -> None:
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

    def test_when_no_strategy_matches_returns_empty(self, db_session: Session) -> None:
        _seed_instrument(db_session, "ZZZZ")
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

    def test_when_text_match_before_cutoff_excluded(self, db_session: Session) -> None:
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


def _hourly(hour: int) -> datetime:
    return datetime(2024, 6, 9, hour, 0, 0, tzinfo=timezone.utc)


class TestMacroNewsFallbackKeepsNewestRows:
    """``limit`` must drop the oldest rows in the window, never the newest."""

    def test_strategy_a_limit_keeps_newest_rows_in_ascending_order(
        self, db_session: Session
    ) -> None:
        _seed_instrument(db_session, "TSLA")
        for hour in (1, 2, 3, 4):
            _seed_macro_news(
                db_session,
                news_id=f"newest-a-{hour}",
                title=f"TSLA story {hour}",
                publish_time=_hourly(hour),
                source_ticker="TSLA",
            )
        repo = SentimentRepository(db_session)
        result = list(repo.get_macro_news_fallback("TSLA", _CUTOFF, limit=2))
        assert [r.news_id for r in result] == ["newest-a-3", "newest-a-4"]

    def test_strategy_b_limit_keeps_newest_rows_in_ascending_order(
        self, db_session: Session
    ) -> None:
        inst = _seed_instrument(db_session, "CRM")
        _seed_profile(db_session, inst, sector="Technology")
        for hour in (1, 2, 3, 4):
            _seed_macro_news(
                db_session,
                news_id=f"newest-b-{hour}",
                title=f"Tech ETF story {hour}",
                publish_time=_hourly(hour),
                source_ticker="XLK",
            )
        repo = SentimentRepository(db_session)
        result = list(repo.get_macro_news_fallback("CRM", _CUTOFF, limit=2))
        assert [r.news_id for r in result] == ["newest-b-3", "newest-b-4"]

    def test_strategy_c_limit_keeps_newest_rows_in_ascending_order(
        self, db_session: Session
    ) -> None:
        _seed_instrument(db_session, "PLTR")
        for hour in (1, 2, 3, 4):
            _seed_macro_news(
                db_session,
                news_id=f"newest-c-{hour}",
                title=f"PLTR story {hour}",
                publish_time=_hourly(hour),
            )
        repo = SentimentRepository(db_session)
        result = list(repo.get_macro_news_fallback("PLTR", _CUTOFF, limit=2))
        assert [r.news_id for r in result] == ["newest-c-3", "newest-c-4"]


class TestAsOfUpperBound:
    """``as_of`` keeps point-in-time runs from reading post-decision news."""

    def test_get_recent_news_excludes_rows_after_as_of(
        self, db_session: Session
    ) -> None:
        inst = _seed_instrument(db_session, "AMZN")
        before = _seed_ticker_news(db_session, inst, "Before", _hourly(6), "asof-1")
        _seed_ticker_news(db_session, inst, "After", _hourly(18), "asof-2")
        repo = SentimentRepository(db_session)
        result = list(repo.get_recent_news(inst.id, _CUTOFF, as_of=_hourly(12)))
        assert [r.id for r in result] == [before.id]

    def test_macro_news_fallback_excludes_rows_after_as_of(
        self, db_session: Session
    ) -> None:
        _seed_instrument(db_session, "TSLA")
        for hour in (6, 18):
            _seed_macro_news(
                db_session,
                news_id=f"asof-macro-{hour}",
                title=f"TSLA story {hour}",
                publish_time=_hourly(hour),
                source_ticker="TSLA",
            )
        repo = SentimentRepository(db_session)
        result = list(repo.get_macro_news_fallback("TSLA", _CUTOFF, as_of=_hourly(12)))
        assert [r.news_id for r in result] == ["asof-macro-6"]


class TestMacroNewsTextSearchPrecision:
    """Strategy C must not treat a short or embedded ticker as a hit."""

    def test_short_ticker_is_not_text_searched(self, db_session: Session) -> None:
        _seed_instrument(db_session, "T")
        _seed_macro_news(
            db_session,
            news_id="short-1",
            title="Treasury yields climb as Fed signals patience",
            publish_time=_NOW,
        )
        repo = SentimentRepository(db_session)
        assert list(repo.get_macro_news_fallback("T", _CUTOFF)) == []

    def test_ticker_embedded_in_a_longer_word_does_not_match(
        self, db_session: Session
    ) -> None:
        _seed_instrument(db_session, "CAT")
        _seed_macro_news(
            db_session,
            news_id="embedded-1",
            title="Education spending rises",
            publish_time=_NOW,
            full_content="The category of services grew",
        )
        repo = SentimentRepository(db_session)
        assert list(repo.get_macro_news_fallback("CAT", _CUTOFF)) == []

    def test_lowercase_word_equal_to_ticker_does_not_match(
        self, db_session: Session
    ) -> None:
        _seed_instrument(db_session, "NOW")
        _seed_macro_news(
            db_session,
            news_id="lower-1",
            title="Rates are now higher than expected",
            publish_time=_NOW,
        )
        repo = SentimentRepository(db_session)
        assert list(repo.get_macro_news_fallback("NOW", _CUTOFF)) == []

    def test_standalone_ticker_amid_punctuation_matches(
        self, db_session: Session
    ) -> None:
        _seed_instrument(db_session, "CAT")
        hit = _seed_macro_news(
            db_session,
            news_id="punct-1",
            title="Heavy machinery (CAT) beats estimates",
            publish_time=_NOW,
        )
        repo = SentimentRepository(db_session)
        result = list(repo.get_macro_news_fallback("CAT", _CUTOFF))
        assert [r.id for r in result] == [hit.id]

    def test_like_wildcards_in_ticker_are_literal(self, db_session: Session) -> None:
        _seed_instrument(db_session, "A_B")
        _seed_macro_news(
            db_session,
            news_id="wild-1",
            title="AXB headline",
            publish_time=_NOW,
        )
        repo = SentimentRepository(db_session)
        assert list(repo.get_macro_news_fallback("A_B", _CUTOFF)) == []

    def test_limit_counts_only_real_mentions(self, db_session: Session) -> None:
        _seed_instrument(db_session, "CAT")
        for hour in (1, 2):
            _seed_macro_news(
                db_session,
                news_id=f"real-{hour}",
                title=f"CAT story {hour}",
                publish_time=_hourly(hour),
            )
        for hour in (3, 4, 5):
            _seed_macro_news(
                db_session,
                news_id=f"noise-{hour}",
                title=f"Education story {hour}",
                publish_time=_hourly(hour),
            )
        repo = SentimentRepository(db_session)
        result = list(repo.get_macro_news_fallback("CAT", _CUTOFF, limit=2))
        assert [r.news_id for r in result] == ["real-1", "real-2"]
