"""Tests for UniverseRepository T212-annotation helpers.

Covers ``set_t212_ticker`` (update by ticker + exchange),
``get_active_instruments`` (non-delisted rows),
``get_active_listings`` ((ticker, exchange name) of non-delisted rows), and
``backfill_isin_from_profiles``.
"""

from __future__ import annotations

from datetime import date

from _fixtures import add_and_flush

from portopt_db.models.market_data.yfinance_data import TickerProfile
from portopt_db.models.universe.universe import Exchange, Instrument
from portopt_db.repositories.universe.universe_repository import UniverseRepository


def _repo(session) -> UniverseRepository:
    return UniverseRepository(session)


def test_instrument_t212_ticker_defaults_to_none(db_session) -> None:
    ex = add_and_flush(db_session, Exchange(name="NASDAQ"))
    inst = add_and_flush(
        db_session, Instrument(ticker="AAPL", short_name="Apple", exchange=ex)
    )
    assert inst.t212_ticker is None


def test_set_t212_ticker_updates_matching_row(db_session) -> None:
    ex = add_and_flush(db_session, Exchange(name="NASDAQ"))
    inst = add_and_flush(
        db_session, Instrument(ticker="AAPL", short_name="Apple", exchange=ex)
    )
    assert (
        _repo(db_session).set_t212_ticker(
            ticker="AAPL", exchange_id=ex.id, t212_ticker="AAPL_US_EQ"
        )
        is True
    )
    db_session.refresh(inst)
    assert inst.t212_ticker == "AAPL_US_EQ"


def test_set_t212_ticker_none_clears_a_stale_cross_reference(db_session) -> None:
    ex = add_and_flush(db_session, Exchange(name="NASDAQ"))
    inst = add_and_flush(
        db_session,
        Instrument(
            ticker="AAPL", short_name="Apple", exchange=ex, t212_ticker="AAPL_US_EQ"
        ),
    )
    assert (
        _repo(db_session).set_t212_ticker(
            ticker="AAPL", exchange_id=ex.id, t212_ticker=None
        )
        is True
    )
    db_session.refresh(inst)
    assert inst.t212_ticker is None


def test_set_t212_ticker_missing_row_returns_false(db_session) -> None:
    ex = add_and_flush(db_session, Exchange(name="NASDAQ"))
    assert (
        _repo(db_session).set_t212_ticker(
            ticker="NOPE", exchange_id=ex.id, t212_ticker="X"
        )
        is False
    )


def test_get_active_instruments_excludes_delisted(db_session) -> None:
    ex = add_and_flush(db_session, Exchange(name="NASDAQ"))
    add_and_flush(
        db_session, Instrument(ticker="AAPL", short_name="Apple", exchange=ex)
    )
    add_and_flush(
        db_session,
        Instrument(
            ticker="OLD",
            short_name="Old",
            exchange=ex,
            delisted_at=date(2020, 1, 1),
        ),
    )
    tickers = {i.ticker for i in _repo(db_session).get_active_instruments()}
    assert tickers == {"AAPL"}


def test_get_active_listings_pairs_ticker_with_exchange_name(db_session) -> None:
    nasdaq = add_and_flush(db_session, Exchange(name="NASDAQ"))
    xetra = add_and_flush(db_session, Exchange(name="Deutsche Börse Xetra"))
    add_and_flush(
        db_session, Instrument(ticker="AAPL", short_name="Apple", exchange=nasdaq)
    )
    add_and_flush(
        db_session, Instrument(ticker="AAPL", short_name="Apple", exchange=xetra)
    )
    add_and_flush(
        db_session,
        Instrument(
            ticker="OLD",
            short_name="Old",
            exchange=nasdaq,
            delisted_at=date(2020, 1, 1),
        ),
    )
    assert _repo(db_session).get_active_listings() == {
        ("AAPL", "NASDAQ"),
        ("AAPL", "Deutsche Börse Xetra"),
    }


def test_backfill_isin_fills_null_from_profile(db_session) -> None:
    ex = add_and_flush(db_session, Exchange(name="NASDAQ"))
    inst = add_and_flush(
        db_session, Instrument(ticker="AAPL", short_name="Apple", exchange=ex)
    )
    assert inst.isin is None
    add_and_flush(db_session, TickerProfile(instrument_id=inst.id, isin="US0378331005"))

    filled = _repo(db_session).backfill_isin_from_profiles()

    assert filled == 1
    db_session.refresh(inst)
    assert inst.isin == "US0378331005"


def test_backfill_isin_does_not_overwrite_existing(db_session) -> None:
    ex = add_and_flush(db_session, Exchange(name="NASDAQ"))
    inst = add_and_flush(
        db_session,
        Instrument(ticker="AAPL", short_name="Apple", exchange=ex, isin="EXISTING123"),
    )
    add_and_flush(db_session, TickerProfile(instrument_id=inst.id, isin="OTHER456"))

    filled = _repo(db_session).backfill_isin_from_profiles()

    assert filled == 0
    db_session.refresh(inst)
    assert inst.isin == "EXISTING123"


def test_backfill_isin_skips_null_profile_isin(db_session) -> None:
    ex = add_and_flush(db_session, Exchange(name="NASDAQ"))
    inst = add_and_flush(
        db_session, Instrument(ticker="AAPL", short_name="Apple", exchange=ex)
    )
    add_and_flush(db_session, TickerProfile(instrument_id=inst.id, isin=None))

    filled = _repo(db_session).backfill_isin_from_profiles()

    assert filled == 0
    db_session.refresh(inst)
    assert inst.isin is None


def _profiled(db_session, ticker: str, profile_isin: str, **kw) -> Instrument:
    ex = add_and_flush(db_session, Exchange(name=f"EX-{ticker}"))
    inst = add_and_flush(
        db_session, Instrument(ticker=ticker, short_name=ticker, exchange=ex, **kw)
    )
    add_and_flush(db_session, TickerProfile(instrument_id=inst.id, isin=profile_isin))
    return inst


def test_backfill_isin_skips_suffixed_symbols(db_session) -> None:
    """Profile ISINs scraped for a venue-suffixed symbol are a secondary line's.

    Before the scrape was restricted to suffix-less symbols, ``ASML.AS`` stored an
    Argentine CEDEAR's ISIN on its profile. Copying that onto the instrument would
    pin the wrong security to it permanently, because the backfill never
    overwrites.
    """
    inst = _profiled(
        db_session,
        "ASML.AS",
        "ARDEUT110061",
        yfinance_ticker="ASML.AS",
    )

    filled = _repo(db_session).backfill_isin_from_profiles()

    assert filled == 0
    db_session.refresh(inst)
    assert inst.isin is None


def test_backfill_isin_judges_a_suffix_by_the_yahoo_symbol_over_the_ticker(
    db_session,
) -> None:
    """The Yahoo symbol is the venue-bearing name; the row ticker may be anything."""
    inst = _profiled(
        db_session,
        "VWCE_XETRA",
        "IE00BK5BQT80",
        yfinance_ticker="VWCE.DE",
    )

    assert _repo(db_session).backfill_isin_from_profiles() == 0
    db_session.refresh(inst)
    assert inst.isin is None


def test_backfill_isin_falls_back_to_the_ticker_when_there_is_no_yahoo_symbol(
    db_session,
) -> None:
    """A row with no Yahoo symbol is still judged by its own ticker."""
    suffixed = _profiled(db_session, "NESN.SW", "CH0038863350")
    plain = _profiled(db_session, "AAPL", "US0378331005")

    filled = _repo(db_session).backfill_isin_from_profiles()

    assert filled == 1
    db_session.refresh(suffixed)
    db_session.refresh(plain)
    assert suffixed.isin is None
    assert plain.isin == "US0378331005"


def test_backfill_isin_keeps_filling_dash_class_us_symbols(db_session) -> None:
    """A dash share class is a US symbol: Yahoo writes ``BRK-B``, not ``BRK.B``."""
    inst = _profiled(db_session, "BRK-B", "US0846707026", yfinance_ticker="BRK-B")

    assert _repo(db_session).backfill_isin_from_profiles() == 1
    db_session.refresh(inst)
    assert inst.isin == "US0846707026"


def test_set_isin_if_missing_fills_a_null_isin(db_session) -> None:
    ex = add_and_flush(db_session, Exchange(name="NASDAQ"))
    inst = add_and_flush(
        db_session, Instrument(ticker="AAPL", short_name="Apple", exchange=ex)
    )

    matched = _repo(db_session).set_isin_if_missing(
        ticker="AAPL", exchange_id=ex.id, isin="US0378331005"
    )

    db_session.refresh(inst)
    assert matched is True
    assert inst.isin == "US0378331005"


def test_set_isin_if_missing_never_overwrites_a_stored_isin(db_session) -> None:
    ex = add_and_flush(db_session, Exchange(name="NASDAQ"))
    inst = add_and_flush(
        db_session,
        Instrument(ticker="AAPL", short_name="Apple", exchange=ex, isin="US0378331005"),
    )

    matched = _repo(db_session).set_isin_if_missing(
        ticker="AAPL", exchange_id=ex.id, isin="US0000000000"
    )

    db_session.refresh(inst)
    assert matched is False
    assert inst.isin == "US0378331005"


def test_set_isin_if_missing_reports_no_match_for_an_unknown_row(db_session) -> None:
    ex = add_and_flush(db_session, Exchange(name="NASDAQ"))

    assert (
        _repo(db_session).set_isin_if_missing(
            ticker="NOPE", exchange_id=ex.id, isin="US0378331005"
        )
        is False
    )


def test_has_active_instruments_is_false_on_an_empty_table(db_session) -> None:
    assert _repo(db_session).has_active_instruments() is False


def test_has_active_instruments_is_true_with_a_live_row(db_session) -> None:
    ex = add_and_flush(db_session, Exchange(name="NASDAQ"))
    add_and_flush(
        db_session, Instrument(ticker="AAPL", short_name="Apple", exchange=ex)
    )
    assert _repo(db_session).has_active_instruments() is True


def test_has_active_instruments_ignores_delisted_rows(db_session) -> None:
    ex = add_and_flush(db_session, Exchange(name="NASDAQ"))
    add_and_flush(
        db_session,
        Instrument(
            ticker="OLD",
            short_name="Old",
            exchange=ex,
            delisted_at=date(2020, 1, 1),
        ),
    )
    assert _repo(db_session).has_active_instruments() is False
