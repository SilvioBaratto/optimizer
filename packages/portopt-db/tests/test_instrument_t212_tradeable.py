"""The Trading 212 tradability overlay on ``instruments``.

``is_t212_tradeable`` is a tri-state: NULL means the overlay has not evaluated the
row (Trading 212 unconfigured, or no ISIN to match), True means the broker's
catalogue carries the instrument, False means it was matched against the
catalogue and is absent. NULL must never read as False, because a deployment
without Trading 212 would otherwise lose its whole universe the moment a caller
restricts to the tradeable set.
"""

from __future__ import annotations

import uuid
from datetime import date

from _fixtures import add_and_flush
from sqlalchemy.dialects import postgresql

from portopt_db.models.universe.universe import Exchange, Instrument
from portopt_db.repositories.universe.universe_repository import UniverseRepository


def _repo(session) -> UniverseRepository:
    return UniverseRepository(session)


def _exchange(session, name: str = "NASDAQ") -> Exchange:
    return add_and_flush(session, Exchange(name=name))


def _instrument(session, exchange: Exchange, ticker: str, **kw) -> Instrument:
    kw.setdefault("yfinance_ticker", ticker)
    return add_and_flush(
        session,
        Instrument(ticker=ticker, short_name=ticker, exchange_id=exchange.id, **kw),
    )


class TestColumn:
    """The column is nullable and defaults to 'not evaluated'."""

    def test_a_new_instrument_is_not_evaluated(self, db_session) -> None:
        """No flag is written until the overlay evaluates the row."""
        inst = _instrument(db_session, _exchange(db_session), "AAPL")

        assert inst.is_t212_tradeable is None

    def test_the_flag_round_trips_both_ways(self, db_session) -> None:
        """True and False are both storable and distinct from NULL."""
        ex = _exchange(db_session)
        yes = _instrument(db_session, ex, "AAPL", is_t212_tradeable=True)
        no = _instrument(db_session, ex, "VOO", is_t212_tradeable=False)

        db_session.refresh(yes)
        db_session.refresh(no)

        assert yes.is_t212_tradeable is True
        assert no.is_t212_tradeable is False


class TestSetT212Tradeable:
    """``set_t212_tradeable`` writes the flag by the (ticker, exchange) key."""

    def test_sets_true_and_reports_the_match(self, db_session) -> None:
        """A matching row is flagged and the call says so."""
        ex = _exchange(db_session)
        inst = _instrument(db_session, ex, "AAPL")

        matched = _repo(db_session).set_t212_tradeable(
            ticker="AAPL", exchange_id=ex.id, tradeable=True
        )

        db_session.refresh(inst)
        assert matched is True
        assert inst.is_t212_tradeable is True

    def test_sets_false_on_a_previously_tradeable_row(self, db_session) -> None:
        """A name that left the broker catalogue is flagged False, not NULL."""
        ex = _exchange(db_session)
        inst = _instrument(db_session, ex, "AAPL", is_t212_tradeable=True)

        _repo(db_session).set_t212_tradeable(
            ticker="AAPL", exchange_id=ex.id, tradeable=False
        )

        db_session.refresh(inst)
        assert inst.is_t212_tradeable is False

    def test_reports_no_match_for_an_unknown_row(self, db_session) -> None:
        """An unknown key updates nothing and returns False."""
        ex = _exchange(db_session)

        matched = _repo(db_session).set_t212_tradeable(
            ticker="NOPE", exchange_id=ex.id, tradeable=True
        )

        assert matched is False

    def test_only_the_keyed_exchange_row_changes(self, db_session) -> None:
        """The same symbol on another exchange keeps its own flag."""
        nasdaq = _exchange(db_session, "NASDAQ")
        xetra = _exchange(db_session, "Deutsche Börse Xetra")
        here = _instrument(db_session, nasdaq, "AAPL")
        there = _instrument(db_session, xetra, "AAPL")

        _repo(db_session).set_t212_tradeable(
            ticker="AAPL", exchange_id=nasdaq.id, tradeable=True
        )

        db_session.refresh(here)
        db_session.refresh(there)
        assert here.is_t212_tradeable is True
        assert there.is_t212_tradeable is None

    def test_leaves_the_t212_ticker_alone(self, db_session) -> None:
        """Flagging is independent of which broker line was chosen."""
        ex = _exchange(db_session)
        inst = _instrument(db_session, ex, "AAPL", t212_ticker="AAPL_US_EQ")

        _repo(db_session).set_t212_tradeable(
            ticker="AAPL", exchange_id=ex.id, tradeable=False
        )

        db_session.refresh(inst)
        assert inst.t212_ticker == "AAPL_US_EQ"


class TestTradeableTickers:
    """``get_t212_tradeable_tickers`` is the allowlist hook for a restricted run."""

    def test_is_none_until_the_overlay_has_evaluated_something(
        self, db_session
    ) -> None:
        """Trading 212 unconfigured: no restriction exists, so the universe is full."""
        ex = _exchange(db_session)
        _instrument(db_session, ex, "AAPL")
        _instrument(db_session, ex, "VOO")

        assert _repo(db_session).get_t212_tradeable_tickers() is None

    def test_lists_flagged_active_yahoo_tickers_sorted(self, db_session) -> None:
        """Only True rows are returned, as sorted Yahoo symbols."""
        ex = _exchange(db_session)
        _instrument(db_session, ex, "MSFT", is_t212_tradeable=True)
        _instrument(db_session, ex, "AAPL", is_t212_tradeable=True)
        _instrument(db_session, ex, "VOO", is_t212_tradeable=False)
        _instrument(db_session, ex, "GLD")

        assert _repo(db_session).get_t212_tradeable_tickers() == ["AAPL", "MSFT"]

    def test_is_empty_not_none_when_the_overlay_found_nothing_tradeable(
        self, db_session
    ) -> None:
        """An evaluated universe with no match is a real, empty allowlist."""
        ex = _exchange(db_session)
        _instrument(db_session, ex, "VOO", is_t212_tradeable=False)

        assert _repo(db_session).get_t212_tradeable_tickers() == []

    def test_excludes_delisted_instruments(self, db_session) -> None:
        """A delisted line is not actionable on the broker."""
        ex = _exchange(db_session)
        _instrument(db_session, ex, "AAPL", is_t212_tradeable=True)
        _instrument(
            db_session,
            ex,
            "OLD",
            is_t212_tradeable=True,
            delisted_at=date(2025, 1, 1),
        )

        assert _repo(db_session).get_t212_tradeable_tickers() == ["AAPL"]

    def test_uses_the_yahoo_symbol_and_skips_rows_without_one(self, db_session) -> None:
        """The allowlist speaks Yahoo symbols, the fund layer's ticker space."""
        ex = _exchange(db_session)
        _instrument(
            db_session,
            ex,
            "VWCE_XETRA",
            yfinance_ticker="VWCE.DE",
            is_t212_tradeable=True,
        )
        _instrument(
            db_session, ex, "NOYAHOO", yfinance_ticker=None, is_t212_tradeable=True
        )

        assert _repo(db_session).get_t212_tradeable_tickers() == ["VWCE.DE"]

    def test_a_symbol_listed_twice_appears_once(self, db_session) -> None:
        """Two rows sharing a Yahoo symbol yield one allowlist entry."""
        nasdaq = _exchange(db_session, "NASDAQ")
        nyse = _exchange(db_session, "NYSE")
        _instrument(db_session, nasdaq, "XYZ", is_t212_tradeable=True)
        _instrument(db_session, nyse, "XYZ", is_t212_tradeable=True)

        assert _repo(db_session).get_t212_tradeable_tickers() == ["XYZ"]


class TestRebuildKeepsTheOverlay:
    """The weekly universe upsert must not reset what the overlay learned."""

    def test_the_conflict_update_does_not_write_the_flag(
        self, db_session, monkeypatch
    ) -> None:
        """Re-running the build would otherwise wipe every flag to NULL."""
        statements: list = []
        real_execute = db_session.execute

        def _capture_upsert(stmt, *args, **kwargs):
            if getattr(stmt, "is_insert", False):
                statements.append(stmt)
                return None
            return real_execute(stmt, *args, **kwargs)

        monkeypatch.setattr(db_session, "execute", _capture_upsert)
        _repo(db_session).save_instruments_batch(
            [{"ticker": "AAPL", "shortName": "AAPL", "type": "STOCK"}],
            exchange_id=uuid.uuid4(),
        )

        sql = str(statements[0].compile(dialect=postgresql.dialect()))
        assert "ON CONFLICT" in sql
        assert "is_t212_tradeable" not in sql.split("DO UPDATE SET")[1]


class TestUntradeableTickers:
    """``get_t212_untradeable_tickers`` names only the explicit refusals.

    The complement of the tradeable allowlist under the tri-state: a row the overlay
    has not evaluated (NULL) is not refused, only one matched against the catalogue and
    found absent (False) is.
    """

    def test_is_none_until_the_overlay_has_evaluated_something(
        self, db_session
    ) -> None:
        """Nothing evaluated means nothing to refuse, and the caller must know that."""
        ex = _exchange(db_session)
        _instrument(db_session, ex, "AAPL")

        assert _repo(db_session).get_t212_untradeable_tickers() is None
        assert _repo(db_session).get_t212_untradeable_tickers(["AAPL"]) is None

    def test_lists_only_explicit_false_not_null_or_true(self, db_session) -> None:
        """NULL (not evaluated) and True (carried) are never refused."""
        ex = _exchange(db_session)
        _instrument(db_session, ex, "AAPL", is_t212_tradeable=True)
        _instrument(db_session, ex, "VOO", is_t212_tradeable=False)
        _instrument(db_session, ex, "GLD")
        _instrument(db_session, ex, "ARKK", is_t212_tradeable=False)

        assert _repo(db_session).get_t212_untradeable_tickers() == ["ARKK", "VOO"]

    def test_restricts_to_the_tickers_asked_about(self, db_session) -> None:
        """A caller with an allowlist pays only for its own names."""
        ex = _exchange(db_session)
        _instrument(db_session, ex, "VOO", is_t212_tradeable=False)
        _instrument(db_session, ex, "ARKK", is_t212_tradeable=False)
        _instrument(db_session, ex, "AAPL", is_t212_tradeable=True)

        repo = _repo(db_session)

        assert repo.get_t212_untradeable_tickers(["VOO", "AAPL", "MISSING"]) == ["VOO"]
        assert repo.get_t212_untradeable_tickers([]) == []

    def test_is_empty_when_everything_evaluated_is_tradeable(self, db_session) -> None:
        """An evaluated universe with no refusal is an empty list, not None."""
        ex = _exchange(db_session)
        _instrument(db_session, ex, "AAPL", is_t212_tradeable=True)

        assert _repo(db_session).get_t212_untradeable_tickers() == []

    def test_a_symbol_with_one_non_false_row_is_not_refused(self, db_session) -> None:
        """Two lines of one Yahoo symbol: the broker carrying either one is enough."""
        nasdaq = _exchange(db_session, "NASDAQ")
        nyse = _exchange(db_session, "NYSE")
        _instrument(db_session, nasdaq, "XYZ", is_t212_tradeable=False)
        _instrument(db_session, nyse, "XYZ", is_t212_tradeable=True)
        _instrument(db_session, nasdaq, "UVW", is_t212_tradeable=False)
        _instrument(db_session, nyse, "UVW")
        _instrument(db_session, nasdaq, "ABC", is_t212_tradeable=False)
        _instrument(db_session, nyse, "ABC", is_t212_tradeable=False)

        assert _repo(db_session).get_t212_untradeable_tickers() == ["ABC"]

    def test_a_delisted_twin_never_shields_the_active_listing(self, db_session) -> None:
        """The overlay only evaluates active rows, so a delisted twin stays NULL.

        Counting it would let a symbol that moved venue slip past the refusal list
        although its live line is absent from the broker.
        """
        nasdaq = _exchange(db_session, "NASDAQ")
        nyse = _exchange(db_session, "NYSE")
        _instrument(db_session, nasdaq, "XYZ", delisted_at=date(2026, 9, 1))
        _instrument(db_session, nyse, "XYZ", is_t212_tradeable=False)
        _instrument(
            db_session,
            nasdaq,
            "UVW",
            is_t212_tradeable=True,
            delisted_at=date(2026, 9, 1),
        )
        _instrument(db_session, nyse, "UVW", is_t212_tradeable=False)

        assert _repo(db_session).get_t212_untradeable_tickers() == ["UVW", "XYZ"]

    def test_a_delisted_false_row_alone_refuses_nothing(self, db_session) -> None:
        """Delisting is the caller's own check, not a broker verdict."""
        ex = _exchange(db_session)
        _instrument(db_session, ex, "AAPL", is_t212_tradeable=True)
        _instrument(
            db_session,
            ex,
            "OLD",
            is_t212_tradeable=False,
            delisted_at=date(2025, 1, 1),
        )

        assert _repo(db_session).get_t212_untradeable_tickers() == []

    def test_uses_the_yahoo_symbol_and_skips_rows_without_one(self, db_session) -> None:
        """The refusals speak Yahoo symbols, the fund layer's ticker space."""
        ex = _exchange(db_session)
        _instrument(
            db_session,
            ex,
            "VWCE_XETRA",
            yfinance_ticker="VWCE.DE",
            is_t212_tradeable=False,
        )
        _instrument(
            db_session, ex, "NOYAHOO", yfinance_ticker=None, is_t212_tradeable=False
        )

        assert _repo(db_session).get_t212_untradeable_tickers() == ["VWCE.DE"]


class TestUnevaluatedTickers:
    """``get_t212_unevaluated_tickers`` names the symbols the overlay could not judge."""

    def test_lists_a_symbol_whose_rows_are_all_null(self, db_session) -> None:
        """No ISIN to match means NULL on every row: the broker status is unknown."""
        ex = _exchange(db_session)
        _instrument(db_session, ex, "KNOWN", is_t212_tradeable=True)
        _instrument(db_session, ex, "UNKNOWN")

        assert _repo(db_session).get_t212_unevaluated_tickers(["KNOWN", "UNKNOWN"]) == [
            "UNKNOWN"
        ]

    def test_a_flag_on_any_active_row_means_it_was_evaluated(self, db_session) -> None:
        """One evaluated listing is enough: True and False are both answers."""
        ex = _exchange(db_session)
        other = _exchange(db_session, "XETRA")
        _instrument(db_session, ex, "MIXED")
        _instrument(db_session, other, "MIXED", is_t212_tradeable=False)
        _instrument(db_session, ex, "REFUSED", is_t212_tradeable=False)

        assert (
            _repo(db_session).get_t212_unevaluated_tickers(["MIXED", "REFUSED"]) == []
        )

    def test_a_delisted_row_does_not_count_either_way(self, db_session) -> None:
        """Only active rows are judged, as in the tradeable and untradeable lists."""
        ex = _exchange(db_session)
        other = _exchange(db_session, "LSE")
        _instrument(db_session, ex, "MOVED", delisted_at=date(2026, 1, 5))
        _instrument(db_session, other, "MOVED", is_t212_tradeable=True)
        _instrument(
            db_session,
            ex,
            "GONE",
            delisted_at=date(2026, 1, 5),
            is_t212_tradeable=True,
        )

        assert _repo(db_session).get_t212_unevaluated_tickers(["MOVED", "GONE"]) == []

    def test_a_ticker_with_no_active_instrument_is_not_listed(self, db_session) -> None:
        """An unknown symbol is the eligibility check's business, not an overlay gap."""
        _instrument(db_session, _exchange(db_session), "KNOWN")

        assert _repo(db_session).get_t212_unevaluated_tickers(["KNOWN", "NOPE"]) == [
            "KNOWN"
        ]

    def test_an_empty_ask_is_an_empty_answer(self, db_session) -> None:
        """No tickers asked about means none listed, without a query."""
        assert _repo(db_session).get_t212_unevaluated_tickers([]) == []
