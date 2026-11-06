"""Regression tests for yfinance repository defects found by a read-only review."""

from __future__ import annotations

import uuid
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from portopt_db.models.market_data.yfinance_data import (
    EarningsEstimate,
    EpsRevisions,
    EpsTrend,
    PriceHistory,
    RevenueEstimate,
    TickerNews,
)
from portopt_db.models.universe.pinned_instrument import PinnedInstrument
from portopt_db.models.universe.universe import Exchange, Instrument
from portopt_db.repositories.market_data.yfinance_repository import (
    YFinanceRepository,
    _safe_str,
    _safe_val,
)


def _instrument(session: Session, ticker: str = "TSM") -> Instrument:
    exchange = Exchange(name=f"EX-{ticker}")
    session.add(exchange)
    session.flush()
    instrument = Instrument(
        ticker=ticker,
        short_name=ticker,
        exchange=exchange,
        yfinance_ticker=ticker,
    )
    session.add(instrument)
    session.flush()
    return instrument


def _insider_frame(shares: list[int]) -> pd.DataFrame:
    n = len(shares)
    return pd.DataFrame(
        {
            "Insider": ["DOE"] * n,
            "Transaction": ["Sale"] * n,
            "Text": [""] * n,
            "Position": ["CEO"] * n,
            "Shares": shares,
            "Value": [s * 10 for s in shares],
            "Start Date": [pd.Timestamp("2026-10-01")] * n,
            "Ownership": ["D"] * n,
        }
    )


class TestInsiderLotsShrink:
    """Lots Yahoo stops listing must leave the table, or sales are double-counted."""

    def test_a_group_that_shrinks_drops_its_surplus_lots(
        self, db_session: Session
    ) -> None:
        """Remove lot 1 once Yahoo lists the sale a single time."""
        instrument = _instrument(db_session)
        repo = YFinanceRepository(db_session)

        repo.upsert_insider_transactions(instrument.id, _insider_frame([100, 100]))
        repo.upsert_insider_transactions(instrument.id, _insider_frame([100]))

        rows = repo.get_insider_transactions(instrument.id)
        assert [(r.lot_index, r.shares) for r in rows] == [(0, 100)]

    def test_other_groups_and_unlisted_history_are_untouched(
        self, db_session: Session
    ) -> None:
        """Reconcile only groups the fetch lists; older history stays as stored."""
        instrument = _instrument(db_session)
        repo = YFinanceRepository(db_session)
        old = _insider_frame([5, 5])
        old["Start Date"] = pd.Timestamp("2025-01-02")
        repo.upsert_insider_transactions(instrument.id, old)
        repo.upsert_insider_transactions(instrument.id, _insider_frame([100, 100]))

        repo.upsert_insider_transactions(instrument.id, _insider_frame([100]))

        stored = sorted(
            (str(r.start_date), r.lot_index)
            for r in repo.get_insider_transactions(instrument.id)
        )
        assert stored == [
            ("2025-01-02", 0),
            ("2025-01-02", 1),
            ("2026-10-01", 0),
        ]

    def test_a_group_that_keeps_its_size_keeps_every_lot(
        self, db_session: Session
    ) -> None:
        """Keep both lots when the group is re-listed in another order."""
        instrument = _instrument(db_session)
        repo = YFinanceRepository(db_session)

        repo.upsert_insider_transactions(instrument.id, _insider_frame([100, 250]))
        repo.upsert_insider_transactions(instrument.id, _insider_frame([250, 100]))

        rows = repo.get_insider_transactions(instrument.id)
        assert sorted((r.lot_index, r.shares) for r in rows) == [(0, 100), (1, 250)]


@pytest.mark.parametrize(
    ("model", "upsert", "frame"),
    [
        (
            EarningsEstimate,
            "upsert_earnings_estimate",
            lambda currency: pd.DataFrame(
                {"avg": [1.0, 2.0], "currency": currency}, index=["0q", "+1q"]
            ),
        ),
        (
            RevenueEstimate,
            "upsert_revenue_estimate",
            lambda currency: pd.DataFrame(
                {"avg": [1.0, 2.0], "currency": currency}, index=["0q", "+1q"]
            ),
        ),
        (
            EpsTrend,
            "upsert_eps_trend",
            lambda currency: pd.DataFrame(
                {"current": [1.0, 2.0], "currency": currency}, index=["0q", "+1q"]
            ),
        ),
        (
            EpsRevisions,
            "upsert_eps_revisions",
            lambda currency: pd.DataFrame(
                {"upLast7days": [1, 2], "currency": currency}, index=["0q", "+1q"]
            ),
        ),
    ],
)
def test_a_row_without_currency_keeps_the_stored_label_in_a_mixed_batch(
    db_session: Session, model: Any, upsert: str, frame: Any
) -> None:
    """One currency-less row in a batch must not NULL its own stored label."""
    instrument = _instrument(db_session)
    repo = YFinanceRepository(db_session)
    getattr(repo, upsert)(instrument.id, frame(["TWD", "TWD"]))

    getattr(repo, upsert)(instrument.id, frame(["TWD", None]))

    db_session.expire_all()
    stored = {
        row.period: row.currency for row in db_session.execute(select(model)).scalars()
    }
    assert stored == {"0q": "TWD", "+1q": "TWD"}


def test_a_row_with_a_new_currency_still_replaces_the_stored_label(
    db_session: Session,
) -> None:
    """Overwrite a stored label when the fetch carries a different currency."""
    instrument = _instrument(db_session)
    repo = YFinanceRepository(db_session)
    frame = pd.DataFrame({"avg": [1.0, 2.0], "currency": ["TWD", "TWD"]})
    frame.index = ["0q", "+1q"]
    repo.upsert_earnings_estimate(instrument.id, frame)

    frame["currency"] = ["USD", None]
    repo.upsert_earnings_estimate(instrument.id, frame)

    db_session.expire_all()
    stored = {
        row.period: row.currency
        for row in db_session.execute(select(EarningsEstimate)).scalars()
    }
    assert stored == {"0q": "USD", "+1q": "TWD"}


class TestSafeValMissingScalars:
    """Every pandas/numpy missing-value scalar must read as None."""

    @pytest.mark.parametrize(
        "missing",
        [
            pd.NA,
            pd.NaT,
            np.float32("nan"),
            np.float32("inf"),
            np.float64("nan"),
            np.datetime64("NaT", "ns"),
        ],
    )
    def test_missing_scalars_become_none(self, missing: Any) -> None:
        """Map pd.NA, NaT and non-finite numpy scalars to None."""
        assert _safe_val(missing) is None
        assert _safe_str(missing) is None

    def test_regular_values_pass_through(self) -> None:
        """Unwrap numpy scalars without altering real values."""
        assert _safe_val(np.float32(1.5)) == 1.5
        assert _safe_val(np.int64(7)) == 7
        assert _safe_val("text") == "text"


class TestScrapedNewsIds:
    """The scrape gate needs ids only, never the stored article bodies."""

    def _news(
        self, session: Session, instrument: Instrument, uuid_: str, body: str | None
    ) -> None:
        session.add(
            TickerNews(
                instrument_id=instrument.id,
                news_uuid=uuid_,
                title=uuid_,
                full_content=body,
            )
        )
        session.flush()

    def test_returns_only_requested_ids_that_have_a_body(
        self, db_session: Session
    ) -> None:
        """Skip unrequested ids, other instruments and empty or missing bodies."""
        instrument = _instrument(db_session)
        other = _instrument(db_session, "OTHER")
        self._news(db_session, instrument, "with-body", "text")
        self._news(db_session, instrument, "empty-body", "")
        self._news(db_session, instrument, "no-body", None)
        self._news(db_session, instrument, "not-requested", "text")
        self._news(db_session, other, "other-instrument", "text")
        repo = YFinanceRepository(db_session)

        scraped = repo.get_scraped_news_uuids(
            instrument.id,
            ["with-body", "empty-body", "no-body", "other-instrument", "unknown"],
        )

        assert scraped == {"with-body"}

    def test_an_empty_request_returns_nothing(self, db_session: Session) -> None:
        """Issue no query for an empty feed."""
        repo = YFinanceRepository(db_session)

        assert repo.get_scraped_news_uuids(uuid.uuid4(), []) == set()

    def test_a_large_feed_is_answered_in_chunks(self, db_session: Session) -> None:
        """Stay under the driver parameter limit on a long id list."""
        instrument = _instrument(db_session)
        self._news(db_session, instrument, "u-1500", "text")
        repo = YFinanceRepository(db_session)

        scraped = repo.get_scraped_news_uuids(
            instrument.id, [f"u-{i}" for i in range(3000)]
        )

        assert scraped == {"u-1500"}


class TestDormantDelistedInstruments:
    """A name delisted long ago that Yahoo no longer prices leaves the sweep."""

    _TODAY = date(2026, 11, 6)

    def _delisted(
        self, session: Session, ticker: str, *, delisted_days_ago: int
    ) -> Instrument:
        instrument = _instrument(session, ticker)
        instrument.delisted_at = self._TODAY - timedelta(days=delisted_days_ago)
        session.flush()
        return instrument

    def _bar(self, session: Session, instrument: Instrument, *, days_ago: int) -> None:
        session.add(
            PriceHistory(
                instrument_id=instrument.id,
                date=self._TODAY - timedelta(days=days_ago),
                close=1,
            )
        )
        session.flush()

    def _swept(self, session: Session) -> set[uuid.UUID]:
        repo = YFinanceRepository(session)
        return {
            row.id
            for row in repo.get_instruments_with_yfinance_ticker(today=self._TODAY)
        }

    def test_long_delisted_with_no_recent_bar_is_dropped(
        self, db_session: Session
    ) -> None:
        """Stop requesting an instrument that has been dead for over a year."""
        dead = self._delisted(db_session, "DEAD", delisted_days_ago=400)
        self._bar(db_session, dead, days_ago=395)

        assert dead.id not in self._swept(db_session)

    def test_long_delisted_without_any_stored_bar_is_dropped(
        self, db_session: Session
    ) -> None:
        """Treat a never-priced old delisting as dead too."""
        dead = self._delisted(db_session, "NOPX", delisted_days_ago=400)

        assert dead.id not in self._swept(db_session)

    def test_long_delisted_that_still_trades_is_kept(self, db_session: Session) -> None:
        """Keep fetching a name that left the universe but Yahoo still prices."""
        alive = self._delisted(db_session, "LIVE", delisted_days_ago=400)
        self._bar(db_session, alive, days_ago=3)

        assert alive.id in self._swept(db_session)

    def test_recently_delisted_is_kept_even_without_recent_bars(
        self, db_session: Session
    ) -> None:
        """Give a new delisting the full grace period."""
        recent = self._delisted(db_session, "RECENT", delisted_days_ago=30)
        self._bar(db_session, recent, days_ago=60)

        assert recent.id in self._swept(db_session)

    def test_active_instruments_are_never_dropped(self, db_session: Session) -> None:
        """Apply the cutoff to delisted rows only."""
        active = _instrument(db_session, "ACTIVE")

        assert active.id in self._swept(db_session)

    def test_pinned_instrument_is_kept(self, db_session: Session) -> None:
        """Honour an operator pin over the dormancy cutoff."""
        pinned = self._delisted(db_session, "PIN", delisted_days_ago=400)
        db_session.add(PinnedInstrument(ticker="PIN", source="manual"))
        db_session.flush()

        assert pinned.id in self._swept(db_session)
