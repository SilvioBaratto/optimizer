"""Unit tests for ``app.services._sector_resolver`` (issue #427).

Covers the two-tier sector-mapping resolver shared by Attribution and
Dashboard:

1. Snapshot mapping is preferred when provided.
2. TickerProfile fallback is queried via ``Instrument.yfinance_ticker``
   (not ``Instrument.ticker``), which is the root-cause fix for the
   ``trading212`` 422 — its weights use yfinance-style keys like
   ``ENGI.PA`` that do not match ``Instrument.ticker`` (``ENGIp_EQ``).
3. Tickers unmatched by either source default to ``Unclassified``.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest
from sqlalchemy.orm import Session

from app.models.market_data.yfinance_data import TickerProfile
from app.models.universe.universe import Exchange, Instrument


@pytest.fixture
def seeded_instruments(db_session: Session) -> Iterator[None]:
    """Seed instruments mirroring the trading212 case + a US-style ticker."""
    exchange = Exchange(name="TEST_SECTOR_EXCH")
    db_session.add(exchange)
    db_session.flush()

    db_session.add_all(
        [
            _make_instrument(exchange.id, ticker="ENGIp_EQ", yf_ticker="ENGI.PA"),
            _make_instrument(exchange.id, ticker="ORAp_EQ", yf_ticker="ORA.PA"),
            _make_instrument(exchange.id, ticker="AAPL", yf_ticker="AAPL"),
            _make_instrument(exchange.id, ticker="EMPTYp_EQ", yf_ticker="EMPTY.PA"),
        ]
    )
    db_session.flush()

    by_yf: dict[str, Instrument] = {
        inst.yfinance_ticker or "": inst for inst in db_session.query(Instrument).all()
    }
    db_session.add_all(
        [
            TickerProfile(instrument_id=by_yf["ENGI.PA"].id, sector="Utilities"),
            TickerProfile(
                instrument_id=by_yf["ORA.PA"].id, sector="Communication Services"
            ),
            TickerProfile(instrument_id=by_yf["AAPL"].id, sector="Technology"),
            # EMPTY.PA intentionally has no profile row
        ]
    )
    db_session.flush()
    yield


def _make_instrument(
    exchange_id: uuid.UUID,
    ticker: str,
    yf_ticker: str,
) -> Instrument:
    return Instrument(
        ticker=ticker,
        short_name=ticker,
        name=f"{ticker} Inc.",
        instrument_type="STOCK",
        currency_code="EUR",
        yfinance_ticker=yf_ticker,
        exchange_id=exchange_id,
    )


# ---------------------------------------------------------------------------
# Tier 2 — TickerProfile fallback via yfinance_ticker
# ---------------------------------------------------------------------------


class TestResolveSectorMapTickerProfileFallback:
    """Without a snapshot mapping, the resolver queries TickerProfile."""

    def test_european_yfinance_tickers_resolve(
        self, db_session: Session, seeded_instruments: None
    ) -> None:
        """Regression for #427: `.PA` tickers now resolve via yfinance_ticker."""
        from app.services._shared import resolve_sector_map

        result = resolve_sector_map(
            session=db_session,
            tickers=["ENGI.PA", "ORA.PA"],
        )

        assert result == {
            "ENGI.PA": "Utilities",
            "ORA.PA": "Communication Services",
        }

    def test_us_ticker_resolves(
        self, db_session: Session, seeded_instruments: None
    ) -> None:
        from app.services._shared import resolve_sector_map

        result = resolve_sector_map(session=db_session, tickers=["AAPL"])

        assert result == {"AAPL": "Technology"}

    def test_unmapped_ticker_falls_back_to_unclassified(
        self, db_session: Session, seeded_instruments: None
    ) -> None:
        from app.services._shared import UNCLASSIFIED, resolve_sector_map

        result = resolve_sector_map(
            session=db_session,
            tickers=["EMPTY.PA", "GHOST.XX"],
        )

        assert result == {
            "EMPTY.PA": UNCLASSIFIED,
            "GHOST.XX": UNCLASSIFIED,
        }


# ---------------------------------------------------------------------------
# Tier 1 — snapshot mapping preferred
# ---------------------------------------------------------------------------


class TestResolveSectorMapSnapshotFirst:
    """Snapshot entries take precedence; the resolver only hits the DB for gaps."""

    def test_snapshot_entries_override_profile(
        self, db_session: Session, seeded_instruments: None
    ) -> None:
        from app.services._shared import resolve_sector_map

        snapshot_mapping = {"ENGI.PA": "Custom-Sector"}

        result = resolve_sector_map(
            session=db_session,
            tickers=["ENGI.PA", "ORA.PA"],
            snapshot_mapping=snapshot_mapping,
        )

        assert result["ENGI.PA"] == "Custom-Sector"  # from snapshot
        assert result["ORA.PA"] == "Communication Services"  # from DB fallback

    def test_missing_tickers_still_default_to_unclassified(
        self, db_session: Session, seeded_instruments: None
    ) -> None:
        from app.services._shared import UNCLASSIFIED, resolve_sector_map

        snapshot_mapping = {"ENGI.PA": "Utilities"}

        result = resolve_sector_map(
            session=db_session,
            tickers=["ENGI.PA", "GHOST.XX"],
            snapshot_mapping=snapshot_mapping,
        )

        assert result == {
            "ENGI.PA": "Utilities",
            "GHOST.XX": UNCLASSIFIED,
        }


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestResolveSectorMapEdgeCases:
    def test_empty_tickers_returns_empty_map(
        self, db_session: Session, seeded_instruments: None
    ) -> None:
        from app.services._shared import resolve_sector_map

        result = resolve_sector_map(session=db_session, tickers=[])

        assert result == {}

    def test_none_sector_in_profile_falls_back_to_unclassified(
        self, db_session: Session
    ) -> None:
        """A NULL ``sector`` column must not override with empty string."""
        from app.services._shared import UNCLASSIFIED, resolve_sector_map

        exchange = Exchange(name="TEST_NULL_SECTOR_EXCH")
        db_session.add(exchange)
        db_session.flush()

        inst = _make_instrument(exchange.id, ticker="NULLp_EQ", yf_ticker="NULL.PA")
        db_session.add(inst)
        db_session.flush()
        db_session.add(TickerProfile(instrument_id=inst.id, sector=None))
        db_session.flush()

        result = resolve_sector_map(session=db_session, tickers=["NULL.PA"])

        assert result == {"NULL.PA": UNCLASSIFIED}
