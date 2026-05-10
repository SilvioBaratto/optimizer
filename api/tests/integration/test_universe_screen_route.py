"""Integration tests for POST /api/v1/universe/screen (issue #426).

Reproduces the production bug against the test SQLite DB: two ``instruments``
rows mapped to the same ``yfinance_ticker`` produce duplicate ``(ticker, date)``
tuples in the ``price_history`` join, which made ``_assemble_price_volume``
crash on ``df.pivot`` and surface as HTTP 422 ``Index contains duplicate
entries, cannot reshape``.

After the fix the service deduplicates at the hydration boundary; both empty
and populated request bodies return 200.
"""

from __future__ import annotations

import uuid
from collections.abc import Generator
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.market_data.yfinance_data import PriceHistory, TickerProfile
from app.models.universe.universe import Exchange, Instrument

BASE = "/api/v1/universe/screen"


@pytest.fixture
def seeded_universe_with_duplicates(
    db_session: Session,
) -> Generator[None, None, None]:
    """Seed two instruments sharing the same yfinance_ticker ('DUP').

    This mirrors the production state (e.g. ``ADM`` listed on two exchanges) that
    breaks the pivot without deduplication.
    """
    exchange_a = Exchange(name="EXCH_A_UNIV")
    exchange_b = Exchange(name="EXCH_B_UNIV")
    db_session.add_all([exchange_a, exchange_b])
    db_session.flush()

    inst_a = _make_instrument(exchange_a.id, ticker="DUP-A", yf_ticker="DUP")
    inst_b = _make_instrument(exchange_b.id, ticker="DUP-B", yf_ticker="DUP")
    other = _make_instrument(exchange_a.id, ticker="OTHER", yf_ticker="OTHER")
    db_session.add_all([inst_a, inst_b, other])
    db_session.flush()

    db_session.add_all(
        [
            TickerProfile(
                instrument_id=inst_a.id, market_cap=2_000_000_000, current_price=50.0
            ),
            TickerProfile(
                instrument_id=inst_b.id, market_cap=2_100_000_000, current_price=50.5
            ),
            TickerProfile(
                instrument_id=other.id, market_cap=5_000_000_000, current_price=120.0
            ),
        ]
    )

    dates = [date(2024, 1, 2) + timedelta(days=i) for i in range(5)]
    db_session.add_all(
        [
            PriceHistory(
                instrument_id=inst_a.id, date=d, close=50.0 + i, volume=1_000_000
            )
            for i, d in enumerate(dates)
        ]
    )
    db_session.add_all(
        [
            PriceHistory(
                instrument_id=inst_b.id, date=d, close=50.5 + i, volume=1_100_000
            )
            for i, d in enumerate(dates)
        ]
    )
    db_session.add_all(
        [
            PriceHistory(
                instrument_id=other.id, date=d, close=120.0 + i, volume=2_000_000
            )
            for i, d in enumerate(dates)
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
        currency_code="USD",
        yfinance_ticker=yf_ticker,
        exchange_id=exchange_id,
    )


def test_empty_body_returns_200_with_duplicate_instruments(
    client: TestClient,
    seeded_universe_with_duplicates: None,
) -> None:
    """Regression for #426: empty body used to surface 422 from the pivot crash."""
    resp = client.post(BASE, json={})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "passing_tickers" in body or "passingTickers" in body


def test_preset_body_returns_200_with_duplicate_instruments(
    client: TestClient,
    seeded_universe_with_duplicates: None,
) -> None:
    """Preset request must succeed over a universe with duplicate (ticker, date)."""
    resp = client.post(BASE, json={"preset": "broad_universe"})

    assert resp.status_code == 200, resp.text


def test_empty_db_returns_200_with_empty_lists(client: TestClient) -> None:
    """Empty DB + empty body → valid empty response."""
    resp = client.post(BASE, json={})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    tickers = body.get("passing_tickers", body.get("passingTickers"))
    total = body.get("total_screened", body.get("totalScreened"))
    assert tickers == []
    assert total == 0
