"""Integration tests for POST /api/v1/attribution/brinson (issue #427).

Seeds a ``trading212``-style portfolio with European (``.PA``) tickers whose
sector data lives only in ``TickerProfile`` (not in any snapshot mapping).
Before the fix the service queried ``Instrument.ticker`` and missed every
ticker; after the fix the shared resolver joins via ``yfinance_ticker`` and
the endpoint returns 200 with a populated ``sectors`` list.
"""

from __future__ import annotations

from collections.abc import Generator
from datetime import date, timedelta

import pytest
from app.models.universe import Exchange, Instrument
from app.models.yfinance_data import PriceHistory, TickerProfile
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

BASE = "/api/v1/attribution/brinson"


@pytest.fixture
def seeded_trading212_like(db_session: Session) -> Generator[None, None, None]:
    """Seed instruments + ticker_profiles + price_history for ENGI.PA / ORA.PA / AAPL."""
    exchange = Exchange(name="TEST_ATTR_EXCH")
    db_session.add(exchange)
    db_session.flush()

    tickers_data = [
        ("ENGIp_EQ", "ENGI.PA", "Utilities"),
        ("ORAp_EQ", "ORA.PA", "Communication Services"),
        ("AAPL", "AAPL", "Technology"),
    ]
    instruments: list[Instrument] = []
    for ticker, yf_ticker, sector in tickers_data:
        inst = Instrument(
            ticker=ticker,
            short_name=ticker,
            name=f"{ticker} Inc.",
            instrument_type="STOCK",
            currency_code="EUR",
            yfinance_ticker=yf_ticker,
            exchange_id=exchange.id,
        )
        db_session.add(inst)
        instruments.append(inst)
    db_session.flush()

    for inst, (_, _, sector) in zip(instruments, tickers_data, strict=True):
        db_session.add(TickerProfile(instrument_id=inst.id, sector=sector))
    db_session.flush()

    start = date(2025, 1, 2)
    dates = [start + timedelta(days=i) for i in range(10)]
    for inst in instruments:
        for i, d in enumerate(dates):
            db_session.add(
                PriceHistory(
                    instrument_id=inst.id,
                    date=d,
                    close=100.0 + i,
                    volume=1_000_000,
                )
            )
    db_session.flush()
    yield


def test_brinson_returns_200_with_european_yfinance_tickers(
    client: TestClient, seeded_trading212_like: None
) -> None:
    """Regression for #427: .PA tickers must resolve via yfinance_ticker."""
    resp = client.post(
        BASE,
        json={
            "portfolio_weights": {"ENGI.PA": 0.5, "ORA.PA": 0.5},
            "benchmark_weights": {"ENGI.PA": 0.3, "ORA.PA": 0.4, "AAPL": 0.3},
            "start_date": "2025-01-02",
            "end_date": "2025-01-11",
        },
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    sectors = body["sectors"]
    assert sectors, "sectors list must be non-empty"
    sector_names = {row["sector"] for row in sectors}
    assert "Utilities" in sector_names
    assert "Communication Services" in sector_names


def test_brinson_preserves_422_when_coverage_truly_low(
    client: TestClient, seeded_trading212_like: None
) -> None:
    """Guard: 422 must still fire when the coverage is genuinely low.

    All three weight keys are unknown tickers that exist in neither
    ``TickerProfile`` nor any snapshot mapping — coverage is 0%, well below
    the 50% threshold.
    """
    resp = client.post(
        BASE,
        json={
            "portfolio_weights": {
                "GHOST1.XX": 0.4,
                "GHOST2.XX": 0.3,
                "GHOST3.XX": 0.3,
            },
            "benchmark_weights": {"ENGI.PA": 1.0},
            "start_date": "2025-01-02",
            "end_date": "2025-01-11",
        },
    )

    assert resp.status_code == 422, resp.text
