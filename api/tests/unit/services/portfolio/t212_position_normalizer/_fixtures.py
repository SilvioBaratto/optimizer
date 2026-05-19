"""Shared T212 payload fixtures for Cycle 1 normalizer e2e tests (issue #778)."""

from __future__ import annotations

from datetime import date
from typing import Any

AS_OF: date = date(2026, 5, 19)

_FIXTURE_ACCOUNT_INFO: dict[str, Any] = {
    "currencyCode": "EUR",
    "id": 12345,
}

_FIXTURE_ACCOUNT_INFO_GBP: dict[str, Any] = {
    "currencyCode": "GBP",
    "id": 12345,
}

# invested ≈ sum of expected eur_value for the EUR-base fixture
_FIXTURE_ACCOUNT_CASH: dict[str, Any] = {
    "total": 50_000.0,
    "free": 1_000.0,
    "invested": 25_805.0,  # tuned to be within 1.5% of computed total below
    "blocked": None,
    "result": 500.0,
    "pieCash": 0.0,
}

# 8 positions covering: _US_EQ, _l_EQ (GBX), _p_EQ, _pp_EQ, _d_EQ, _m_EQ,
# unmapped (`FAKE_XX_EQ` → no DB row, suffix unrecognized),
# missing-FX (CHF currency, provider returns None).
_FIXTURE_PORTFOLIO: list[dict[str, Any]] = [
    {
        "ticker": "AAPL_US_EQ",
        "quantity": 10.0,
        "averagePrice": 170.0,
        "currentPrice": 180.0,
        "currencyCode": "USD",
        "ppl": 100.0,
        "fxPpl": -5.0,
        "initialFillDate": "2024-01-10T10:00:00Z",
    },
    {
        "ticker": "MSFT_US_EQ",
        "quantity": 5.0,
        "averagePrice": 350.0,
        "currentPrice": 400.0,
        "currencyCode": "USD",
        "ppl": 250.0,
        "fxPpl": -10.0,
        "initialFillDate": "2023-09-15T10:00:00Z",
    },
    {
        "ticker": "VOD_l_EQ",
        "quantity": 100.0,
        "averagePrice": 7000.0,
        "currentPrice": 7500.0,  # 75.00 GBP in pence
        "currencyCode": "GBX",
        "ppl": 50.0,
        "fxPpl": 3.0,
        "initialFillDate": "2024-02-01T10:00:00Z",
    },
    {
        "ticker": "AIR_p_EQ",
        "quantity": 20.0,
        "averagePrice": 145.0,
        "currentPrice": 150.0,
        "currencyCode": "EUR",
        "ppl": 100.0,
        "fxPpl": 0.0,
        "initialFillDate": "2024-03-15T10:00:00Z",
    },
    {
        "ticker": "AIRp_pp_EQ",
        "quantity": 15.0,
        "averagePrice": 95.0,
        "currentPrice": 100.0,
        "currencyCode": "EUR",
        "ppl": 75.0,
        "fxPpl": 0.0,
        "initialFillDate": "2024-04-01T10:00:00Z",
    },
    {
        "ticker": "SAP_d_EQ",
        "quantity": 8.0,
        "averagePrice": 140.0,
        "currentPrice": 150.0,
        "currencyCode": "EUR",
        "ppl": 80.0,
        "fxPpl": 0.0,
        "initialFillDate": "2024-05-10T10:00:00Z",
    },
    {
        "ticker": "ENI_m_EQ",
        "quantity": 50.0,
        "averagePrice": 14.0,
        "currentPrice": 15.0,
        "currencyCode": "EUR",
        "ppl": 50.0,
        "fxPpl": 0.0,
        "initialFillDate": "2024-06-01T10:00:00Z",
    },
    {
        "ticker": "FAKE_XX_EQ",
        "quantity": 3.0,
        "averagePrice": 10.0,
        "currentPrice": 12.0,
        "currencyCode": "CHF",  # FX provider returns None for CHF
        "ppl": 6.0,
        "fxPpl": 0.0,
        "initialFillDate": "2024-07-01T10:00:00Z",
    },
]

# DB-seedable T212 → yfinance ticker mappings.
# Excludes `FAKE_XX_EQ` deliberately so it stays unmapped.
_INSTRUMENT_SEED: dict[str, str] = {
    "AAPL_US_EQ": "AAPL",
    "MSFT_US_EQ": "MSFT",
    "VOD_l_EQ": "VOD.L",
    "AIR_p_EQ": "AIR.PA",
    "AIRp_pp_EQ": "AIRp.PA",
    "SAP_d_EQ": "SAP.DE",
    "ENI_m_EQ": "ENI.MI",
}

# Per-currency FX rates (rate to base EUR).
_FX_RATES: dict[tuple[str, date], float | None] = {
    ("USD", AS_OF): 0.92,
    ("GBP", AS_OF): 1.17,
    ("CHF", AS_OF): None,  # forces FX_MISSING flag
}


def make_fx_provider() -> Any:
    """Mock FxRateProvider that looks up rates from _FX_RATES."""
    from unittest.mock import MagicMock

    fx = MagicMock()

    def lookup(from_ccy: str, as_of: date) -> float | None:
        return _FX_RATES.get((from_ccy.upper(), as_of))

    fx.get_rate_to_base.side_effect = lookup
    return fx


def make_client(
    *,
    positions: list[dict[str, Any]] | None = None,
    account_info: dict[str, Any] | None = None,
    account_cash: dict[str, Any] | None = None,
) -> Any:
    from unittest.mock import MagicMock

    client = MagicMock()
    client.get_account_info.return_value = account_info or _FIXTURE_ACCOUNT_INFO
    client.get_account_cash.return_value = account_cash or _FIXTURE_ACCOUNT_CASH
    client.get_portfolio_positions.return_value = (
        positions if positions is not None else _FIXTURE_PORTFOLIO
    )
    return client


def seed_instruments(session: Any) -> None:
    """Seed `Instrument` rows for mapped fixture tickers."""
    from app.models.universe.universe import Exchange, Instrument

    exchange = Exchange(name="T212_TEST_EXCHANGE", t212_id=999)
    session.add(exchange)
    session.flush()

    for t212_ticker, yf_ticker in _INSTRUMENT_SEED.items():
        session.add(
            Instrument(
                ticker=t212_ticker,
                short_name=t212_ticker.split("_")[0],
                yfinance_ticker=yf_ticker,
                exchange_id=exchange.id,
            )
        )
    session.flush()
