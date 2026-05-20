"""Route-level fixtures + no-silent-drops audit for GET /portfolio/{name}/drift (#797).

Each test exercises ONE diagnostic category in isolation against the new
portfolio drift route (`/api/v1/portfolio/{name}/drift`, defined at
`api/app/api/v1/portfolio/portfolio.py:392-414`). This is NOT the dashboard
drift route (`/api/v1/portfolio-analytics/{name}/drift`).
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.models.universe.universe import Exchange, Instrument
from app.schemas.portfolio import PositionFlag

BASE_URL = "/api/v1/portfolio"
_PORTFOLIO_REPO = "app.api.v1.portfolio.portfolio.PortfolioRepository"
_FX_PROVIDER = (
    "app.services.portfolio.t212_position_normalizer.normalizer_service."
    "FxRateProvider"
)
_T212_API_KEY_PATCH = "app.api.v1.portfolio.portfolio.settings"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fake_portfolio() -> MagicMock:
    p = MagicMock()
    p.id = uuid.uuid4()
    p.name = "test"
    return p


def _fake_snapshot(weights: dict[str, float]) -> MagicMock:
    s = MagicMock()
    s.weights = dict(weights)
    return s


def _seed_instruments(session, mapping: dict[str, str]) -> None:
    exchange = Exchange(name="T212_TEST_EXCHANGE", t212_id=999)
    session.add(exchange)
    session.flush()
    for t212, yf in mapping.items():
        session.add(
            Instrument(
                ticker=t212,
                short_name=t212.split("_")[0],
                yfinance_ticker=yf,
                exchange_id=exchange.id,
            )
        )
    session.flush()


class FakeFxProvider:
    """In-process FX provider — no yfinance calls."""

    def __init__(
        self,
        base_currency: str = "EUR",
        *,
        rates: dict[str, float | None] | None = None,
    ) -> None:
        self.base_currency = base_currency
        self._rates = rates or {"USD": 0.92, "GBP": 1.17, "CHF": None}

    def get_rate_to_base(self, from_ccy: str, _as_of: object) -> float | None:
        normalized = from_ccy.upper()
        if normalized == self.base_currency.upper():
            return 1.0
        return self._rates.get(normalized)

    def clear_cache(self) -> None:  # pragma: no cover — interface parity
        pass


def _fake_t212_client(
    *,
    positions: list[dict[str, Any]],
    instruments: list[dict[str, str]] | None = None,
    invested: float = 1000.0,
    total: float | None = None,
) -> MagicMock:
    c = MagicMock()
    c.get_portfolio_positions.return_value = positions
    c.get_instruments.return_value = instruments or []
    c.get_account_info.return_value = {"currencyCode": "EUR", "id": 1}
    c.get_account_cash.return_value = {
        "invested": invested,
        "total": total if total is not None else invested,
    }
    c.api_key = "SECRET_API_KEY_DO_NOT_LEAK"
    c.api_secret = "SECRET_API_SECRET_DO_NOT_LEAK"  # noqa: S105
    return c


@pytest.fixture
def override_t212(client: TestClient) -> Iterator[Any]:
    from app.api.v1.portfolio.portfolio import get_t212_client
    from app.main import app

    holder: dict[str, MagicMock] = {}

    def install(fake: MagicMock) -> MagicMock:
        app.dependency_overrides[get_t212_client] = lambda: fake
        holder["client"] = fake
        return fake

    try:
        yield install
    finally:
        app.dependency_overrides.pop(get_t212_client, None)


@pytest.fixture
def configure_t212_key() -> Iterator[None]:
    with patch(_T212_API_KEY_PATCH) as mock_settings:
        mock_settings.trading_212_api_key = "test-key"
        mock_settings.trading_212_secret_key = "test-secret"  # noqa: S105
        mock_settings.trading_212_mode = "live"
        yield


@pytest.fixture
def patch_portfolio_repo() -> Iterator[Any]:
    with patch(_PORTFOLIO_REPO) as MockRepo:

        def install(target_weights: dict[str, float]) -> MagicMock:
            MockRepo.return_value.get_by_name.return_value = _fake_portfolio()
            MockRepo.return_value.get_latest_snapshot.return_value = (
                _fake_snapshot(target_weights)
            )
            return MockRepo

        yield install


@pytest.fixture
def patch_fx() -> Iterator[Any]:
    with patch(_FX_PROVIDER, FakeFxProvider):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _raw_pos(
    ticker: str,
    *,
    quantity: float = 10.0,
    current_price: float | None = 100.0,
    currency: str = "EUR",
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "quantity": quantity,
        "currentPrice": current_price,
        "currencyCode": currency,
        "ppl": 0.0,
        "fxPpl": 0.0,
    }


def _call_drift(client: TestClient, name: str = "test") -> Any:
    return client.get(f"{BASE_URL}/{name}/drift")


def _entries_for(payload: dict[str, Any], code: PositionFlag) -> list[dict[str, Any]]:
    return [
        e for e in payload["diagnostics"]["entries"] if e["code"] == code.value
    ]


# ---------------------------------------------------------------------------
# UNMAPPED
# ---------------------------------------------------------------------------


class TestUnmappedCategory:
    def test_when_raw_ticker_has_no_instrument_then_unmapped_entry_emitted(
        self,
        client: TestClient,
        db_session,
        override_t212,
        patch_portfolio_repo,
        patch_fx,
        configure_t212_key,
    ) -> None:
        _seed_instruments(db_session, {"AAPL_US_EQ": "AAPL"})
        positions = [
            _raw_pos("AAPL_US_EQ"),
            _raw_pos("GHOST_XX_EQ", currency="EUR"),
        ]
        override_t212(_fake_t212_client(positions=positions, invested=2000.0))
        patch_portfolio_repo({"AAPL": 1.0})

        resp = _call_drift(client)

        assert resp.status_code == 200, resp.text
        body = resp.json()
        unmapped_entries = _entries_for(body, PositionFlag.UNMAPPED)
        assert any(e["reference"] == "GHOST_XX_EQ" for e in unmapped_entries)
        assert body["diagnostics"]["unmapped_count"] >= 1

    def test_when_unmapped_position_then_row_remains_visible_in_holdings(
        self,
        client: TestClient,
        db_session,
        override_t212,
        patch_portfolio_repo,
        patch_fx,
        configure_t212_key,
    ) -> None:
        _seed_instruments(db_session, {"AAPL_US_EQ": "AAPL"})
        positions = [
            _raw_pos("AAPL_US_EQ"),
            _raw_pos("GHOST_XX_EQ", currency="EUR"),
        ]
        override_t212(_fake_t212_client(positions=positions, invested=2000.0))
        patch_portfolio_repo({"AAPL": 1.0})

        body = _call_drift(client).json()

        # Both positions present in holdings (greyed via flags, NOT hidden)
        assert len(body["holdings"]) == 2


# ---------------------------------------------------------------------------
# FX_MISSING
# ---------------------------------------------------------------------------


class TestFxMissingCategory:
    def test_when_fx_rate_missing_then_fx_missing_entry_emitted(
        self,
        client: TestClient,
        db_session,
        override_t212,
        patch_portfolio_repo,
        patch_fx,
        configure_t212_key,
    ) -> None:
        _seed_instruments(db_session, {"NESN_s_EQ": "NESN.SW"})
        positions = [_raw_pos("NESN_s_EQ", currency="CHF", current_price=100.0)]
        override_t212(_fake_t212_client(positions=positions, invested=0.0))
        patch_portfolio_repo({"NESN.SW": 1.0})

        body = _call_drift(client).json()

        entries = _entries_for(body, PositionFlag.FX_MISSING)
        assert any(e["reference"] == "CHF" for e in entries)
        assert body["diagnostics"]["fx_missing_count"] >= 1


# ---------------------------------------------------------------------------
# STALE_PRICE
# ---------------------------------------------------------------------------


class TestStalePriceCategory:
    def test_when_current_price_zero_then_stale_price_entry_emitted(
        self,
        client: TestClient,
        db_session,
        override_t212,
        patch_portfolio_repo,
        patch_fx,
        configure_t212_key,
    ) -> None:
        _seed_instruments(db_session, {"AAPL_US_EQ": "AAPL"})
        positions = [
            _raw_pos("AAPL_US_EQ", current_price=0.0, currency="USD"),
        ]
        override_t212(_fake_t212_client(positions=positions, invested=0.0))
        patch_portfolio_repo({"AAPL": 1.0})

        body = _call_drift(client).json()

        entries = _entries_for(body, PositionFlag.STALE_PRICE)
        assert entries
        assert body["diagnostics"]["stale_price_count"] >= 1


# ---------------------------------------------------------------------------
# TARGET_NOT_ON_BROKER
# ---------------------------------------------------------------------------


class TestTargetNotOnBrokerCategory:
    def test_when_target_absent_from_broker_universe_then_entry_emitted(
        self,
        client: TestClient,
        db_session,
        override_t212,
        patch_portfolio_repo,
        patch_fx,
        configure_t212_key,
    ) -> None:
        _seed_instruments(db_session, {"AAPL_US_EQ": "AAPL"})
        positions = [_raw_pos("AAPL_US_EQ", currency="USD")]
        # `get_instruments` returns [] so no ticker is on broker
        override_t212(
            _fake_t212_client(positions=positions, instruments=[], invested=900.0)
        )
        patch_portfolio_repo({"AAPL": 0.5, "GHOST": 0.5})

        body = _call_drift(client).json()

        entries = _entries_for(body, PositionFlag.TARGET_NOT_ON_BROKER)
        assert any(e["ticker"] == "GHOST" for e in entries)
        assert body["diagnostics"]["target_not_on_broker_count"] >= 1

    def test_when_target_not_on_broker_then_drift_row_visible(
        self,
        client: TestClient,
        db_session,
        override_t212,
        patch_portfolio_repo,
        patch_fx,
        configure_t212_key,
    ) -> None:
        _seed_instruments(db_session, {"AAPL_US_EQ": "AAPL"})
        positions = [_raw_pos("AAPL_US_EQ", currency="USD")]
        override_t212(
            _fake_t212_client(positions=positions, instruments=[], invested=900.0)
        )
        patch_portfolio_repo({"AAPL": 0.5, "GHOST": 0.5})

        body = _call_drift(client).json()

        tickers = [row["ticker"] for row in body["drift"]]
        assert "GHOST" in tickers, "target ticker must remain visible (greyed, not hidden)"


# ---------------------------------------------------------------------------
# RECONCILIATION_MISMATCH
# ---------------------------------------------------------------------------


class TestReconciliationMismatchCategory:
    def test_when_sum_diverges_beyond_tolerance_then_reconciliation_aggregate_populated(
        self,
        client: TestClient,
        db_session,
        override_t212,
        patch_portfolio_repo,
        patch_fx,
        configure_t212_key,
    ) -> None:
        _seed_instruments(db_session, {"AAPL_US_EQ": "AAPL"})
        # eur_value = 10 * 100 = 1000. invested = 500. delta = 1.0 → > 1.5%
        positions = [
            _raw_pos("AAPL_US_EQ", quantity=10.0, current_price=100.0, currency="EUR"),
        ]
        override_t212(_fake_t212_client(positions=positions, invested=500.0))
        patch_portfolio_repo({"AAPL": 1.0})

        body = _call_drift(client).json()
        diag = body["diagnostics"]

        assert diag["reconciliation_ok"] is False
        assert diag["sum_eur"] == pytest.approx(1000.0)
        assert diag["invested_eur"] == pytest.approx(500.0)
        assert diag["delta_eur"] == pytest.approx(500.0)
        assert diag["tolerance_pct"] == pytest.approx(0.015)
        recon_entries = _entries_for(body, PositionFlag.RECONCILIATION_MISMATCH)
        assert recon_entries  # at least one (per-position + aggregate may both fire)


# ---------------------------------------------------------------------------
# No-silent-drops audit
# ---------------------------------------------------------------------------


class TestNoSilentDropsInvariant:
    def test_when_some_positions_malformed_then_unmapped_entries_cover_drops(
        self,
        client: TestClient,
        db_session,
        override_t212,
        patch_portfolio_repo,
        patch_fx,
        configure_t212_key,
    ) -> None:
        """len(input) == len(holdings) + len([e for e in entries if UNMAPPED-from-drop])."""
        _seed_instruments(
            db_session, {"AAPL_US_EQ": "AAPL", "MSFT_US_EQ": "MSFT"}
        )
        # 3 raw inputs: 2 valid, 1 malformed (no `quantity`)
        positions: list[dict[str, Any]] = [
            _raw_pos("AAPL_US_EQ", currency="USD"),
            _raw_pos("MSFT_US_EQ", currency="USD"),
            {
                "ticker": "MAL_FORMED",
                "currentPrice": 100.0,
                "currencyCode": "EUR",
            },
        ]
        override_t212(_fake_t212_client(positions=positions, invested=2000.0))
        patch_portfolio_repo({"AAPL": 0.5, "MSFT": 0.5})

        body = _call_drift(client).json()
        unmapped_from_drops = [
            e
            for e in body["diagnostics"]["entries"]
            if e["code"] == PositionFlag.UNMAPPED.value
            and e["ticker"] == "MAL_FORMED"
        ]
        assert len(body["holdings"]) + len(unmapped_from_drops) == len(positions)


# ---------------------------------------------------------------------------
# Credential safety
# ---------------------------------------------------------------------------


class TestCredentialsNeverLeakInResponse:
    def test_when_response_returned_then_api_key_not_in_payload(
        self,
        client: TestClient,
        db_session,
        override_t212,
        patch_portfolio_repo,
        patch_fx,
        configure_t212_key,
    ) -> None:
        _seed_instruments(db_session, {"AAPL_US_EQ": "AAPL"})
        positions = [_raw_pos("AAPL_US_EQ", currency="USD")]
        override_t212(_fake_t212_client(positions=positions, invested=900.0))
        patch_portfolio_repo({"AAPL": 1.0})

        resp = _call_drift(client)

        assert resp.status_code == 200
        text = resp.text
        assert "SECRET_API_KEY_DO_NOT_LEAK" not in text
        assert "SECRET_API_SECRET_DO_NOT_LEAK" not in text
        assert "api_key" not in text
