"""Route-level unit tests for GET /portfolio/{name}/drift (issue #784)."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest
import requests
from fastapi.testclient import TestClient

from app.schemas.portfolio import (
    BaseToggle,
    DriftDiagnostics,
    DriftResponse,
    DriftTotals,
)

_PORTFOLIO_REPO = "app.api.v1.portfolio.portfolio.PortfolioRepository"
_COMPUTE_DRIFT = "app.api.v1.portfolio.portfolio.compute_drift"

BASE_URL = "/api/v1/portfolio"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_portfolio() -> MagicMock:
    p = MagicMock()
    p.id = uuid.uuid4()
    p.name = "test"
    return p


def _make_snapshot() -> MagicMock:
    s = MagicMock()
    s.weights = {"AAPL": 0.5, "MSFT": 0.5}
    return s


def _drift_response(request_id: int = 0) -> DriftResponse:
    return DriftResponse(
        holdings=[],
        target=[],
        drift=[],
        trades=[],
        totals=DriftTotals(
            deployable_eur=10_000.0,
            total_holdings_eur=10_000.0,
            total_drift_abs=0.0,
            buy_eur=0.0,
            sell_eur=0.0,
        ),
        diagnostics=DriftDiagnostics(
            reconciliation_ok=True,
            reconciliation_delta_pct=0.0,
            unmapped_count=0,
            fx_missing_count=0,
            target_not_on_broker_count=0,
            base_currency="EUR",
        ),
        request_id=request_id,
    )


def _mock_t212_client_dep() -> MagicMock:
    return MagicMock()


def _override_t212():
    from app.api.v1.portfolio.portfolio import get_t212_client
    from app.main import app

    app.dependency_overrides[get_t212_client] = _mock_t212_client_dep
    return get_t212_client


def _clear_t212(dep):
    from app.main import app

    app.dependency_overrides.pop(dep, None)


# ---------------------------------------------------------------------------
# 200 success
# ---------------------------------------------------------------------------


class TestDriftSuccess:
    def test_when_called_then_response_has_all_top_level_fields(
        self, client: TestClient
    ) -> None:
        dep = _override_t212()
        try:
            with (
                patch(_PORTFOLIO_REPO) as MockRepo,
                patch(_COMPUTE_DRIFT, return_value=_drift_response()),
            ):
                MockRepo.return_value.get_by_name.return_value = _make_portfolio()
                MockRepo.return_value.get_latest_snapshot.return_value = (
                    _make_snapshot()
                )
                resp = client.get(f"{BASE_URL}/test/drift")
        finally:
            _clear_t212(dep)

        assert resp.status_code == 200
        body = resp.json()
        for key in (
            "holdings",
            "target",
            "drift",
            "trades",
            "totals",
            "diagnostics",
            "request_id",
        ):
            assert key in body

    def test_when_header_set_then_request_id_echoed(
        self, client: TestClient
    ) -> None:
        dep = _override_t212()
        try:
            with (
                patch(_PORTFOLIO_REPO) as MockRepo,
                patch(_COMPUTE_DRIFT) as mock_compute,
            ):
                MockRepo.return_value.get_by_name.return_value = _make_portfolio()
                MockRepo.return_value.get_latest_snapshot.return_value = (
                    _make_snapshot()
                )
                mock_compute.side_effect = lambda *_a, **kw: _drift_response(
                    request_id=kw["request_id"]
                )
                resp = client.get(
                    f"{BASE_URL}/test/drift",
                    headers={"X-Drift-Request-Id": "7"},
                )
        finally:
            _clear_t212(dep)

        assert resp.status_code == 200
        assert resp.json()["request_id"] == 7


# ---------------------------------------------------------------------------
# Base toggle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query, expected_base",
    [
        ("?base=total", BaseToggle.TOTAL),
        ("?base=invested", BaseToggle.INVESTED),
        ("", BaseToggle.INVESTED),  # default
    ],
)
def test_when_base_param_then_compute_drift_called_with_enum(
    query, expected_base, client: TestClient
) -> None:
    dep = _override_t212()
    try:
        with (
            patch(_PORTFOLIO_REPO) as MockRepo,
            patch(_COMPUTE_DRIFT, return_value=_drift_response()) as mock_compute,
        ):
            MockRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockRepo.return_value.get_latest_snapshot.return_value = (
                _make_snapshot()
            )
            resp = client.get(f"{BASE_URL}/test/drift{query}")
    finally:
        _clear_t212(dep)

    assert resp.status_code == 200
    assert mock_compute.call_args.kwargs["base"] is expected_base


# ---------------------------------------------------------------------------
# Error codes
# ---------------------------------------------------------------------------


class TestDriftErrors:
    def test_when_portfolio_missing_then_404(self, client: TestClient) -> None:
        dep = _override_t212()
        try:
            with patch(_PORTFOLIO_REPO) as MockRepo:
                MockRepo.return_value.get_by_name.return_value = None
                resp = client.get(f"{BASE_URL}/ghost/drift")
        finally:
            _clear_t212(dep)

        assert resp.status_code == 404
        assert "Portfolio 'ghost' not found" in resp.text

    def test_when_snapshot_missing_then_404(self, client: TestClient) -> None:
        dep = _override_t212()
        try:
            with patch(_PORTFOLIO_REPO) as MockRepo:
                MockRepo.return_value.get_by_name.return_value = _make_portfolio()
                MockRepo.return_value.get_latest_snapshot.return_value = None
                resp = client.get(f"{BASE_URL}/test/drift")
        finally:
            _clear_t212(dep)

        assert resp.status_code == 404
        assert "No snapshot found; run optimization first" in resp.text

    def test_when_t212_key_unset_then_503(self, client: TestClient) -> None:
        with patch("app.api.v1.portfolio.portfolio.settings") as mock_settings:
            mock_settings.trading_212_api_key = ""
            resp = client.get(f"{BASE_URL}/test/drift")
        assert resp.status_code == 503

    def test_when_compute_drift_raises_request_exception_then_502(
        self, client: TestClient
    ) -> None:
        dep = _override_t212()
        try:
            with (
                patch(_PORTFOLIO_REPO) as MockRepo,
                patch(
                    _COMPUTE_DRIFT,
                    side_effect=requests.RequestException("upstream down"),
                ),
            ):
                MockRepo.return_value.get_by_name.return_value = _make_portfolio()
                MockRepo.return_value.get_latest_snapshot.return_value = (
                    _make_snapshot()
                )
                resp = client.get(f"{BASE_URL}/test/drift")
        finally:
            _clear_t212(dep)

        assert resp.status_code == 502
        assert "upstream down" in resp.text


# ---------------------------------------------------------------------------
# Credential leakage
# ---------------------------------------------------------------------------


def test_when_response_returned_then_t212_api_key_not_in_body(
    client: TestClient,
) -> None:
    dep = _override_t212()
    secret = "super-secret-t212-key-do-not-leak"  # noqa: S105
    try:
        with (
            patch("app.api.v1.portfolio.portfolio.settings") as mock_settings,
            patch(_PORTFOLIO_REPO) as MockRepo,
            patch(_COMPUTE_DRIFT, return_value=_drift_response()),
        ):
            mock_settings.trading_212_api_key = secret
            MockRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockRepo.return_value.get_latest_snapshot.return_value = (
                _make_snapshot()
            )
            resp = client.get(f"{BASE_URL}/test/drift")
    finally:
        _clear_t212(dep)

    assert resp.status_code == 200
    assert secret not in resp.text
