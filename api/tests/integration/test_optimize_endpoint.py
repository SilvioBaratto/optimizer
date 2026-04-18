"""Integration tests for POST /api/v1/optimize.

Exercises the full route → service → optimizer library → response flow with
real skfolio wiring. The DB price fetcher and persistence layer are mocked
so the optimiser fits against a deterministic synthetic price series
without leaking rows into downstream tests.

Regression coverage for issue #419 — duplicate ``risk_measure`` kwarg.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from app.models.execution import OptimizationRun
from fastapi.testclient import TestClient

BASE_URL = "/api/v1/optimize"
_TICKERS = ["AAPL", "MSFT", "GOOG"]
_FETCHER = "app.services.optimization_service.fetch_close_prices"
_REPO = "app.api.v1.optimize.ExecutionRepository"


@contextmanager
def _mock_dependencies():
    """Patch price fetcher and repository so the route runs without touching DB."""
    now = datetime.now(timezone.utc)

    def _make_run(data: dict) -> OptimizationRun:
        run = OptimizationRun(**data)
        run.id = uuid.uuid4()
        run.created_at = now
        run.updated_at = now
        return run

    mock_repo = MagicMock()
    mock_repo.create_optimization_run.side_effect = _make_run

    with (
        patch(_FETCHER, return_value=_synthetic_prices()),
        patch(_REPO, return_value=mock_repo),
    ):
        yield mock_repo


def _synthetic_prices() -> pd.DataFrame:
    """Build a deterministic positive-trending price panel for optimiser fit."""
    rng = np.random.default_rng(42)
    n_days = 252
    dates = pd.date_range("2023-01-02", periods=n_days, freq="B")
    returns = rng.normal(loc=0.0005, scale=0.01, size=(n_days, len(_TICKERS)))
    prices = 100.0 * np.cumprod(1.0 + returns, axis=0)
    return pd.DataFrame(prices, index=dates, columns=_TICKERS)


def _payload(risk_measure: str = "variance") -> dict:
    return {
        "tickers": _TICKERS,
        "start_date": "2023-01-02",
        "end_date": "2023-12-31",
        "optimizer_type": "mean_risk",
        "config": {"risk_measure": risk_measure},
    }


class TestPostOptimizeEndToEnd:
    """End-to-end POST /api/v1/optimize with real skfolio wiring."""

    def test_returns_200_for_mean_risk_with_risk_measure(
        self, client: TestClient
    ) -> None:
        """Regression for #419: the duplicate-kwarg crash must be gone."""
        with _mock_dependencies():
            resp = client.post(BASE_URL, json=_payload(risk_measure="variance"))

        assert resp.status_code == 200, resp.text

    def test_weights_sum_to_one(self, client: TestClient) -> None:
        """Fitted MeanRisk portfolio weights sum to 1.0 (budget constraint)."""
        with _mock_dependencies():
            resp = client.post(BASE_URL, json=_payload())

        assert resp.status_code == 200, resp.text
        weights = resp.json()["weights"]
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_response_exposes_requested_universe(self, client: TestClient) -> None:
        """Response weights cover the requested ticker universe."""
        with _mock_dependencies():
            resp = client.post(BASE_URL, json=_payload())

        assert resp.status_code == 200, resp.text
        assert set(resp.json()["weights"].keys()) == set(_TICKERS)
