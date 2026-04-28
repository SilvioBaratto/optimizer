"""Integration tests for POST /api/v1/backtest (issue #422).

Exercises the full route → background worker → service → library flow with
real skfolio wiring. The DB price fetcher is mocked so the optimiser fits
against a deterministic synthetic price series whose index is a plain
``Index[object]`` of ``datetime.date`` — this is what ``fetch_close_prices``
produces in production and what originally triggered the ``TypeError`` in
issue #422.

The test asserts that the job reaches ``status='completed'`` (not ``failed``)
after the service normalises the index to a ``DatetimeIndex``.
"""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import date, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.execution import BacktestRun

BASE_URL = "/api/v1/backtest"
_TICKERS = ["AAPL", "MSFT", "GOOG"]
_FETCHER = "app.services.backtest_service.fetch_close_prices"
_SESSION_CM = "app.api.v1.backtest.database_manager.get_session"


@pytest.fixture(autouse=True)
def cleanup_backtest_runs(db_session: Session) -> Generator[None, None, None]:
    """Delete BacktestRun rows after each test so the shared in-memory DB stays clean."""
    yield
    db_session.query(BacktestRun).delete()
    db_session.commit()


def _synthetic_prices_plain_date_index() -> pd.DataFrame:
    """Build a price panel with a plain ``Index[object]`` of ``datetime.date``.

    This replicates the exact shape ``fetch_close_prices`` returns in
    production (dict-of-dicts keyed by ``date``), which is what triggered
    the original ``TypeError`` in issue #422.
    """
    rng = np.random.default_rng(42)
    n_days = 252
    base = date(2023, 1, 2)
    dates = [base + timedelta(days=i) for i in range(n_days)]  # python dates
    returns = rng.normal(loc=0.0005, scale=0.01, size=(n_days, len(_TICKERS)))
    prices = 100.0 * np.cumprod(1.0 + returns, axis=0)
    df = pd.DataFrame(prices, index=pd.Index(dates), columns=_TICKERS)
    assert not isinstance(df.index, pd.DatetimeIndex)  # precondition
    return df


@contextmanager
def _patch_worker_session(db_session: Session):
    """Make the background worker's ``get_session`` context manager reuse the test session."""

    @contextmanager
    def _cm():
        yield db_session

    with patch(_SESSION_CM, side_effect=_cm):
        yield


def _payload() -> dict:
    return {
        "tickers": _TICKERS,
        "start_date": "2023-01-02",
        "end_date": "2023-12-31",
        "pipeline_config": {"optimizer_type": "mean_risk", "run_cv": False},
    }


def _poll_until_done(
    client: TestClient, job_id: str, timeout_seconds: float = 30.0
) -> dict:
    """Poll ``GET /backtest/{job_id}`` until status is terminal or the timeout elapses."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        resp = client.get(f"{BASE_URL}/{job_id}")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        if body["status"] in {"completed", "failed"}:
            return body
        time.sleep(0.1)
    pytest.fail(f"Backtest job {job_id} did not finish within {timeout_seconds}s")


class TestBacktestRouteEndToEnd:
    """End-to-end POST /api/v1/backtest with real skfolio wiring."""

    def test_job_reaches_completed_with_plain_date_index(
        self, client: TestClient, db_session: Session
    ) -> None:
        """Regression for #422: the TypeError on ``Index`` must be gone."""
        prices = _synthetic_prices_plain_date_index()

        with (
            patch(_FETCHER, return_value=prices),
            _patch_worker_session(db_session),
        ):
            post = client.post(BASE_URL, json=_payload())
            assert post.status_code == 202, post.text
            job_id = post.json()["jobId"]

            final = _poll_until_done(client, job_id)

        assert final["status"] == "completed", final
        assert final.get("error") is None
