"""Integration tests for risk analytics endpoints (issues #423, #424).

Exercises the full route → service → repository flow against the test SQLite
database with a seeded portfolio snapshot and partial price history.

The seeded scenario mirrors the production bug: a portfolio with 3 tickers
where one ticker has zero rows in ``price_history``. The previous service
implementation failed with 400 (``Insufficient data`` / ``condensed distance
matrix must contain only finite values``); after the fix the surviving
tickers are enough to produce a populated VaR and correlation response.

Guard-clause coverage: when **every** ticker is absent, the route still
returns 400 (no real information to compute on).

Issue #424 additions: ``/risk/factor-exposure`` must return 200 with a
populated ``exposures`` mapping when ``factor_scores`` rows exist, and 200
with an empty mapping when none exist (portfolio exists → empty state, not
404).
"""

from __future__ import annotations

import uuid
from collections.abc import Generator
from datetime import date, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.factors.factor import FactorScore
from app.models.market_data.yfinance_data import PriceHistory
from app.models.portfolio.portfolio import Portfolio, PortfolioSnapshot, SnapshotWeight
from app.models.universe.universe import Exchange, Instrument

BASE = "/api/v1/portfolio"

_PORTFOLIO_WITH_GAP = "integration-test-portfolio-gap"
_PORTFOLIO_ALL_MISSING = "integration-test-portfolio-all-missing"


# ---------------------------------------------------------------------------
# Test fixtures — seed two portfolios into the shared in-memory SQLite
# ---------------------------------------------------------------------------


@pytest.fixture
def seeded_portfolios(
    db_session: Session,
) -> Generator[dict[str, uuid.UUID], None, None]:
    """Seed two portfolios covering the happy and fail paths for issue #423.

    - ``integration-test-portfolio-gap``: 3 tickers, 2 have 260 price rows, 1 has none
    - ``integration-test-portfolio-all-missing``: 2 tickers, neither has any price rows
    """
    exchange = Exchange(name="TEST_NYSE")
    db_session.add(exchange)
    db_session.flush()

    rng = np.random.default_rng(101)
    dates = [date(2024, 1, 2) + timedelta(days=i) for i in range(260)]

    instruments_with_prices = [
        _seed_instrument(db_session, exchange.id, ticker="AAPL_T", yf_ticker="AAPL_T"),
        _seed_instrument(db_session, exchange.id, ticker="MSFT_T", yf_ticker="MSFT_T"),
        _seed_instrument(db_session, exchange.id, ticker="GOOG_T", yf_ticker="GOOG_T"),
    ]
    # Ticker deliberately registered with NO price_history rows — replicates EDV.L bug
    _seed_instrument(db_session, exchange.id, ticker="NODATA_T", yf_ticker="NODATA_T")
    # Ticker also registered but no price_history rows (for all-missing portfolio)
    _seed_instrument(db_session, exchange.id, ticker="GAP1_T", yf_ticker="GAP1_T")
    _seed_instrument(db_session, exchange.id, ticker="GAP2_T", yf_ticker="GAP2_T")

    for inst in instruments_with_prices:
        _seed_price_history(db_session, inst.id, dates, rng)

    portfolio_gap = _seed_portfolio_with_weights(
        db_session,
        name=_PORTFOLIO_WITH_GAP,
        weights={"AAPL_T": 0.4, "MSFT_T": 0.3, "GOOG_T": 0.2, "NODATA_T": 0.1},
    )
    portfolio_all_missing = _seed_portfolio_with_weights(
        db_session,
        name=_PORTFOLIO_ALL_MISSING,
        weights={"GAP1_T": 0.6, "GAP2_T": 0.4},
    )

    db_session.flush()
    yield {
        "gap_id": portfolio_gap.id,
        "all_missing_id": portfolio_all_missing.id,
    }
    # SAVEPOINT rollback in conftest.db_session cleans up — no explicit delete needed.


def _seed_instrument(
    session: Session,
    exchange_id: uuid.UUID,
    ticker: str,
    yf_ticker: str,
) -> Instrument:
    inst = Instrument(
        ticker=ticker,
        short_name=ticker,
        name=f"{ticker} Inc.",
        instrument_type="STOCK",
        currency_code="USD",
        yfinance_ticker=yf_ticker,
        exchange_id=exchange_id,
    )
    session.add(inst)
    session.flush()
    return inst


def _seed_price_history(
    session: Session,
    instrument_id: uuid.UUID,
    dates: list[date],
    rng: np.random.Generator,
) -> None:
    returns = rng.normal(loc=0.0005, scale=0.01, size=len(dates))
    closes = 100.0 * np.cumprod(1.0 + returns)
    rows = [
        PriceHistory(
            instrument_id=instrument_id,
            date=d,
            close=float(closes[i]),
            volume=1_000_000,
        )
        for i, d in enumerate(dates)
    ]
    session.add_all(rows)


def _seed_portfolio_with_weights(
    session: Session,
    name: str,
    weights: dict[str, float],
) -> Portfolio:
    portfolio = Portfolio(
        name=name,
        description="integration test",
        currency="USD",
        benchmark_ticker="AAPL_T",
        is_active=True,
    )
    session.add(portfolio)
    session.flush()

    snapshot = PortfolioSnapshot(
        portfolio_id=portfolio.id,
        snapshot_date=date.today(),
        snapshot_type="manual",
        holding_count=len(weights),
        turnover=None,
    )
    session.add(snapshot)
    session.flush()

    session.add_all(
        [
            SnapshotWeight(snapshot_id=snapshot.id, ticker=t, weight=w)
            for t, w in weights.items()
        ]
    )
    return portfolio


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVarRoute:
    """GET /portfolio/{name}/risk/var — regression for #423."""

    def test_historical_returns_200_when_one_ticker_has_no_prices(
        self, client: TestClient, seeded_portfolios: dict[str, uuid.UUID]
    ) -> None:
        resp = client.get(
            f"{BASE}/{_PORTFOLIO_WITH_GAP}/risk/var",
            params={"method": "historical", "lookback": 100},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "var" in body and "cvar" in body
        assert set(body["var"].keys()) == {"90", "95", "99"}
        assert all(v > 0 for v in body["var"].values())
        assert all(v > 0 for v in body["cvar"].values())

    def test_parametric_returns_200_when_one_ticker_has_no_prices(
        self, client: TestClient, seeded_portfolios: dict[str, uuid.UUID]
    ) -> None:
        resp = client.get(
            f"{BASE}/{_PORTFOLIO_WITH_GAP}/risk/var",
            params={"method": "parametric", "lookback": 100},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["method"] == "parametric"

    def test_returns_400_when_every_ticker_is_absent(
        self, client: TestClient, seeded_portfolios: dict[str, uuid.UUID]
    ) -> None:
        """Guard clause: when no ticker has data, 400 must still fire."""
        resp = client.get(
            f"{BASE}/{_PORTFOLIO_ALL_MISSING}/risk/var",
            params={"method": "historical", "lookback": 100},
        )
        assert resp.status_code == 400


class TestCorrelationRoute:
    """GET /portfolio/{name}/risk/correlation — regression for #423."""

    def test_returns_200_with_non_empty_matrix_when_one_ticker_has_no_prices(
        self, client: TestClient, seeded_portfolios: dict[str, uuid.UUID]
    ) -> None:
        resp = client.get(
            f"{BASE}/{_PORTFOLIO_WITH_GAP}/risk/correlation",
            params={"lookback": 100},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assets = body["assets"]
        matrix = body["matrix"]

        assert len(assets) == 3  # NODATA_T excluded
        assert len(matrix) == 3
        assert all(len(row) == 3 for row in matrix)
        # Every cell must be finite (no NaN/inf leaking through)
        flat = [cell for row in matrix for cell in row]
        assert all(np.isfinite(cell) for cell in flat)

    def test_returns_400_when_every_ticker_is_absent(
        self, client: TestClient, seeded_portfolios: dict[str, uuid.UUID]
    ) -> None:
        resp = client.get(
            f"{BASE}/{_PORTFOLIO_ALL_MISSING}/risk/correlation",
            params={"lookback": 100},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Factor exposure fixtures + tests (issue #424)
# ---------------------------------------------------------------------------


_PORTFOLIO_WITH_FACTOR_SCORES = "integration-test-portfolio-factors"
_PORTFOLIO_WITHOUT_FACTOR_SCORES = "integration-test-portfolio-no-factors"


@pytest.fixture
def seeded_portfolios_with_factor_scores(
    db_session: Session,
) -> Generator[dict[str, uuid.UUID], None, None]:
    """Seed two portfolios for issue #424 coverage.

    - ``...-factors``: snapshot with 2 tickers; ``factor_scores`` rows for both (within 90 days)
    - ``...-no-factors``: snapshot with 2 tickers; zero ``factor_scores`` rows
    """
    exchange = Exchange(name="TEST_FACTORS_NYSE")
    db_session.add(exchange)
    db_session.flush()

    _seed_instrument(db_session, exchange.id, ticker="FACT1", yf_ticker="FACT1")
    _seed_instrument(db_session, exchange.id, ticker="FACT2", yf_ticker="FACT2")
    _seed_instrument(db_session, exchange.id, ticker="NOFACT1", yf_ticker="NOFACT1")
    _seed_instrument(db_session, exchange.id, ticker="NOFACT2", yf_ticker="NOFACT2")

    score_date = date.today() - timedelta(days=1)
    db_session.add_all(
        [
            _make_factor_score("FACT1", "momentum", "style", 0.6, score_date),
            _make_factor_score("FACT1", "quality", "fundamental", 0.4, score_date),
            _make_factor_score("FACT2", "momentum", "style", 0.2, score_date),
            _make_factor_score("FACT2", "quality", "fundamental", 0.8, score_date),
        ]
    )

    p_with = _seed_portfolio_with_weights(
        db_session,
        name=_PORTFOLIO_WITH_FACTOR_SCORES,
        weights={"FACT1": 0.7, "FACT2": 0.3},
    )
    p_without = _seed_portfolio_with_weights(
        db_session,
        name=_PORTFOLIO_WITHOUT_FACTOR_SCORES,
        weights={"NOFACT1": 0.5, "NOFACT2": 0.5},
    )

    db_session.flush()
    yield {"with_id": p_with.id, "without_id": p_without.id}


def _make_factor_score(
    ticker: str,
    factor_type: str,
    factor_group: str,
    score: float,
    score_date: date,
) -> FactorScore:
    return FactorScore(
        ticker=ticker,
        score_date=score_date,
        factor_type=factor_type,
        factor_group=factor_group,
        raw_score=score,
        standardized_score=score,
        composite_score=None,
    )


def test_factor_exposure_happy_path(
    client: TestClient,
    seeded_portfolios_with_factor_scores: dict[str, uuid.UUID],
) -> None:
    """Regression for #424 happy path: seeded factor scores → 200 with populated exposures."""
    resp = client.get(f"{BASE}/{_PORTFOLIO_WITH_FACTOR_SCORES}/risk/factor-exposure")

    assert resp.status_code == 200, resp.text
    body = resp.json()

    exposures = body["exposures"]
    assert exposures, "exposures must be non-empty when seeded scores exist"
    # Weighted: momentum = 0.7*0.6 + 0.3*0.2 = 0.48; quality = 0.7*0.4 + 0.3*0.8 = 0.52
    assert abs(exposures["momentum"] - 0.48) < 1e-6
    assert abs(exposures["quality"] - 0.52) < 1e-6

    asset_exposures = body["assetExposures"]
    assert set(asset_exposures.keys()) == {"FACT1", "FACT2"}


def test_factor_exposure_no_scores_returns_200_with_empty_dict(
    client: TestClient,
    seeded_portfolios_with_factor_scores: dict[str, uuid.UUID],
) -> None:
    """Regression for #424: portfolio exists but has no scores → 200 + empty, not 404."""
    resp = client.get(f"{BASE}/{_PORTFOLIO_WITHOUT_FACTOR_SCORES}/risk/factor-exposure")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["exposures"] == {}
    assert body["assetExposures"] == {}


def test_factor_exposure_missing_portfolio_still_returns_404(
    client: TestClient,
    seeded_portfolios_with_factor_scores: dict[str, uuid.UUID],
) -> None:
    """Sanity guard: the 404-on-missing-portfolio branch is untouched by #424."""
    resp = client.get(f"{BASE}/does-not-exist/risk/factor-exposure")
    assert resp.status_code == 404
