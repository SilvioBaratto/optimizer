"""Phase 06 Task 3 — top-level ``macro_attribution`` entrypoint, end-to-end.

Seeds a real (in-memory SQLite) paper ledger + price history + FRED + benchmark,
then drives the whole chain (build_nav_series → nav_return_series → factor builder →
estimate_exposures → attribute) through the single ``macro_attribution`` entrypoint.
USD base with USD holdings keeps FX out of the picture so the attribution math is
what is under test. Covers the sum-to-total invariant plus the honest-degradation
edges (all factors missing, short history, empty portfolio).
"""

from __future__ import annotations

import datetime as dt
import uuid
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from portopt_db.models.macro.macro_regime import FredObservation
from portopt_db.models.market_data.yfinance_data import PriceHistory
from portopt_db.models.orders.paper_order import PaperOrder
from portopt_db.models.universe.universe import Exchange, Instrument

from fund.attribution import AttributionResult, macro_attribution

_PID = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
_START = dt.date(2026, 1, 5)


def _seed_portfolio(
    db_session,
    *,
    n_days: int = 24,
    with_benchmark: bool = True,
    with_fred: bool = True,
) -> pd.DatetimeIndex:
    """Seed a USD ledger: two holdings, prices, a fill, optional benchmark + FRED."""
    biz = pd.bdate_range(_START, periods=n_days)
    ex = Exchange(name="X")
    db_session.add(ex)
    db_session.flush()
    rng = np.random.default_rng(7)

    def _inst(ticker: str) -> Instrument:
        inst = Instrument(
            ticker=ticker, short_name=ticker, exchange=ex, yfinance_ticker=ticker
        )
        db_session.add(inst)
        db_session.flush()
        return inst

    def _walk(p0: float) -> np.ndarray:
        return p0 * np.cumprod(1.0 + rng.normal(0, 0.01, n_days))

    def _seed_prices(inst: Instrument, p0: float) -> None:
        db_session.add_all(
            PriceHistory(
                instrument_id=inst.id,
                date=day.date(),
                close=Decimal(str(round(val, 6))),
                price_unit="USD",
            )
            for day, val in zip(biz, _walk(p0), strict=True)
        )

    _seed_prices(_inst("US1"), 100.0)
    _seed_prices(_inst("US2"), 50.0)
    if with_benchmark:
        _seed_prices(_inst("SPX"), 4000.0)

    db_session.add(
        PaperOrder(
            portfolio_id=_PID,
            asof=_START,
            weights_hash="h1",
            weights={"US1": 0.6, "US2": 0.4},
            fill_date=biz[0].date(),
            lines=[
                {"ticker": "US1", "notional": 6000.0, "fill_price": 100.0},
                {"ticker": "US2", "notional": 4000.0, "fill_price": 50.0},
            ],
            notional=10000.0,
            total_commission=0.0,
            total_slippage_cost=0.0,
        )
    )

    if with_fred:
        pre = (biz[0] - pd.offsets.BDay(2)).date()
        for series_id, base in (("DGS10", 4.0), ("BAA10Y", 1.8), ("DTWEXBGS", 100.0)):
            rows = [FredObservation(series_id=series_id, date=pre, value=float(base))]
            rows += [
                FredObservation(series_id=series_id, date=day.date(), value=float(v))
                for day, v in zip(biz, base + rng.normal(0, 0.03, n_days), strict=True)
            ]
            db_session.add_all(rows)

    db_session.flush()
    return biz


def _reconstructs(result: AttributionResult) -> None:
    """The per-factor contributions and residual sum back to the total return."""
    assert sum(result.contributions.values()) + result.residual == pytest.approx(
        result.total_return, abs=1e-9
    )


def test_attribution_engine_reconstructs_total(db_session):
    """Over ample seeded history the pieces reconstruct the total; factors present."""
    _seed_portfolio(db_session)

    result = macro_attribution(
        _PID, session=db_session, base_currency="USD", benchmark="SPX"
    )

    assert isinstance(result, AttributionResult)
    _reconstructs(result)
    assert set(result.factor_names) <= {"rates", "credit", "usd", "equity"}
    assert result.factor_names  # some factor was estimated
    assert not result.insufficient_history


def test_attribution_engine_all_factors_missing_is_residual_only(db_session):
    """No FRED and no benchmark ⇒ empty contributions, residual == total, no crash."""
    _seed_portfolio(db_session, with_benchmark=False, with_fred=False)

    result = macro_attribution(
        _PID, session=db_session, base_currency="USD", benchmark=None
    )

    assert result.contributions == {}
    assert result.factor_names == ()
    assert result.residual == pytest.approx(result.total_return, abs=1e-12)


def test_attribution_engine_short_history_flags_insufficient(db_session):
    """A two-bar ledger is below min_observations: flagged, degraded, never raises."""
    _seed_portfolio(db_session, n_days=2)

    result = macro_attribution(
        _PID, session=db_session, base_currency="USD", benchmark="SPX"
    )

    assert result.insufficient_history
    _reconstructs(result)


def test_attribution_engine_empty_portfolio_is_graceful(db_session):
    """A portfolio with no fills yields a zeroed, flagged result — no raise."""
    result = macro_attribution(
        _PID, session=db_session, base_currency="USD", benchmark="SPX"
    )

    assert result.total_return == pytest.approx(0.0)
    assert result.contributions == {}
    assert result.insufficient_history
