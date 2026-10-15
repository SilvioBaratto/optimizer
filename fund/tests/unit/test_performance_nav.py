"""Performance tracker — FX-aware price reader (SPEC Phase 05 T6) + NAV series (T7).

The FX-conversion cases (``test_nav_fx_*``) drive the pure base-currency price
reader against in-memory fixtures (incl. a GBp minor-unit ticker); the DB reader
cases run against real SQLite. The module must import no agent stack
(optimizer/skfolio are allowed — they are not the deep-agent stack).
"""

from __future__ import annotations

import ast
import datetime as dt
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
from portopt_db.models.market_data.yfinance_data import PriceHistory
from portopt_db.models.universe.universe import Exchange, Instrument

from fund import performance
from fund.performance import (
    _load_fx_rates_frame,
    _load_price_frame,
    _to_base_prices,
)

_D1 = dt.date(2026, 10, 1)
_D2 = dt.date(2026, 10, 2)


def _price_frame():
    return pd.DataFrame(
        {"US1": [100.0, 101.0], "LON": [500.0, 510.0]},
        index=pd.to_datetime([_D1, _D2]),
    )


def _fx_frame():
    return pd.DataFrame(
        {"USD": [0.90, 0.91], "GBP": [1.17, 1.18]},
        index=pd.to_datetime([_D1, _D2]),
    )


def test_nav_fx_converts_minor_unit_and_multiplies_by_rate():
    """GBp prices are /100 then × GBP rate; USD prices × USD rate; base per date."""
    out = _to_base_prices(
        _price_frame(),
        base_currency="EUR",
        currency_map={"US1": "USD", "LON": "GBp"},
        fx_rates=_fx_frame(),
    )
    assert out["US1"].tolist() == pytest.approx([90.0, 91.91])
    assert out["LON"].tolist() == pytest.approx([5.85, 6.018])


def test_nav_fx_missing_rate_degrades_without_raise():
    """A ticker whose currency has no FX column is left unconverted, not raised."""
    prices = pd.DataFrame({"JPN": [1000.0, 1010.0]}, index=pd.to_datetime([_D1, _D2]))
    out = _to_base_prices(
        prices,
        base_currency="EUR",
        currency_map={"JPN": "JPY"},
        fx_rates=_fx_frame(),
    )
    assert out["JPN"].tolist() == pytest.approx([1000.0, 1010.0])


def test_nav_fx_empty_prices_returns_empty():
    """An empty price frame round-trips to empty without touching optimizer.fx."""
    out = _to_base_prices(
        pd.DataFrame(),
        base_currency="EUR",
        currency_map={},
        fx_rates=_fx_frame(),
    )
    assert out.empty


def test_nav_fx_load_price_frame_carries_raw_price_unit(db_session):
    """The DB reader keys currency_map on the RAW price_unit (e.g. 'GBp')."""
    ex = Exchange(name="LSE")
    db_session.add(ex)
    db_session.flush()
    lon = Instrument(
        ticker="LON", short_name="Lon", exchange=ex, yfinance_ticker="LON.L"
    )
    db_session.add(lon)
    db_session.flush()
    db_session.add_all(
        [
            PriceHistory(
                instrument_id=lon.id, date=_D1, close=Decimal("500"), price_unit="GBp"
            ),
            PriceHistory(
                instrument_id=lon.id, date=_D2, close=Decimal("510"), price_unit="GBp"
            ),
        ]
    )
    db_session.flush()

    prices, currency_map = _load_price_frame(db_session, ["LON.L"])
    assert currency_map["LON.L"] == "GBp"  # raw, not pre-normalized to GBP
    assert prices["LON.L"].tolist() == pytest.approx([500.0, 510.0])


def test_nav_fx_load_fx_rates_frame_pivots_to_currency_columns(db_session):
    """The DB reader pivots fx_rate rows into a dates × quote-currency frame."""
    from portopt_db.repositories.market_data.fx_rates_repository import (
        FxRatesRepository,
    )

    FxRatesRepository(db_session).upsert_rates(
        [
            {
                "base_currency": "EUR",
                "quote_currency": "USD",
                "date": _D1,
                "rate": 0.90,
            },
            {
                "base_currency": "EUR",
                "quote_currency": "USD",
                "date": _D2,
                "rate": 0.91,
            },
            {
                "base_currency": "EUR",
                "quote_currency": "GBP",
                "date": _D1,
                "rate": 1.17,
            },
        ]
    )
    db_session.flush()

    frame = _load_fx_rates_frame(db_session, "EUR", ["USD", "GBP"])
    assert set(frame.columns) == {"USD", "GBP"}
    assert float(frame.loc[pd.Timestamp(_D1), "USD"]) == pytest.approx(0.90)
    assert float(frame.loc[pd.Timestamp(_D2), "USD"]) == pytest.approx(0.91)


def test_performance_module_declares_no_agent_stack_imports():
    """performance.py imports no agent stack (optimizer/skfolio are allowed)."""
    forbidden = ("deepagents", "langgraph", "langchain", "fund.agents", "fund.audit")
    source = Path(performance.__file__).read_text(encoding="utf-8")
    names: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    offenders = [n for n in names if any(n.startswith(f) for f in forbidden)]
    assert offenders == [], (
        f"agent-stack imports leaked into performance.py: {offenders}"
    )
