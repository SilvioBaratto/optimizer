"""Phase 06 Task 2 — FRED-first macro factor return builder (DB-backed).

Seeds ``fred_observations`` on an in-memory SQLite session and asserts the canonical
alignment recipe: reindex each level to the NAV grid → ffill across gaps → Δlevel
(``diff``) or ratio return (``pct_change``) per the factor's transform → drop the
first NaN row. A series with no rows lands in ``missing[]`` (no crash); the ``equity``
factor is sourced from the benchmark return series, not FRED.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest
from portopt_db.models.macro.macro_regime import FredObservation

from fund.attribution import (
    FactorSeries,
    FactorSpec,
    FactorTransform,
    MacroAttributionConfig,
    build_macro_factor_returns,
)

_GRID = pd.DatetimeIndex(
    [
        pd.Timestamp("2026-01-05"),
        pd.Timestamp("2026-01-06"),
        pd.Timestamp("2026-01-07"),
        pd.Timestamp("2026-01-08"),
    ]
)
_TARGET = _GRID[1:]


def _seed_fred(session, series_id: str, values: dict[dt.date, float]) -> None:
    """Insert FRED observations for ``series_id`` from a ``{date: value}`` map."""
    session.add_all(
        FredObservation(series_id=series_id, date=d, value=v) for d, v in values.items()
    )
    session.flush()


def test_factor_series_diff_and_pct_change_transforms(db_session):
    """rates is Δlevel; usd is pct-change; both on the NAV return grid."""
    _seed_fred(
        db_session,
        "DGS10",
        {
            dt.date(2026, 1, 5): 4.00,
            dt.date(2026, 1, 6): 4.10,
            dt.date(2026, 1, 8): 4.30,  # 01-07 absent → ffill from 01-06
        },
    )
    _seed_fred(
        db_session,
        "DTWEXBGS",
        {
            dt.date(2026, 1, 5): 100.0,
            dt.date(2026, 1, 6): 101.0,
            dt.date(2026, 1, 7): 102.0,
            dt.date(2026, 1, 8): 102.0,
        },
    )
    config = MacroAttributionConfig(
        factors=(
            FactorSpec("rates", "DGS10", FactorTransform.DIFF),
            FactorSpec("usd", "DTWEXBGS", FactorTransform.PCT_CHANGE),
        )
    )

    out = build_macro_factor_returns(
        session=db_session, dates=_GRID, benchmark_returns=None, config=config
    )

    assert isinstance(out, FactorSeries)
    assert list(out.returns.columns) == ["rates", "usd"]
    assert out.returns.index.tolist() == list(_TARGET)
    # rates: diff of [4.00, 4.10, 4.10(ffill), 4.30] → [0.10, 0.00, 0.20]
    assert out.returns["rates"].tolist() == pytest.approx([0.10, 0.00, 0.20])
    # usd: pct_change of [100, 101, 102, 102] → [0.01, 102/101-1, 0.0]
    assert out.returns["usd"].tolist() == pytest.approx(
        [0.01, 102.0 / 101.0 - 1.0, 0.0]
    )
    assert out.missing == []


def test_factor_series_ffills_level_from_pre_grid_observation(db_session):
    """The first grid day uses the most recent level at or before it (pre-buffer)."""
    _seed_fred(
        db_session,
        "DGS10",
        {
            dt.date(2026, 1, 2): 3.90,  # before the grid → ffilled onto 01-05
            dt.date(2026, 1, 6): 4.10,
            dt.date(2026, 1, 7): 4.20,
            dt.date(2026, 1, 8): 4.25,
        },
    )
    config = MacroAttributionConfig(
        factors=(FactorSpec("rates", "DGS10", FactorTransform.DIFF),)
    )

    out = build_macro_factor_returns(
        session=db_session, dates=_GRID, benchmark_returns=None, config=config
    )

    # levels on grid: 01-05=3.90(pre-buffer ffill), 01-06=4.10, 01-07=4.20, 01-08=4.25
    assert out.returns["rates"].tolist() == pytest.approx([0.20, 0.10, 0.05])


def test_factor_series_stale_tail_is_unobserved_not_a_zero_move(db_session):
    """Grid days after a series' last observation are NaN, never a flat 0.0 return."""
    _seed_fred(
        db_session,
        "DGS10",
        {
            dt.date(2026, 1, 5): 4.00,
            dt.date(2026, 1, 6): 4.10,  # last stored print; 01-07 and 01-08 are stale
        },
    )
    config = MacroAttributionConfig(
        factors=(FactorSpec("rates", "DGS10", FactorTransform.DIFF),)
    )

    out = build_macro_factor_returns(
        session=db_session, dates=_GRID, benchmark_returns=None, config=config
    )

    rates = out.returns["rates"]
    assert rates.iloc[0] == pytest.approx(0.10)
    assert rates.iloc[1:].isna().all()
    assert out.missing == []


def test_factor_series_entirely_stale_over_the_grid_is_missing(db_session):
    """A series whose last print predates the whole grid has no usable data."""
    _seed_fred(
        db_session,
        "DGS10",
        {dt.date(2025, 12, 30): 3.90, dt.date(2025, 12, 31): 3.95},
    )
    config = MacroAttributionConfig(
        factors=(FactorSpec("rates", "DGS10", FactorTransform.DIFF),)
    )

    out = build_macro_factor_returns(
        session=db_session, dates=_GRID, benchmark_returns=None, config=config
    )

    assert "rates" not in out.returns.columns
    assert out.missing == ["rates"]


def test_factor_series_absent_series_flagged_missing(db_session):
    """A factor whose FRED series has no rows is flagged, not built — no crash."""
    _seed_fred(
        db_session,
        "DGS10",
        {
            dt.date(2026, 1, 5): 4.00,
            dt.date(2026, 1, 6): 4.10,
            dt.date(2026, 1, 7): 4.20,
            dt.date(2026, 1, 8): 4.30,
        },
    )
    config = MacroAttributionConfig(
        factors=(
            FactorSpec("rates", "DGS10", FactorTransform.DIFF),
            FactorSpec("credit", "BAA10Y", FactorTransform.DIFF),  # not seeded
        )
    )

    out = build_macro_factor_returns(
        session=db_session, dates=_GRID, benchmark_returns=None, config=config
    )

    assert list(out.returns.columns) == ["rates"]
    assert out.missing == ["credit"]


def test_factor_series_equity_from_benchmark(db_session):
    """equity is the benchmark return series reindexed to the grid, not FRED."""
    benchmark = pd.Series([0.01, -0.02, 0.03], index=_TARGET)
    config = MacroAttributionConfig(
        factors=(FactorSpec("equity", None, FactorTransform.PCT_CHANGE),)
    )

    out = build_macro_factor_returns(
        session=db_session, dates=_GRID, benchmark_returns=benchmark, config=config
    )

    assert list(out.returns.columns) == ["equity"]
    assert out.returns["equity"].tolist() == pytest.approx([0.01, -0.02, 0.03])
    assert out.missing == []


def test_factor_series_no_benchmark_marks_equity_missing(db_session):
    """Without a benchmark series the equity factor degrades to missing."""
    config = MacroAttributionConfig(
        factors=(FactorSpec("equity", None, FactorTransform.PCT_CHANGE),)
    )

    out = build_macro_factor_returns(
        session=db_session, dates=_GRID, benchmark_returns=None, config=config
    )

    assert "equity" not in out.returns.columns
    assert out.missing == ["equity"]


def test_factor_series_empty_grid_degrades(db_session):
    """An empty NAV grid yields an empty frame with every factor missing."""
    config = MacroAttributionConfig.for_default()

    out = build_macro_factor_returns(
        session=db_session,
        dates=pd.DatetimeIndex([]),
        benchmark_returns=None,
        config=config,
    )

    assert out.returns.empty
    assert set(out.missing) == {f.name for f in config.factors}
