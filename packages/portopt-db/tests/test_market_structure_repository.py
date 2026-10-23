"""MarketStructureRepository upserts (sector snapshot / industries / top companies)."""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from portopt_db.models.market_data.market_structure import SectorTopCompany
from portopt_db.repositories.market_data.market_structure_repository import (
    MarketStructureRepository,
    _int,
    _num,
)

_AS_OF = dt.date(2026, 8, 27)


def test_upserts_are_idempotent(db_session) -> None:
    repo = MarketStructureRepository(db_session)
    n_ind = n_co = 0
    for _ in range(2):
        repo.upsert_sector_snapshot(
            "technology",
            "US",
            _AS_OF,
            name="Technology",
            symbol="^YH311",
            market_cap=1.5e13,
            market_weight=0.31,
            companies_count=800,
            industries_count=12,
            employee_count=5_000_000,
        )
        n_ind = repo.upsert_industries(
            "technology",
            "US",
            _AS_OF,
            [{"key": "semiconductors", "name": "Semiconductors"}],
        )
        n_co = repo.upsert_top_companies(
            "technology",
            "US",
            _AS_OF,
            [{"symbol": "AAPL", "name": "Apple", "weight": 0.07, "rating": "buy"}],
        )
    db_session.flush()

    assert n_ind == 1 and n_co == 1
    snap = repo.get_sector_snapshot("technology", "US")
    assert snap is not None and float(snap.market_weight) == 0.31
    assert repo.get_latest_sector_as_of("technology", "US") == _AS_OF


_NAN = float("nan")
_INF = float("inf")


@pytest.mark.parametrize("bad", [_NAN, _INF, -_INF])
def test_num_treats_non_finite_values_as_missing(bad: float) -> None:
    """NaN/inf from a pandas frame must become NULL, not a poisoned number."""
    assert _num(bad) is None


@pytest.mark.parametrize("bad", [_NAN, _INF, -_INF, "n/a"])
def test_int_treats_non_finite_or_unparsable_values_as_missing(bad: object) -> None:
    """``int(nan)``/``int(inf)`` raise, so ``_int`` must map them to None."""
    assert _int(bad) is None


def test_int_truncates_finite_values() -> None:
    """A finite float count keeps its integer part."""
    assert _int(12.0) == 12
    assert _int("7") == 7


def test_top_companies_store_nan_rating_as_null(db_session) -> None:
    """A missing rating arrives as NaN from pandas and must not persist as 'nan'."""
    repo = MarketStructureRepository(db_session)
    repo.upsert_top_companies(
        "technology",
        "GB",
        _AS_OF,
        [
            {"symbol": "AAA", "name": "Alpha", "weight": 0.1, "rating": _NAN},
            {"symbol": "BBB", "name": "Beta plc", "weight": 0.2, "rating": "Buy"},
        ],
    )
    db_session.flush()

    rows = {
        r.symbol: r
        for r in db_session.query(SectorTopCompany).filter_by(region="GB").all()
    }
    assert rows["AAA"].rating is None
    assert rows["BBB"].rating == "Buy"
    assert rows["BBB"].name == "Beta plc"


@pytest.fixture
def captured_rows(db_session, monkeypatch: pytest.MonkeyPatch):
    """Repository plus the raw rows it hands to ``_upsert``.

    SQLite silently turns float NaN into NULL, so the NaN-vs-NULL distinction
    PostgreSQL cares about is only observable on the upsert payload.
    """
    repo = MarketStructureRepository(db_session)
    captured: list[dict[str, Any]] = []

    def _capture(model: Any, rows: list[dict[str, Any]], **_: Any) -> None:
        captured.extend(rows)

    monkeypatch.setattr(repo, "_upsert", _capture)
    return repo, captured


def test_top_companies_payload_carries_no_nan_weight_or_name(captured_rows) -> None:
    """NaN weights/names must reach the upsert as None so PostgreSQL never stores them."""
    repo, captured = captured_rows

    repo.upsert_top_companies(
        "technology",
        "GB",
        _AS_OF,
        [
            {"symbol": "AAA", "name": _NAN, "weight": _NAN, "rating": "Buy"},
            {"symbol": "BBB", "name": "Beta", "weight": _INF, "rating": "Buy"},
        ],
    )

    assert [row["weight"] for row in captured] == [None, None]
    assert [row["name"] for row in captured] == [None, "Beta"]


def test_industries_payload_carries_no_nan_name(captured_rows) -> None:
    """A NaN industry name must reach the upsert as None, not the string 'nan'."""
    repo, captured = captured_rows

    repo.upsert_industries("technology", "GB", _AS_OF, [{"key": "chips", "name": _NAN}])

    assert [row["industry_name"] for row in captured] == [None]
