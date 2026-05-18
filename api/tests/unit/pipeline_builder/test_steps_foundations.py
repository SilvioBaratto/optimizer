"""Foundational tests for ``app.services.pipeline_builder.steps`` (issue #696).

Covers:

* The module-level ``sys.path`` shim that makes ``research.*`` importable
  from inside the api package.
* ``_to_json`` serializer per supported leaf/container type, including
  ``safe_float`` coercion for non-finite floats.
* ``_require`` precondition validator: raises on the first missing field
  and passes silently when every requested field is set.
* ``_load_market_proxy_from_db`` returns ``None`` when no instrument
  matches and closes the DB session before returning.
"""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from app.services.pipeline_builder import session_store, steps

# ---------------------------------------------------------------------------
# sys.path shim
# ---------------------------------------------------------------------------


def test_when_steps_module_imported_then_research_is_importable():
    research = importlib.import_module("research")
    assert research is not None


def test_when_steps_module_loaded_then_repo_root_added_to_sys_path():
    expected_root = Path(steps.__file__).resolve().parents[4]
    assert str(expected_root) in sys.path
    assert (expected_root / "research" / "__init__.py").exists()


# ---------------------------------------------------------------------------
# _to_json
# ---------------------------------------------------------------------------


def test_when_to_json_receives_dataframe_then_returns_list_of_records():
    df = pd.DataFrame({"a": [1, 2]}, index=pd.Index(["x", "y"], name="idx"))
    result = steps._to_json(df)
    assert result == [{"idx": "x", "a": 1.0}, {"idx": "y", "a": 2.0}]


def test_when_to_json_receives_series_then_returns_index_values_dict():
    s = pd.Series([1.0, 2.0], index=["a", "b"])
    result = steps._to_json(s)
    assert result == {"index": ["a", "b"], "values": [1.0, 2.0]}


def test_when_to_json_receives_ndarray_then_returns_list():
    arr = np.array([1, 2, 3])
    assert steps._to_json(arr) == [1, 2, 3]


def test_when_to_json_receives_numpy_float_then_returns_python_float():
    result = steps._to_json(np.float64(1.5))
    assert isinstance(result, float)
    assert result == 1.5


def test_when_to_json_receives_numpy_int_then_returns_python_int():
    result = steps._to_json(np.int64(7))
    assert isinstance(result, int)
    assert result == 7


def test_when_to_json_receives_nan_float_then_returns_none():
    assert steps._to_json(float("nan")) is None


def test_when_to_json_receives_inf_float_then_returns_none():
    assert steps._to_json(float("inf")) is None


def test_when_to_json_receives_finite_float_then_returns_value():
    assert steps._to_json(1.25) == 1.25


def test_when_to_json_receives_timestamp_then_returns_string():
    ts = pd.Timestamp("2026-01-01")
    assert steps._to_json(ts) == str(ts)


def test_when_to_json_receives_datetime_then_returns_string():
    dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert steps._to_json(dt) == str(dt)


def test_when_to_json_receives_date_then_returns_string():
    d = date(2026, 1, 1)
    assert steps._to_json(d) == str(d)


def test_when_to_json_receives_enum_then_returns_value():
    class Color(Enum):
        RED = "red"

    assert steps._to_json(Color.RED) == "red"


def test_when_to_json_receives_dict_then_recurses():
    payload = {"x": np.float64(1.0), "y": float("nan")}
    assert steps._to_json(payload) == {"x": 1.0, "y": None}


def test_when_to_json_receives_list_then_recurses():
    assert steps._to_json([np.int64(1), float("nan")]) == [1, None]


def test_when_to_json_receives_tuple_then_recurses_to_list():
    assert steps._to_json((np.int64(1), 2.0)) == [1, 2.0]


def test_when_to_json_receives_unknown_object_then_falls_back_to_str():
    class Dummy:
        def __str__(self) -> str:
            return "dummy"

    assert steps._to_json(Dummy()) == "dummy"


def test_when_to_json_receives_none_then_returns_none():
    assert steps._to_json(None) is None


# ---------------------------------------------------------------------------
# _require
# ---------------------------------------------------------------------------


def _make_session(**overrides):
    base = {
        "session_id": "sid",
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "run_config": {},
        "assembly": None,
        "investable": None,
        "clean_returns": None,
        "factor_scores_dict": None,
        "returns_history": None,
        "is_result": None,
        "oos_result": None,
        "coverage_result": None,
        "regime_result": None,
        "optimize_result": None,
        "rebalance_result": None,
        "cost_result": None,
        "report_result": None,
        "current_step": "idle",
        "step_status": {},
        "step_results": {},
    }
    base.update(overrides)
    return session_store._PipelineSession(**base)


def test_when_require_field_missing_then_raises_value_error():
    session = _make_session()
    with pytest.raises(ValueError, match="assembly not set"):
        steps._require(session, "assembly")


def test_when_require_field_set_then_passes_silently():
    session = _make_session(assembly={"k": "v"})
    steps._require(session, "assembly")  # no raise


def test_when_require_first_missing_field_is_reported():
    session = _make_session(assembly={"k": "v"})
    with pytest.raises(ValueError, match="investable not set"):
        steps._require(session, "assembly", "investable", "clean_returns")


# ---------------------------------------------------------------------------
# _load_market_proxy_from_db
# ---------------------------------------------------------------------------


class _FakeDBManager:
    def __init__(self, rows_by_ticker: dict[str, list[tuple]]):
        self.rows_by_ticker = rows_by_ticker
        self.session_closed = False
        self.execute_calls: list[str] = []

    @contextmanager
    def get_session(self):
        session = MagicMock()
        # Each .execute() returns an object whose .all() yields the rows for
        # the *next* ticker in the iteration order (URTH, then SPY).
        ordered = list(self.rows_by_ticker.values())

        def _execute(_stmt):
            result = MagicMock()
            rows = ordered.pop(0) if ordered else []
            result.all.return_value = rows
            return result

        session.execute.side_effect = _execute
        try:
            yield session
        finally:
            self.session_closed = True


def test_when_no_instrument_matches_then_returns_none():
    db = _FakeDBManager({"URTH": [], "SPY": []})
    assert steps._load_market_proxy_from_db(db) is None
    assert db.session_closed is True


def test_when_first_ticker_has_rows_then_returns_dataframe():
    rows = [
        (date(2026, 1, 2), 100.0),
        (date(2026, 1, 3), 101.0),
    ]
    db = _FakeDBManager({"URTH": rows, "SPY": []})
    out = steps._load_market_proxy_from_db(db)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["close"]
    assert len(out) == 2
    assert db.session_closed is True


def test_when_first_ticker_empty_then_falls_back_to_next():
    rows = [(date(2026, 1, 2), 99.0)]
    db = _FakeDBManager({"URTH": [], "SPY": rows})
    out = steps._load_market_proxy_from_db(db)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 1


def test_when_custom_tickers_supplied_then_uses_them():
    rows = [(date(2026, 1, 2), 10.0)]
    db = _FakeDBManager({"VTI": rows})
    out = steps._load_market_proxy_from_db(db, tickers=("VTI",))
    assert isinstance(out, pd.DataFrame)
