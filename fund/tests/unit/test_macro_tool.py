"""T3.5 — ``get_macro_series`` tool over ``MacroRegimeRepository``.

The macro tool feeds the regime/views agents. These tests pin its contract as a
pure function of a seeded SQLite DB of FRED observations:

* it hands back a shape/coverage **summary**, never the raw observation frame;
* it respects the ``asof`` upper bound (no look-ahead);
* an unknown series / a series with no observations is **flagged**, not raised;
* empty ``names`` degrades to ``{ok: false}``;
* a malformed ``asof`` degrades to ``{ok: false}``, never an exception;
* identical inputs on identical data give identical output (determinism).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
from portopt_db.models.macro.macro_regime import FredObservation

from fund.tools.macro import get_macro_series

_ASOF = dt.date(2024, 1, 5)


def _seed_series(
    db_session, series_id: str, observations: dict[dt.date, float]
) -> None:
    for day, value in observations.items():
        db_session.add(
            FredObservation(series_id=series_id, date=day, value=value)
        )
    db_session.flush()


def _two_series_panel(db_session) -> None:
    _seed_series(
        db_session,
        "T10Y2Y",
        {
            dt.date(2024, 1, 2): 0.10,
            dt.date(2024, 1, 3): 0.12,
            dt.date(2024, 1, 4): 0.11,
        },
    )
    _seed_series(
        db_session,
        "VIXCLS",
        {
            dt.date(2024, 1, 2): 13.0,
            dt.date(2024, 1, 3): 14.0,
            dt.date(2024, 1, 4): 12.5,
        },
    )


class TestGetMacroSeriesHappyPath:
    def test_returns_summary_not_raw_frame(self, db_session) -> None:
        _two_series_panel(db_session)

        result = get_macro_series(db_session, ["T10Y2Y", "VIXCLS"], _ASOF)

        assert result["ok"] is True
        data = result["data"]
        assert data["rows"] == 3
        assert data["cols"] == 2
        assert data["columns"] == ["T10Y2Y", "VIXCLS"]
        assert data["asof"] == "2024-01-05"
        assert data["missing"] == []
        # No raw frame leaked back to the model.
        assert not any(isinstance(v, pd.DataFrame) for v in data.values())

    def test_columns_follow_requested_order(self, db_session) -> None:
        _two_series_panel(db_session)

        result = get_macro_series(db_session, ["VIXCLS", "T10Y2Y"], _ASOF)

        assert result["data"]["columns"] == ["VIXCLS", "T10Y2Y"]

    def test_respects_asof_upper_bound(self, db_session) -> None:
        _seed_series(
            db_session,
            "T10Y2Y",
            {
                dt.date(2024, 1, 2): 0.10,
                dt.date(2024, 1, 3): 0.12,
                dt.date(2024, 1, 10): 9.99,  # strictly after asof → excluded
            },
        )

        result = get_macro_series(db_session, ["T10Y2Y"], dt.date(2024, 1, 5))

        assert result["ok"] is True
        assert result["data"]["rows"] == 2
        assert result["data"]["index_end"].startswith("2024-01-03")

    def test_accepts_iso_string_asof(self, db_session) -> None:
        _two_series_panel(db_session)

        result = get_macro_series(db_session, ["T10Y2Y"], "2024-01-05")

        assert result["ok"] is True
        assert result["data"]["asof"] == "2024-01-05"

    def test_datetime_asof_collapses_to_date(self, db_session) -> None:
        _two_series_panel(db_session)

        result = get_macro_series(
            db_session, ["T10Y2Y"], dt.datetime(2024, 1, 5, 16, 30)
        )

        assert result["ok"] is True
        assert result["data"]["asof"] == "2024-01-05"


class TestGetMacroSeriesFlagsAndErrors:
    def test_unknown_series_is_flagged_not_raised(self, db_session) -> None:
        _two_series_panel(db_session)

        result = get_macro_series(db_session, ["T10Y2Y", "NOPE"], _ASOF)

        assert result["ok"] is True
        assert result["data"]["missing"] == ["NOPE"]
        assert result["data"]["columns"] == ["T10Y2Y"]

    def test_series_with_only_null_values_is_flagged(self, db_session) -> None:
        _seed_series(db_session, "EMPTY", {dt.date(2024, 1, 2): None})  # type: ignore[dict-item]

        result = get_macro_series(db_session, ["EMPTY"], _ASOF)

        assert result["ok"] is True
        assert result["data"]["missing"] == ["EMPTY"]
        assert result["data"]["columns"] == []

    def test_empty_names_is_error(self, db_session) -> None:
        result = get_macro_series(db_session, [], _ASOF)

        assert result["ok"] is False
        assert "error" in result

    def test_bad_asof_string_is_error_not_raised(self, db_session) -> None:
        result = get_macro_series(db_session, ["T10Y2Y"], "not-a-date")

        assert result["ok"] is False
        assert "error" in result


class TestGetMacroSeriesDeterminism:
    def test_identical_inputs_give_identical_output(self, db_session) -> None:
        _two_series_panel(db_session)

        first = get_macro_series(db_session, ["T10Y2Y", "VIXCLS"], _ASOF)
        second = get_macro_series(db_session, ["T10Y2Y", "VIXCLS"], _ASOF)

        assert first == second
