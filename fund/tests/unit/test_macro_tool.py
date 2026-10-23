"""``get_macro_series`` tool over ``MacroRegimeRepository``.

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

from fund.tools.macro import get_macro_series, load_macro_frame

_ASOF = dt.date(2024, 1, 5)


def _seed_series(
    db_session, series_id: str, observations: dict[dt.date, float]
) -> None:
    for day, value in observations.items():
        db_session.add(FredObservation(series_id=series_id, date=day, value=value))
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

    def test_reports_requested_names(self, db_session) -> None:
        _two_series_panel(db_session)

        result = get_macro_series(db_session, ["T10Y2Y", "VIXCLS"], _ASOF)

        assert result["data"]["requested"] == ["T10Y2Y", "VIXCLS"]

    def test_unaligned_series_report_partial_coverage(self, db_session) -> None:
        # Two series on disjoint dates → the union index carries NaNs where a
        # series has no observation, so overall coverage drops below 1.0.
        _seed_series(db_session, "AAA", {dt.date(2024, 1, 2): 1.0})
        _seed_series(db_session, "BBB", {dt.date(2024, 1, 3): 2.0})

        result = get_macro_series(db_session, ["AAA", "BBB"], _ASOF)

        assert result["ok"] is True
        data = result["data"]
        assert data["rows"] == 2
        assert data["cols"] == 2
        assert data["coverage"] < 1.0
        assert data["na_counts"] == {"AAA": 1, "BBB": 1}

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


def _daily(start: dt.date, days: int, value: float) -> dict[dt.date, float]:
    return {start + dt.timedelta(days=i): value for i in range(days)}


class TestGetMacroSeriesPointInTime:
    """Observations dated before ``asof`` may still have been unpublished."""

    def test_quarterly_gdp_print_is_hidden_until_its_release_window(
        self, db_session
    ) -> None:
        # The Q2 advance estimate dated 2026-04-01 is released around 30 July.
        _seed_series(
            db_session,
            "A191RL1Q225SBEA",
            {
                dt.date(2025, 10, 1): 2.0,
                dt.date(2026, 1, 1): 1.5,
                dt.date(2026, 4, 1): 3.0,
            },
        )

        result = get_macro_series(db_session, ["A191RL1Q225SBEA"], "2026-04-02")

        assert result["ok"] is True
        assert result["data"]["rows"] == 1
        assert result["data"]["index_end"].startswith("2025-10-01")

    def test_quarterly_gdp_print_is_visible_once_published(self, db_session) -> None:
        _seed_series(
            db_session,
            "A191RL1Q225SBEA",
            {dt.date(2026, 1, 1): 1.5, dt.date(2026, 4, 1): 3.0},
        )

        result = get_macro_series(db_session, ["A191RL1Q225SBEA"], "2026-09-01")

        assert result["data"]["index_end"].startswith("2026-04-01")

    def test_monthly_series_hides_the_period_still_being_reported(
        self, db_session
    ) -> None:
        # OECD CLI for September is published mid-October, so at 2026-09-30 the
        # latest known month is July.
        _seed_series(
            db_session,
            "USALOLITOAASTSAM",
            {
                dt.date(2026, 5, 1): 99.1,
                dt.date(2026, 6, 1): 99.3,
                dt.date(2026, 7, 1): 99.4,
                dt.date(2026, 8, 1): 99.6,
                dt.date(2026, 9, 1): 99.7,
            },
        )

        result = get_macro_series(db_session, ["USALOLITOAASTSAM"], "2026-09-30")

        assert result["data"]["index_end"].startswith("2026-07-01")

    def test_nber_recession_flag_is_an_ex_post_label(self, db_session) -> None:
        # NBER announced the Feb-2020 peak in June 2020; the daily flag cannot be
        # known on 2020-04-15.
        _seed_series(db_session, "USRECDM", _daily(dt.date(2020, 3, 1), 30, 1.0))

        result = get_macro_series(db_session, ["USRECDM"], "2020-04-15")

        assert result["ok"] is True
        assert result["data"]["missing"] == ["USRECDM"]

    def test_daily_series_keeps_the_previous_day(self, db_session) -> None:
        _seed_series(db_session, "DGS10", _daily(dt.date(2024, 1, 1), 5, 4.0))

        result = get_macro_series(db_session, ["DGS10"], dt.date(2024, 1, 6))

        assert result["data"]["rows"] == 5

    def test_load_macro_frame_can_skip_the_publication_lag(self, db_session) -> None:
        _seed_series(
            db_session,
            "A191RL1Q225SBEA",
            {dt.date(2026, 1, 1): 1.5, dt.date(2026, 4, 1): 3.0},
        )

        frame, missing = load_macro_frame(
            db_session, ["A191RL1Q225SBEA"], "2026-04-02", point_in_time=False
        )

        assert missing == []
        assert len(frame) == 2


class TestGetMacroSeriesLiveRead:
    """The publication lag is not a replay-only filter: a live run is lagged too."""

    def test_live_read_hides_the_latest_monthly_prints_like_a_replay(
        self, db_session
    ) -> None:
        today = dt.date.today()
        _seed_series(
            db_session,
            "USALOLITOAASTSAM",
            {today - dt.timedelta(days=days): 99.0 for days in (100, 70, 40, 10)},
        )

        result = get_macro_series(db_session, ["USALOLITOAASTSAM"], today)

        assert result["data"]["rows"] == 2
        assert result["data"]["index_end"].startswith(
            (today - dt.timedelta(days=70)).isoformat()
        )

    def test_live_read_never_shows_the_ex_post_recession_flags(
        self, db_session
    ) -> None:
        today = dt.date.today()
        _seed_series(
            db_session, "USREC", _daily(today - dt.timedelta(days=30), 30, 0.0)
        )

        result = get_macro_series(db_session, ["USREC"], today)

        assert result["data"]["missing"] == ["USREC"]
