"""T3.1 — ``get_prices`` tool over ``YFinanceRepository``.

The tool is the first node of the allocator critical path. These tests pin its
contract as a pure function of a seeded SQLite DB:

* it hands back a shape/coverage **summary**, never the raw price frame;
* it respects the ``asof`` upper bound (no look-ahead);
* a ticker with no instrument / no prices is **flagged**, not raised;
* empty ``tickers`` and an unknown price column degrade to ``{ok: false}``;
* identical inputs on identical data give identical output (determinism).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
from portopt_db.models.market_data.yfinance_data import PriceHistory
from portopt_db.models.universe.universe import Exchange, Instrument

from fund.tools.prices import get_prices

_ASOF = dt.date(2024, 1, 5)


def _seed_instrument(db_session, ticker: str) -> Instrument:
    ex = Exchange(name=f"EX-{ticker}")
    db_session.add(ex)
    db_session.flush()
    inst = Instrument(
        ticker=ticker,
        short_name=ticker,
        exchange_id=ex.id,
        instrument_type="EQUITY",
        asset_class="equity",
        yfinance_ticker=ticker,
    )
    db_session.add(inst)
    db_session.flush()
    return inst


def _seed_prices(
    db_session, inst: Instrument, closes: dict[dt.date, float], *, volume: int = 100
) -> None:
    for day, close in closes.items():
        db_session.add(
            PriceHistory(instrument_id=inst.id, date=day, close=close, volume=volume)
        )
    db_session.flush()


def _two_ticker_panel(db_session) -> None:
    aaa = _seed_instrument(db_session, "AAA")
    bbb = _seed_instrument(db_session, "BBB")
    _seed_prices(
        db_session,
        aaa,
        {
            dt.date(2024, 1, 2): 10.0,
            dt.date(2024, 1, 3): 11.0,
            dt.date(2024, 1, 4): 12.0,
        },
    )
    _seed_prices(
        db_session,
        bbb,
        {
            dt.date(2024, 1, 2): 20.0,
            dt.date(2024, 1, 3): 21.0,
            dt.date(2024, 1, 4): 22.0,
        },
    )


class TestGetPricesHappyPath:
    def test_returns_summary_not_raw_frame(self, db_session) -> None:
        _two_ticker_panel(db_session)

        result = get_prices(db_session, _ASOF, ["AAA", "BBB"])

        assert result["ok"] is True
        data = result["data"]
        assert data["rows"] == 3
        assert data["cols"] == 2
        assert data["columns"] == ["AAA", "BBB"]
        assert data["field"] == "close"
        assert data["asof"] == "2024-01-05"
        assert data["missing"] == []
        # No raw frame leaked back to the model.
        assert not any(isinstance(v, pd.DataFrame) for v in data.values())

    def test_columns_follow_requested_order(self, db_session) -> None:
        _two_ticker_panel(db_session)

        result = get_prices(db_session, _ASOF, ["BBB", "AAA"])

        assert result["data"]["columns"] == ["BBB", "AAA"]

    def test_respects_asof_upper_bound(self, db_session) -> None:
        aaa = _seed_instrument(db_session, "AAA")
        _seed_prices(
            db_session,
            aaa,
            {
                dt.date(2024, 1, 2): 10.0,
                dt.date(2024, 1, 3): 11.0,
                dt.date(2024, 1, 10): 99.0,  # strictly after asof → excluded
            },
        )

        result = get_prices(db_session, dt.date(2024, 1, 5), ["AAA"])

        assert result["ok"] is True
        assert result["data"]["rows"] == 2
        assert result["data"]["index_end"].startswith("2024-01-03")

    def test_accepts_iso_string_asof(self, db_session) -> None:
        _two_ticker_panel(db_session)

        result = get_prices(db_session, "2024-01-05", ["AAA"])

        assert result["ok"] is True
        assert result["data"]["asof"] == "2024-01-05"

    def test_datetime_asof_collapses_to_date(self, db_session) -> None:
        _two_ticker_panel(db_session)

        result = get_prices(db_session, dt.datetime(2024, 1, 5, 16, 30), ["AAA"])

        assert result["ok"] is True
        assert result["data"]["asof"] == "2024-01-05"

    def test_field_selection_pulls_requested_column(self, db_session) -> None:
        aaa = _seed_instrument(db_session, "AAA")
        _seed_prices(db_session, aaa, {dt.date(2024, 1, 2): 10.0}, volume=555)

        result = get_prices(db_session, _ASOF, ["AAA"], field="volume")

        assert result["ok"] is True
        assert result["data"]["field"] == "volume"
        assert result["data"]["coverage"] == 1.0


class TestGetPricesFlagsAndErrors:
    def test_missing_instrument_is_flagged_not_raised(self, db_session) -> None:
        _two_ticker_panel(db_session)

        result = get_prices(db_session, _ASOF, ["AAA", "ZZZ"])

        assert result["ok"] is True
        assert result["data"]["missing"] == ["ZZZ"]
        assert result["data"]["columns"] == ["AAA"]

    def test_instrument_without_prices_is_flagged(self, db_session) -> None:
        _seed_instrument(db_session, "AAA")  # instrument, zero price rows

        result = get_prices(db_session, _ASOF, ["AAA"])

        assert result["ok"] is True
        assert result["data"]["missing"] == ["AAA"]
        assert result["data"]["columns"] == []

    def test_field_all_null_flags_ticker_as_missing(self, db_session) -> None:
        # The instrument has priced rows, but the requested column is NULL on every
        # one — the loader drops the None cells, leaving no values, so the ticker
        # lands in ``missing`` (rows exist but the field is empty).
        aaa = _seed_instrument(db_session, "AAA")
        _seed_prices(db_session, aaa, {dt.date(2024, 1, 2): 10.0})  # dividends unset

        result = get_prices(db_session, _ASOF, ["AAA"], field="dividends")

        assert result["ok"] is True
        assert result["data"]["missing"] == ["AAA"]
        assert result["data"]["columns"] == []

    def test_empty_tickers_is_error(self, db_session) -> None:
        result = get_prices(db_session, _ASOF, [])

        assert result["ok"] is False
        assert "error" in result

    def test_unknown_field_is_error(self, db_session) -> None:
        _two_ticker_panel(db_session)

        result = get_prices(db_session, _ASOF, ["AAA"], field="vwap")

        assert result["ok"] is False
        assert "vwap" in result["error"]

    def test_bad_asof_string_is_error_not_raised(self, db_session) -> None:
        result = get_prices(db_session, "not-a-date", ["AAA"])

        assert result["ok"] is False
        assert "error" in result


class TestGetPricesDeterminism:
    def test_identical_inputs_give_identical_output(self, db_session) -> None:
        _two_ticker_panel(db_session)

        first = get_prices(db_session, _ASOF, ["AAA", "BBB"])
        second = get_prices(db_session, _ASOF, ["AAA", "BBB"])

        assert first == second
