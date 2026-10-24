"""``get_prices`` tool over ``YFinanceRepository``.

The tool is the first node of the allocator critical path. These tests pin its
contract as a pure function of a seeded SQLite DB:

* it hands back a shape/coverage **summary**, never the raw price frame;
* it respects the ``asof`` upper bound (no look-ahead);
* a ticker with no instrument / no prices is **flagged**, not raised;
* empty ``tickers`` and an unknown price column degrade to ``{ok: false}``;
* identical inputs on identical data give identical output (determinism);
* with a ``base_currency`` the allocator panel is FX-converted (minor units
  rescaled) before any return is derived, so the optimiser sees the same
  base-currency series the NAV tracker marks;
* a delisted or stale ticker is excluded and flagged ``missing`` instead of being
  forward-filled into a flat, zero-variance tail the optimiser would favour;
* the summary carries per-ticker derived statistics (level, horizon changes,
  realised volatility, drawdown), so the economist reads numbers rather than only
  column names and coverage.
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest
from portopt_db.models.market_data.yfinance_data import PriceHistory
from portopt_db.models.universe.universe import Exchange, Instrument
from portopt_db.repositories.market_data.fx_rates_repository import FxRatesRepository

from fund.tools.moments import estimate_moments
from fund.tools.optimize import optimize_portfolio
from fund.tools.prices import get_prices, load_price_frame
from fund.tools.risk import backtest

_ASOF = dt.date(2024, 1, 5)


def _seed_instrument(
    db_session, ticker: str, *, delisted_at: dt.date | None = None
) -> Instrument:
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
        delisted_at=delisted_at,
    )
    db_session.add(inst)
    db_session.flush()
    return inst


def _seed_prices(
    db_session,
    inst: Instrument,
    closes: dict[dt.date, float],
    *,
    volume: int = 100,
    price_unit: str | None = None,
) -> None:
    for day, close in closes.items():
        db_session.add(
            PriceHistory(
                instrument_id=inst.id,
                date=day,
                close=close,
                volume=volume,
                price_unit=price_unit,
            )
        )
    db_session.flush()


def _seed_fx(db_session, quote: str, rates: dict[dt.date, float]) -> None:
    """Seed EUR-base rates: each value is EUR per one unit of ``quote``."""
    FxRatesRepository(db_session).upsert_rates(
        [
            {"base_currency": "EUR", "quote_currency": quote, "date": d, "rate": r}
            for d, r in rates.items()
        ]
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


_FX_START = dt.date(2024, 1, 2)
_FX_DAYS = 11
_FX_ASOF = _FX_START + dt.timedelta(days=_FX_DAYS - 1)


def _fx_days() -> list[dt.date]:
    return [_FX_START + dt.timedelta(days=i) for i in range(_FX_DAYS)]


def _usd_closes() -> pd.Series:
    """A USD asset oscillating +/-2% around 100."""
    return pd.Series([100.0 * (1.0 + 0.02 * ((-1) ** i)) for i in range(_FX_DAYS)])


def _eur_per_usd() -> pd.Series:
    """EUR per USD compounding +1% a day, so USD returns understate EUR ones."""
    return pd.Series([0.90 * 1.01**i for i in range(_FX_DAYS)])


def _expected_mean_returns() -> tuple[float, float]:
    """Mean daily return of the USD asset in USD and in EUR, derived independently."""
    usd = _usd_closes()
    eur = usd * _eur_per_usd()
    return float(usd.pct_change().mean()), float(eur.pct_change().mean())


def _seed_usd_and_eur_panel(db_session) -> None:
    """A USD asset (``USDX``), a wavy EUR asset (``EURX``) and EUR/USD rates."""
    usd = _seed_instrument(db_session, "USDX")
    eur = _seed_instrument(db_session, "EURX")
    days = _fx_days()
    _seed_prices(
        db_session, usd, dict(zip(days, _usd_closes(), strict=True)), price_unit="USD"
    )
    _seed_prices(
        db_session,
        eur,
        {d: 50.0 * (1.0 + 0.03 * ((-1) ** (i // 2))) for i, d in enumerate(days)},
        price_unit="EUR",
    )
    _seed_fx(db_session, "USD", dict(zip(days, _eur_per_usd(), strict=True)))


class TestLoadPriceFrameBaseCurrency:
    def test_foreign_and_minor_unit_prices_are_converted_to_base(
        self, db_session
    ) -> None:
        days = [dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 4)]
        us = _seed_instrument(db_session, "US1")
        lon = _seed_instrument(db_session, "LON.L")
        de = _seed_instrument(db_session, "DE1")
        _seed_prices(
            db_session,
            us,
            dict(zip(days, [10.0, 11.0, 12.0], strict=True)),
            price_unit="USD",
        )
        _seed_prices(
            db_session,
            lon,
            dict(zip(days, [500.0, 510.0, 520.0], strict=True)),
            price_unit="GBp",
        )
        _seed_prices(
            db_session,
            de,
            dict(zip(days, [20.0, 21.0, 22.0], strict=True)),
            price_unit="EUR",
        )
        _seed_fx(db_session, "USD", dict(zip(days, [0.90, 0.90, 0.80], strict=True)))
        _seed_fx(db_session, "GBP", dict(zip(days, [1.20, 1.20, 1.20], strict=True)))

        frame, missing = load_price_frame(
            db_session, _ASOF, ["US1", "LON.L", "DE1"], base_currency="EUR"
        )

        assert missing == []
        assert frame["US1"].tolist() == pytest.approx([9.0, 9.9, 9.6])
        # GBp is pence: 500p = 5.00 GBP = 6.00 EUR at 1.20.
        assert frame["LON.L"].tolist() == pytest.approx([6.0, 6.12, 6.24])
        assert frame["DE1"].tolist() == pytest.approx([20.0, 21.0, 22.0])

    def test_without_base_currency_prices_stay_in_listing_currency(
        self, db_session
    ) -> None:
        us = _seed_instrument(db_session, "US1")
        _seed_prices(db_session, us, {dt.date(2024, 1, 2): 10.0}, price_unit="USD")

        frame, _ = load_price_frame(db_session, _ASOF, ["US1"])

        assert frame["US1"].tolist() == [10.0]

    def test_non_price_fields_are_never_fx_converted(self, db_session) -> None:
        us = _seed_instrument(db_session, "US1")
        _seed_prices(
            db_session, us, {dt.date(2024, 1, 2): 10.0}, volume=555, price_unit="USD"
        )
        _seed_fx(db_session, "USD", {dt.date(2024, 1, 2): 0.50})

        frame, _ = load_price_frame(
            db_session, _ASOF, ["US1"], field="volume", base_currency="EUR"
        )

        assert frame["US1"].tolist() == [555.0]

    def test_foreign_panel_with_no_fx_rates_fails_loud(self, db_session) -> None:
        us = _seed_instrument(db_session, "US1")
        _seed_prices(db_session, us, {dt.date(2024, 1, 2): 10.0}, price_unit="USD")

        result = estimate_moments(db_session, _ASOF, ["US1"], base_currency="EUR")

        assert result["ok"] is False
        assert "FxCoverageError" in result["error"]


class TestAllocatorToolsSeeBaseCurrencyReturns:
    def test_fixture_fx_move_is_material(self) -> None:
        raw, eur = _expected_mean_returns()

        assert eur - raw > 0.005

    def test_estimate_moments_mu_includes_the_fx_return(self, db_session) -> None:
        _seed_usd_and_eur_panel(db_session)
        raw_mean, eur_mean = _expected_mean_returns()

        raw = estimate_moments(db_session, _FX_ASOF, ["USDX", "EURX"])
        eur = estimate_moments(
            db_session, _FX_ASOF, ["USDX", "EURX"], base_currency="EUR"
        )

        assert raw["data"]["mu"]["USDX"] == pytest.approx(raw_mean)
        assert eur["data"]["mu"]["USDX"] == pytest.approx(eur_mean)

    def test_optimize_portfolio_metrics_include_the_fx_return(self, db_session) -> None:
        _seed_usd_and_eur_panel(db_session)
        raw_mean, eur_mean = _expected_mean_returns()

        raw = optimize_portfolio(db_session, _FX_ASOF, ["USDX"])
        eur = optimize_portfolio(db_session, _FX_ASOF, ["USDX"], base_currency="EUR")

        assert raw["data"]["metrics"]["mean"] == pytest.approx(raw_mean)
        assert eur["data"]["metrics"]["mean"] == pytest.approx(eur_mean)

    def test_backtest_metrics_include_the_fx_return(self, db_session) -> None:
        _seed_usd_and_eur_panel(db_session)
        raw_mean, eur_mean = _expected_mean_returns()

        raw = backtest(db_session, _FX_ASOF, {"USDX": 1.0})
        eur = backtest(db_session, _FX_ASOF, {"USDX": 1.0}, base_currency="EUR")

        assert raw["data"]["metrics"]["mean"] == pytest.approx(raw_mean)
        assert eur["data"]["metrics"]["mean"] == pytest.approx(eur_mean)

    def test_universe_filter_forwards_base_currency_to_the_loader(
        self, db_session, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from fund.tools import universe as universe_mod

        seen: dict[str, object] = {}

        def _spy(session, asof, tickers, **kwargs):
            seen.update(kwargs)
            return pd.DataFrame(), []

        monkeypatch.setattr(universe_mod, "load_price_frame", _spy)

        universe_mod.universe_filter(db_session, _ASOF, ["AAA"], base_currency="EUR")

        assert seen["base_currency"] == "EUR"


_LIVE_START = dt.date(2023, 11, 1)
_LIVE_DAYS = 60
_LIVE_ASOF = _LIVE_START + dt.timedelta(days=_LIVE_DAYS - 1)


def _wavy_closes(phase: float, days: int) -> dict[dt.date, float]:
    """A deterministic oscillating close series starting at ``_LIVE_START``."""
    import math

    return {
        _LIVE_START + dt.timedelta(days=i): 100.0
        * (1.0 + 0.03 * math.sin(i / 2 + phase))
        for i in range(days)
    }


def _seed_live_pair_and_dead_ticker(db_session, *, dead_bars: int) -> None:
    """Two tickers trading through ``_LIVE_ASOF`` plus ``DEAD`` that stops early."""
    _seed_prices(
        db_session, _seed_instrument(db_session, "LIVA"), _wavy_closes(0.0, 60)
    )
    _seed_prices(
        db_session, _seed_instrument(db_session, "LIVB"), _wavy_closes(1.3, 60)
    )
    _seed_prices(
        db_session,
        _seed_instrument(db_session, "DEAD"),
        _wavy_closes(2.6, dead_bars),
    )


class TestDeadAndStaleTickersAreNotInvestable:
    def test_stale_ticker_is_excluded_and_flagged_missing(self, db_session) -> None:
        _seed_live_pair_and_dead_ticker(db_session, dead_bars=10)

        frame, missing = load_price_frame(
            db_session, _LIVE_ASOF, ["LIVA", "LIVB", "DEAD"]
        )

        assert list(frame.columns) == ["LIVA", "LIVB"]
        assert missing == ["DEAD"]

    def test_optimizer_never_weights_a_ticker_that_stopped_trading(
        self, db_session
    ) -> None:
        _seed_live_pair_and_dead_ticker(db_session, dead_bars=10)

        result = optimize_portfolio(db_session, _LIVE_ASOF, ["LIVA", "LIVB", "DEAD"])

        assert result["ok"] is True
        assert set(result["data"]["weights"]) == {"LIVA", "LIVB"}
        assert result["data"]["missing"] == ["DEAD"]

    def test_delisted_ticker_is_excluded_even_with_recent_bars(
        self, db_session
    ) -> None:
        _seed_prices(
            db_session, _seed_instrument(db_session, "LIVA"), _wavy_closes(0.0, 60)
        )
        gone = _seed_instrument(
            db_session, "GONE", delisted_at=_LIVE_ASOF - dt.timedelta(days=3)
        )
        _seed_prices(db_session, gone, _wavy_closes(1.3, 60))

        frame, missing = load_price_frame(db_session, _LIVE_ASOF, ["LIVA", "GONE"])

        assert list(frame.columns) == ["LIVA"]
        assert missing == ["GONE"]

    def test_ticker_delisted_after_asof_is_still_investable_at_asof(
        self, db_session
    ) -> None:
        later = _seed_instrument(
            db_session, "LATER", delisted_at=_LIVE_ASOF + dt.timedelta(days=30)
        )
        _seed_prices(db_session, later, _wavy_closes(0.0, 60))

        frame, missing = load_price_frame(db_session, _LIVE_ASOF, ["LATER"])

        assert list(frame.columns) == ["LATER"]
        assert missing == []

    def test_short_holiday_gap_does_not_exclude_a_ticker(self, db_session) -> None:
        _seed_prices(
            db_session, _seed_instrument(db_session, "LIVA"), _wavy_closes(0.0, 60)
        )
        _seed_prices(
            db_session, _seed_instrument(db_session, "HOLI"), _wavy_closes(1.3, 56)
        )

        frame, missing = load_price_frame(db_session, _LIVE_ASOF, ["LIVA", "HOLI"])

        assert list(frame.columns) == ["LIVA", "HOLI"]
        assert missing == []

    def test_recent_null_closes_do_not_hide_a_stale_last_real_close(
        self, db_session
    ) -> None:
        _seed_prices(
            db_session, _seed_instrument(db_session, "LIVA"), _wavy_closes(0.0, 60)
        )
        inst = _seed_instrument(db_session, "NULC")
        for i, day in enumerate(_wavy_closes(1.3, 60)):
            db_session.add(
                PriceHistory(
                    instrument_id=inst.id,
                    date=day,
                    close=100.0 + i if i < 40 else None,
                    volume=100,
                )
            )
        db_session.flush()

        frame, missing = load_price_frame(db_session, _LIVE_ASOF, ["LIVA", "NULC"])

        assert list(frame.columns) == ["LIVA"]
        assert missing == ["NULC"]

    def test_sparse_field_does_not_make_a_trading_ticker_stale(
        self, db_session
    ) -> None:
        inst = _seed_instrument(db_session, "DIVS")
        for day in _wavy_closes(0.0, 60):
            db_session.add(
                PriceHistory(
                    instrument_id=inst.id,
                    date=day,
                    close=100.0,
                    dividends=0.5 if day == _LIVE_START else None,
                )
            )
        db_session.flush()

        frame, missing = load_price_frame(
            db_session, _LIVE_ASOF, ["DIVS"], field="dividends"
        )

        assert list(frame.columns) == ["DIVS"]
        assert missing == []


_STATS_ASOF = dt.date(2024, 1, 5)


def _stat_path() -> dict[dt.date, float]:
    """Peak of 150, a trough of 100, then a partial recovery to 120."""
    return {
        dt.date(2023, 1, 4): 80.0,
        dt.date(2023, 6, 1): 150.0,
        dt.date(2023, 10, 4): 100.0,
        dt.date(2023, 12, 4): 110.0,
        dt.date(2024, 1, 4): 120.0,
    }


class TestGetPricesStatistics:
    def test_horizon_changes_are_simple_returns_to_the_last_close(
        self, db_session
    ) -> None:
        _seed_prices(db_session, _seed_instrument(db_session, "AAA"), _stat_path())

        result = get_prices(db_session, _STATS_ASOF, ["AAA"])

        stats = result["data"]["stats"]["AAA"]
        assert stats["last"] == pytest.approx(120.0)
        assert stats["last_date"] == "2024-01-04"
        assert stats["change_1m"] == pytest.approx(120.0 / 110.0 - 1.0)
        assert stats["change_3m"] == pytest.approx(0.2)
        assert stats["change_12m"] == pytest.approx(0.5)

    def test_drawdown_is_a_non_negative_magnitude_from_the_running_peak(
        self, db_session
    ) -> None:
        _seed_prices(db_session, _seed_instrument(db_session, "AAA"), _stat_path())

        stats = get_prices(db_session, _STATS_ASOF, ["AAA"])["data"]["stats"]["AAA"]

        assert stats["drawdown"] == pytest.approx(1.0 - 120.0 / 150.0)
        assert stats["max_drawdown"] == pytest.approx(1.0 - 100.0 / 150.0)
        assert stats["max_drawdown"] >= stats["drawdown"] >= 0.0

    def test_series_at_its_peak_has_zero_drawdown(self, db_session) -> None:
        _seed_prices(
            db_session,
            _seed_instrument(db_session, "AAA"),
            {dt.date(2023, 12, 4): 100.0, dt.date(2024, 1, 4): 110.0},
        )

        stats = get_prices(db_session, _STATS_ASOF, ["AAA"])["data"]["stats"]["AAA"]

        assert stats["drawdown"] == 0.0
        assert stats["max_drawdown"] == 0.0

    def test_realised_vol_is_annualised_from_daily_returns(self, db_session) -> None:
        rng = np.random.default_rng(7)
        days = [dt.date(2023, 3, 1) + dt.timedelta(days=i) for i in range(300)]
        closes = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, len(days)))
        _seed_prices(
            db_session,
            _seed_instrument(db_session, "AAA"),
            dict(zip(days, closes.tolist(), strict=True)),
        )

        stats = get_prices(db_session, dt.date(2024, 1, 1), ["AAA"])["data"]["stats"][
            "AAA"
        ]

        trailing = pd.Series(closes).tail(253)
        expected = trailing.pct_change().dropna().std(ddof=1) * np.sqrt(252)
        assert stats["realised_vol"] == pytest.approx(expected)

    def test_each_ticker_gets_its_own_statistics(self, db_session) -> None:
        _two_ticker_panel(db_session)

        stats = get_prices(db_session, _ASOF, ["AAA", "BBB"])["data"]["stats"]

        assert set(stats) == {"AAA", "BBB"}
        assert stats["AAA"]["last"] == pytest.approx(12.0)
        assert stats["BBB"]["last"] == pytest.approx(22.0)

    def test_missing_ticker_has_no_statistics(self, db_session) -> None:
        _two_ticker_panel(db_session)

        data = get_prices(db_session, _ASOF, ["AAA", "GHOST"])["data"]

        assert data["missing"] == ["GHOST"]
        assert set(data["stats"]) == {"AAA"}

    def test_statistics_are_bounded_scalars_never_a_series(self, db_session) -> None:
        _seed_prices(db_session, _seed_instrument(db_session, "AAA"), _stat_path())

        stats = get_prices(db_session, _STATS_ASOF, ["AAA"])["data"]["stats"]["AAA"]

        assert len(stats) <= 12
        assert all(
            v is None or isinstance(v, (int, float, str)) for v in stats.values()
        )

    def test_single_bar_degrades_to_null_statistics_not_nan(self, db_session) -> None:
        _seed_prices(
            db_session,
            _seed_instrument(db_session, "AAA"),
            {dt.date(2024, 1, 4): 12.0},
        )

        result = get_prices(db_session, _STATS_ASOF, ["AAA"])

        stats = result["data"]["stats"]["AAA"]
        assert stats["last"] == pytest.approx(12.0)
        assert stats["change_1m"] is None
        assert stats["realised_vol"] is None
        json.dumps(result, allow_nan=False)

    def test_non_price_field_reports_levels_without_return_statistics(
        self, db_session
    ) -> None:
        _two_ticker_panel(db_session)

        stats = get_prices(db_session, _ASOF, ["AAA"], field="volume")["data"]["stats"]

        assert stats["AAA"]["last"] == pytest.approx(100.0)
        assert "realised_vol" not in stats["AAA"]
        assert "drawdown" not in stats["AAA"]
