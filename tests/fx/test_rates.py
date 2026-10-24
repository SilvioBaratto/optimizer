"""Tests for FX rate utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimizer.fx import (
    align_fx_rates,
    build_fx_pair_ticker,
    compute_cross_rate,
    invert_fx_rates,
    required_fx_currencies,
)


class TestBuildFxPairTicker:
    """Tests for build_fx_pair_ticker()."""

    def test_same_currency_returns_none(self) -> None:
        assert build_fx_pair_ticker("EUR", "EUR") is None
        assert build_fx_pair_ticker("usd", "USD") is None

    def test_from_usd_quotes_target_per_usd(self) -> None:
        # USDEUR=X is EUR per 1 USD, the to-per-from direction the converter
        # needs; EURUSD=X would be its reciprocal.
        assert build_fx_pair_ticker("USD", "EUR") == "USDEUR=X"
        assert build_fx_pair_ticker("USD", "JPY") == "USDJPY=X"

    @pytest.mark.parametrize(
        ("from_ccy", "to_ccy"),
        [("USD", "EUR"), ("EUR", "USD"), ("USD", "JPY"), ("GBP", "USD")],
    )
    def test_single_ticker_quotes_to_per_from(self, from_ccy: str, to_ccy: str) -> None:
        # Yahoo's "<BASE><QUOTE>=X" quotes QUOTE per 1 BASE, so a ticker that
        # needs no inversion has BASE == from_ccy and QUOTE == to_ccy.
        assert build_fx_pair_ticker(from_ccy, to_ccy) == f"{from_ccy}{to_ccy}=X"

    def test_to_usd(self) -> None:
        result = build_fx_pair_ticker("GBP", "USD")
        assert result == "GBPUSD=X"

    def test_cross_via_usd(self) -> None:
        result = build_fx_pair_ticker("GBP", "EUR", cross_via_usd=True)
        assert result == ("GBPUSD=X", "EURUSD=X")

    def test_direct_cross(self) -> None:
        result = build_fx_pair_ticker("GBP", "EUR", cross_via_usd=False)
        assert result == "GBPEUR=X"

    def test_case_insensitive(self) -> None:
        result = build_fx_pair_ticker("gbp", "usd")
        assert result == "GBPUSD=X"


class TestComputeCrossRate:
    """Tests for compute_cross_rate()."""

    def test_basic_cross(self) -> None:
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        gbp_usd = pd.Series([1.27, 1.28, 1.26, 1.29, 1.27], index=dates)
        eur_usd = pd.Series([1.09, 1.10, 1.08, 1.11, 1.09], index=dates)

        cross = compute_cross_rate(gbp_usd, eur_usd)

        # GBP/EUR = GBP/USD / EUR/USD
        expected = gbp_usd / eur_usd
        pd.testing.assert_series_equal(cross, expected)

    def test_misaligned_dates(self) -> None:
        dates_a = pd.date_range("2024-01-01", periods=5, freq="B")
        dates_b = pd.date_range("2024-01-02", periods=5, freq="B")
        a = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4], index=dates_a)
        b = pd.Series([2.0, 2.1, 2.2, 2.3, 2.4], index=dates_b)

        cross = compute_cross_rate(a, b)
        # Inner join: only overlapping dates
        assert len(cross) == 4


class TestComputeCrossRateInvalidLegs:
    """A zero, negative or non-finite leg is missing data, never an inf rate."""

    @pytest.mark.parametrize("bad", [0.0, -1.1, np.inf, np.nan])
    def test_bad_to_leg_gives_nan_not_inf(self, bad: float) -> None:
        dates = pd.bdate_range("2024-01-01", periods=2)
        gbp_usd = pd.Series([1.3, 1.3], index=dates)
        eur_usd = pd.Series([bad, 1.1], index=dates)

        cross = compute_cross_rate(gbp_usd, eur_usd)

        assert np.isnan(cross.iloc[0])
        assert cross.iloc[1] == pytest.approx(1.3 / 1.1)

    @pytest.mark.parametrize("bad", [0.0, -1.3, np.inf])
    def test_bad_from_leg_gives_nan(self, bad: float) -> None:
        dates = pd.bdate_range("2024-01-01", periods=2)
        gbp_usd = pd.Series([bad, 1.3], index=dates)
        eur_usd = pd.Series([1.1, 1.1], index=dates)

        cross = compute_cross_rate(gbp_usd, eur_usd)

        assert np.isnan(cross.iloc[0])
        assert np.isfinite(cross.iloc[1])


class TestAlignFxRates:
    """Tests for align_fx_rates()."""

    def test_basic_alignment(self) -> None:
        fx_dates = pd.date_range("2024-01-01", periods=10, freq="B")
        price_dates = pd.date_range("2024-01-01", periods=10, freq="B")
        fx = pd.DataFrame({"GBP": np.linspace(1.15, 1.20, 10)}, index=fx_dates)

        result = align_fx_rates(fx, price_dates)

        assert len(result) == len(price_dates)
        assert list(result.columns) == ["GBP"]

    def test_forward_fill_gaps(self) -> None:
        fx_dates = pd.to_datetime(["2024-01-02", "2024-01-05"])
        price_dates = pd.bdate_range("2024-01-02", periods=4)
        fx = pd.DataFrame({"GBP": [1.15, 1.18]}, index=fx_dates)

        result = align_fx_rates(fx, price_dates, fill_limit=3)

        # Jan 3, Jan 4 should be forward-filled from Jan 2
        assert not result["GBP"].isna().any()

    def test_fill_limit_respected(self) -> None:
        fx_dates = pd.to_datetime(["2024-01-02"])
        price_dates = pd.bdate_range("2024-01-02", periods=10)
        fx = pd.DataFrame({"GBP": [1.15]}, index=fx_dates)

        result = align_fx_rates(fx, price_dates, fill_limit=3)

        # First 4 should be filled (original + 3 forward), rest NaN
        assert result["GBP"].notna().sum() == 4

    def test_zero_fill_limit_means_exact_date_rates_only(self) -> None:
        fx_dates = pd.to_datetime(["2024-01-02", "2024-01-05"])
        price_dates = pd.bdate_range("2024-01-02", periods=4)
        fx = pd.DataFrame({"GBP": [1.15, 1.18]}, index=fx_dates)

        result = align_fx_rates(fx, price_dates, fill_limit=0)

        assert result["GBP"].tolist()[0] == 1.15
        assert result["GBP"].isna().tolist() == [False, True, True, False]

    def test_non_positive_and_non_finite_rates_are_missing(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=2)
        fx = pd.DataFrame(
            {"ZERO": [0.0, 1.15], "NEG": [-1.0, 1.15], "INF": [np.inf, 1.15]},
            index=dates,
        )

        result = align_fx_rates(fx, dates)

        # A bad first print has nothing to fill from, so it is NaN rather than
        # a 0, negative or inf rate that would silently price assets at 0/inf.
        assert result.iloc[0].isna().all()
        assert result.iloc[1].tolist() == [1.15, 1.15, 1.15]

    def test_bad_rate_is_filled_from_last_valid_rate(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=3)
        fx = pd.DataFrame({"GBP": [1.15, 0.0, 1.17]}, index=dates)

        result = align_fx_rates(fx, dates, fill_limit=5)

        assert result["GBP"].tolist() == [1.15, 1.15, 1.17]

    def test_tz_aware_prices_with_naive_rates(self) -> None:
        naive = pd.bdate_range("2024-01-02", periods=4)
        aware = naive.tz_localize("UTC")
        fx = pd.DataFrame({"GBP": [1.15, 1.16, 1.17, 1.18]}, index=naive)

        result = align_fx_rates(fx, aware)

        assert result["GBP"].tolist() == [1.15, 1.16, 1.17, 1.18]
        # Callers multiply against the price frame, so labels must stay the
        # price index's own, tz included.
        assert result.index.equals(aware)

    def test_naive_prices_with_tz_aware_rates(self) -> None:
        naive = pd.bdate_range("2024-01-02", periods=4)
        fx = pd.DataFrame(
            {"GBP": [1.15, 1.16, 1.17, 1.18]},
            index=naive.tz_localize("Europe/London"),
        )

        result = align_fx_rates(fx, naive)

        assert result["GBP"].tolist() == [1.15, 1.16, 1.17, 1.18]
        assert result.index.equals(naive)

    def test_tz_aware_price_dates_match_rates_by_local_date(self) -> None:
        # A daily bar stamped at local midnight is that local date; converting
        # to UTC first would shift London summer-time bars onto the prior day.
        naive = pd.bdate_range("2024-06-03", periods=3)
        aware = naive.tz_localize("Europe/London")
        fx = pd.DataFrame({"GBP": [1.1, 1.2, 1.3]}, index=naive)

        result = align_fx_rates(fx, aware)

        assert result["GBP"].tolist() == [1.1, 1.2, 1.3]

    @pytest.mark.parametrize(
        ("price_tz", "fx_tz"),
        [
            ("Europe/London", "UTC"),
            ("UTC", "Europe/London"),
            ("Asia/Tokyo", "America/New_York"),
        ],
    )
    def test_tz_aware_indexes_in_different_zones_match_by_local_date(
        self, price_tz: str, fx_tz: str
    ) -> None:
        # Comparing instants would put London summer-time midnight bars (23:00
        # UTC the evening before) one rate behind their UTC-midnight prints and
        # carry yesterday's rate with no warning.
        naive = pd.bdate_range("2024-06-03", periods=5)
        fx = pd.DataFrame(
            {"GBP": [1.1, 1.2, 1.3, 1.4, 1.5]}, index=naive.tz_localize(fx_tz)
        )
        prices = naive.tz_localize(price_tz)

        result = align_fx_rates(fx, prices)

        assert result["GBP"].tolist() == [1.1, 1.2, 1.3, 1.4, 1.5]
        assert result.index.equals(prices)

    def test_empty_raises(self) -> None:
        from optimizer.exceptions import DataError

        price_dates = pd.bdate_range("2024-01-01", periods=5)
        with pytest.raises(DataError, match="empty"):
            align_fx_rates(pd.DataFrame(), price_dates)


class TestAlignFxRatesAsOf:
    """Rates observed on dates outside the price calendar still seed the fill."""

    def test_rate_on_non_price_date_fills_next_price_date(self) -> None:
        # 2024-03-29 is a market holiday (no price) but FX still printed.
        fx = pd.DataFrame(
            {"GBP": [1.10, 1.20, 1.40]},
            index=pd.to_datetime(["2024-03-28", "2024-03-29", "2024-04-02"]),
        )
        price_dates = pd.to_datetime(["2024-04-01", "2024-04-02"])

        result = align_fx_rates(fx, price_dates)

        assert result["GBP"].tolist() == [1.20, 1.40]

    def test_rate_before_first_price_date_is_not_lost(self) -> None:
        fx = pd.DataFrame(
            {"GBP": [1.10, 1.30]},
            index=pd.to_datetime(["2023-12-29", "2024-01-03"]),
        )
        price_dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])

        result = align_fx_rates(fx, price_dates)

        assert result["GBP"].tolist() == [1.10, 1.10, 1.30]

    def test_month_end_uses_latest_print_not_prior_month_end(self) -> None:
        fx_dates = pd.bdate_range("2024-01-31", "2024-02-28")
        fx = pd.DataFrame(
            {"GBP": np.linspace(1.30, 1.58, len(fx_dates))}, index=fx_dates
        )
        price_dates = pd.to_datetime(["2024-01-31", "2024-02-29"])

        result = align_fx_rates(fx, price_dates)

        assert result["GBP"].iloc[0] == pytest.approx(1.30)
        assert result["GBP"].iloc[1] == pytest.approx(1.58)

    def test_fill_limit_is_measured_in_business_days_not_price_rows(self) -> None:
        # A single stale print must not carry across a sparse (monthly) index
        # just because only one price row sits between it and the target.
        fx = pd.DataFrame({"GBP": [1.30]}, index=pd.to_datetime(["2024-01-31"]))
        price_dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29"])

        result = align_fx_rates(fx, price_dates, fill_limit=5)

        assert result["GBP"].isna().tolist() == [False, True, True]

    def test_weekend_gap_counts_one_business_day(self) -> None:
        fx = pd.DataFrame({"GBP": [1.30]}, index=pd.to_datetime(["2024-01-05"]))
        # Friday print -> Mon..Fri is 1..5 business days later.
        price_dates = pd.bdate_range("2024-01-08", periods=6)

        result = align_fx_rates(fx, price_dates, fill_limit=5)

        assert result["GBP"].isna().tolist() == [False] * 5 + [True]


class TestAlignFxRatesFillLimitBounds:
    """``None`` means no age cap; a negative cap is rejected, not silently NaN."""

    def test_none_fill_limit_carries_rate_without_an_age_cap(self) -> None:
        fx = pd.DataFrame({"GBP": [1.30]}, index=pd.to_datetime(["2024-01-31"]))
        price_dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-12-31"])

        result = align_fx_rates(fx, price_dates, fill_limit=None)

        assert result["GBP"].tolist() == [1.30, 1.30, 1.30]

    def test_none_fill_limit_does_not_fill_before_first_print(self) -> None:
        fx = pd.DataFrame({"GBP": [1.30]}, index=pd.to_datetime(["2024-01-05"]))
        price_dates = pd.to_datetime(["2024-01-02", "2024-01-05", "2024-01-10"])

        result = align_fx_rates(fx, price_dates, fill_limit=None)

        assert result["GBP"].isna().tolist() == [True, False, False]

    def test_negative_fill_limit_raises(self) -> None:
        from optimizer.exceptions import DataError

        dates = pd.bdate_range("2024-01-02", periods=3)
        fx = pd.DataFrame({"GBP": [1.1, 1.2, 1.3]}, index=dates)

        with pytest.raises(DataError, match="fill_limit"):
            align_fx_rates(fx, dates, fill_limit=-1)


class TestAlignFxRatesIndexTypes:
    """The as-of fill needs date axes; anything else must not fill look-ahead."""

    def test_date_string_price_index_matches_rates_by_date(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=5)
        fx = pd.DataFrame({"GBP": [1.1, 1.2, 1.3, 1.4, 1.5]}, index=dates)
        prices = pd.Index([d.strftime("%Y-%m-%d") for d in dates])

        result = align_fx_rates(fx, prices)

        # A future print must never fill an earlier row.
        assert result["GBP"].tolist() == [1.1, 1.2, 1.3, 1.4, 1.5]
        assert result.index.equals(prices)

    def test_date_string_rates_index_matches_datetime_prices(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=3)
        fx = pd.DataFrame(
            {"GBP": [1.1, 1.2, 1.3]},
            index=pd.Index([d.strftime("%Y-%m-%d") for d in dates]),
        )

        result = align_fx_rates(fx, dates)

        assert result["GBP"].tolist() == [1.1, 1.2, 1.3]

    def test_integer_price_index_raises(self) -> None:
        from optimizer.exceptions import DataError

        dates = pd.bdate_range("2024-01-02", periods=5)
        fx = pd.DataFrame({"GBP": [1.1, 1.2, 1.3, 1.4, 1.5]}, index=dates)

        with pytest.raises(DataError, match="price_index"):
            align_fx_rates(fx, pd.RangeIndex(5))  # type: ignore[arg-type]

    def test_unparseable_price_index_raises(self) -> None:
        from optimizer.exceptions import DataError

        dates = pd.bdate_range("2024-01-02", periods=2)
        fx = pd.DataFrame({"GBP": [1.1, 1.2]}, index=dates)

        with pytest.raises(DataError, match="price_index"):
            align_fx_rates(fx, pd.Index(["AAA", "BBB"]))  # type: ignore[arg-type]

    def test_integer_rates_index_raises(self) -> None:
        from optimizer.exceptions import DataError

        fx = pd.DataFrame({"GBP": [1.1, 1.2]})

        with pytest.raises(DataError, match="fx_rates"):
            align_fx_rates(fx, pd.bdate_range("2024-01-02", periods=2))


class TestInvertFxRates:
    """Tests for invert_fx_rates()."""

    def test_series_reciprocal(self) -> None:
        s = pd.Series([1.25, 2.0, 0.5])
        result = invert_fx_rates(s)
        pd.testing.assert_series_equal(result, pd.Series([0.8, 0.5, 2.0]))

    def test_dataframe_reciprocal(self) -> None:
        df = pd.DataFrame({"GBP": [1.25, 2.0], "USD": [4.0, 0.5]})
        result = invert_fx_rates(df)
        expected = pd.DataFrame({"GBP": [0.8, 0.5], "USD": [0.25, 2.0]})
        pd.testing.assert_frame_equal(result, expected)

    def test_zero_maps_to_nan_not_inf(self) -> None:
        s = pd.Series([0.0, 2.0])
        result = invert_fx_rates(s)
        assert np.isnan(result.iloc[0])
        assert result.iloc[1] == 0.5

    def test_double_inversion_roundtrip(self) -> None:
        s = pd.Series([1.15, 1.17, 0.92])
        pd.testing.assert_series_equal(invert_fx_rates(invert_fx_rates(s)), s)


class TestRequiredFxCurrencies:
    """Tests for required_fx_currencies()."""

    def test_mixed_currencies(self) -> None:
        cmap = {"LLOY.L": "GBP", "ORA.PA": "EUR", "SPY": "USD"}
        result = required_fx_currencies(cmap, "EUR")
        assert result == {"GBP", "USD"}

    def test_all_base(self) -> None:
        cmap = {"ORA.PA": "EUR", "BNP.PA": "EUR"}
        result = required_fx_currencies(cmap, "EUR")
        assert result == set()

    def test_case_insensitive(self) -> None:
        cmap = {"LLOY.L": "gbp"}
        result = required_fx_currencies(cmap, "eur")
        assert result == {"GBP"}

    def test_minor_units_resolve_to_major(self) -> None:
        # GBp -> GBP, ZAc -> ZAR; the caller must fetch the major-unit pairs.
        cmap = {"LLOY.L": "GBp", "NPN.JO": "ZAc", "ORA.PA": "EUR"}
        result = required_fx_currencies(cmap, "EUR")
        assert result == {"GBP", "ZAR"}

    def test_pence_ticker_with_gbp_base_needs_no_fx(self) -> None:
        # GBp resolves to GBP == base, so no FX pair is required.
        cmap = {"LLOY.L": "GBp"}
        result = required_fx_currencies(cmap, "GBP")
        assert result == set()


class TestBuildFxPairTickerMinorUnits:
    """build_fx_pair_ticker resolves sub-unit codes to their major pair."""

    def test_pence_resolves_to_gbp_pair(self) -> None:
        assert build_fx_pair_ticker("GBp", "USD") == "GBPUSD=X"

    def test_cents_resolves_to_zar_pair(self) -> None:
        assert build_fx_pair_ticker("ZAc", "USD") == "ZARUSD=X"

    def test_pence_to_gbp_base_returns_none(self) -> None:
        # GBp -> GBP; same major currency as target -> no pair.
        assert build_fx_pair_ticker("GBp", "GBP") is None
