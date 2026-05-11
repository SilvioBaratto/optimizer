"""Tests for deterministic cross-listed ticker deduplication in data_assembly.py."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("typer")

from research._currency import CURRENCY_DEDUP_PRIORITY, currency_dedup_rank
from research.data_assembly import (
    _DEDUP_DROP_THRESHOLD_PCT,
    _dedup_fundamentals_df,
    _pivot_with_dedup,
)

# ---------------------------------------------------------------------------
# currency_dedup_rank
# ---------------------------------------------------------------------------


class TestCurrencyDedupRank:
    def test_usd_highest_priority(self) -> None:
        assert currency_dedup_rank("USD") == 0

    def test_gbp_rank(self) -> None:
        assert currency_dedup_rank("GBP") == 1

    def test_eur_rank(self) -> None:
        assert currency_dedup_rank("EUR") == 2

    def test_gbx_rank(self) -> None:
        assert currency_dedup_rank("GBX") == 3

    def test_gbp_variant_same_as_gbx(self) -> None:
        assert currency_dedup_rank("GBp") == 3

    def test_unknown_currency(self) -> None:
        assert currency_dedup_rank("SEK") == 99

    def test_none_currency(self) -> None:
        assert currency_dedup_rank(None) == 99

    def test_priority_ordering(self) -> None:
        ordered = sorted(CURRENCY_DEDUP_PRIORITY.items(), key=lambda x: x[1])
        codes = [c for c, _ in ordered]
        assert codes[0] == "USD"


# ---------------------------------------------------------------------------
# _dedup_fundamentals_df
# ---------------------------------------------------------------------------


def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a fundamentals-like DataFrame with _raw_currency column."""
    df = pd.DataFrame(rows).set_index("ticker")
    return df


class TestDedupFundamentalsDf:
    def test_usd_wins_over_eur(self) -> None:
        """USD listing should win over EUR for same ticker."""
        df = _make_df(
            [
                {
                    "ticker": "ASML",
                    "market_cap": 100.0,
                    "trailing_eps": 5.0,
                    "_raw_currency": "EUR",
                    "exchange": "AMS",
                },
                {
                    "ticker": "ASML",
                    "market_cap": 100.0,
                    "trailing_eps": 5.0,
                    "_raw_currency": "USD",
                    "exchange": "NMS",
                },
            ]
        )
        result = _dedup_fundamentals_df(df)

        assert len(result) == 1
        assert result.loc["ASML", "exchange"] == "NMS"

    def test_gbp_wins_over_gbx(self) -> None:
        """GBP listing should win over GBX for same ticker."""
        df = _make_df(
            [
                {
                    "ticker": "SHELL",
                    "market_cap": 200.0,
                    "trailing_eps": 3.0,
                    "_raw_currency": "GBX",
                    "exchange": "LSE",
                },
                {
                    "ticker": "SHELL",
                    "market_cap": 200.0,
                    "trailing_eps": 3.0,
                    "_raw_currency": "GBP",
                    "exchange": "LSE-GBP",
                },
            ]
        )
        result = _dedup_fundamentals_df(df)

        assert len(result) == 1
        assert result.loc["SHELL", "exchange"] == "LSE-GBP"

    def test_fewer_nans_wins_as_tiebreaker(self) -> None:
        """When currencies are identical, fewer NaNs should win."""
        df = _make_df(
            [
                {
                    "ticker": "VOD",
                    "market_cap": np.nan,
                    "trailing_eps": np.nan,
                    "operating_cashflow": np.nan,
                    "_raw_currency": "USD",
                    "exchange": "NYSE-bad",
                },
                {
                    "ticker": "VOD",
                    "market_cap": 50.0,
                    "trailing_eps": 2.0,
                    "operating_cashflow": 10.0,
                    "_raw_currency": "USD",
                    "exchange": "NYSE-good",
                },
            ]
        )
        result = _dedup_fundamentals_df(df)

        assert len(result) == 1
        assert result.loc["VOD", "exchange"] == "NYSE-good"

    def test_three_way_dedup(self) -> None:
        """USD should win in a three-way contest (USD, EUR, GBX)."""
        df = _make_df(
            [
                {
                    "ticker": "RIO",
                    "market_cap": 300.0,
                    "_raw_currency": "GBX",
                    "exchange": "LSE",
                },
                {
                    "ticker": "RIO",
                    "market_cap": 300.0,
                    "_raw_currency": "EUR",
                    "exchange": "AMS",
                },
                {
                    "ticker": "RIO",
                    "market_cap": 300.0,
                    "_raw_currency": "USD",
                    "exchange": "NYSE",
                },
            ]
        )
        result = _dedup_fundamentals_df(df)

        assert len(result) == 1
        assert result.loc["RIO", "exchange"] == "NYSE"

    def test_unique_tickers_pass_through(self) -> None:
        """Single-listing tickers should pass through unchanged."""
        df = _make_df(
            [
                {
                    "ticker": "AAPL",
                    "market_cap": 3000.0,
                    "_raw_currency": "USD",
                    "exchange": "NMS",
                },
                {
                    "ticker": "MSFT",
                    "market_cap": 2800.0,
                    "_raw_currency": "USD",
                    "exchange": "NMS",
                },
            ]
        )
        result = _dedup_fundamentals_df(df)

        assert len(result) == 2
        assert set(result.index) == {"AAPL", "MSFT"}

    def test_none_currency_loses(self) -> None:
        """Row with None currency should lose to any known currency."""
        df = _make_df(
            [
                {
                    "ticker": "XYZ",
                    "market_cap": 10.0,
                    "_raw_currency": None,
                    "exchange": "UNKNOWN",
                },
                {
                    "ticker": "XYZ",
                    "market_cap": 10.0,
                    "_raw_currency": "EUR",
                    "exchange": "AMS",
                },
            ]
        )
        result = _dedup_fundamentals_df(df)

        assert len(result) == 1
        assert result.loc["XYZ", "exchange"] == "AMS"

    def test_helper_columns_removed(self) -> None:
        """Temporary columns (_raw_currency, _ccy_rank, _nan_count) must not leak."""
        df = _make_df(
            [
                {
                    "ticker": "ASML",
                    "market_cap": 100.0,
                    "_raw_currency": "USD",
                    "exchange": "NMS",
                },
                {
                    "ticker": "ASML",
                    "market_cap": 100.0,
                    "_raw_currency": "EUR",
                    "exchange": "AMS",
                },
            ]
        )
        result = _dedup_fundamentals_df(df)

        for col in ("_raw_currency", "_ccy_rank", "_nan_count"):
            assert col not in result.columns

    def test_deterministic_regardless_of_input_order(self) -> None:
        """Result should be identical regardless of row ordering."""
        base_rows = [
            {
                "ticker": "ASML",
                "market_cap": 100.0,
                "_raw_currency": "EUR",
                "exchange": "AMS",
            },
            {
                "ticker": "ASML",
                "market_cap": 100.0,
                "_raw_currency": "USD",
                "exchange": "NMS",
            },
            {
                "ticker": "AAPL",
                "market_cap": 3000.0,
                "_raw_currency": "USD",
                "exchange": "NMS",
            },
        ]

        df_original = _make_df(base_rows)
        result1 = _dedup_fundamentals_df(df_original)

        # Reverse the order
        df_reversed = _make_df(list(reversed(base_rows)))
        result2 = _dedup_fundamentals_df(df_reversed)

        # Sort both to compare (index order may differ)
        pd.testing.assert_frame_equal(result1.sort_index(), result2.sort_index())
        assert result1.loc["ASML", "exchange"] == "NMS"
        assert result2.loc["ASML", "exchange"] == "NMS"

    def test_no_duplicates_in_result_index(self) -> None:
        """Result index must have no duplicates."""
        df = _make_df(
            [
                {
                    "ticker": "A",
                    "market_cap": 1.0,
                    "_raw_currency": "USD",
                    "exchange": "X",
                },
                {
                    "ticker": "A",
                    "market_cap": 2.0,
                    "_raw_currency": "EUR",
                    "exchange": "Y",
                },
                {
                    "ticker": "B",
                    "market_cap": 3.0,
                    "_raw_currency": "GBX",
                    "exchange": "Z",
                },
                {
                    "ticker": "B",
                    "market_cap": 4.0,
                    "_raw_currency": "GBP",
                    "exchange": "W",
                },
            ]
        )
        result = _dedup_fundamentals_df(df)

        assert not result.index.duplicated().any()

    def test_threshold_constant_is_five_percent(self) -> None:
        assert _DEDUP_DROP_THRESHOLD_PCT == 0.05

    def test_mixed_dedup_and_unique(self) -> None:
        """Mix of duplicated and unique tickers."""
        df = _make_df(
            [
                {
                    "ticker": "ASML",
                    "market_cap": 100.0,
                    "_raw_currency": "EUR",
                    "exchange": "AMS",
                },
                {
                    "ticker": "ASML",
                    "market_cap": 100.0,
                    "_raw_currency": "USD",
                    "exchange": "NMS",
                },
                {
                    "ticker": "AAPL",
                    "market_cap": 3000.0,
                    "_raw_currency": "USD",
                    "exchange": "NMS",
                },
                {
                    "ticker": "SHELL",
                    "market_cap": 200.0,
                    "_raw_currency": "GBX",
                    "exchange": "LSE",
                },
                {
                    "ticker": "SHELL",
                    "market_cap": 200.0,
                    "_raw_currency": "GBP",
                    "exchange": "LSE-GBP",
                },
            ]
        )
        result = _dedup_fundamentals_df(df)

        assert len(result) == 3
        assert result.loc["ASML", "exchange"] == "NMS"
        assert result.loc["AAPL", "exchange"] == "NMS"
        assert result.loc["SHELL", "exchange"] == "LSE-GBP"


# ---------------------------------------------------------------------------
# _pivot_with_dedup — structured logging + 5% threshold guard
# ---------------------------------------------------------------------------


def _pivot_input(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestPivotWithDedupGuard:
    def test_when_no_duplicates_then_no_log_no_raise(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        df = _pivot_input(
            [
                {"date": "2024-01-01", "ticker": "AAPL", "close": 1.0, "_ccy_rank": 0},
                {"date": "2024-01-01", "ticker": "MSFT", "close": 2.0, "_ccy_rank": 0},
            ]
        )
        with caplog.at_level(logging.INFO, logger="research.data"):
            out = _pivot_with_dedup(df, "date", "ticker", "close", "test")
        assert out.shape == (1, 2)
        assert not any("dedup" in r.getMessage().lower() for r in caplog.records)

    def test_when_drop_below_threshold_then_logger_info_with_extra(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # 1 duplicate out of 100 rows = 1% — below 5%.
        rows = [
            {"date": "2024-01-01", "ticker": f"T{i}", "close": float(i), "_ccy_rank": 0}
            for i in range(100)
        ]
        rows.append(
            {"date": "2024-01-01", "ticker": "T0", "close": 99.0, "_ccy_rank": 1}
        )
        df = _pivot_input(rows)
        with caplog.at_level(logging.INFO, logger="research.data"):
            _pivot_with_dedup(df, "date", "ticker", "close", "assemble_prices")
        info_records = [
            r
            for r in caplog.records
            if r.levelname == "INFO" and "dedup" in r.getMessage().lower()
        ]
        assert info_records, "expected one structured INFO log on dedup"
        rec = info_records[0]
        assert rec.__dict__.get("dedup_name") == "assemble_prices"
        assert rec.__dict__.get("n_dropped") == 1
        assert rec.__dict__.get("n_before") == 101
        assert abs(rec.__dict__.get("dropped_pct", 0) - (1 / 101)) < 1e-9

    def test_when_drop_exceeds_5pct_then_value_error_raised(self) -> None:
        # 10 rows total, 6 duplicates (60%) → exceeds 5%.
        rows = [
            {"date": "2024-01-01", "ticker": "T1", "close": 1.0, "_ccy_rank": 0},
            {"date": "2024-01-01", "ticker": "T2", "close": 2.0, "_ccy_rank": 0},
            {"date": "2024-01-01", "ticker": "T3", "close": 3.0, "_ccy_rank": 0},
            {"date": "2024-01-01", "ticker": "T4", "close": 4.0, "_ccy_rank": 0},
        ] + [{"date": "2024-01-01", "ticker": "T1", "close": 1.0, "_ccy_rank": 1}] * 6
        df = _pivot_input(rows)
        with pytest.raises(ValueError, match="5%"):
            _pivot_with_dedup(df, "date", "ticker", "close", "assemble_prices")

    def test_when_value_error_then_message_carries_name_and_pct(self) -> None:
        rows = [
            {"date": "2024-01-01", "ticker": "T1", "close": 1.0, "_ccy_rank": 0},
        ] + [{"date": "2024-01-01", "ticker": "T1", "close": 1.0, "_ccy_rank": 1}] * 5
        df = _pivot_input(rows)
        with pytest.raises(ValueError) as excinfo:
            _pivot_with_dedup(df, "date", "ticker", "close", "assemble_prices")
        msg = str(excinfo.value)
        assert "assemble_prices" in msg
        assert "5/6" in msg or "5" in msg
        assert "%" in msg

    def test_when_empty_input_then_no_division_by_zero(self) -> None:
        df = pd.DataFrame(columns=["date", "ticker", "close", "_ccy_rank"])
        out = _pivot_with_dedup(df, "date", "ticker", "close", "test")
        assert out.empty


# ---------------------------------------------------------------------------
# _pivot_with_dedup — boundary, production-rate, exact-message assertions
# ---------------------------------------------------------------------------


def _pivot_rows_with_drop_ratio(n_unique: int, n_duplicates: int) -> pd.DataFrame:
    """Build a pivot input with `n_unique` unique tickers + `n_duplicates`
    extra rows colliding on the first ticker (lower-priority rank).
    """
    rows: list[dict] = [
        {"date": "2024-01-01", "ticker": f"T{i}", "close": float(i), "_ccy_rank": 0}
        for i in range(n_unique)
    ]
    rows.extend(
        {"date": "2024-01-01", "ticker": "T0", "close": 99.0, "_ccy_rank": 1}
        for _ in range(n_duplicates)
    )
    return pd.DataFrame(rows)


class TestPivotWithDedup:
    def test_when_zero_drops_then_silent(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        df = _pivot_rows_with_drop_ratio(n_unique=10, n_duplicates=0)
        with caplog.at_level(logging.INFO, logger="research.data"):
            _pivot_with_dedup(df, "date", "ticker", "close", "noop")
        assert not any(
            r.levelname == "INFO" and "dedup" in r.getMessage().lower()
            for r in caplog.records
        )

    def test_when_drop_below_threshold_then_record_extra_payload_indexable(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # 1 dup / 101 ≈ 0.99% — well below 5%.
        df = _pivot_rows_with_drop_ratio(n_unique=100, n_duplicates=1)
        with caplog.at_level(logging.INFO, logger="research.data"):
            _pivot_with_dedup(df, "date", "ticker", "close", "assemble_prices")
        info = [
            r
            for r in caplog.records
            if r.levelname == "INFO" and "dedup" in r.getMessage().lower()
        ]
        assert len(info) == 1
        rec = info[0]
        # `record.dropped_pct` must be accessible directly via getattr
        # (LogRecord exposes `extra={...}` keys as instance attributes).
        assert rec.dedup_name == "assemble_prices"
        assert rec.n_dropped == 1
        assert rec.n_before == 101
        assert rec.dropped_pct == pytest.approx(1 / 101)

    def test_when_production_rate_18_per_10000_then_no_raise(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # 18 / 10018 ≈ 0.18% — current production observation.
        df = _pivot_rows_with_drop_ratio(n_unique=10_000, n_duplicates=18)
        with caplog.at_level(logging.INFO, logger="research.data"):
            _pivot_with_dedup(df, "date", "ticker", "close", "assemble_prices")
        # Must not raise; must emit the structured info log.
        info = [
            r
            for r in caplog.records
            if r.levelname == "INFO" and "dedup" in r.getMessage().lower()
        ]
        assert len(info) == 1
        assert info[0].dropped_pct < 0.05

    def test_when_drop_exactly_at_threshold_then_no_raise(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # 5 dups / 100 unique = 5/(100+5) ≈ 4.76% — strictly below 5%.
        # To get exactly 5%: 5 dups / 100 total → unique=95, dups=5.
        df = _pivot_rows_with_drop_ratio(n_unique=95, n_duplicates=5)
        with caplog.at_level(logging.INFO, logger="research.data"):
            _pivot_with_dedup(df, "date", "ticker", "close", "boundary")
        info = [
            r
            for r in caplog.records
            if r.levelname == "INFO" and "dedup" in r.getMessage().lower()
        ]
        assert len(info) == 1
        assert info[0].dropped_pct == pytest.approx(0.05)

    def test_when_drop_just_above_threshold_then_value_error_raised(self) -> None:
        # 51 dups / 1000 total = 5.1% — just above 5%.
        df = _pivot_rows_with_drop_ratio(n_unique=949, n_duplicates=51)
        with pytest.raises(ValueError, match="5%"):
            _pivot_with_dedup(df, "date", "ticker", "close", "above_boundary")

    def test_when_value_error_then_message_matches_exact_phrase(self) -> None:
        df = _pivot_rows_with_drop_ratio(n_unique=10, n_duplicates=10)
        with pytest.raises(
            ValueError,
            match=r"expected ≤ 5%; check upstream normalization",
        ):
            _pivot_with_dedup(df, "date", "ticker", "close", "phrase_check")
