"""Tests for deterministic cross-listed ticker deduplication in data_assembly.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("typer")

from cli._currency import CURRENCY_DEDUP_PRIORITY, currency_dedup_rank
from cli.data_assembly import _dedup_fundamentals_df

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
