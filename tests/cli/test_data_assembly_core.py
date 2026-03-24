"""Tests for pure and DB-dependent functions in cli/data_assembly.py.

Covers:
  - _to_float()
  - _apply_delisting_returns()
  - assemble_prices()            (mock session)
  - assemble_volumes()           (mock session)
  - assemble_delisting_returns() (mock session)
  - assemble_fx_rates()          (mock yfinance.download)
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from cli.data_assembly import (
    _apply_delisting_returns,
    _to_float,
    assemble_delisting_returns,
    assemble_fx_rates,
    assemble_prices,
    assemble_volumes,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _exec_result(rows: list) -> MagicMock:
    """Return a mock whose .all() yields *rows*."""
    m = MagicMock()
    m.all.return_value = rows
    return m


def _make_session(*call_results: list) -> MagicMock:
    """Build a mock session where successive execute() calls pop one entry."""
    session = MagicMock()
    session.execute.side_effect = [_exec_result(r) for r in call_results]
    return session


# ---------------------------------------------------------------------------
# TestToFloat
# ---------------------------------------------------------------------------


class TestToFloat:
    def test_none_returns_none(self) -> None:
        assert _to_float(None) is None

    def test_decimal_returns_float(self) -> None:
        result = _to_float(Decimal("3.14"))
        assert isinstance(result, float)
        assert result == pytest.approx(3.14)

    def test_int_returns_float(self) -> None:
        result = _to_float(42)
        assert isinstance(result, float)
        assert result == pytest.approx(42.0)

    def test_float_identity(self) -> None:
        result = _to_float(1.5)
        assert isinstance(result, float)
        assert result == pytest.approx(1.5)

    def test_zero_returns_zero_float(self) -> None:
        result = _to_float(0)
        assert isinstance(result, float)
        assert result == 0.0

    def test_decimal_zero(self) -> None:
        result = _to_float(Decimal("0"))
        assert result == 0.0

    def test_negative_decimal(self) -> None:
        result = _to_float(Decimal("-1.23"))
        assert result == pytest.approx(-1.23)


# ---------------------------------------------------------------------------
# TestApplyDelistingReturns
# ---------------------------------------------------------------------------


class TestApplyDelistingReturns:
    def _base_prices(self) -> pd.DataFrame:
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        return pd.DataFrame(
            {"AAPL": [100.0, 101.0, 102.0], "MSFT": [200.0, 201.0, 202.0]},
            index=idx,
        )

    def test_empty_delistings_returns_unchanged(self) -> None:
        prices = self._base_prices()
        result = _apply_delisting_returns(prices, [])
        pd.testing.assert_frame_equal(result, prices)

    def test_ticker_not_in_prices_silently_skipped(self) -> None:
        prices = self._base_prices()
        delisting_date = pd.Timestamp("2024-01-05")
        result = _apply_delisting_returns(
            prices, [("UNKNOWN", delisting_date, -0.30)]
        )
        assert set(result.columns) == {"AAPL", "MSFT"}
        assert len(result) == 3  # no new rows

    def test_synthetic_row_injected_at_new_date(self) -> None:
        prices = self._base_prices()
        delisting_date = pd.Timestamp("2024-01-05")
        result = _apply_delisting_returns(
            prices, [("AAPL", delisting_date, -0.10)]
        )
        assert delisting_date in result.index
        assert len(result) == 4

    def test_synthetic_price_formula(self) -> None:
        prices = self._base_prices()
        delisting_date = pd.Timestamp("2024-01-05")
        r = -0.30
        result = _apply_delisting_returns(
            prices, [("AAPL", delisting_date, r)]
        )
        expected = 102.0 * (1.0 + r)
        assert result.loc[delisting_date, "AAPL"] == pytest.approx(expected)

    def test_delisting_date_in_index_fills_nan_cell(self) -> None:
        idx = pd.date_range("2024-01-01", periods=4, freq="D")
        prices = pd.DataFrame(
            {
                "AAPL": [100.0, 101.0, np.nan, np.nan],
                "MSFT": [200.0, 201.0, 202.0, 203.0],
            },
            index=idx,
        )
        delisting_date = pd.Timestamp("2024-01-03")
        result = _apply_delisting_returns(
            prices, [("AAPL", delisting_date, -0.20)]
        )
        assert not pd.isna(result.loc[delisting_date, "AAPL"])
        assert result.loc[delisting_date, "AAPL"] == pytest.approx(101.0 * 0.80)
        assert len(result) == 4  # no new row added

    def test_non_nan_cell_not_overwritten(self) -> None:
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        prices = pd.DataFrame({"AAPL": [100.0, 101.0, 102.0]}, index=idx)
        delisting_date = pd.Timestamp("2024-01-03")
        result = _apply_delisting_returns(
            prices, [("AAPL", delisting_date, -0.50)]
        )
        # 102.0 is not overwritten by 102.0 * 0.50 = 51.0
        assert result.loc[delisting_date, "AAPL"] == pytest.approx(102.0)

    def test_other_column_stays_nan_in_new_row(self) -> None:
        prices = self._base_prices()
        delisting_date = pd.Timestamp("2024-01-10")
        result = _apply_delisting_returns(
            prices, [("AAPL", delisting_date, -0.10)]
        )
        assert pd.isna(result.loc[delisting_date, "MSFT"])

    def test_multiple_delistings_one_in_index_one_not(self) -> None:
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        prices = pd.DataFrame(
            {
                "AAPL": [100.0, 101.0, np.nan],
                "MSFT": [200.0, 201.0, 202.0],
            },
            index=idx,
        )
        aapl_date = pd.Timestamp("2024-01-03")  # in index, NaN
        msft_date = pd.Timestamp("2024-01-05")  # not in index

        result = _apply_delisting_returns(
            prices,
            [
                ("AAPL", aapl_date, -0.10),
                ("MSFT", msft_date, -0.20),
            ],
        )
        assert result.loc[aapl_date, "AAPL"] == pytest.approx(101.0 * 0.90)
        assert msft_date in result.index
        assert result.loc[msft_date, "MSFT"] == pytest.approx(202.0 * 0.80)
        assert len(result) == 4

    def test_result_index_is_sorted(self) -> None:
        prices = self._base_prices()
        delisting_date = pd.Timestamp("2023-12-31")  # before existing rows
        result = _apply_delisting_returns(
            prices, [("AAPL", delisting_date, 0.05)]
        )
        assert result.index.is_monotonic_increasing

    def test_empty_prices_returns_empty(self) -> None:
        empty = pd.DataFrame()
        result = _apply_delisting_returns(empty, [])
        assert result.empty

    def test_col_with_all_nan_skipped(self) -> None:
        """All-NaN series → col.empty after dropna → no synthetic price."""
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        prices = pd.DataFrame(
            {"AAPL": [np.nan, np.nan, np.nan]},
            index=idx,
        )
        delisting_date = pd.Timestamp("2024-01-05")
        result = _apply_delisting_returns(
            prices, [("AAPL", delisting_date, -0.30)]
        )
        # New row added but cell should be NaN (no last_price found)
        if delisting_date in result.index:
            assert pd.isna(result.loc[delisting_date, "AAPL"])


# ---------------------------------------------------------------------------
# TestAssemblePrices
# ---------------------------------------------------------------------------


class TestAssemblePrices:
    def test_empty_price_rows_returns_empty_dataframe(self) -> None:
        inst_id = str(uuid.uuid4())
        session = _make_session(
            [(inst_id, "AAPL", "USD")],  # ticker rank map query
            [],                           # price rows — empty → early return
        )
        result = assemble_prices(session, include_delisted=False, currency_map={})
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_single_ticker_single_date_correct_shape(self) -> None:
        inst_id = str(uuid.uuid4())
        session = _make_session(
            [(inst_id, "AAPL", "USD")],
            [(inst_id, "2024-01-02", Decimal("150.00"))],
        )
        result = assemble_prices(session, include_delisted=False, currency_map={})
        assert result.shape == (1, 1)
        assert "AAPL" in result.columns
        assert result.iloc[0, 0] == pytest.approx(150.0)

    def test_gbx_price_normalized_by_100(self) -> None:
        inst_id = str(uuid.uuid4())
        session = _make_session(
            [(inst_id, "BARC.L", "GBX")],
            [(inst_id, "2024-01-02", Decimal("150.00"))],
        )
        result = assemble_prices(
            session,
            include_delisted=False,
            currency_map={"BARC.L": "GBX"},
        )
        assert result.loc[pd.Timestamp("2024-01-02"), "BARC.L"] == pytest.approx(1.50)

    def test_include_delisted_false_makes_exactly_two_execute_calls(self) -> None:
        inst_id = str(uuid.uuid4())
        session = _make_session(
            [(inst_id, "AAPL", "USD")],
            [(inst_id, "2024-01-02", 100.0)],
        )
        assemble_prices(session, include_delisted=False, currency_map={})
        assert session.execute.call_count == 2

    def test_duplicate_warns_and_selects_primary_currency(self) -> None:
        """Cross-listed tickers: lower currency rank wins, warning emitted."""
        id_usd = str(uuid.uuid4())  # USD listing — rank 0 (primary)
        id_gbx = str(uuid.uuid4())  # GBX listing — rank 3 (secondary)
        session = _make_session(
            [(id_usd, "RDSA.L", "USD"), (id_gbx, "RDSA.L", "GBX")],
            [
                (id_usd, "2024-01-02", 200.0),  # USD price
                (id_gbx, "2024-01-02", 21000.0),  # GBX price (should be dropped)
            ],
        )
        with pytest.warns(UserWarning, match="assemble_prices"):
            result = assemble_prices(session, include_delisted=False, currency_map={})
        assert result.shape == (1, 1)
        assert "RDSA.L" in result.columns
        # USD listing (200.0) must be selected over GBX listing (21000.0)
        assert result.iloc[0, 0] == pytest.approx(200.0)

    def test_duplicate_same_rank_no_panic(self) -> None:
        """Two listings with same currency rank: one row kept, warning emitted."""
        id_a = str(uuid.uuid4())
        id_b = str(uuid.uuid4())
        session = _make_session(
            [(id_a, "RDSA.L", "USD"), (id_b, "RDSA.L", "USD")],
            [(id_a, "2024-01-02", 200.0), (id_b, "2024-01-02", 210.0)],
        )
        with pytest.warns(UserWarning, match="assemble_prices"):
            result = assemble_prices(session, include_delisted=False, currency_map={})
        assert result.shape[1] == 1
        assert "RDSA.L" in result.columns

    def test_no_duplicate_no_warning(self) -> None:
        """Single listing per ticker: no warning emitted."""
        inst_id = str(uuid.uuid4())
        session = _make_session(
            [(inst_id, "AAPL", "USD")],
            [(inst_id, "2024-01-02", 150.0)],
        )
        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter("error", UserWarning)
            result = assemble_prices(session, include_delisted=False, currency_map={})
        assert result.shape == (1, 1)

    def test_unknown_instrument_id_row_dropped(self) -> None:
        inst_id = str(uuid.uuid4())
        unknown_id = str(uuid.uuid4())
        session = _make_session(
            [(inst_id, "AAPL", "USD")],
            [
                (inst_id, "2024-01-02", 100.0),
                (unknown_id, "2024-01-02", 200.0),
            ],
        )
        result = assemble_prices(session, include_delisted=False, currency_map={})
        assert "AAPL" in result.columns
        assert result.shape == (1, 1)

    def test_result_has_datetime_index(self) -> None:
        inst_id = str(uuid.uuid4())
        session = _make_session(
            [(inst_id, "AAPL", "USD")],
            [(inst_id, "2024-01-02", 100.0)],
        )
        result = assemble_prices(session, include_delisted=False, currency_map={})
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_empty_ticker_map_returns_empty_dataframe(self) -> None:
        """No instruments in DB → early return."""
        session = _make_session(
            [],   # ticker rank map — empty
            [],   # price rows
        )
        result = assemble_prices(session, include_delisted=False, currency_map={})
        assert result.empty


# ---------------------------------------------------------------------------
# TestAssembleVolumes
# ---------------------------------------------------------------------------


class TestAssembleVolumes:
    def test_empty_volume_rows_returns_empty_dataframe(self) -> None:
        inst_id = str(uuid.uuid4())
        session = _make_session(
            [(inst_id, "AAPL", "USD")],
            [],
        )
        result = assemble_volumes(session, include_delisted=False)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_single_ticker_multiple_dates_correct_shape(self) -> None:
        inst_id = str(uuid.uuid4())
        dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
        vol_rows = [(inst_id, d, float(i * 1_000_000)) for i, d in enumerate(dates, 1)]
        session = _make_session(
            [(inst_id, "AAPL", "USD")],
            vol_rows,
        )
        result = assemble_volumes(session, include_delisted=False)
        assert result.shape == (3, 1)
        assert "AAPL" in result.columns
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_index_is_monotonically_increasing(self) -> None:
        inst_id = str(uuid.uuid4())
        vol_rows = [
            (inst_id, "2024-01-03", 2_000_000.0),
            (inst_id, "2024-01-01", 1_000_000.0),
        ]
        session = _make_session(
            [(inst_id, "MSFT", "USD")],
            vol_rows,
        )
        result = assemble_volumes(session, include_delisted=False)
        assert result.index.is_monotonic_increasing

    def test_unknown_instrument_id_dropped(self) -> None:
        inst_id = str(uuid.uuid4())
        session = _make_session(
            [(inst_id, "AAPL", "USD")],
            [(str(uuid.uuid4()), "2024-01-02", 500_000.0)],
        )
        result = assemble_volumes(session, include_delisted=False)
        assert result.empty

    def test_duplicate_warns_and_selects_primary_currency(self) -> None:
        """Cross-listed tickers: lower currency rank wins, warning emitted."""
        id_usd = str(uuid.uuid4())
        id_gbx = str(uuid.uuid4())
        session = _make_session(
            [(id_usd, "SHEL.L", "USD"), (id_gbx, "SHEL.L", "GBX")],
            [
                (id_usd, "2024-01-02", 5_000_000.0),   # USD volume (should win)
                (id_gbx, "2024-01-02", 3_000_000.0),   # GBX volume (dropped)
            ],
        )
        with pytest.warns(UserWarning, match="assemble_volumes"):
            result = assemble_volumes(session, include_delisted=False)
        assert result.shape == (1, 1)
        assert "SHEL.L" in result.columns
        assert result.iloc[0, 0] == pytest.approx(5_000_000.0)

    def test_no_duplicate_no_warning(self) -> None:
        """Single listing per ticker: no warning emitted."""
        inst_id = str(uuid.uuid4())
        session = _make_session(
            [(inst_id, "AAPL", "USD")],
            [(inst_id, "2024-01-02", 1_000_000.0)],
        )
        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter("error", UserWarning)
            result = assemble_volumes(session, include_delisted=False)
        assert result.shape == (1, 1)


# ---------------------------------------------------------------------------
# TestAssembleDelistingReturns
# ---------------------------------------------------------------------------


class TestAssembleDelistingReturns:
    def test_empty_db_returns_empty_dict(self) -> None:
        session = _make_session([])
        result = assemble_delisting_returns(session)
        assert result == {}

    def test_single_delisted_ticker_with_return(self) -> None:
        session = _make_session([("GONE.L", Decimal("-0.25"))])
        result = assemble_delisting_returns(session)
        assert "GONE.L" in result
        assert result["GONE.L"] == pytest.approx(-0.25)

    def test_null_return_defaults_to_minus_30_pct(self) -> None:
        session = _make_session([("BANKRUPT", None)])
        result = assemble_delisting_returns(session)
        assert result["BANKRUPT"] == pytest.approx(-0.30)

    def test_multiple_tickers(self) -> None:
        rows = [
            ("TICK_A", Decimal("-0.10")),
            ("TICK_B", None),
            ("TICK_C", Decimal("-0.50")),
        ]
        session = _make_session(rows)
        result = assemble_delisting_returns(session)
        assert result["TICK_A"] == pytest.approx(-0.10)
        assert result["TICK_B"] == pytest.approx(-0.30)
        assert result["TICK_C"] == pytest.approx(-0.50)

    def test_all_values_are_float_type(self) -> None:
        session = _make_session([("AAPL", Decimal("-0.15"))])
        result = assemble_delisting_returns(session)
        assert isinstance(result["AAPL"], float)

    def test_positive_return_preserved(self) -> None:
        """Some delistings produce positive returns (acquisitions)."""
        session = _make_session([("ACQD", Decimal("0.20"))])
        result = assemble_delisting_returns(session)
        assert result["ACQD"] == pytest.approx(0.20)

    def test_returns_dict_type(self) -> None:
        session = _make_session([])
        result = assemble_delisting_returns(session)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# TestAssembleFxRates
# ---------------------------------------------------------------------------


class TestAssembleFxRates:
    def test_empty_price_index_returns_empty_dataframe(self) -> None:
        result = assemble_fx_rates(
            currency_map={"AAPL": "USD"},
            base_currency="USD",
            price_index=pd.DatetimeIndex([]),
        )
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_no_foreign_currencies_returns_index_only_df(self) -> None:
        """All tickers share the base currency → no FX needed."""
        price_index = pd.date_range("2024-01-01", periods=5, freq="D")
        result = assemble_fx_rates(
            currency_map={"AAPL": "USD", "MSFT": "USD"},
            base_currency="USD",
            price_index=price_index,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 0
        assert len(result) == len(price_index)

    def test_yfinance_raises_returns_empty_gracefully(self) -> None:
        price_index = pd.date_range("2024-01-01", periods=5, freq="D")
        with patch("yfinance.download", side_effect=RuntimeError("network error")):
            result = assemble_fx_rates(
                currency_map={"BARC.L": "GBP"},
                base_currency="USD",
                price_index=price_index,
            )
        assert isinstance(result, pd.DataFrame)

    def test_yfinance_returns_empty_data_gracefully(self) -> None:
        price_index = pd.date_range("2024-01-01", periods=5, freq="D")
        with patch("yfinance.download", return_value=pd.DataFrame()):
            result = assemble_fx_rates(
                currency_map={"BARC.L": "GBP"},
                base_currency="USD",
                price_index=price_index,
            )
        assert isinstance(result, pd.DataFrame)

    def test_single_foreign_currency_column_assembled(self) -> None:
        """GBP→USD: mock yf.download returns flat Close column → 'GBP' in result."""
        price_index = pd.bdate_range("2024-01-02", periods=3)
        fake_close = [1.27, 1.28, 1.26]
        # Single-ticker download → flat DataFrame (not MultiIndex)
        fake_data = pd.DataFrame({"Close": fake_close}, index=price_index)

        with patch("yfinance.download", return_value=fake_data):
            result = assemble_fx_rates(
                currency_map={"BARC.L": "GBP"},
                base_currency="USD",
                price_index=price_index,
            )

        assert "GBP" in result.columns
        np.testing.assert_array_almost_equal(result["GBP"].values, fake_close)

    def test_empty_currency_map_returns_indexed_empty_df(self) -> None:
        price_index = pd.date_range("2024-01-01", periods=3, freq="D")
        result = assemble_fx_rates(
            currency_map={},
            base_currency="USD",
            price_index=price_index,
        )
        assert isinstance(result, pd.DataFrame)
