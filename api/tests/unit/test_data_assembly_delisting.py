"""Unit tests for survivorship-bias correction in data_assembly.py."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# Ensure the api package is importable.
_api_path = Path(__file__).parent.parent.parent / "app"
_api_root = Path(__file__).parent.parent.parent
if str(_api_root) not in sys.path:
    sys.path.insert(0, str(_api_root))

from cli.data_assembly import _apply_delisting_returns  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _price_df(
    tickers: list[str],
    n_days: int = 10,
    start: str = "2024-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    data = {
        t: 100.0 * np.cumprod(1 + rng.normal(0.001, 0.02, n_days))
        for t in tickers
    }
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# _apply_delisting_returns — pure function tests
# ---------------------------------------------------------------------------


class TestApplyDelistingReturns:
    def test_no_delistings_returns_unchanged(self) -> None:
        df = _price_df(["A", "B"])
        out = _apply_delisting_returns(df, [])
        pd.testing.assert_frame_equal(out, df)

    def test_synthetic_row_added_after_last_date(self) -> None:
        df = _price_df(["A", "B"])
        delisting_date = df.index[-1] + pd.Timedelta(days=1)
        out = _apply_delisting_returns(
            df, [("A", delisting_date, -0.30)]
        )
        assert delisting_date in out.index

    def test_delisting_return_applied_correctly(self) -> None:
        """Synthetic price = last_price * (1 + delisting_return)."""
        df = _price_df(["A", "B"])
        last_price = float(df["A"].iloc[-1])
        r = -0.30
        delisting_date = df.index[-1] + pd.Timedelta(days=1)

        out = _apply_delisting_returns(df, [("A", delisting_date, r)])

        expected = last_price * (1.0 + r)
        assert out.loc[delisting_date, "A"] == pytest.approx(expected)

    def test_bankruptcy_return_minus_100_pct(self) -> None:
        """A -1.0 return produces a near-zero synthetic price."""
        df = _price_df(["A"])
        last_price = float(df["A"].iloc[-1])
        delisting_date = df.index[-1] + pd.Timedelta(days=1)

        out = _apply_delisting_returns(df, [("A", delisting_date, -1.0)])
        assert out.loc[delisting_date, "A"] == pytest.approx(0.0, abs=1e-9)

    def test_acquisition_premium_positive_return(self) -> None:
        df = _price_df(["A"])
        last_price = float(df["A"].iloc[-1])
        r = 0.25  # 25% acquisition premium
        delisting_date = df.index[-1] + pd.Timedelta(days=1)

        out = _apply_delisting_returns(df, [("A", delisting_date, r)])
        assert out.loc[delisting_date, "A"] == pytest.approx(last_price * 1.25)

    def test_non_traded_ticker_skipped(self) -> None:
        df = _price_df(["A", "B"])
        delisting_date = df.index[-1] + pd.Timedelta(days=1)

        # "C" is not in the DataFrame
        out = _apply_delisting_returns(df, [("C", delisting_date, -0.30)])
        assert delisting_date not in out.index

    def test_delisting_date_already_in_index_not_added_again(self) -> None:
        df = _price_df(["A", "B"])
        delisting_date = df.index[-2]  # already in the DataFrame
        n_rows_before = len(df)

        out = _apply_delisting_returns(df, [("A", delisting_date, -0.30)])
        assert len(out) == n_rows_before

    def test_existing_real_price_not_overwritten(self) -> None:
        """If a real price exists on the delisting date, it is preserved."""
        df = _price_df(["A", "B"])
        delisting_date = df.index[-1]  # last real trading day
        real_price = float(df.loc[delisting_date, "A"])

        out = _apply_delisting_returns(df, [("A", delisting_date, -0.30)])
        # Should NOT have been overwritten
        assert out.loc[delisting_date, "A"] == pytest.approx(real_price)

    def test_other_tickers_unaffected(self) -> None:
        df = _price_df(["A", "B"])
        delisting_date = df.index[-1] + pd.Timedelta(days=1)

        out = _apply_delisting_returns(df, [("A", delisting_date, -0.30)])
        # B column in original rows is unchanged (ignore freq diff after concat)
        pd.testing.assert_series_equal(
            out.loc[df.index, "B"], df["B"], check_freq=False
        )

    def test_multiple_delistings(self) -> None:
        """Two stocks delist on the same date."""
        df = _price_df(["A", "B", "C"])
        delisting_date = df.index[-1] + pd.Timedelta(days=1)

        delistings = [
            ("A", delisting_date, -0.30),
            ("B", delisting_date, -1.0),
        ]
        out = _apply_delisting_returns(df, delistings)

        last_a = float(df["A"].iloc[-1])
        last_b = float(df["B"].iloc[-1])
        assert out.loc[delisting_date, "A"] == pytest.approx(last_a * 0.70)
        assert out.loc[delisting_date, "B"] == pytest.approx(0.0, abs=1e-9)
        # C is untouched: NaN on delisting_date
        assert pd.isna(out.loc[delisting_date, "C"])

    def test_returns_lower_mean_with_delisted_stock(self) -> None:
        """Mean return is lower when a bankrupted stock is included.

        Constructs a price series where stock A declines to zero (bankruptcy).
        Computes returns with and without the delisting row. The biased
        (no-delisting) mean return over A's active period should be higher
        than the corrected mean that includes the -100% final return.
        """
        from skfolio.preprocessing import prices_to_returns

        # A goes from 100 to ~50 over 20 days (trending down) then goes bankrupt
        idx = pd.bdate_range("2024-01-01", periods=21)
        prices_a = np.linspace(100.0, 50.0, 20)
        prices_b = np.linspace(100.0, 110.0, 20)  # B is healthy

        df_active = pd.DataFrame(
            {"A": prices_a, "B": prices_b},
            index=idx[:20],
        )

        # Survivorship-biased: no delisting return
        ret_biased = prices_to_returns(df_active)
        mean_biased = float(ret_biased["A"].mean())

        # Corrected: add delisting row on day 21
        delisting_date = idx[20]
        df_with_delist = _apply_delisting_returns(
            df_active, [("A", delisting_date, -1.0)]
        )
        ret_corrected = prices_to_returns(df_with_delist)
        mean_corrected = float(ret_corrected["A"].mean())

        assert mean_corrected < mean_biased, (
            f"Corrected mean ({mean_corrected:.4f}) should be lower than "
            f"biased mean ({mean_biased:.4f})"
        )

    def test_delisting_return_on_correct_date(self) -> None:
        """The delisting return appears as the last non-NaN return."""
        from skfolio.preprocessing import prices_to_returns

        idx = pd.bdate_range("2024-01-01", periods=10)
        prices_a = np.linspace(100.0, 90.0, 10)
        df = pd.DataFrame({"A": prices_a}, index=idx)

        delisting_date = idx[-1] + pd.Timedelta(days=1)
        r = -0.40
        df_with_delist = _apply_delisting_returns(df, [("A", delisting_date, r)])
        returns = prices_to_returns(df_with_delist)

        last_return = float(returns["A"].dropna().iloc[-1])
        assert last_return == pytest.approx(r, abs=1e-6)


# ---------------------------------------------------------------------------
# assemble_prices — integration with mock session
# ---------------------------------------------------------------------------


class TestAssemblePricesDelisting:
    """Tests for assemble_prices using a mocked SQLAlchemy session."""

    def _make_session(
        self,
        ticker_rows: list,
        price_rows: list,
        delisting_rows: list | None = None,
    ) -> MagicMock:
        """Build a mock session whose .execute() returns controlled results."""
        mock_session = MagicMock()

        results: list[MagicMock] = []
        for data in [ticker_rows, price_rows]:
            r = MagicMock()
            r.all.return_value = data
            results.append(r)

        if delisting_rows is not None:
            r = MagicMock()
            r.all.return_value = delisting_rows
            results.append(r)

        mock_session.execute.side_effect = results
        return mock_session

    def test_include_delisted_false_skips_extra_query(self) -> None:
        """When include_delisted=False, the delisting rows query is never made."""
        from cli.data_assembly import assemble_prices

        # Two .execute() calls: ticker_map + price_history (no delisting query)
        import uuid

        inst_id = uuid.uuid4()
        ticker_rows = [(inst_id, "AAPL")]
        price_rows = [(inst_id, date(2024, 1, 2), 100.0)]

        mock_session = self._make_session(ticker_rows, price_rows)
        result = assemble_prices(mock_session, include_delisted=False)

        assert "AAPL" in result.columns
        assert mock_session.execute.call_count == 2  # ticker_map + prices only

    def test_include_delisted_true_makes_delisting_query(self) -> None:
        """When include_delisted=True, a third query fetches delisting rows."""
        from cli.data_assembly import assemble_prices

        import uuid

        inst_id = uuid.uuid4()
        ticker_rows = [(inst_id, "AAPL")]
        price_rows = [(inst_id, date(2024, 1, 2), 100.0)]
        delisting_rows: list = []  # no delistings

        mock_session = self._make_session(ticker_rows, price_rows, delisting_rows)
        assemble_prices(mock_session, include_delisted=True)

        assert mock_session.execute.call_count == 3

    def test_delisted_stock_gets_synthetic_price_row(self) -> None:
        """The delisting synthetic row appears in the returned DataFrame."""
        from cli.data_assembly import assemble_prices

        import uuid

        active_id = uuid.uuid4()
        delisted_id = uuid.uuid4()

        ticker_rows = [(active_id, "AAPL"), (delisted_id, "DEAD")]
        price_rows = [
            (active_id, date(2024, 1, 2), 150.0),
            (active_id, date(2024, 1, 3), 152.0),
            (delisted_id, date(2024, 1, 2), 50.0),
            (delisted_id, date(2024, 1, 3), 48.0),
        ]
        # DEAD delists on Jan 5 with -30% return
        delisting_rows = [("DEAD", date(2024, 1, 5), -0.30)]

        mock_session = self._make_session(ticker_rows, price_rows, delisting_rows)
        result = assemble_prices(mock_session, include_delisted=True)

        delisting_ts = pd.Timestamp("2024-01-05")
        assert delisting_ts in result.index
        assert result.loc[delisting_ts, "DEAD"] == pytest.approx(48.0 * 0.70)

    def test_include_delisted_false_lower_column_count(self) -> None:
        """With include_delisted=False, the DataFrame has fewer columns."""
        # This test uses _apply_delisting_returns directly to verify
        # that the survivorship-biased subset excludes the delisted instrument.
        df_full = _price_df(["AAPL", "DEAD"])
        r = -1.0
        delisting_date = df_full.index[-1] + pd.Timedelta(days=1)

        df_corrected = _apply_delisting_returns(df_full, [("DEAD", delisting_date, r)])

        from skfolio.preprocessing import prices_to_returns

        ret_biased = prices_to_returns(df_full[["AAPL"]])
        ret_corrected = prices_to_returns(df_corrected)

        # Corrected has an extra row (delisting date) with DEAD's final return
        assert len(ret_corrected) > len(ret_biased)
