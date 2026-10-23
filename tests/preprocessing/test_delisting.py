"""Tests for survivorship-bias guard (delisting returns)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from optimizer.exceptions import DataError
from optimizer.preprocessing import to_returns
from optimizer.preprocessing._delisting import (
    apply_delisting_returns,
    delisting_protection_mask,
)


@pytest.fixture()
def sample_returns() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    return pd.DataFrame(
        {
            "AAPL": [0.01, 0.02, 0.01, np.nan, np.nan],
            "MSFT": [0.01, 0.01, 0.01, 0.01, 0.01],
            "GOOG": [np.nan, np.nan, np.nan, np.nan, np.nan],
        },
        index=dates,
    )


class TestApplyDelistingReturns:
    def test_replaces_last_valid(self, sample_returns: pd.DataFrame) -> None:
        result = apply_delisting_returns(sample_returns, {"AAPL": -0.50})
        # AAPL's last valid is at index 2 (value 0.01 → -0.50)
        assert result["AAPL"].iloc[2] == pytest.approx(-0.50)

    def test_preserves_others(self, sample_returns: pd.DataFrame) -> None:
        result = apply_delisting_returns(sample_returns, {"AAPL": -0.50})
        assert result["AAPL"].iloc[0] == pytest.approx(0.01)
        assert result["AAPL"].iloc[1] == pytest.approx(0.02)
        assert result["MSFT"].iloc[4] == pytest.approx(0.01)

    def test_raises_on_unknown_ticker(self, sample_returns: pd.DataFrame) -> None:
        with pytest.raises(DataError, match="TSLA"):
            apply_delisting_returns(sample_returns, {"TSLA": -0.30})

    def test_skips_all_nan_column(self, sample_returns: pd.DataFrame) -> None:
        result = apply_delisting_returns(sample_returns, {"GOOG": -1.0})
        assert result["GOOG"].isna().all()

    def test_returns_copy(self, sample_returns: pd.DataFrame) -> None:
        result = apply_delisting_returns(sample_returns, {"AAPL": -0.50})
        assert result is not sample_returns
        # Original unchanged
        assert sample_returns["AAPL"].iloc[2] == pytest.approx(0.01)

    def test_crsp_default_applied_verbatim(self, sample_returns: pd.DataFrame) -> None:
        # -0.30 CRSP-style default is written as-is (no sign flip / scaling).
        result = apply_delisting_returns(sample_returns, {"AAPL": -0.30})
        assert result["AAPL"].iloc[2] == pytest.approx(-0.30)

    def test_bankruptcy_total_loss(self, sample_returns: pd.DataFrame) -> None:
        result = apply_delisting_returns(sample_returns, {"AAPL": -1.0})
        assert result["AAPL"].iloc[2] == pytest.approx(-1.0)

    def test_post_death_filled_with_zero(self, sample_returns: pd.DataFrame) -> None:
        # After the terminal return, trailing rows are held as cash (0.0), so no
        # trailing NaN remains for a downstream SelectComplete to drop.
        result = apply_delisting_returns(sample_returns, {"AAPL": -0.30})
        assert result["AAPL"].iloc[2] == pytest.approx(-0.30)  # terminal loss
        assert result["AAPL"].iloc[3] == pytest.approx(0.0)
        assert result["AAPL"].iloc[4] == pytest.approx(0.0)
        assert not result["AAPL"].isna().any()  # loss stays in the sample

    def test_leading_nan_not_filled(self) -> None:
        # A late listing (leading NaN) keeps its leading gap so SelectComplete
        # still drops it as short-history; only the trailing gap is cash-filled.
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        df = pd.DataFrame(
            {"LATE": [np.nan, np.nan, 0.01, 0.02, np.nan]},
            index=dates,
        )
        result = apply_delisting_returns(df, {"LATE": -0.30})
        assert result["LATE"].iloc[3] == pytest.approx(-0.30)  # terminal loss
        assert result["LATE"].iloc[4] == pytest.approx(0.0)  # post-death cash
        assert np.isnan(result["LATE"].iloc[0])  # leading gap preserved
        assert np.isnan(result["LATE"].iloc[1])

    def test_ticker_left_out_of_mapping_keeps_its_history(
        self, sample_returns: pd.DataFrame
    ) -> None:
        # A line that left the universe but still trades has no terminal return,
        # so the caller omits it; its trailing NaN must not be cash-filled.
        result = apply_delisting_returns(sample_returns, {"MSFT": -0.30})
        pd.testing.assert_series_equal(result["AAPL"], sample_returns["AAPL"])

    def test_none_delisting_return_raises(self, sample_returns: pd.DataFrame) -> None:
        # Unresolved DB NULL must not be silently written as NaN.
        with pytest.raises(DataError, match="AAPL"):
            apply_delisting_returns(sample_returns, {"AAPL": None})  # type: ignore[dict-item]

    def test_nan_delisting_return_raises(self, sample_returns: pd.DataFrame) -> None:
        with pytest.raises(DataError, match="AAPL"):
            apply_delisting_returns(sample_returns, {"AAPL": float("nan")})

    def test_inf_delisting_return_raises(self, sample_returns: pd.DataFrame) -> None:
        with pytest.raises(DataError, match="AAPL"):
            apply_delisting_returns(sample_returns, {"AAPL": float("inf")})


class TestDelistingProtectionMask:
    def test_marks_last_valid_cell(self, sample_returns: pd.DataFrame) -> None:
        mask = delisting_protection_mask(sample_returns, {"AAPL": -0.30})
        # AAPL's last valid is idx 2; only that cell is True.
        assert bool(mask["AAPL"].iloc[2])
        assert mask["AAPL"].sum() == 1
        # No other ticker is marked.
        assert not mask["MSFT"].any()
        assert not mask["GOOG"].any()

    def test_marks_cell_that_receives_terminal_return(
        self, sample_returns: pd.DataFrame
    ) -> None:
        # The marked cell is exactly where apply_delisting_returns writes -0.30.
        mask = delisting_protection_mask(sample_returns, {"AAPL": -0.30})
        applied = apply_delisting_returns(sample_returns, {"AAPL": -0.30})
        marked = mask["AAPL"]
        assert applied["AAPL"][marked].iloc[0] == pytest.approx(-0.30)

    def test_skips_all_nan_column(self, sample_returns: pd.DataFrame) -> None:
        mask = delisting_protection_mask(sample_returns, {"GOOG": -1.0})
        assert not mask.to_numpy().any()

    def test_ticker_left_out_of_mapping_is_not_marked(
        self, sample_returns: pd.DataFrame
    ) -> None:
        mask = delisting_protection_mask(sample_returns, {"MSFT": -0.30})
        assert not mask["AAPL"].any()

    def test_skips_unknown_ticker(self, sample_returns: pd.DataFrame) -> None:
        # No raise here (apply_delisting_returns is the validator); no mark.
        mask = delisting_protection_mask(sample_returns, {"TSLA": -0.30})
        assert not mask.to_numpy().any()

    def test_aligned_to_returns(self, sample_returns: pd.DataFrame) -> None:
        mask = delisting_protection_mask(sample_returns, {"AAPL": -0.30})
        assert list(mask.columns) == list(sample_returns.columns)
        assert mask.index.equals(sample_returns.index)
        assert mask.to_numpy().dtype == bool

    def test_marks_pre_fill_position_not_padding(self) -> None:
        # With a trailing gap, the mark is the pre-fill last valid row, not the
        # 0.0 post-death padding that apply_delisting_returns writes after it.
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        df = pd.DataFrame({"X": [0.01, 0.02, np.nan, np.nan, np.nan]}, index=dates)
        mask = delisting_protection_mask(df, {"X": -0.30})
        assert bool(mask["X"].iloc[1])  # last valid at idx 1
        assert mask["X"].sum() == 1


@pytest.fixture()
def ffilled_panel() -> tuple[pd.DataFrame, pd.Timestamp]:
    """Returns from the default ``to_returns`` for a ticker that stops quoting.

    ``D`` has no price after ``last_quote``; the default ``fill_nan=True``
    forward-fills its price, so its returns are exactly 0.0 to the end of the
    window and carry no trailing ``NaN``.
    """
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2023-01-02", periods=12)
    prices = pd.DataFrame(
        {
            "B": 100 * np.cumprod(1 + rng.normal(0, 0.01, 12)),
            "D": 100 * np.cumprod(1 + rng.normal(0, 0.01, 12)),
        },
        index=dates,
    )
    last_quote = dates[6]
    prices.loc[dates[7] :, "D"] = np.nan
    returns = to_returns(prices)
    assert isinstance(returns, pd.DataFrame)
    return returns, last_quote


class TestDelistedAt:
    def test_terminal_return_booked_on_delisting_date_not_window_end(
        self, ffilled_panel: tuple[pd.DataFrame, pd.Timestamp]
    ) -> None:
        returns, last_quote = ffilled_panel
        result = apply_delisting_returns(
            returns, {"D": -0.30}, delisted_at={"D": last_quote}
        )
        assert result.loc[last_quote, "D"] == pytest.approx(-0.30)
        assert (result["D"] == -0.30).sum() == 1
        assert (result["D"].loc[last_quote:].iloc[1:] == 0.0).all()
        assert result["D"].iloc[-1] == 0.0

    def test_history_before_delisting_and_other_assets_untouched(
        self, ffilled_panel: tuple[pd.DataFrame, pd.Timestamp]
    ) -> None:
        returns, last_quote = ffilled_panel
        result = apply_delisting_returns(
            returns, {"D": -0.30}, delisted_at={"D": last_quote}
        )
        before = returns.index < last_quote
        pd.testing.assert_series_equal(result["D"][before], returns["D"][before])
        pd.testing.assert_series_equal(result["B"], returns["B"])

    def test_protection_mask_marks_the_delisting_date(
        self, ffilled_panel: tuple[pd.DataFrame, pd.Timestamp]
    ) -> None:
        returns, last_quote = ffilled_panel
        mask = delisting_protection_mask(
            returns, {"D": -0.30}, delisted_at={"D": last_quote}
        )
        assert mask["D"].sum() == 1
        assert bool(mask.loc[last_quote, "D"])
        applied = apply_delisting_returns(
            returns, {"D": -0.30}, delisted_at={"D": last_quote}
        )
        assert applied["D"][mask["D"]].iloc[0] == pytest.approx(-0.30)

    def test_non_trading_delisting_date_uses_prior_row(
        self, ffilled_panel: tuple[pd.DataFrame, pd.Timestamp]
    ) -> None:
        returns, last_quote = ffilled_panel
        weekend = last_quote + pd.Timedelta(days=1)
        # last_quote is a Monday-Friday business day; make sure +1d is a gap.
        while weekend in returns.index:
            weekend += pd.Timedelta(days=1)
        row_before = returns.index[returns.index < weekend][-1]
        result = apply_delisting_returns(
            returns, {"D": -0.30}, delisted_at={"D": weekend}
        )
        assert result.loc[row_before, "D"] == pytest.approx(-0.30)

    def test_tickers_without_a_date_keep_last_valid_inference(
        self, sample_returns: pd.DataFrame
    ) -> None:
        result = apply_delisting_returns(
            sample_returns, {"AAPL": -0.50}, delisted_at={"MSFT": "2020-01-03"}
        )
        assert result["AAPL"].iloc[2] == pytest.approx(-0.50)

    def test_delisting_date_before_first_row_raises(
        self, ffilled_panel: tuple[pd.DataFrame, pd.Timestamp]
    ) -> None:
        returns, _ = ffilled_panel
        with pytest.raises(DataError, match="precedes"):
            apply_delisting_returns(
                returns, {"D": -0.30}, delisted_at={"D": "2000-01-01"}
            )

    def test_delisted_at_needs_datetime_index(
        self, sample_returns: pd.DataFrame
    ) -> None:
        plain = sample_returns.reset_index(drop=True)
        with pytest.raises(DataError, match="DatetimeIndex"):
            apply_delisting_returns(
                plain, {"AAPL": -0.30}, delisted_at={"AAPL": "2020-01-03"}
            )

    def test_final_row_terminal_without_date_warns(
        self,
        ffilled_panel: tuple[pd.DataFrame, pd.Timestamp],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Forward-filled returns leave no trailing NaN, so the inferred
        # terminal row is the window end; the caller must be told.
        returns, _ = ffilled_panel
        with caplog.at_level(logging.WARNING, logger="optimizer.preprocessing"):
            apply_delisting_returns(returns, {"D": -0.30})
        assert "delisted_at" in caplog.text

    def test_nan_tailed_ticker_without_date_does_not_warn(
        self,
        sample_returns: pd.DataFrame,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="optimizer.preprocessing"):
            apply_delisting_returns(sample_returns, {"AAPL": -0.30})
        assert "delisted_at" not in caplog.text


class TestDelistedAtValidation:
    """A date that cannot be placed must fail loudly, never fall to the window end."""

    @pytest.mark.parametrize("bad", [None, pd.NaT, float("nan"), "NaT"])
    def test_null_delisting_date_raises(
        self, ffilled_panel: tuple[pd.DataFrame, pd.Timestamp], bad: object
    ) -> None:
        returns, _ = ffilled_panel
        with pytest.raises(DataError, match=r"Delisting date.*'D'"):
            apply_delisting_returns(
                returns,
                {"D": -0.30},
                delisted_at={"D": bad},  # type: ignore[dict-item]
            )

    def test_unparseable_delisting_date_raises(
        self, ffilled_panel: tuple[pd.DataFrame, pd.Timestamp]
    ) -> None:
        returns, _ = ffilled_panel
        with pytest.raises(DataError, match=r"Delisting date.*'D'"):
            apply_delisting_returns(
                returns, {"D": -0.30}, delisted_at={"D": "not a date"}
            )

    def test_protection_mask_rejects_null_delisting_date_too(
        self, ffilled_panel: tuple[pd.DataFrame, pd.Timestamp]
    ) -> None:
        returns, _ = ffilled_panel
        with pytest.raises(DataError, match=r"Delisting date.*'D'"):
            delisting_protection_mask(
                returns,
                {"D": -0.30},
                delisted_at={"D": pd.NaT},  # type: ignore[dict-item]
            )

    def test_aware_date_on_naive_index_raises(
        self, ffilled_panel: tuple[pd.DataFrame, pd.Timestamp]
    ) -> None:
        returns, last_quote = ffilled_panel
        with pytest.raises(DataError, match="timezone"):
            apply_delisting_returns(
                returns,
                {"D": -0.30},
                delisted_at={"D": last_quote.tz_localize("UTC")},
            )

    def test_naive_date_on_aware_index_raises(
        self, ffilled_panel: tuple[pd.DataFrame, pd.Timestamp]
    ) -> None:
        returns, last_quote = ffilled_panel
        aware = returns.tz_localize("UTC")
        with pytest.raises(DataError, match="timezone"):
            apply_delisting_returns(aware, {"D": -0.30}, delisted_at={"D": last_quote})

    def test_aware_date_on_aware_index_is_booked(
        self, ffilled_panel: tuple[pd.DataFrame, pd.Timestamp]
    ) -> None:
        returns, last_quote = ffilled_panel
        aware = returns.tz_localize("UTC")
        result = apply_delisting_returns(
            aware, {"D": -0.30}, delisted_at={"D": last_quote.tz_localize("UTC")}
        )
        assert result.loc[last_quote.tz_localize("UTC"), "D"] == pytest.approx(-0.30)

    def test_null_date_for_all_nan_column_is_skipped_like_any_other(
        self, sample_returns: pd.DataFrame
    ) -> None:
        result = apply_delisting_returns(
            sample_returns, {"GOOG": -0.30}, delisted_at={"GOOG": pd.NaT}
        )
        pd.testing.assert_frame_equal(result, sample_returns)
