"""Tests for the pipeline-composable investability-screen selector."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from optimizer.exceptions import DataError
from optimizer.universe import (
    InvestabilityScreenConfig,
    InvestabilityScreenSelector,
    build_investability_screen,
)


@pytest.fixture()
def screening_data() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(42)
    tickers = ["AAPL", "MSFT", "GOOG", "TINY", "NEW"]
    dates = pd.bdate_range("2023-01-01", periods=300)
    prices = np.abs(100.0 + rng.normal(0, 1, (300, 5)).cumsum(axis=0))
    price_df = pd.DataFrame(prices, index=dates, columns=tickers)
    # NEW: recent IPO (only 50 days of history)
    price_df.loc[price_df.index[:-50], "NEW"] = np.nan

    vol = rng.integers(1_000_000, 5_000_000, size=(300, 5))
    volume_df = pd.DataFrame(vol, index=dates, columns=tickers)
    volume_df.loc[price_df["NEW"].isna(), "NEW"] = 0
    # TINY: sparse trading
    tiny_mask = rng.random(300) < 0.20
    volume_df.loc[tiny_mask, "TINY"] = 0

    fundamentals = pd.DataFrame(
        {
            "market_cap": [2e9, 1.5e9, 3e9, 50e6, 500e6],
            "current_price": [150.0, 300.0, 100.0, 0.5, 25.0],
        },
        index=pd.Index(tickers, name="ticker"),
    )
    return {
        "fundamentals": fundamentals,
        "price_history": price_df,
        "volume_history": volume_df,
    }


def _returns_from(price_history: pd.DataFrame) -> pd.DataFrame:
    """Linear returns with tickers as columns (skfolio input convention)."""
    return price_history.pct_change().iloc[1:]


class TestInvestabilityScreenSelector:
    def test_selects_investable_columns(
        self, screening_data: dict[str, pd.DataFrame]
    ) -> None:
        X = _returns_from(screening_data["price_history"])
        selector = build_investability_screen(**screening_data)
        selector.fit(X)

        kept = set(selector.investable_universe_)
        assert "AAPL" in kept
        assert "MSFT" in kept
        assert "GOOG" in kept
        # TINY fails mcap/price; NEW fails listing age
        assert "TINY" not in kept
        assert "NEW" not in kept

    def test_transform_returns_subset_preserving_order(
        self, screening_data: dict[str, pd.DataFrame]
    ) -> None:
        X = _returns_from(screening_data["price_history"])
        selector = build_investability_screen(**screening_data)
        selector.set_output(transform="pandas")
        selector.fit(X)
        out = selector.transform(X)

        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["AAPL", "MSFT", "GOOG"]
        # Column order follows X, not screen output ordering.
        assert list(out.columns) == [c for c in X.columns if c in out.columns]

    def test_get_support_mask_matches_columns(
        self, screening_data: dict[str, pd.DataFrame]
    ) -> None:
        X = _returns_from(screening_data["price_history"])
        selector = build_investability_screen(**screening_data).fit(X)
        mask = selector.get_support()
        assert mask.dtype == bool
        assert mask.tolist() == [True, True, True, False, False]
        assert list(selector.get_feature_names_out()) == ["AAPL", "MSFT", "GOOG"]

    def test_composes_in_sklearn_pipeline(
        self, screening_data: dict[str, pd.DataFrame]
    ) -> None:
        X = _returns_from(screening_data["price_history"])
        pipe = Pipeline([("screen", build_investability_screen(**screening_data))])
        pipe.set_output(transform="pandas")
        out = pipe.fit_transform(X)
        assert list(out.columns) == ["AAPL", "MSFT", "GOOG"]

    def test_get_params_exposes_config(
        self, screening_data: dict[str, pd.DataFrame]
    ) -> None:
        cfg = InvestabilityScreenConfig.for_large_cap()
        selector = build_investability_screen(**screening_data, config=cfg)
        params = selector.get_params()
        assert params["config"] is cfg
        assert "fundamentals" in params

    def test_config_none_uses_defaults(
        self, screening_data: dict[str, pd.DataFrame]
    ) -> None:
        X = _returns_from(screening_data["price_history"])
        selector = InvestabilityScreenSelector(config=None, **screening_data)
        selector.fit(X)
        assert isinstance(selector.investable_universe_, pd.Index)

    def test_missing_screening_data_raises(self) -> None:
        X = pd.DataFrame({"A": [0.01, -0.01], "B": [0.02, 0.0]})
        selector = InvestabilityScreenSelector()
        with pytest.raises(DataError, match="requires fundamentals"):
            selector.fit(X)

    def test_array_input_without_feature_names_raises(
        self, screening_data: dict[str, pd.DataFrame]
    ) -> None:
        X = _returns_from(screening_data["price_history"]).to_numpy()
        selector = build_investability_screen(**screening_data)
        with pytest.raises(DataError, match="ticker column names"):
            selector.fit(X)

    def test_allows_nan_returns(self, screening_data: dict[str, pd.DataFrame]) -> None:
        X = _returns_from(screening_data["price_history"])
        # NEW column legitimately carries NaN (pre-IPO) — must not raise.
        assert X["NEW"].isna().any()
        selector = build_investability_screen(**screening_data).fit(X)
        assert "NEW" not in set(selector.investable_universe_)

    def test_current_members_hysteresis(
        self, screening_data: dict[str, pd.DataFrame]
    ) -> None:
        X = _returns_from(screening_data["price_history"])
        first = build_investability_screen(**screening_data).fit(X)
        second = build_investability_screen(
            **screening_data, current_members=first.investable_universe_
        ).fit(X)
        assert set(first.investable_universe_) == set(second.investable_universe_)


@pytest.fixture()
def long_history() -> dict[str, pd.DataFrame]:
    """800 business days; liquidity and listings change after day 400."""
    n_days = 800
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    tickers = ["ALWAYS", "LIQ_THEN_DEAD", "NEW_LISTING"]
    rng = np.random.default_rng(7)
    prices = pd.DataFrame(
        100.0 * np.exp(rng.normal(0, 0.01, (n_days, 3)).cumsum(axis=0)),
        index=dates,
        columns=tickers,
    )
    volume = pd.DataFrame(1_000_000.0, index=dates, columns=tickers)
    volume.iloc[400:, volume.columns.get_loc("LIQ_THEN_DEAD")] = 0.0
    prices.iloc[:500, prices.columns.get_loc("NEW_LISTING")] = np.nan
    volume.iloc[:500, volume.columns.get_loc("NEW_LISTING")] = np.nan
    fundamentals = pd.DataFrame(
        {"market_cap": [5e9] * 3, "current_price": [100.0] * 3},
        index=pd.Index(tickers, name="ticker"),
    )
    return {
        "fundamentals": fundamentals,
        "price_history": prices,
        "volume_history": volume,
    }


class TestFitWindowIsHonoured:
    """Screens must only see data known at the end of the training window."""

    def test_price_and_volume_after_training_window_are_ignored(
        self, long_history: dict[str, pd.DataFrame]
    ) -> None:
        X = _returns_from(long_history["price_history"]).iloc[:400]
        selector = build_investability_screen(**long_history).fit(X)

        # LIQ_THEN_DEAD was liquid for the whole window; NEW_LISTING did not
        # exist yet.  Later data must not change either verdict.
        assert list(selector.investable_universe_) == ["ALWAYS", "LIQ_THEN_DEAD"]

    def test_timezone_aware_fit_window_is_honoured(
        self, long_history: dict[str, pd.DataFrame]
    ) -> None:
        X = _returns_from(long_history["price_history"]).iloc[:400]
        X.index = X.index.tz_localize("UTC")
        selector = build_investability_screen(**long_history).fit(X)

        assert list(selector.investable_universe_) == ["ALWAYS", "LIQ_THEN_DEAD"]

    def test_unchanged_for_non_datetime_returns_index(
        self, long_history: dict[str, pd.DataFrame]
    ) -> None:
        X = _returns_from(long_history["price_history"]).reset_index(drop=True)
        selector = build_investability_screen(**long_history).fit(X)

        # Without dates there is no window to apply: the full history is used.
        assert list(selector.investable_universe_) == ["ALWAYS", "NEW_LISTING"]

    def test_delisting_is_ignored_after_window_end_and_honoured_before_it(
        self, long_history: dict[str, pd.DataFrame]
    ) -> None:
        X = _returns_from(long_history["price_history"]).iloc[:400]
        fundamentals = long_history["fundamentals"].copy()
        fundamentals["delisted_at"] = pd.to_datetime(["2099-01-01", "2020-06-30", None])
        selector = build_investability_screen(
            **{**long_history, "fundamentals": fundamentals}
        ).fit(X)

        # ALWAYS is only delisted long after the window, so it was still
        # trading and stays; LIQ_THEN_DEAD was delisted inside the window.
        assert list(selector.investable_universe_) == ["ALWAYS"]

    def test_future_delisting_of_listed_name_is_not_survivorship_biased(
        self, long_history: dict[str, pd.DataFrame]
    ) -> None:
        X = _returns_from(long_history["price_history"]).iloc[:400]
        fundamentals = long_history["fundamentals"].copy()
        fundamentals["delisted_at"] = pd.to_datetime([None, "2023-01-02", None])
        selector = build_investability_screen(
            **{**long_history, "fundamentals": fundamentals}
        ).fit(X)

        assert "LIQ_THEN_DEAD" in set(selector.investable_universe_)

    def test_statements_dated_after_training_window_are_ignored(
        self, long_history: dict[str, pd.DataFrame]
    ) -> None:
        X = _returns_from(long_history["price_history"]).iloc[:400]
        rows = [
            {"ticker": "ALWAYS", "period_type": "annual", "period_date": d}
            for d in ("2018-12-31", "2019-12-31", "2020-12-31")
        ]
        rows += [
            {"ticker": "LIQ_THEN_DEAD", "period_type": "annual", "period_date": d}
            for d in ("2020-12-31", "2022-12-31", "2023-12-31")
        ]
        statements = pd.DataFrame(rows)
        # Keep LIQ_THEN_DEAD liquid so only the statement count can drop it.
        volume = long_history["volume_history"].copy()
        volume["LIQ_THEN_DEAD"] = 1_000_000.0
        selector = build_investability_screen(
            **{**long_history, "volume_history": volume},
            financial_statements=statements,
        ).fit(X)

        # LIQ_THEN_DEAD only had one annual report on file inside the window.
        assert list(selector.investable_universe_) == ["ALWAYS"]


def _reindexed(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    out = frame.copy()
    if kind == "date":
        out.index = pd.Index([ts.date() for ts in frame.index], dtype=object)
    elif kind == "iso_string":
        out.index = pd.Index([ts.strftime("%Y-%m-%d") for ts in frame.index])
    elif kind == "iso_string_utc":
        out.index = pd.Index(
            [ts.strftime("%Y-%m-%dT%H:%M:%S+00:00") for ts in frame.index]
        )
    else:
        raise AssertionError(kind)
    return out


class TestHistoryIndexTypes:
    """The window must also cut histories whose dates are not a DatetimeIndex."""

    @pytest.mark.parametrize("kind", ["date", "iso_string", "iso_string_utc"])
    def test_date_like_history_index_is_windowed(
        self, long_history: dict[str, pd.DataFrame], kind: str
    ) -> None:
        X = _returns_from(long_history["price_history"]).iloc[:400]
        data = {
            **long_history,
            "price_history": _reindexed(long_history["price_history"], kind),
            "volume_history": _reindexed(long_history["volume_history"], kind),
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            selector = build_investability_screen(**data).fit(X)

        assert list(selector.investable_universe_) == ["ALWAYS", "LIQ_THEN_DEAD"]

    @pytest.mark.parametrize("kind", ["range", "unparseable_strings"])
    def test_undated_history_warns_and_is_screened_in_full(
        self, long_history: dict[str, pd.DataFrame], kind: str
    ) -> None:
        X = _returns_from(long_history["price_history"]).iloc[:400]
        data = {**long_history}
        for name in ("price_history", "volume_history"):
            frame = long_history[name].copy()
            if kind == "range":
                frame = frame.reset_index(drop=True)
            else:
                frame.index = pd.Index([f"row-{i}" for i in range(len(frame))])
            data[name] = frame
        with pytest.warns(UserWarning) as caught:
            selector = build_investability_screen(**data).fit(X)

        messages = " ".join(str(w.message) for w in caught)
        assert "price_history has no date index" in messages
        assert "volume_history has no date index" in messages
        assert list(selector.investable_universe_) == ["ALWAYS", "NEW_LISTING"]

    def test_ambiguous_day_first_strings_are_not_guessed(
        self, long_history: dict[str, pd.DataFrame]
    ) -> None:
        X = _returns_from(long_history["price_history"]).iloc[:400]
        data = dict(long_history)
        for name in ("price_history", "volume_history"):
            frame = long_history[name].copy()
            frame.index = pd.Index([ts.strftime("%d/%m/%Y") for ts in frame.index])
            data[name] = frame
        with pytest.warns(UserWarning, match="no date index"):
            selector = build_investability_screen(**data).fit(X)

        assert list(selector.investable_universe_) == ["ALWAYS", "NEW_LISTING"]

    def test_empty_history_is_not_reported_as_undated(
        self, long_history: dict[str, pd.DataFrame]
    ) -> None:
        X = _returns_from(long_history["price_history"]).iloc[:400]
        selector = build_investability_screen(
            **{**long_history, "volume_history": pd.DataFrame()}
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            with pytest.raises(DataError, match="volume_history is empty"):
                selector.fit(X)

    def test_undated_returns_do_not_warn(
        self, long_history: dict[str, pd.DataFrame]
    ) -> None:
        X = _returns_from(long_history["price_history"]).reset_index(drop=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            build_investability_screen(**long_history).fit(X)
