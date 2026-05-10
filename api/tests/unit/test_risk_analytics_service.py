"""Unit tests for RiskAnalyticsService (issues #368, #369).

Covers:
  - compute_var: historical VaR/CVaR with known data
  - compute_var: parametric VaR/CVaR with known data
  - compute_var: CVaR values exceed VaR values at same confidence level
  - compute_var: raises ValueError for insufficient data
  - compute_correlation: NxN symmetric matrix with cluster labels
  - compute_factor_exposure: weighted exposures sum correctly
  - compute_factor_exposure: raises FactorScoresNotFoundError when no scores
  - compute_concentration: HHI, effective-N, top-N math (issue #369)
  - compute_liquidity: ADDV, days-to-liquidate, missing data (issue #369)
"""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.services.risk.risk_analytics_service import RiskAnalyticsService

_PORTFOLIO_ID = "00000000-0000-0000-0000-000000000001"
_WEIGHTS = {"AAPL": 0.6, "MSFT": 0.4}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_price_df(n: int = 300) -> pd.DataFrame:
    """Build a synthetic price DataFrame with deterministic daily returns."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    prices = pd.DataFrame(
        {
            "AAPL": 100 * np.cumprod(1 + rng.normal(0.0005, 0.015, n)),
            "MSFT": 200 * np.cumprod(1 + rng.normal(0.0004, 0.012, n)),
        },
        index=dates,
    )
    return prices


def _make_factor_scores(
    tickers: list[str],
    factor_types: list[str],
    score_date: datetime.date | None = None,
) -> list[MagicMock]:
    if score_date is None:
        score_date = datetime.date(2024, 1, 2)
    scores = []
    for ticker in tickers:
        for ft in factor_types:
            s = MagicMock()
            s.ticker = ticker
            s.factor_type = ft
            s.standardized_score = 0.5
            s.score_date = score_date
            scores.append(s)
    return scores


# ===========================================================================
# Regression: #423 — portfolio with one ticker missing from price_history
# ===========================================================================


class TestFetchWeightedReturnsSkipsMissingTickers:
    """_fetch_weighted_returns must survive when some weight tickers have no DB prices.

    Regression for #423: trading212 portfolio contains EDV.L which has zero rows in
    price_history. The previous implementation called ``prices.reindex(columns=tickers)``
    which reintroduced the missing ticker as an all-NaN column, then ``.dropna()``
    deleted every row and the downstream VaR/correlation computation failed with 400.
    """

    def setup_method(self) -> None:
        self.session = MagicMock()
        self.service = RiskAnalyticsService(self.session)

    def test_returns_nonempty_series_when_one_ticker_missing(self) -> None:
        """19 tickers have data, 1 is absent entirely — weighted series must still be built."""
        rng = np.random.default_rng(7)
        dates = pd.date_range("2024-01-02", periods=260, freq="B")
        prices = pd.DataFrame(
            {
                "AAPL": 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 260)),
                "MSFT": 200 * np.cumprod(1 + rng.normal(0.0004, 0.01, 260)),
                # NOTE: EDV.L intentionally absent — simulates DB gap
            },
            index=dates,
        )
        weights = {"AAPL": 0.6, "MSFT": 0.3, "EDV.L": 0.1}

        with patch.object(self.service, "_fetch_prices", return_value=prices):
            returns = self.service._fetch_weighted_returns(weights, lookback=252)

        assert not returns.empty
        assert len(returns) >= 250

    def test_compute_var_happy_path_with_missing_ticker(self) -> None:
        rng = np.random.default_rng(11)
        dates = pd.date_range("2024-01-02", periods=260, freq="B")
        prices = pd.DataFrame(
            {
                "AAPL": 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 260)),
                "MSFT": 200 * np.cumprod(1 + rng.normal(0.0004, 0.01, 260)),
            },
            index=dates,
        )
        weights = {"AAPL": 0.6, "MSFT": 0.3, "EDV.L": 0.1}

        with patch.object(self.service, "_fetch_prices", return_value=prices):
            result = self.service.compute_var(
                portfolio_id=_PORTFOLIO_ID,
                weights=weights,
                lookback=252,
                method="historical",
            )

        assert result["var"]["95"] > 0
        assert result["var"]["99"] > 0
        assert result["cvar"]["95"] > 0

    def test_compute_var_raises_when_all_tickers_missing(self) -> None:
        """When every weight ticker is absent from price_history, the 400 path must still fire."""
        empty = pd.DataFrame()
        with patch.object(self.service, "_fetch_prices", return_value=empty):
            with pytest.raises(ValueError, match="[Ii]nsufficient|[Nn]o price data"):
                self.service.compute_var(
                    portfolio_id=_PORTFOLIO_ID,
                    weights={"EDV.L": 1.0},
                    lookback=252,
                    method="historical",
                )

    def test_compute_correlation_happy_path_with_missing_ticker(self) -> None:
        rng = np.random.default_rng(13)
        dates = pd.date_range("2024-01-02", periods=260, freq="B")
        prices = pd.DataFrame(
            {
                "AAPL": 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 260)),
                "MSFT": 200 * np.cumprod(1 + rng.normal(0.0004, 0.01, 260)),
                "GOOG": 150 * np.cumprod(1 + rng.normal(0.0003, 0.01, 260)),
            },
            index=dates,
        )
        weights = {"AAPL": 0.4, "MSFT": 0.3, "GOOG": 0.2, "EDV.L": 0.1}

        with patch.object(self.service, "_fetch_prices", return_value=prices):
            result = self.service.compute_correlation(
                portfolio_id=_PORTFOLIO_ID,
                weights=weights,
                lookback=252,
            )

        assert set(result["assets"]) == {"AAPL", "MSFT", "GOOG"}
        mat = np.array(result["matrix"])
        assert np.isfinite(mat).all()

    def test_compute_correlation_raises_when_under_two_tickers_survive(self) -> None:
        """If only one ticker has DB prices, correlation must raise the existing 'at least 2' error."""
        rng = np.random.default_rng(17)
        dates = pd.date_range("2024-01-02", periods=260, freq="B")
        prices = pd.DataFrame(
            {"AAPL": 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 260))},
            index=dates,
        )
        weights = {"AAPL": 0.5, "EDV.L": 0.5}

        with patch.object(self.service, "_fetch_prices", return_value=prices):
            with pytest.raises(ValueError, match="[Aa]t least 2"):
                self.service.compute_correlation(
                    portfolio_id=_PORTFOLIO_ID,
                    weights=weights,
                    lookback=252,
                )

    def test_compute_var_short_lookback_smaller_than_history(self) -> None:
        """Short-lookback edge case: 30-day lookback should succeed when history is abundant."""
        rng = np.random.default_rng(19)
        dates = pd.date_range("2024-01-02", periods=260, freq="B")
        prices = pd.DataFrame(
            {
                "AAPL": 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 260)),
                "MSFT": 200 * np.cumprod(1 + rng.normal(0.0004, 0.01, 260)),
            },
            index=dates,
        )
        weights = {"AAPL": 0.6, "MSFT": 0.4}

        with patch.object(self.service, "_fetch_prices", return_value=prices):
            result = self.service.compute_var(
                portfolio_id=_PORTFOLIO_ID,
                weights=weights,
                lookback=30,
                method="historical",
            )

        assert result["lookback"] == 30
        assert result["var"]["95"] > 0


# ===========================================================================
# compute_var — historical method
# ===========================================================================


class TestComputeVarHistorical:
    """Historical VaR / CVaR with known synthetic data."""

    def setup_method(self) -> None:
        self.session = MagicMock()
        self.service = RiskAnalyticsService(self.session)
        self.prices = _make_price_df(300)

    def _patch_prices(self):  # type: ignore[return]
        return patch.object(
            self.service,
            "_fetch_weighted_returns",
            return_value=self._weighted_returns(),
        )

    def _weighted_returns(self) -> pd.Series:
        returns = self.prices.pct_change().dropna()
        weighted = returns["AAPL"] * 0.6 + returns["MSFT"] * 0.4
        return weighted.iloc[-252:]

    def test_returns_var_dict_with_three_confidence_levels(self) -> None:
        with self._patch_prices():
            result = self.service.compute_var(
                portfolio_id=_PORTFOLIO_ID,
                weights=_WEIGHTS,
                lookback=252,
                method="historical",
            )
        assert set(result["var"].keys()) == {"90", "95", "99"}

    def test_returns_cvar_dict_with_three_confidence_levels(self) -> None:
        with self._patch_prices():
            result = self.service.compute_var(
                portfolio_id=_PORTFOLIO_ID,
                weights=_WEIGHTS,
                lookback=252,
                method="historical",
            )
        assert set(result["cvar"].keys()) == {"90", "95", "99"}

    def test_cvar_exceeds_var_at_each_level(self) -> None:
        with self._patch_prices():
            result = self.service.compute_var(
                portfolio_id=_PORTFOLIO_ID,
                weights=_WEIGHTS,
                lookback=252,
                method="historical",
            )
        for level in ("90", "95", "99"):
            assert result["cvar"][level] >= result["var"][level]

    def test_var_99_exceeds_var_95(self) -> None:
        with self._patch_prices():
            result = self.service.compute_var(
                portfolio_id=_PORTFOLIO_ID,
                weights=_WEIGHTS,
                lookback=252,
                method="historical",
            )
        assert result["var"]["99"] >= result["var"]["95"]

    def test_var_95_exceeds_var_90(self) -> None:
        with self._patch_prices():
            result = self.service.compute_var(
                portfolio_id=_PORTFOLIO_ID,
                weights=_WEIGHTS,
                lookback=252,
                method="historical",
            )
        assert result["var"]["95"] >= result["var"]["90"]

    def test_method_in_result_is_historical(self) -> None:
        with self._patch_prices():
            result = self.service.compute_var(
                portfolio_id=_PORTFOLIO_ID,
                weights=_WEIGHTS,
                lookback=252,
                method="historical",
            )
        assert result["method"] == "historical"

    def test_n_observations_matches_lookback(self) -> None:
        with self._patch_prices():
            result = self.service.compute_var(
                portfolio_id=_PORTFOLIO_ID,
                weights=_WEIGHTS,
                lookback=252,
                method="historical",
            )
        assert result["n_observations"] == 252

    def test_raises_value_error_when_insufficient_data(self) -> None:
        empty_series = pd.Series(dtype=float)
        with (
            patch.object(
                self.service, "_fetch_weighted_returns", return_value=empty_series
            ),
            pytest.raises(ValueError, match="[Ii]nsufficient"),
        ):
            self.service.compute_var(
                portfolio_id=_PORTFOLIO_ID,
                weights=_WEIGHTS,
                lookback=252,
                method="historical",
            )


# ===========================================================================
# compute_var — parametric method
# ===========================================================================


class TestComputeVarParametric:
    """Parametric (normal) VaR / CVaR."""

    def setup_method(self) -> None:
        self.session = MagicMock()
        self.service = RiskAnalyticsService(self.session)
        self.prices = _make_price_df(300)

    def _patch_prices(self):  # type: ignore[return]
        weighted = self.prices.pct_change().dropna()
        wret = weighted["AAPL"] * 0.6 + weighted["MSFT"] * 0.4
        return patch.object(
            self.service, "_fetch_weighted_returns", return_value=wret.iloc[-252:]
        )

    def test_returns_var_and_cvar(self) -> None:
        with self._patch_prices():
            result = self.service.compute_var(
                portfolio_id=_PORTFOLIO_ID,
                weights=_WEIGHTS,
                lookback=252,
                method="parametric",
            )
        assert "var" in result
        assert "cvar" in result

    def test_method_in_result_is_parametric(self) -> None:
        with self._patch_prices():
            result = self.service.compute_var(
                portfolio_id=_PORTFOLIO_ID,
                weights=_WEIGHTS,
                lookback=252,
                method="parametric",
            )
        assert result["method"] == "parametric"

    def test_cvar_exceeds_var_parametric(self) -> None:
        with self._patch_prices():
            result = self.service.compute_var(
                portfolio_id=_PORTFOLIO_ID,
                weights=_WEIGHTS,
                lookback=252,
                method="parametric",
            )
        for level in ("90", "95", "99"):
            assert result["cvar"][level] >= result["var"][level]


# ===========================================================================
# compute_correlation
# ===========================================================================


class TestComputeCorrelation:
    """Clustered correlation matrix tests."""

    def setup_method(self) -> None:
        self.session = MagicMock()
        self.service = RiskAnalyticsService(self.session)
        self.prices = _make_price_df(300)

    def _patch_prices(self):  # type: ignore[return]
        return patch.object(self.service, "_fetch_prices", return_value=self.prices)

    def test_returns_assets_list(self) -> None:
        with self._patch_prices():
            result = self.service.compute_correlation(
                portfolio_id=_PORTFOLIO_ID,
                weights=_WEIGHTS,
                lookback=252,
            )
        assert isinstance(result["assets"], list)

    def test_matrix_is_n_by_n(self) -> None:
        with self._patch_prices():
            result = self.service.compute_correlation(
                portfolio_id=_PORTFOLIO_ID,
                weights=_WEIGHTS,
                lookback=252,
            )
        n = len(result["assets"])
        assert len(result["matrix"]) == n
        assert all(len(row) == n for row in result["matrix"])

    def test_matrix_is_symmetric(self) -> None:
        with self._patch_prices():
            result = self.service.compute_correlation(
                portfolio_id=_PORTFOLIO_ID,
                weights=_WEIGHTS,
                lookback=252,
            )
        mat = np.array(result["matrix"])
        np.testing.assert_allclose(mat, mat.T, atol=1e-9)

    def test_diagonal_is_ones(self) -> None:
        with self._patch_prices():
            result = self.service.compute_correlation(
                portfolio_id=_PORTFOLIO_ID,
                weights=_WEIGHTS,
                lookback=252,
            )
        mat = np.array(result["matrix"])
        np.testing.assert_allclose(
            np.diag(mat), np.ones(len(result["assets"])), atol=1e-9
        )

    def test_cluster_labels_length_matches_assets(self) -> None:
        with self._patch_prices():
            result = self.service.compute_correlation(
                portfolio_id=_PORTFOLIO_ID,
                weights=_WEIGHTS,
                lookback=252,
            )
        assert len(result["cluster_labels"]) == len(result["assets"])

    def test_cluster_labels_are_integers(self) -> None:
        with self._patch_prices():
            result = self.service.compute_correlation(
                portfolio_id=_PORTFOLIO_ID,
                weights=_WEIGHTS,
                lookback=252,
            )
        assert all(isinstance(v, int) for v in result["cluster_labels"])

    def test_raises_value_error_for_single_asset(self) -> None:
        single_asset_prices = self.prices[["AAPL"]]
        with (
            patch.object(
                self.service, "_fetch_prices", return_value=single_asset_prices
            ),
            pytest.raises(ValueError, match="[Aa]t least 2"),
        ):
            self.service.compute_correlation(
                portfolio_id=_PORTFOLIO_ID,
                weights={"AAPL": 1.0},
                lookback=252,
            )


# ===========================================================================
# compute_factor_exposure
# ===========================================================================


class TestComputeFactorExposure:
    """Portfolio-weighted factor exposure tests."""

    def setup_method(self) -> None:
        self.session = MagicMock()
        self.service = RiskAnalyticsService(self.session)

    def test_returns_exposures_dict(self) -> None:
        scores = _make_factor_scores(["AAPL", "MSFT"], ["momentum", "quality"])
        with patch.object(
            self.service, "_fetch_latest_factor_scores", return_value=scores
        ):
            result = self.service.compute_factor_exposure(
                portfolio_id=_PORTFOLIO_ID,
                weights=_WEIGHTS,
            )
        assert isinstance(result["exposures"], dict)

    def test_returns_asset_exposures_dict(self) -> None:
        scores = _make_factor_scores(["AAPL", "MSFT"], ["momentum", "quality"])
        with patch.object(
            self.service, "_fetch_latest_factor_scores", return_value=scores
        ):
            result = self.service.compute_factor_exposure(
                portfolio_id=_PORTFOLIO_ID,
                weights=_WEIGHTS,
            )
        assert isinstance(result["asset_exposures"], dict)

    def test_weighted_exposure_sums_correctly(self) -> None:
        """Weighted exposure = sum(weight_i * score_i) for each factor."""
        aapl_score, msft_score = 0.8, 0.4
        aapl_weight, msft_weight = 0.6, 0.4
        expected = aapl_weight * aapl_score + msft_weight * msft_score

        scores = []
        for ticker, score in [("AAPL", aapl_score), ("MSFT", msft_score)]:
            s = MagicMock()
            s.ticker = ticker
            s.factor_type = "momentum"
            s.standardized_score = score
            s.score_date = datetime.date(2024, 1, 2)
            scores.append(s)

        with patch.object(
            self.service, "_fetch_latest_factor_scores", return_value=scores
        ):
            result = self.service.compute_factor_exposure(
                portfolio_id=_PORTFOLIO_ID,
                weights={"AAPL": aapl_weight, "MSFT": msft_weight},
            )
        assert abs(result["exposures"]["momentum"] - expected) < 1e-9

    def test_returns_empty_when_no_factor_scores(self) -> None:
        """Regression for #424: absence of factor scores is an empty state, not 404.

        The portfolio exists; there just isn't a computation to show yet. The
        frontend renders 'no data yet' rather than a navigation error.
        """
        with patch.object(self.service, "_fetch_latest_factor_scores", return_value=[]):
            result = self.service.compute_factor_exposure(
                portfolio_id=_PORTFOLIO_ID,
                weights=_WEIGHTS,
            )
        assert result == {"exposures": {}, "asset_exposures": {}}

    def test_asset_exposures_contain_all_weights_tickers(self) -> None:
        scores = _make_factor_scores(["AAPL", "MSFT"], ["momentum"])
        with patch.object(
            self.service, "_fetch_latest_factor_scores", return_value=scores
        ):
            result = self.service.compute_factor_exposure(
                portfolio_id=_PORTFOLIO_ID,
                weights=_WEIGHTS,
            )
        assert set(result["asset_exposures"].keys()) >= {"AAPL", "MSFT"}


# ===========================================================================
# compute_concentration (issue #369)
# ===========================================================================


class TestComputeConcentration:
    """HHI, effective-N, and top-N concentration math tests."""

    def setup_method(self) -> None:
        self.session = MagicMock()
        self.service = RiskAnalyticsService(self.session)

    def _patch_names(self, names: dict):  # type: ignore[return]
        return patch.object(self.service, "_fetch_asset_names", return_value=names)

    def test_assets_list_contains_one_entry_per_weight(self) -> None:
        with self._patch_names({"AAPL": "Apple Inc", "MSFT": "Microsoft Corp"}):
            result = self.service.compute_concentration(_WEIGHTS, n=5)
        assert len(result["assets"]) == 2

    def test_each_asset_entry_has_ticker_name_weight(self) -> None:
        with self._patch_names({"AAPL": "Apple Inc", "MSFT": "Microsoft Corp"}):
            result = self.service.compute_concentration(_WEIGHTS, n=5)
        for asset in result["assets"]:
            assert "ticker" in asset
            assert "name" in asset
            assert "weight" in asset

    def test_hhi_equals_sum_of_squared_weights(self) -> None:
        expected = 0.6**2 + 0.4**2
        with self._patch_names({"AAPL": "Apple Inc", "MSFT": "Microsoft Corp"}):
            result = self.service.compute_concentration(_WEIGHTS, n=5)
        assert abs(result["summary"]["hhi"] - expected) < 1e-9

    def test_effective_n_is_reciprocal_of_hhi(self) -> None:
        hhi = 0.6**2 + 0.4**2
        expected = 1.0 / hhi
        with self._patch_names({"AAPL": "Apple Inc", "MSFT": "Microsoft Corp"}):
            result = self.service.compute_concentration(_WEIGHTS, n=5)
        assert abs(result["summary"]["effective_n"] - expected) < 1e-7

    def test_top_n_ratio_sums_largest_n_weights(self) -> None:
        weights = {"AAPL": 0.4, "MSFT": 0.3, "GOOGL": 0.2, "AMZN": 0.1}
        names = {t: t for t in weights}
        with self._patch_names(names):
            result = self.service.compute_concentration(weights, n=2)
        # top-2: 0.4 + 0.3 = 0.7
        assert abs(result["summary"]["top_n_ratio"] - 0.7) < 1e-9

    def test_single_asset_portfolio_hhi_is_one(self) -> None:
        with self._patch_names({"AAPL": "Apple Inc"}):
            result = self.service.compute_concentration({"AAPL": 1.0}, n=5)
        assert result["summary"]["hhi"] == pytest.approx(1.0)
        assert result["summary"]["effective_n"] == pytest.approx(1.0)

    def test_empty_portfolio_returns_zero_summary(self) -> None:
        with self._patch_names({}):
            result = self.service.compute_concentration({}, n=5)
        assert result["assets"] == []
        assert result["summary"]["hhi"] == 0.0
        assert result["summary"]["effective_n"] == 0.0
        assert result["summary"]["top_n_ratio"] == 0.0

    def test_top_n_larger_than_portfolio_clamps_to_total_weight(self) -> None:
        """When n > number of assets, top_n_ratio equals sum of all weights."""
        with self._patch_names({"AAPL": "Apple Inc", "MSFT": "Microsoft Corp"}):
            result = self.service.compute_concentration(_WEIGHTS, n=99)
        assert abs(result["summary"]["top_n_ratio"] - 1.0) < 1e-9


# ===========================================================================
# compute_liquidity (issue #369)
# ===========================================================================


class TestComputeLiquidity:
    """ADDV-based days-to-liquidate and liquidity cost tests."""

    _PARTICIPATION = 0.1

    def setup_method(self) -> None:
        self.session = MagicMock()
        self.service = RiskAnalyticsService(self.session)

    def _patch_addv(self, value: float | None):  # type: ignore[return]
        return patch.object(self.service, "_fetch_addv", return_value=value)

    def _patch_names(self, names: dict):  # type: ignore[return]
        return patch.object(self.service, "_fetch_asset_names", return_value=names)

    def test_returns_assets_list(self) -> None:
        with self._patch_addv(1_000_000), self._patch_names({"AAPL": "Apple Inc"}):
            result = self.service.compute_liquidity(
                {"AAPL": 0.5}, 20, self._PARTICIPATION
            )
        assert isinstance(result["assets"], list)
        assert len(result["assets"]) == 1

    def test_days_to_liquidate_follows_formula(self) -> None:
        # weight=0.5, ADDV=1e6, participation_rate=0.1 → 0.5 / (1e6 × 0.1) = 5e-6
        expected = 0.5 / (1_000_000 * self._PARTICIPATION)
        with self._patch_addv(1_000_000), self._patch_names({"AAPL": "Apple Inc"}):
            result = self.service.compute_liquidity(
                {"AAPL": 0.5}, 20, self._PARTICIPATION
            )
        assert abs(result["assets"][0]["days_to_liquidate"] - expected) < 1e-12

    def test_liquidity_cost_is_days_times_market_impact_constant(self) -> None:
        with self._patch_addv(1_000_000), self._patch_names({"AAPL": "Apple Inc"}):
            result = self.service.compute_liquidity(
                {"AAPL": 0.5}, 20, self._PARTICIPATION
            )
        asset = result["assets"][0]
        assert (
            abs(asset["liquidity_cost"] - asset["days_to_liquidate"] * 0.0005) < 1e-12
        )

    def test_avg_daily_volume_matches_mocked_addv(self) -> None:
        with self._patch_addv(2_500_000), self._patch_names({"AAPL": "Apple Inc"}):
            result = self.service.compute_liquidity(
                {"AAPL": 0.5}, 20, self._PARTICIPATION
            )
        assert result["assets"][0]["avg_daily_volume"] == pytest.approx(2_500_000)

    def test_missing_price_history_yields_null_liquidity_fields(self) -> None:
        with self._patch_addv(None), self._patch_names({"AAPL": "Apple Inc"}):
            result = self.service.compute_liquidity(
                {"AAPL": 0.5}, 20, self._PARTICIPATION
            )
        asset = result["assets"][0]
        assert asset["avg_daily_volume"] is None
        assert asset["days_to_liquidate"] is None
        assert asset["liquidity_cost"] is None

    def test_partial_response_when_one_asset_missing_data(self) -> None:
        """Partial response: AAPL has data, MSFT does not."""

        def side_effect(ticker: str, lookback: int) -> float | None:
            return 1_000_000 if ticker == "AAPL" else None

        with (
            patch.object(self.service, "_fetch_addv", side_effect=side_effect),
            self._patch_names({"AAPL": "Apple Inc", "MSFT": "Microsoft Corp"}),
        ):
            result = self.service.compute_liquidity(
                {"AAPL": 0.6, "MSFT": 0.4}, 20, self._PARTICIPATION
            )

        tickers = {a["ticker"]: a for a in result["assets"]}
        assert tickers["AAPL"]["days_to_liquidate"] is not None
        assert tickers["MSFT"]["days_to_liquidate"] is None

    def test_weighted_avg_days_excludes_null_assets(self) -> None:
        """Weighted avg days uses only assets with valid ADDV."""
        expected_days_aapl = 0.6 / (1_000_000 * self._PARTICIPATION)

        def side_effect(ticker: str, lookback: int) -> float | None:
            return 1_000_000 if ticker == "AAPL" else None

        with (
            patch.object(self.service, "_fetch_addv", side_effect=side_effect),
            self._patch_names({"AAPL": "Apple Inc", "MSFT": "Microsoft Corp"}),
        ):
            result = self.service.compute_liquidity(
                {"AAPL": 0.6, "MSFT": 0.4}, 20, self._PARTICIPATION
            )

        assert (
            abs(
                result["summary"]["weighted_avg_days_to_liquidate"] - expected_days_aapl
            )
            < 1e-12
        )

    def test_returns_summary_with_weighted_avg_days(self) -> None:
        with self._patch_addv(1_000_000), self._patch_names({"AAPL": "Apple Inc"}):
            result = self.service.compute_liquidity(
                {"AAPL": 0.5}, 20, self._PARTICIPATION
            )
        assert "weighted_avg_days_to_liquidate" in result["summary"]
