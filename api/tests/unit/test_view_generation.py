"""Unit tests for LLM-driven Black-Litterman view generation."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from baml_client.types import AssetFactorData, AssetView, ViewOutput
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers — fixture factories
# ---------------------------------------------------------------------------

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN"]


def _make_factor_data(ticker: str = "AAPL") -> AssetFactorData:
    return AssetFactorData(
        ticker=ticker,
        trailing_pe=28.0,
        price_to_book=45.0,
        ev_to_ebitda=21.0,
        momentum_12_1m=0.20,
        momentum_1m=0.02,
        rsi_14=60.0,
        return_on_equity=1.4,
        debt_to_equity=1.8,
        profit_margins=0.25,
        revenue_growth_yoy=0.06,
        earnings_growth_yoy=0.13,
        pct_from_52w_high=-0.03,
        pct_from_52w_low=0.28,
        recommendation_mean=1.9,
        target_upside=0.12,
        analyst_count=45,
    )


def _make_asset_view(
    asset: str = "AAPL",
    direction: int = 1,
    magnitude_bps: float = 200.0,
    confidence: float = 0.7,
) -> AssetView:
    return AssetView(
        asset=asset,
        direction=direction,
        magnitude_bps=magnitude_bps,
        confidence=confidence,
        reasoning="Strong momentum and quality signals.",
    )


def _make_view_output(views: list[AssetView] | None = None) -> ViewOutput:
    if views is None:
        views = [_make_asset_view()]
    alphas = {v.asset: v.confidence for v in views}
    return ViewOutput(views=views, idzorek_alphas=alphas, rationale="LLM rationale.")


# ---------------------------------------------------------------------------
# Service layer — _views_to_arrays
# ---------------------------------------------------------------------------


class TestViewsToArrays:
    def test_p_shape(self) -> None:
        from app.services.view_generation import _views_to_arrays

        views = [_make_asset_view("AAPL"), _make_asset_view("MSFT", direction=-1)]
        _, P, Q, _ = _views_to_arrays(views, TICKERS)

        assert P.shape == (2, len(TICKERS))

    def test_q_shape(self) -> None:
        from app.services.view_generation import _views_to_arrays

        views = [_make_asset_view("AAPL"), _make_asset_view("MSFT")]
        _, P, Q, _ = _views_to_arrays(views, TICKERS)

        assert Q.shape == (2,)

    def test_p_is_one_hot_per_row(self) -> None:
        from app.services.view_generation import _views_to_arrays

        views = [_make_asset_view("AAPL"), _make_asset_view("MSFT")]
        _, P, _, _ = _views_to_arrays(views, TICKERS)

        for row in P:
            assert np.sum(row != 0) == 1
            assert np.sum(row) == pytest.approx(1.0)

    def test_q_values_in_decimal(self) -> None:
        """200 bps with direction +1 → Q = +0.02."""
        from app.services.view_generation import _views_to_arrays

        views = [_make_asset_view("AAPL", direction=1, magnitude_bps=200.0)]
        _, _, Q, _ = _views_to_arrays(views, TICKERS)

        assert Q[0] == pytest.approx(0.02)

    def test_negative_direction_negates_q(self) -> None:
        from app.services.view_generation import _views_to_arrays

        views = [_make_asset_view("MSFT", direction=-1, magnitude_bps=150.0)]
        _, _, Q, _ = _views_to_arrays(views, TICKERS)

        assert Q[0] == pytest.approx(-0.015)

    def test_view_strings_parseable(self) -> None:
        from app.services.view_generation import _views_to_arrays

        views = [_make_asset_view("AAPL", magnitude_bps=200.0)]
        view_strings, _, _, _ = _views_to_arrays(views, TICKERS)

        assert len(view_strings) == 1
        assert "AAPL ==" in view_strings[0]
        assert "0.02" in view_strings[0]

    def test_view_confidences_match_views(self) -> None:
        from app.services.view_generation import _views_to_arrays

        views = [
            _make_asset_view("AAPL", confidence=0.8),
            _make_asset_view("MSFT", confidence=0.5),
        ]
        _, _, _, confidences = _views_to_arrays(views, TICKERS)

        assert len(confidences) == 2
        assert confidences[0] == pytest.approx(0.8)
        assert confidences[1] == pytest.approx(0.5)

    def test_unknown_asset_skipped(self) -> None:
        from app.services.view_generation import _views_to_arrays

        views = [_make_asset_view("UNKNOWN_XYZ"), _make_asset_view("AAPL")]
        _, P, Q, _ = _views_to_arrays(views, TICKERS)

        # Only AAPL survives
        assert P.shape[0] == 1
        assert Q.shape == (1,)

    def test_empty_views_returns_empty_arrays(self) -> None:
        from app.services.view_generation import _views_to_arrays

        _, P, Q, conf = _views_to_arrays([], TICKERS)
        assert P.shape == (0, len(TICKERS))
        assert Q.shape == (0,)
        assert conf == []


# ---------------------------------------------------------------------------
# Service layer — _validate_idzorek_alphas
# ---------------------------------------------------------------------------


class TestValidateIdzorekAlphas:
    def test_values_clamped_to_open_unit_interval(self) -> None:
        from app.services.view_generation import _validate_idzorek_alphas

        raw = {"AAPL": 0.0, "MSFT": 1.0, "GOOGL": 1.5}
        result = _validate_idzorek_alphas(raw, ["AAPL", "MSFT", "GOOGL"])

        assert all(0.0 < v < 1.0 for v in result.values())

    def test_missing_key_defaults_to_half(self) -> None:
        from app.services.view_generation import _validate_idzorek_alphas

        result = _validate_idzorek_alphas({}, ["AAPL"])
        assert result["AAPL"] == pytest.approx(0.5)

    def test_valid_values_unchanged(self) -> None:
        from app.services.view_generation import _validate_idzorek_alphas

        result = _validate_idzorek_alphas({"AAPL": 0.7}, ["AAPL"])
        assert result["AAPL"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Service layer — generate_views (mocked BAML)
# ---------------------------------------------------------------------------


class TestGenerateViews:
    def test_raises_on_empty_factor_data(self) -> None:
        from app.services.view_generation import generate_views

        with pytest.raises(ValueError, match="empty"):
            generate_views(TICKERS, [])

    def test_returns_generated_views(self) -> None:
        from app.services.view_generation import generate_views

        factor_data = [_make_factor_data("AAPL")]
        mock_output = _make_view_output([_make_asset_view("AAPL")])

        with patch(
            "app.services.view_generation.b.GenerateViews", return_value=mock_output
        ):
            result = generate_views(TICKERS, factor_data)

        assert len(result.view_strings) == 1
        assert result.P.shape == (1, len(TICKERS))
        assert result.Q.shape == (1,)

    def test_p_shape_matches_n_views_n_assets(self) -> None:
        from app.services.view_generation import generate_views

        factor_data = [_make_factor_data("AAPL"), _make_factor_data("MSFT")]
        views = [_make_asset_view("AAPL"), _make_asset_view("MSFT")]
        mock_output = _make_view_output(views)

        with patch(
            "app.services.view_generation.b.GenerateViews", return_value=mock_output
        ):
            result = generate_views(TICKERS, factor_data)

        assert result.P.shape == (2, len(TICKERS))

    def test_q_shape_matches_n_views(self) -> None:
        from app.services.view_generation import generate_views

        factor_data = [_make_factor_data("AAPL")]
        mock_output = _make_view_output([_make_asset_view("AAPL")])

        with patch(
            "app.services.view_generation.b.GenerateViews", return_value=mock_output
        ):
            result = generate_views(TICKERS, factor_data)

        assert result.Q.shape == (len(result.view_strings),)

    def test_all_idzorek_alphas_in_unit_interval(self) -> None:
        from app.services.view_generation import generate_views

        factor_data = [_make_factor_data("AAPL")]
        mock_output = _make_view_output([_make_asset_view("AAPL", confidence=0.9)])

        with patch(
            "app.services.view_generation.b.GenerateViews", return_value=mock_output
        ):
            result = generate_views(TICKERS, factor_data)

        assert all(0.0 < alpha < 1.0 for alpha in result.idzorek_alphas.values())

    def test_view_strings_compatible_with_bl(self) -> None:
        """Each view string must contain '==' — required by skfolio BlackLitterman."""
        from app.services.view_generation import generate_views

        factor_data = [_make_factor_data("AAPL")]
        mock_output = _make_view_output()

        with patch(
            "app.services.view_generation.b.GenerateViews", return_value=mock_output
        ):
            result = generate_views(TICKERS, factor_data)

        for vs in result.view_strings:
            assert "==" in vs

    def test_views_on_non_requested_tickers_filtered(self) -> None:
        """LLM may hallucinate a ticker not in the universe — must be dropped."""
        from app.services.view_generation import generate_views

        factor_data = [_make_factor_data("AAPL")]
        mock_output = _make_view_output(
            [_make_asset_view("AAPL"), _make_asset_view("HALLUCINATED")]
        )

        with patch(
            "app.services.view_generation.b.GenerateViews", return_value=mock_output
        ):
            result = generate_views(TICKERS, factor_data)

        assert all(v.asset in set(TICKERS) for v in result.asset_views)


# ---------------------------------------------------------------------------
# Service layer — _compute_momentum and _compute_rsi
# ---------------------------------------------------------------------------


class TestMomentumRSI:
    def test_momentum_returns_none_on_insufficient_data(self) -> None:
        from app.services.view_generation import _compute_momentum

        mom_12_1m, mom_1m = _compute_momentum([100.0] * 5)
        assert mom_12_1m is None
        assert mom_1m is None

    def test_rsi_returns_none_on_insufficient_data(self) -> None:
        from app.services.view_generation import _compute_rsi

        assert _compute_rsi([100.0] * 5) is None

    def test_rsi_overbought(self) -> None:
        """All-upward price series → RSI approaches 100."""
        from app.services.view_generation import _compute_rsi

        prices = [float(i) for i in range(1, 31)]  # strictly increasing
        rsi = _compute_rsi(prices)
        assert rsi is not None
        assert rsi > 70.0

    def test_rsi_oversold(self) -> None:
        from app.services.view_generation import _compute_rsi

        prices = [float(30 - i) for i in range(30)]  # strictly decreasing
        rsi = _compute_rsi(prices)
        assert rsi is not None
        assert rsi < 30.0


# ---------------------------------------------------------------------------
# API endpoint — POST /api/v1/views/generate
# ---------------------------------------------------------------------------

URL = "/api/v1/views/generate"

TICKERS_PAYLOAD = ["AAPL", "MSFT", "GOOGL"]


def _make_generated_views(
    tickers: list[str],
    asset_views: list[AssetView] | None = None,
) -> GeneratedViews:
    """Build a GeneratedViews result for mocking the endpoint's generate_views call."""
    from app.services.view_generation import (
        GeneratedViews,
        _validate_idzorek_alphas,
        _views_to_arrays,
    )

    if asset_views is None:
        asset_views = [_make_asset_view("AAPL")]

    view_strings, P, Q, confidences = _views_to_arrays(asset_views, tickers)
    alphas = _validate_idzorek_alphas(
        {v.asset: v.confidence for v in asset_views},
        [v.asset for v in asset_views],
    )
    return GeneratedViews(
        view_strings=view_strings,
        P=P,
        Q=Q,
        view_confidences=confidences,
        idzorek_alphas=alphas,
        asset_views=asset_views,
        rationale="Test rationale.",
    )


class TestGenerateViewsEndpoint:
    """Endpoint tests use the 'client' fixture from conftest (DB override included).

    Both fetch_factor_data and generate_views are patched at their import site
    in the views module (app.api.v1.views.*) to avoid touching the real DB or LLM.
    """

    PAYLOAD = {"tickers": TICKERS_PAYLOAD}
    # Patch targets: where the names are imported in the endpoint module
    _FETCH = "app.api.v1.views.fetch_factor_data"
    _GENERATE = "app.api.v1.views.generate_views"

    def test_successful_response_shape(self, client: TestClient) -> None:
        asset_views = [_make_asset_view("AAPL"), _make_asset_view("MSFT")]
        mock_gv = _make_generated_views(TICKERS_PAYLOAD, asset_views)

        with (
            patch(
                self._FETCH,
                return_value=[_make_factor_data(t) for t in TICKERS_PAYLOAD],
            ),
            patch(self._GENERATE, return_value=mock_gv),
        ):
            resp = client.post(URL, json=self.PAYLOAD)

        assert resp.status_code == 200
        data = resp.json()
        n_views = data["n_views"]
        n_assets = data["n_assets"]

        assert n_assets == len(TICKERS_PAYLOAD)
        assert n_views == len(data["view_strings"])
        assert len(data["P"]) == n_views
        assert all(len(row) == n_assets for row in data["P"])
        assert len(data["Q"]) == n_views
        assert len(data["view_confidences"]) == n_views

    def test_all_idzorek_alphas_in_open_unit_interval(self, client: TestClient) -> None:
        mock_gv = _make_generated_views(TICKERS_PAYLOAD)

        with (
            patch(self._FETCH, return_value=[_make_factor_data("AAPL")]),
            patch(self._GENERATE, return_value=mock_gv),
        ):
            resp = client.post(URL, json=self.PAYLOAD)

        assert resp.status_code == 200
        for alpha in resp.json()["idzorek_alphas"].values():
            assert 0.0 < alpha < 1.0

    def test_view_confidences_in_unit_interval(self, client: TestClient) -> None:
        mock_gv = _make_generated_views(
            TICKERS_PAYLOAD, [_make_asset_view("AAPL", confidence=0.75)]
        )

        with (
            patch(self._FETCH, return_value=[_make_factor_data("AAPL")]),
            patch(self._GENERATE, return_value=mock_gv),
        ):
            resp = client.post(URL, json=self.PAYLOAD)

        assert resp.status_code == 200
        for c in resp.json()["view_confidences"]:
            assert 0.0 < c < 1.0

    def test_p_matrix_shape_n_views_x_n_assets(self, client: TestClient) -> None:
        asset_views = [_make_asset_view("AAPL"), _make_asset_view("MSFT")]
        mock_gv = _make_generated_views(TICKERS_PAYLOAD, asset_views)

        with (
            patch(
                self._FETCH,
                return_value=[_make_factor_data(t) for t in TICKERS_PAYLOAD],
            ),
            patch(self._GENERATE, return_value=mock_gv),
        ):
            resp = client.post(URL, json=self.PAYLOAD)

        data = resp.json()
        P = np.array(data["P"])
        assert P.shape == (data["n_views"], data["n_assets"])

    def test_q_shape_n_views(self, client: TestClient) -> None:
        mock_gv = _make_generated_views(TICKERS_PAYLOAD)

        with (
            patch(self._FETCH, return_value=[_make_factor_data("AAPL")]),
            patch(self._GENERATE, return_value=mock_gv),
        ):
            resp = client.post(URL, json=self.PAYLOAD)

        data = resp.json()
        assert len(data["Q"]) == data["n_views"]

    def test_fewer_than_2_tickers_rejected(self, client: TestClient) -> None:
        resp = client.post(URL, json={"tickers": ["AAPL"]})
        assert resp.status_code == 422

    def test_empty_tickers_rejected(self, client: TestClient) -> None:
        resp = client.post(URL, json={"tickers": []})
        assert resp.status_code == 422

    def test_no_db_data_returns_422(self, client: TestClient) -> None:
        with patch(self._FETCH, return_value=[]):
            resp = client.post(URL, json=self.PAYLOAD)
        assert resp.status_code == 422

    def test_llm_error_returns_502(self, client: TestClient) -> None:
        with (
            patch(self._FETCH, return_value=[_make_factor_data("AAPL")]),
            patch(self._GENERATE, side_effect=RuntimeError("LLM timeout")),
        ):
            resp = client.post(URL, json=self.PAYLOAD)

        assert resp.status_code == 502

    def test_tickers_missing_data_reported(self, client: TestClient) -> None:
        """Assets with no DB record should appear in tickers_missing_data."""
        mock_gv = _make_generated_views(TICKERS_PAYLOAD)

        with (
            patch(self._FETCH, return_value=[_make_factor_data("AAPL")]),  # only AAPL
            patch(self._GENERATE, return_value=mock_gv),
        ):
            resp = client.post(URL, json=self.PAYLOAD)

        data = resp.json()
        assert "MSFT" in data["tickers_missing_data"]
        assert "GOOGL" in data["tickers_missing_data"]
        assert "AAPL" in data["tickers_with_data"]

    def test_output_passes_directly_to_build_black_litterman(
        self, client: TestClient
    ) -> None:
        """Smoke-test that view_strings feed into BlackLittermanConfig without error."""
        from optimizer.views._config import BlackLittermanConfig

        mock_gv = _make_generated_views(
            TICKERS_PAYLOAD, [_make_asset_view("AAPL", magnitude_bps=200)]
        )

        with (
            patch(self._FETCH, return_value=[_make_factor_data("AAPL")]),
            patch(self._GENERATE, return_value=mock_gv),
        ):
            resp = client.post(URL, json=self.PAYLOAD)

        data = resp.json()
        views_tuple = tuple(data["view_strings"])
        config = BlackLittermanConfig(views=views_tuple)
        assert len(config.views) == data["n_views"]
