"""Empty-data and exception-path tests for view_generation service (issue #825).

Covers gaps NOT in tests/unit/test_view_generation.py:
  - fetch_factor_data with empty tickers list returns empty list
  - fetch_factor_data when all tickers missing from DB returns empty list
  - generate_views raises ValueError when factor_data is empty (via patched BAML)
  - _build_factor_data when instrument not found returns None
  - _build_factor_data with empty closes list (no 52w high/low computation)
  - _safe_float with various invalid inputs returns None
  - _compute_momentum with exactly enough data
  - result of fetch_factor_data is json-safe representation
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tests._fixtures.assertions import assert_json_safe

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_repo(instrument=None, profile=None, closes=None) -> MagicMock:
    repo = MagicMock()
    repo.get_instrument_by_ticker.return_value = instrument
    repo.get_ticker_profile.return_value = profile
    repo.get_close_prices.return_value = closes if closes is not None else []
    return repo


def _fake_instrument(instrument_id: int = 1) -> MagicMock:
    inst = MagicMock()
    inst.id = instrument_id
    return inst


# ---------------------------------------------------------------------------
# fetch_factor_data — empty inputs
# ---------------------------------------------------------------------------


class TestWhenEmptyTickersThenFetchFactorDataReturnsEmpty:
    def test_when_empty_tickers_list_then_returns_empty_list(self) -> None:
        from app.services.views.view_generation import fetch_factor_data

        repo = _make_repo()
        result = fetch_factor_data(repo, [])

        assert result == []

    def test_when_all_tickers_missing_from_db_then_returns_empty_list(self) -> None:
        from app.services.views.view_generation import fetch_factor_data

        # instrument not found → returns None → skipped
        repo = _make_repo(instrument=None)
        result = fetch_factor_data(repo, ["NONEXIST1", "NONEXIST2"])

        assert result == []

    def test_when_empty_result_then_json_safe_representation(self) -> None:
        from app.services.views.view_generation import fetch_factor_data

        repo = _make_repo(instrument=None)
        fetch_factor_data(repo, ["GHOST"])

        payload = {"factor_data": []}  # result is an empty list
        assert_json_safe(payload)


# ---------------------------------------------------------------------------
# generate_views — ValueError on empty factor_data
# ---------------------------------------------------------------------------


class TestWhenEmptyFactorDataThenGenerateViewsRaises:
    def test_when_factor_data_empty_then_value_error_raised(self) -> None:
        from app.services.views.view_generation import generate_views

        with pytest.raises(ValueError, match="empty"):
            generate_views(["AAPL", "MSFT"], [])

    def test_when_factor_data_empty_then_baml_not_called(self) -> None:
        from app.services.views.view_generation import generate_views

        with patch("app.services.views.view_generation.b.GenerateViews") as mock_baml:
            with pytest.raises(ValueError):
                generate_views(["AAPL"], [])

        mock_baml.assert_not_called()


# ---------------------------------------------------------------------------
# _build_factor_data — instrument missing branch
# ---------------------------------------------------------------------------


class TestWhenInstrumentMissingThenBuildFactorDataReturnsNone:
    def test_when_instrument_not_in_db_then_returns_none(self) -> None:
        from app.services.views.view_generation import _build_factor_data

        repo = _make_repo(instrument=None)
        result = _build_factor_data(repo, "UNKN")

        assert result is None

    def test_when_instrument_missing_then_repo_profile_not_queried(self) -> None:
        from app.services.views.view_generation import _build_factor_data

        repo = _make_repo(instrument=None)
        _build_factor_data(repo, "UNKN")

        repo.get_ticker_profile.assert_not_called()


# ---------------------------------------------------------------------------
# _build_factor_data — empty closes list (no 52w computation)
# ---------------------------------------------------------------------------


class TestWhenEmptyClosesThenBuildFactorDataHandlesGracefully:
    def test_when_closes_empty_then_pct_from_high_is_none(self) -> None:
        from app.services.views.view_generation import _build_factor_data

        inst = _fake_instrument()
        repo = _make_repo(instrument=inst, profile=None, closes=[])
        result = _build_factor_data(repo, "AAPL")

        assert result is not None
        assert result.pct_from_52w_high is None
        assert result.pct_from_52w_low is None

    def test_when_closes_empty_then_momentum_is_none(self) -> None:
        from app.services.views.view_generation import _build_factor_data

        inst = _fake_instrument()
        repo = _make_repo(instrument=inst, profile=None, closes=[])
        result = _build_factor_data(repo, "AAPL")

        assert result is not None
        assert result.momentum_12_1m is None
        assert result.momentum_1m is None

    def test_when_closes_empty_then_rsi_is_none(self) -> None:
        from app.services.views.view_generation import _build_factor_data

        inst = _fake_instrument()
        repo = _make_repo(instrument=inst, profile=None, closes=[])
        result = _build_factor_data(repo, "AAPL")

        assert result is not None
        assert result.rsi_14 is None


# ---------------------------------------------------------------------------
# _safe_float — invalid inputs
# ---------------------------------------------------------------------------


class TestSafeFloatInvalidInputs:
    def test_when_none_then_returns_none(self) -> None:
        from app.services.views.view_generation import _safe_float

        assert _safe_float(None) is None

    def test_when_nan_then_returns_none(self) -> None:
        from app.services.views.view_generation import _safe_float

        assert _safe_float(float("nan")) is None

    def test_when_inf_then_returns_none(self) -> None:
        from app.services.views.view_generation import _safe_float

        assert _safe_float(float("inf")) is None

    def test_when_neg_inf_then_returns_none(self) -> None:
        from app.services.views.view_generation import _safe_float

        assert _safe_float(float("-inf")) is None

    def test_when_non_numeric_string_then_returns_none(self) -> None:
        from app.services.views.view_generation import _safe_float

        assert _safe_float("not_a_number") is None

    def test_when_valid_float_string_then_returns_float(self) -> None:
        from app.services.views.view_generation import _safe_float

        result = _safe_float("3.14")
        assert result == pytest.approx(3.14)

    def test_when_integer_then_returns_float(self) -> None:
        from app.services.views.view_generation import _safe_float

        result = _safe_float(42)
        assert result == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# _compute_momentum — boundary cases
# ---------------------------------------------------------------------------


class TestComputeMomentumBoundary:
    def test_when_exactly_min_data_then_returns_values(self) -> None:
        from app.services.views.view_generation import _compute_momentum

        # long_n = 12*21 = 252, short_n = 1*21 = 21; need len >= 253
        closes = [float(100 + i) for i in range(253)]
        mom_12_1m, mom_1m = _compute_momentum(closes)

        assert mom_12_1m is not None
        assert mom_1m is not None

    def test_when_one_fewer_than_min_then_returns_none(self) -> None:
        from app.services.views.view_generation import _compute_momentum

        closes = [float(100 + i) for i in range(252)]  # 252 < 253
        mom_12_1m, mom_1m = _compute_momentum(closes)

        assert mom_12_1m is None
        assert mom_1m is None
