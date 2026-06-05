"""Empty-data and exception-path tests for rebalancing_service (issue #825).

No existing unit tests cover rebalancing_service module directly (only
route-level tests exist in test_rebalancing_routes.py).

Explicit raises in rebalancing_service:
  1. build_config: unknown policy_type → raises ValueError
  2. _evaluate_policy: hybrid with last_review_date=None → raises ValueError

Gap analysis:
  - build_config: all three valid policy types construct without raising
  - build_config: unknown type → ValueError (raise #1)
  - _evaluate_policy: hybrid + last_review_date=None → ValueError (raise #2)
  - compute_trade_weights: empty current + target → empty dict
  - compute_trade_weights: asset in target but not current → treated as 0
  - build_trade_list: empty target and broker → empty list
  - build_trade_list: matching weights → no-op (delta < 1e-9 filtered)
  - compute_broker_weights: empty positions → empty dict
  - compute_broker_weights: positions with zero price → zero-value weights
  - decide: calendar policy with default date → returns JSON-safe result
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from tests._fixtures.assertions import assert_json_safe

# ---------------------------------------------------------------------------
# build_config — explicit raise for unknown policy_type
# ---------------------------------------------------------------------------


class TestBuildConfigUnknownType:
    """build_config raises ValueError for any unregistered policy_type string."""

    def test_when_unknown_policy_type_then_raises_value_error(self) -> None:
        from app.services.rebalancing.rebalancing_service import build_config

        with pytest.raises(ValueError, match="Unknown policy_type"):
            build_config("monthly_magic", {})

    def test_when_empty_string_policy_type_then_raises_value_error(self) -> None:
        from app.services.rebalancing.rebalancing_service import build_config

        with pytest.raises(ValueError, match="Unknown policy_type"):
            build_config("", {})


class TestBuildConfigValidTypes:
    """build_config succeeds for the three registered policy types."""

    @pytest.mark.parametrize("policy_type", ["calendar", "threshold", "hybrid"])
    def test_when_valid_policy_type_then_returns_config_without_raising(
        self, policy_type: str
    ) -> None:
        from app.services.rebalancing.rebalancing_service import build_config

        cfg = build_config(policy_type, {})

        assert cfg is not None


# ---------------------------------------------------------------------------
# _evaluate_policy — hybrid + missing last_review_date → explicit raise
# ---------------------------------------------------------------------------


class TestEvaluatePolicyHybridMissingDate:
    """_evaluate_policy raises ValueError when hybrid policy has no last_review_date."""

    def test_when_hybrid_and_last_review_date_none_then_raises_value_error(
        self,
    ) -> None:
        import numpy as np

        from app.services.rebalancing.rebalancing_service import (
            _evaluate_policy,
            build_config,
        )

        config = build_config("hybrid", {})
        current_arr = np.array([0.5, 0.5])
        target_arr = np.array([0.6, 0.4])

        with pytest.raises(ValueError, match="last_review_date"):
            _evaluate_policy(
                config=config,
                current_arr=current_arr,
                target_arr=target_arr,
                current_date=date(2024, 1, 15),
                last_review_date=None,
            )


# ---------------------------------------------------------------------------
# compute_trade_weights — empty-data and asymmetric asset sets
# ---------------------------------------------------------------------------


class TestComputeTradeWeightsEmptyData:
    """compute_trade_weights handles empty dicts and asymmetric asset sets."""

    def test_when_both_empty_then_returns_empty_dict(self) -> None:
        from app.services.rebalancing.rebalancing_service import compute_trade_weights

        result = compute_trade_weights({}, {})

        assert result == {}

    def test_when_asset_only_in_target_then_delta_equals_target_weight(self) -> None:
        from app.services.rebalancing.rebalancing_service import compute_trade_weights

        result = compute_trade_weights(current_weights={}, target_weights={"AAPL": 0.6})

        assert result["AAPL"] == pytest.approx(0.6)

    def test_when_asset_only_in_current_then_delta_is_negative(self) -> None:
        from app.services.rebalancing.rebalancing_service import compute_trade_weights

        result = compute_trade_weights(
            current_weights={"AAPL": 0.4}, target_weights={}
        )

        assert result["AAPL"] == pytest.approx(-0.4)

    def test_when_equal_weights_then_delta_is_zero(self) -> None:
        from app.services.rebalancing.rebalancing_service import compute_trade_weights

        result = compute_trade_weights(
            current_weights={"AAPL": 0.5},
            target_weights={"AAPL": 0.5},
        )

        assert result["AAPL"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# build_trade_list — empty / no-op cases
# ---------------------------------------------------------------------------


class TestBuildTradeListEmptyData:
    """build_trade_list returns [] when inputs are empty or identical."""

    def test_when_both_empty_then_returns_empty_list(self) -> None:
        from app.services.rebalancing.rebalancing_service import build_trade_list

        result = build_trade_list({}, {})

        assert result == []

    def test_when_weights_are_identical_then_no_trades(self) -> None:
        """Deltas below 1e-9 are filtered out."""
        from app.services.rebalancing.rebalancing_service import build_trade_list

        weights = {"AAPL": 0.5, "MSFT": 0.5}
        result = build_trade_list(target_weights=weights, broker_weights=weights)

        assert result == []

    def test_when_new_asset_added_then_trade_is_buy(self) -> None:
        from app.services.rebalancing.rebalancing_service import build_trade_list

        result = build_trade_list(
            target_weights={"AAPL": 0.5},
            broker_weights={},
        )

        assert len(result) == 1
        assert result[0]["side"] == "buy"
        assert result[0]["ticker"] == "AAPL"

    def test_when_asset_removed_then_trade_is_sell(self) -> None:
        from app.services.rebalancing.rebalancing_service import build_trade_list

        result = build_trade_list(
            target_weights={},
            broker_weights={"AAPL": 0.5},
        )

        assert len(result) == 1
        assert result[0]["side"] == "sell"

    def test_when_prices_and_portfolio_value_provided_then_shares_computed(
        self,
    ) -> None:
        from app.services.rebalancing.rebalancing_service import build_trade_list

        result = build_trade_list(
            target_weights={"AAPL": 0.5},
            broker_weights={},
            prices={"AAPL": 200.0},
            portfolio_value=10_000.0,
        )

        # shares = (0.5 * 10000) / 200 = 25.0
        assert result[0]["shares"] == pytest.approx(25.0, abs=0.01)

    def test_when_price_missing_then_shares_is_none(self) -> None:
        from app.services.rebalancing.rebalancing_service import build_trade_list

        result = build_trade_list(
            target_weights={"AAPL": 0.5},
            broker_weights={},
            prices={},
            portfolio_value=10_000.0,
        )

        assert result[0]["shares"] is None

    def test_result_is_json_safe(self) -> None:
        from app.services.rebalancing.rebalancing_service import build_trade_list

        result = build_trade_list(
            target_weights={"AAPL": 0.6, "MSFT": 0.4},
            broker_weights={"AAPL": 0.5, "MSFT": 0.5},
            prices={"AAPL": 180.0, "MSFT": 300.0},
            portfolio_value=50_000.0,
        )

        assert_json_safe(result)


# ---------------------------------------------------------------------------
# compute_broker_weights — empty positions / zero prices
# ---------------------------------------------------------------------------


class TestComputeBrokerWeightsEmptyData:
    """compute_broker_weights with empty or all-zero-price positions."""

    def test_when_positions_empty_then_returns_empty_dict(self) -> None:
        from app.services.rebalancing.rebalancing_service import compute_broker_weights

        result = compute_broker_weights([])

        assert result == {}

    def test_when_all_prices_zero_then_weights_are_zero(self) -> None:
        """Zero-price positions: total_value=0 → all weights=0 (not division error)."""
        from app.services.rebalancing.rebalancing_service import compute_broker_weights

        pos = MagicMock()
        pos.quantity = 100
        pos.current_price = 0.0
        pos.average_price = 0.0
        pos.yfinance_ticker = "AAPL"
        pos.ticker = "AAPL"

        result = compute_broker_weights([pos])

        assert result == {"AAPL": 0.0}

    def test_when_positions_have_prices_then_weights_sum_to_one(self) -> None:
        from app.services.rebalancing.rebalancing_service import compute_broker_weights

        def _pos(ticker: str, qty: float, price: float) -> MagicMock:
            p = MagicMock()
            p.quantity = qty
            p.current_price = price
            p.average_price = price
            p.yfinance_ticker = ticker
            p.ticker = ticker
            return p

        positions = [_pos("AAPL", 10, 200.0), _pos("MSFT", 5, 300.0)]
        result = compute_broker_weights(positions)

        total = sum(result.values())
        assert total == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# decide — calendar policy JSON-safe result
# ---------------------------------------------------------------------------


class TestDecideCalendarJsonSafe:
    """decide with calendar policy returns a JSON-safe result dict."""

    def test_when_calendar_policy_then_result_is_json_safe(self) -> None:
        from app.services.rebalancing.rebalancing_service import decide

        result = decide(
            current_weights={"AAPL": 0.5, "MSFT": 0.5},
            target_weights={"AAPL": 0.6, "MSFT": 0.4},
            policy_type="calendar",
            policy_config={"frequency": "quarterly"},
            current_date=date(2024, 6, 1),
            last_review_date=None,
        )

        assert_json_safe(result)
        assert "should_rebalance" in result
        assert "turnover" in result
        assert "estimated_cost" in result
        assert "trade_weights" in result

    def test_when_empty_weights_then_decide_returns_zero_turnover(self) -> None:
        from app.services.rebalancing.rebalancing_service import decide

        result = decide(
            current_weights={},
            target_weights={},
            policy_type="threshold",
            policy_config={"threshold_type": "absolute", "threshold": 0.05},
        )

        assert result["turnover"] == pytest.approx(0.0)
        assert result["estimated_cost"] == pytest.approx(0.0)
        assert result["trade_weights"] == {}
