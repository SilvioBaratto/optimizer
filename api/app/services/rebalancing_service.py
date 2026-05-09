"""Rebalancing service: wraps optimizer library rebalancing functions.

Stateless functions following Single Responsibility Principle:
  - build_config         — dispatch policy_type string to config dataclass
  - decide               — compute rebalancing decision given weights + policy
  - compute_trade_weights — per-asset weight deltas (target - current)
  - preview_trades       — build trade list from snapshot + positions
  - _calendar_should_rebalance — calendar-only rebalance check via business-day math
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from optimizer.rebalancing import (
    CalendarRebalancingConfig,
    HybridRebalancingConfig,
    RebalancingFrequency,
    ThresholdRebalancingConfig,
    ThresholdType,
    compute_rebalancing_cost,
    compute_turnover,
    should_rebalance,
    should_rebalance_hybrid,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Policy type registry (Open/Closed — extend by adding entries here)
# ---------------------------------------------------------------------------

_VALID_POLICY_TYPES = frozenset({"calendar", "threshold", "hybrid"})


# ---------------------------------------------------------------------------
# Config builders (Single Responsibility)
# ---------------------------------------------------------------------------


def _build_threshold_config(cfg: dict[str, Any]) -> ThresholdRebalancingConfig:
    threshold_type = ThresholdType(cfg.get("threshold_type", "absolute"))
    threshold = float(cfg.get("threshold", 0.05))
    return ThresholdRebalancingConfig(
        threshold_type=threshold_type,
        threshold=threshold,
    )


def _build_calendar_config(cfg: dict[str, Any]) -> CalendarRebalancingConfig:
    frequency = RebalancingFrequency(cfg.get("frequency", "quarterly"))
    return CalendarRebalancingConfig(frequency=frequency)


def _build_hybrid_config(cfg: dict[str, Any]) -> HybridRebalancingConfig:
    calendar = _build_calendar_config(cfg.get("calendar", {}))
    threshold = _build_threshold_config(cfg.get("threshold", {}))
    return HybridRebalancingConfig(calendar=calendar, threshold=threshold)


def build_config(
    policy_type: str, policy_config: dict[str, Any]
) -> CalendarRebalancingConfig | ThresholdRebalancingConfig | HybridRebalancingConfig:
    """Build the appropriate rebalancing config dataclass from a policy_type string.

    Args:
        policy_type: One of "calendar", "threshold", "hybrid".
        policy_config: Raw dict of policy-specific parameters.

    Returns:
        Frozen config dataclass matching policy_type.

    Raises:
        ValueError: For unknown policy_type.
    """
    if policy_type == "threshold":
        return _build_threshold_config(policy_config)
    if policy_type == "calendar":
        return _build_calendar_config(policy_config)
    if policy_type == "hybrid":
        return _build_hybrid_config(policy_config)
    raise ValueError(
        f"Unknown policy_type: {policy_type!r}. "
        f"Valid types: {sorted(_VALID_POLICY_TYPES)}"
    )


# ---------------------------------------------------------------------------
# Calendar rebalance check (no library function — uses date arithmetic)
# ---------------------------------------------------------------------------


def _calendar_should_rebalance(
    config: CalendarRebalancingConfig,
    current_date: date,
    last_review_date: date | None,
) -> bool:
    """Return True when enough business days have elapsed since last_review_date.

    Uses pandas BDay offset for accurate business-day counting.
    When last_review_date is None, always returns True (first run).
    """
    if last_review_date is None:
        return True
    next_review = pd.Timestamp(last_review_date) + pd.offsets.BDay(
        config.trading_days
    )
    return pd.Timestamp(current_date) >= next_review


# ---------------------------------------------------------------------------
# Weight helpers
# ---------------------------------------------------------------------------


def compute_trade_weights(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
) -> dict[str, float]:
    """Compute per-asset weight deltas: target - current.

    Positive values = buy, negative = sell.
    Assets in target but absent from current are treated as having weight 0.
    """
    all_tickers = set(current_weights) | set(target_weights)
    return {
        ticker: target_weights.get(ticker, 0.0) - current_weights.get(ticker, 0.0)
        for ticker in all_tickers
    }


def _weights_to_arrays(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Align weight dicts to a common ticker order and return numpy arrays."""
    tickers = sorted(set(current_weights) | set(target_weights))
    current_arr = np.array([current_weights.get(t, 0.0) for t in tickers])
    target_arr = np.array([target_weights.get(t, 0.0) for t in tickers])
    return current_arr, target_arr


# ---------------------------------------------------------------------------
# Rebalancing decision (stateless, no DB)
# ---------------------------------------------------------------------------


def decide(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    policy_type: str,
    policy_config: dict[str, Any],
    *,
    current_date: date | None = None,
    last_review_date: date | None = None,
    transaction_costs: float = 0.001,
) -> dict[str, Any]:
    """Compute the rebalancing decision for a given policy.

    Args:
        current_weights: Current portfolio weights per ticker.
        target_weights: Target portfolio weights per ticker.
        policy_type: "calendar", "threshold", or "hybrid".
        policy_config: Dict of policy-specific parameters.
        current_date: Date for calendar/hybrid checks. Defaults to today.
        last_review_date: Required for hybrid. Optional for calendar.
        transaction_costs: Per-unit transaction cost (default 10 bps).

    Returns:
        Dict with keys: should_rebalance, turnover, estimated_cost, trade_weights.
    """
    effective_date = current_date or date.today()
    config = build_config(policy_type, policy_config)
    current_arr, target_arr = _weights_to_arrays(current_weights, target_weights)

    rebalance = _evaluate_policy(
        config=config,
        current_arr=current_arr,
        target_arr=target_arr,
        current_date=effective_date,
        last_review_date=last_review_date,
    )
    turnover = compute_turnover(current_arr, target_arr)
    cost = compute_rebalancing_cost(current_arr, target_arr, transaction_costs)
    trades = compute_trade_weights(current_weights, target_weights)

    return {
        "should_rebalance": rebalance,
        "turnover": round(turnover, 6),
        "estimated_cost": round(cost, 6),
        "trade_weights": trades,
    }


def _evaluate_policy(
    config: CalendarRebalancingConfig | ThresholdRebalancingConfig | HybridRebalancingConfig,
    current_arr: np.ndarray,
    target_arr: np.ndarray,
    current_date: date,
    last_review_date: date | None,
) -> bool:
    """Dispatch to the correct rebalancing check based on config type."""
    if isinstance(config, HybridRebalancingConfig):
        if last_review_date is None:
            raise ValueError("last_review_date is required for hybrid policy")
        # pd.Timestamp(str) never returns NaT; stubs over-approximate the return type
        return should_rebalance_hybrid(
            current_arr,
            target_arr,
            config,
            pd.Timestamp(current_date.isoformat()),  # type: ignore[arg-type]
            pd.Timestamp(last_review_date.isoformat()),  # type: ignore[arg-type]
        )
    if isinstance(config, CalendarRebalancingConfig):
        return _calendar_should_rebalance(config, current_date, last_review_date)
    return should_rebalance(current_arr, target_arr, config)


# ---------------------------------------------------------------------------
# Preview trade list builder
# ---------------------------------------------------------------------------


def build_trade_list(
    target_weights: dict[str, float],
    broker_weights: dict[str, float],
    prices: dict[str, float] | None = None,
    portfolio_value: float | None = None,
) -> list[dict[str, Any]]:
    """Build a list of trade items from target and current broker weights.

    Each item contains: ticker, weight_delta, side (buy/sell), shares.

    Shares are computed as ``(weight_delta * portfolio_value) / price`` when
    ``prices`` and ``portfolio_value`` are provided and the price is positive.
    Otherwise ``shares`` is ``None``.
    """
    all_tickers = set(target_weights) | set(broker_weights)
    trades: list[dict[str, Any]] = []
    for ticker in sorted(all_tickers):
        target = target_weights.get(ticker, 0.0)
        current = broker_weights.get(ticker, 0.0)
        delta = round(target - current, 6)
        if abs(delta) < 1e-9:
            continue
        shares: float | None = None
        if prices and portfolio_value:
            price = prices.get(ticker)
            if price and price > 0:
                shares = round((delta * portfolio_value) / price, 4)
        trades.append({
            "ticker": ticker,
            "weight_delta": delta,
            "side": "buy" if delta > 0 else "sell",
            "shares": shares,
        })
    return trades


def compute_broker_weights(positions: list[Any]) -> dict[str, float]:
    """Compute current weight per ticker from broker positions.

    Always normalises by the sum of position values so weights are in [0, 1]
    and sum to 1. Account ``portfolio_value`` (cash + positions in account
    currency) is deliberately not used here because it can diverge from the
    sum of position market values due to FX, stale cash balance, or pending
    settlements. Pass ``portfolio_value`` to ``build_trade_list`` instead for
    weight→share quantity conversion.
    """
    if not positions:
        return {}

    totals: dict[str, float] = {}
    for pos in positions:
        price = pos.current_price or pos.average_price or 0.0
        value = pos.quantity * price
        ticker = pos.yfinance_ticker or pos.ticker
        totals[ticker] = totals.get(ticker, 0.0) + value

    total_value = sum(totals.values())
    if total_value == 0.0:
        return dict.fromkeys(totals, 0.0)
    return {t: v / total_value for t, v in totals.items()}
