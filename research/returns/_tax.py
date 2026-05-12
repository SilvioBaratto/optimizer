"""After-tax return reconstruction (Cycle 4 §9.2).

Pure helper module — no DB, no FS, no logging.  Used by
``research/stock_selection_pipeline.py`` to derive a third return series
``after_tax_returns`` from ``result.gross_returns`` (#541) and
``result.weight_history`` (#285).
"""

from __future__ import annotations

import pandas as pd

DEFAULT_TAX_RATE: float = 0.26  # Italy capital-gains rate


def compute_after_tax_returns(
    gross_returns: pd.Series | None,
    weight_history: pd.DataFrame | None,
    prices: pd.DataFrame,
    *,
    tax_rate: float = DEFAULT_TAX_RATE,
    cost_bps: float = 10.0,
) -> pd.Series | None:
    """Reconstruct after-tax portfolio returns from realised PnL at rebalances.

    Spec §9.2: ``after_tax_return = gross - txn_cost
    - tax_rate × max(realised_pnl, 0) / portfolio_value``.

    At each rebalance date ``t``:
      * ``sell_qty[i] = max(prev_w[i] - new_w[i], 0)`` — fraction reduced.
      * ``realised_pnl = Σᵢ sell_qty[i] × (price[t] - price[t-1])``.
      * ``txn_cost = turnover/2 × cost_bps/1e4`` — same as
        ``compute_net_backtest_returns``.
      * ``portfolio_value`` is ``(1 + gross).cumprod()``.

    Cold-start (first rebalance, no prior weights) → realised_pnl = 0.
    Negative realised PnL is NOT taxed (only positive gains).
    When ``portfolio_value`` is non-positive, tax is skipped to avoid
    division by zero.
    """
    if gross_returns is None or weight_history is None:
        return None

    after_tax = gross_returns.copy()
    cost_frac = cost_bps / 10_000.0
    cum_value = (1.0 + gross_returns).cumprod()
    rb_prices = prices.reindex(weight_history.index, method="ffill")

    prev_w: pd.Series | None = None
    prev_p: pd.Series | None = None
    for date in weight_history.index:
        if date not in after_tax.index:
            continue
        new_w = weight_history.loc[date]
        new_p = rb_prices.loc[date]
        after_tax.at[date] -= _txn_cost(prev_w, new_w, cost_frac)
        after_tax.at[date] -= _tax_drag(
            prev_w, new_w, prev_p, new_p, cum_value.at[date], tax_rate
        )
        prev_w = new_w
        prev_p = new_p

    return after_tax


def _txn_cost(prev_w: pd.Series | None, new_w: pd.Series, cost_frac: float) -> float:
    """One-way turnover × cost fraction; cold-start treats prior as cash."""
    if prev_w is None:
        delta = new_w.abs()
    else:
        delta = new_w.subtract(prev_w, fill_value=0.0).abs()
    turnover = float(delta.sum()) / 2.0
    return turnover * cost_frac


def _tax_drag(
    prev_w: pd.Series | None,
    new_w: pd.Series,
    prev_p: pd.Series | None,
    new_p: pd.Series,
    portfolio_value: float,
    tax_rate: float,
) -> float:
    """Tax fraction on positive realised PnL; 0 on cold-start or zero PV."""
    if prev_w is None or prev_p is None or portfolio_value <= 0:
        return 0.0
    sell = prev_w.subtract(new_w, fill_value=0.0).clip(lower=0.0)
    price_delta = new_p.subtract(prev_p, fill_value=0.0)
    common = sell.index.intersection(price_delta.index)
    realised = float((sell.loc[common] * price_delta.loc[common]).sum())
    if realised <= 0:
        return 0.0
    return tax_rate * realised / portfolio_value
