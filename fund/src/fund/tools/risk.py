"""``risk_check`` and ``backtest``: the risk agent's blocking-gate primitives.

Two deterministic tools the risk agent calls before an allocation is allowed
through:

* ``risk_check`` validates a weight vector against a ConstraintSet placeholder
  (``min_weights`` / ``max_weights`` / ``budget`` — the same keys ``optimize``
  honours, SPEC D17). It is pure arithmetic on the weights: no DB, no RNG. In-norm
  weights return ``passed=True`` with an empty ``violations`` list; every breach
  adds a structured entry and flips ``passed`` to ``False``.
* ``backtest`` holds the given weights fixed and scores them on the walk-forward
  test blocks from ``optimizer.validation`` (``shuffle=False`` — each test block
  strictly follows its training block). The weights are never re-estimated per
  fold, so the metrics are out-of-sample only when the weights were fixed without
  seeing those test blocks; the caller states how far the weights' own estimation
  sample reaches (``fit_end``). Returns are computed **outside** any pipeline via
  ``optimizer.preprocessing.prices_to_returns`` (linear returns); metrics come
  from a skfolio ``Portfolio``. When the panel is too short for even one fold the
  tool falls back to the full IN-SAMPLE series (``n_folds == 0``); those metrics
  are then in-sample, not walk-forward, and are flagged ``out_of_sample=False``.
  ``metrics.max_drawdown`` is the **compounded** wealth-path decline (see
  :func:`backtest`), flagged ``drawdown_basis="compounded"``.

Contract (via ``tool_envelope``): every failure — empty weights, no priced
assets, a bad constraint value, an invalid walk-forward window — is returned as
``{ok: false, error}``, never raised.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from optimizer.preprocessing import prices_to_returns
from optimizer.validation import WalkForwardConfig, build_walk_forward
from skfolio import Portfolio

from fund.tools._base import ToolResult, coerce_date, err, ok, tool_envelope
from fund.tools.prices import load_price_frame

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

# Defaults are long-only, fully invested (D17): 0 <= w <= 1, sum(w) = 1.
_DEFAULT_CONSTRAINTS: dict[str, float] = {
    "min_weights": 0.0,
    "max_weights": 1.0,
    "budget": 1.0,
}
_WEIGHT_TOL = 1e-9
_BUDGET_TOL = 1e-6

# ``WalkForwardConfig`` field names — ``window`` keys outside this set are ignored,
# mirroring ``optimize`` / ``universe_filter``.
_WINDOW_FIELDS: frozenset[str] = frozenset(
    f.name for f in dataclasses.fields(WalkForwardConfig)
)


@tool_envelope
def risk_check(
    weights: dict[str, float],
    *,
    constraints: dict[str, Any] | None = None,
) -> ToolResult:
    """Check ``weights`` against ``constraints``; return ``{passed, violations}``.

    Args:
        weights: ``{ticker: weight}`` to validate.
        constraints: Optional overrides for ``min_weights`` / ``max_weights`` /
            ``budget``; defaults to long-only, fully invested (D17).

    Returns:
        ``ok`` with ``passed`` (``True`` iff no breach) and ``violations`` (a list
        of structured breach records — ``min_weight`` / ``max_weight`` per asset,
        or a single ``budget`` entry). ``err`` on empty ``weights``; a
        non-numeric bound is caught by the envelope and returned as ``{ok: false}``.
    """
    if not weights:
        return err("empty weights")

    bounds = {**_DEFAULT_CONSTRAINTS, **(constraints or {})}
    min_w = bounds["min_weights"]
    max_w = bounds["max_weights"]
    budget = bounds["budget"]

    violations: list[dict[str, Any]] = []
    for asset, weight in weights.items():
        value = float(weight)
        if not math.isfinite(value):
            # A non-finite weight compares False against every bound, so without
            # this guard the gate would silently pass a degenerate allocation.
            violations.append({"type": "non_finite", "asset": asset, "value": value})
            continue
        if value < min_w - _WEIGHT_TOL:
            violations.append(
                {"type": "min_weight", "asset": asset, "value": value, "limit": min_w}
            )
        if value > max_w + _WEIGHT_TOL:
            violations.append(
                {"type": "max_weight", "asset": asset, "value": value, "limit": max_w}
            )

    total = sum(float(w) for w in weights.values())
    if abs(total - budget) > _BUDGET_TOL:
        violations.append({"type": "budget", "value": total, "target": budget})

    return ok({"passed": not violations, "violations": violations})


def _resolve_window(window: dict[str, Any] | None) -> WalkForwardConfig:
    """Build a ``WalkForwardConfig`` from raw ``window`` overrides.

    Unknown keys are ignored; a *known* field carrying a bad value (e.g.
    ``test_size=0``) raises inside ``WalkForwardConfig.__post_init__`` and is
    turned into ``{ok: false}`` by the envelope.
    """
    if not window:
        return WalkForwardConfig()
    overrides = {k: v for k, v in window.items() if k in _WINDOW_FIELDS}
    return WalkForwardConfig(**overrides) if overrides else WalkForwardConfig()


@tool_envelope
def backtest(
    session: Session,
    asof: dt.date | str,
    weights: dict[str, float],
    *,
    window: dict[str, Any] | None = None,
    base_currency: str | None = None,
    fit_end: dt.date | str | None = None,
) -> ToolResult:
    """Backtest fixed ``weights`` over prices up to ``asof``.

    Holds ``weights`` constant and evaluates them on the walk-forward *test*
    windows only (``shuffle=False`` — each test block strictly follows its
    training block). The weights are not re-estimated per fold, so the result is
    out-of-sample only if they were fixed before the test blocks (see ``fit_end``).
    When the panel is too short for a single fold the full sample is used instead
    (``n_folds == 0``).

    Args:
        session: A sync ``portopt_db`` session (D1); the tool does not own it.
        asof: Inclusive upper bound on price history (no look-ahead).
        weights: ``{ticker: weight}`` to hold fixed across the backtest.
        window: Optional overrides mapped onto ``WalkForwardConfig`` fields
            (``train_size`` / ``test_size`` / ``purged_size`` / ...); unknown keys
            are ignored. ``None`` uses the default walk-forward window.
        base_currency: Mandate base currency prices are FX-converted to before
            returns are derived (``None`` keeps listing currency; see
            :func:`fund.tools.prices.load_price_frame`).
        fit_end: Last bar whose returns the ``weights`` were estimated from, when
            they were fitted on this same panel (accepts a ``date`` or an ISO
            string). A walk-forward test block that reaches back to ``fit_end`` lies
            inside the weights' own estimation sample, so its metrics are in-sample.
            ``None`` asserts the weights were fixed independently of this panel.

    Returns:
        ``ok`` with ``metrics`` (``mean`` / ``standard_deviation`` /
        ``sharpe_ratio`` / ``max_drawdown``, over the evaluated window),
        ``full_sample_max_drawdown`` (the same drawdown over every aligned return row,
        including the initial training block the walk-forward blocks skip; the figure
        to compare with a drawdown ceiling), ``drawdown_basis``, ``n_observations``
        (evaluated rows), ``n_folds``, ``out_of_sample`` (``True`` iff at least one
        walk-forward fold was produced and every test bar falls after ``fit_end``;
        ``False`` when the panel was too short and the full IN-SAMPLE series was
        evaluated, or when the weights were fitted on the test blocks — the metrics
        are then in-sample), ``sample_start`` / ``window_start`` / ``window_end`` (ISO
        dates), and ``missing`` (zero-weight tickers absent from the panel). ``err`` on
        empty ``weights``, no priced assets, a non-zero weight on a ticker with no
        prices (the proposal cannot be evaluated as given), fewer than two return rows
        once the assets' histories are aligned, a bad ``window``, or metrics that are
        not finite.

        ``max_drawdown`` is the peak-to-trough decline of the **compounded** wealth
        path (``drawdown_basis == "compounded"``), the loss an investor holding the
        weights would have seen. The ``nu1`` / ``nu2`` / ``nu3`` drawdown ceilings
        the optimizer enforces are measured on the **additive** cumulative-return
        path (skfolio's convention), so the two can differ: treat this figure as the
        realised decline, not a like-for-like reading of those ceilings, and do not
        block or pass on a margin smaller than that basis gap.
    """
    if not weights:
        return err("empty weights")

    config = _resolve_window(window)
    frame, missing = load_price_frame(
        session, asof, list(weights), base_currency=base_currency
    )
    if frame.shape[1] == 0:
        return err("no priced assets for weights")
    # A dropped weight would be evaluated as idle cash, understating volatility and
    # drawdown by the dropped share; only a zero weight can be dropped losslessly.
    unpriced = [ticker for ticker in missing if float(weights[ticker]) != 0.0]
    if unpriced:
        return err(f"no prices for weighted tickers: {', '.join(unpriced)}")
    if frame.shape[0] < 2:
        return err("insufficient price history to backtest")

    # Single-arg ``prices_to_returns`` always yields a DataFrame; the ``| tuple``
    # branch only applies when ``y`` is passed. Narrow it so the ``.index`` chains
    # below don't resolve against ``tuple.index``.
    returns = cast("pd.DataFrame", prices_to_returns(frame))
    # ``frame`` counts the union of price dates, but ``prices_to_returns`` trims every
    # row before the youngest asset's first bar, so the returns can be empty or a
    # single row even when ``frame`` is long.
    if returns.shape[0] < 2:
        return err("insufficient price history to backtest")
    weight_vector = np.array([float(weights[col]) for col in returns.columns])

    # Fall back to the full IN-SAMPLE series when the panel cannot supply even one
    # fold; those metrics are in-sample, surfaced via ``out_of_sample``.
    min_needed = config.train_size + config.purged_size + config.test_size
    splits = (
        list(build_walk_forward(config).split(returns))
        if returns.shape[0] >= min_needed
        else []
    )
    if splits:
        oos = pd.concat([returns.iloc[test_idx] for _, test_idx in splits]).sort_index()
    else:
        oos = returns

    # Fixed weights never see a test block only if they were estimated before it:
    # a block that reaches back into the weights' own fit window is in-sample.
    out_of_sample = bool(splits)
    if splits and fit_end is not None:
        out_of_sample = oos.index.min().date() > coerce_date(fit_end)

    # Compounded so max_drawdown is the wealth-path peak-to-trough decline, not the
    # additive-return one the optimizer's ceilings use; ``drawdown_basis`` says so.
    portfolio = Portfolio(oos, weight_vector, compounded=True)
    full_sample_max_drawdown = (
        float(Portfolio(returns, weight_vector, compounded=True).max_drawdown)
        if splits
        else float(portfolio.max_drawdown)
    )
    metrics = {
        "mean": float(portfolio.mean),
        "standard_deviation": float(portfolio.standard_deviation),
        "sharpe_ratio": float(portfolio.sharpe_ratio),
        "max_drawdown": float(portfolio.max_drawdown),
    }
    # A degenerate series (zero variance) yields NaN/inf, which are not valid JSON
    # for the agent and compare False against every ceiling, so nothing is flagged.
    if not all(math.isfinite(value) for value in metrics.values()):
        return err("backtest metrics are not finite (degenerate return series)")

    return ok(
        {
            "metrics": metrics,
            "full_sample_max_drawdown": full_sample_max_drawdown,
            "drawdown_basis": "compounded",
            "n_observations": int(oos.shape[0]),
            "n_folds": len(splits),
            "out_of_sample": out_of_sample,
            "sample_start": str(returns.index.min().date()),
            "window_start": str(oos.index.min().date()),
            "window_end": str(oos.index.max().date()),
            "missing": missing,
        }
    )


__all__ = ["backtest", "risk_check"]
