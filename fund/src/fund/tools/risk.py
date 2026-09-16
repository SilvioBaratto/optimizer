"""T3.4 — ``risk_check`` + ``backtest``: the risk agent's blocking-gate primitives.

Two deterministic tools the risk agent calls before an allocation is allowed
through:

* :func:`risk_check` validates a weight vector against a ConstraintSet placeholder
  (``min_weights`` / ``max_weights`` / ``budget`` — the same keys ``optimize``
  honours, SPEC D17). It is pure arithmetic on the weights: no DB, no RNG. In-norm
  weights return ``passed=True`` with an empty ``violations`` list; every breach
  adds a structured entry and flips ``passed`` to ``False``.
* :func:`backtest` holds the given weights fixed and evaluates them *out of
  sample* with a walk-forward split from :mod:`optimizer.validation`
  (``shuffle=False`` — the test window strictly follows the training block, so no
  future data leaks). Returns are computed **outside** any pipeline via
  :func:`optimizer.preprocessing.prices_to_returns` (linear returns); metrics come
  from a skfolio :class:`~skfolio.Portfolio`. When the panel is too short for even
  one fold the tool falls back to the full sample (``n_folds == 0``).

Contract (via :func:`fund.tools._base.tool_envelope`): every failure — empty
weights, no priced assets, a bad constraint value, an invalid walk-forward
window — is returned as ``{ok: false, error}``, never raised.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from optimizer.preprocessing import prices_to_returns
from optimizer.validation import WalkForwardConfig, build_walk_forward
from skfolio import Portfolio

from fund.tools._base import ToolResult, err, ok, tool_envelope
from fund.tools.prices import load_price_frame

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

# ConstraintSet placeholder honoured by ``risk_check`` (Fase-4 schema fills this in
# later). Defaults are long-only, fully invested (D17): 0 <= w <= 1, sum(w) = 1.
_DEFAULT_CONSTRAINTS: dict[str, float] = {
    "min_weights": 0.0,
    "max_weights": 1.0,
    "budget": 1.0,
}
_WEIGHT_TOL = 1e-9
_BUDGET_TOL = 1e-6

# ``WalkForwardConfig`` field names — ``window`` keys outside this set are ignored
# (placeholder for the Fase-4 schema), mirroring ``optimize`` / ``universe_filter``.
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
) -> ToolResult:
    """Backtest fixed ``weights`` out-of-sample over prices up to ``asof``.

    Holds ``weights`` constant and evaluates them on the walk-forward *test*
    windows only (``shuffle=False`` — each test block strictly follows its
    training block, so no future data leaks). When the panel is too short for a
    single fold the full sample is used instead (``n_folds == 0``).

    Args:
        session: A sync ``portopt_db`` session (D1); the tool does not own it.
        asof: Inclusive upper bound on price history (no look-ahead).
        weights: ``{ticker: weight}`` to hold fixed across the backtest.
        window: Optional overrides mapped onto ``WalkForwardConfig`` fields
            (``train_size`` / ``test_size`` / ``purged_size`` / ...); unknown keys
            are ignored. ``None`` uses the default walk-forward window.

    Returns:
        ``ok`` with ``metrics`` (``mean`` / ``standard_deviation`` /
        ``sharpe_ratio`` / ``max_drawdown``), ``n_observations`` (evaluated rows),
        ``n_folds``, ``sample_start`` / ``window_start`` / ``window_end`` (ISO
        dates), and ``missing`` (weighted tickers absent from the panel). ``err``
        on empty ``weights``, no priced assets, too little history, or a bad
        ``window``.
    """
    if not weights:
        return err("empty weights")

    config = _resolve_window(window)
    frame, missing = load_price_frame(session, asof, list(weights))
    if frame.shape[1] == 0:
        return err("no priced assets for weights")
    if frame.shape[0] < 2:
        return err("insufficient price history to backtest")

    returns = prices_to_returns(frame)
    weight_vector = np.array([float(weights[col]) for col in returns.columns])

    # A test block strictly follows its training block, so evaluating only on the
    # concatenated test windows keeps the backtest out-of-sample. Fall back to the
    # full sample when the panel cannot supply even one fold.
    min_needed = config.train_size + config.purged_size + config.test_size
    splits = (
        list(build_walk_forward(config).split(returns))
        if returns.shape[0] >= min_needed
        else []
    )
    if splits:
        oos = pd.concat(
            [returns.iloc[test_idx] for _, test_idx in splits]
        ).sort_index()
    else:
        oos = returns

    portfolio = Portfolio(oos, weight_vector)
    metrics = {
        "mean": float(portfolio.mean),
        "standard_deviation": float(portfolio.standard_deviation),
        "sharpe_ratio": float(portfolio.sharpe_ratio),
        "max_drawdown": float(portfolio.max_drawdown),
    }

    return ok(
        {
            "metrics": metrics,
            "n_observations": int(oos.shape[0]),
            "n_folds": len(splits),
            "sample_start": str(returns.index.min().date()),
            "window_start": str(oos.index.min().date()),
            "window_end": str(oos.index.max().date()),
            "missing": missing,
        }
    )


__all__ = ["backtest", "risk_check"]
