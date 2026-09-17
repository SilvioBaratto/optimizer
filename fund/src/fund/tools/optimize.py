"""T3.2 — ``optimize_portfolio``: the ★ walking-skeleton milestone.

The load-bearing tool of the whole architecture: **skfolio computes the weights,
never the LLM.** Given a price panel as of ``asof`` it builds a ``MeanRisk``
optimiser (default: minimum-variance, long-only, fully invested — ``min_weights=0``,
``max_weights=1``, ``budget=1``, SPEC D17), fits it on linear returns, and returns
the resulting weights plus a small metrics summary.

The tool is a pure, deterministic function of ``(session, asof, universe,
constraints)``: the min-variance QP is convex, so identical seeded data ⇒
identical weights. Returns are computed **outside** any pipeline via
:func:`optimizer.preprocessing.prices_to_returns` (linear returns, D-gotcha).

Contract (via :func:`fund.tools._base.tool_envelope`):

* empty ``universe`` ⇒ ``{ok: false, error}``;
* no priced assets, or fewer than two observations, ⇒ ``{ok: false, error}``;
* a bad ``constraints`` value is caught by the envelope and returned as
  ``{ok: false}``, never raised.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from optimizer.optimization import MeanRiskConfig, build_mean_risk
from optimizer.preprocessing import prices_to_returns

from fund.tools._base import ToolResult, err, ok, tool_envelope
from fund.tools.prices import load_price_frame

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

# ``constraints`` keys honoured today (placeholder for the Fase-4 ConstraintSet
# schema). Each maps 1:1 onto a ``MeanRiskConfig`` field; anything else is ignored.
_CONSTRAINT_FIELDS: frozenset[str] = frozenset({"min_weights", "max_weights", "budget"})


def _resolve_config(constraints: dict[str, Any] | None) -> MeanRiskConfig:
    """Build the ``MeanRiskConfig`` — min-variance long-only, D17 — with overrides."""
    base = MeanRiskConfig()
    if not constraints:
        return base
    overrides = {k: constraints[k] for k in _CONSTRAINT_FIELDS if k in constraints}
    return replace(base, **overrides) if overrides else base


@tool_envelope
def optimize_portfolio(
    session: Session,
    asof: dt.date | str,
    universe: Sequence[str],
    *,
    constraints: dict[str, Any] | None = None,
) -> ToolResult:
    """Optimise weights for ``universe`` from prices as of ``asof``.

    Args:
        session: A sync ``portopt_db`` session (D1); the tool does not own it.
        asof: Inclusive upper bound on price history (no look-ahead).
        universe: yfinance tickers to allocate across.
        constraints: Optional overrides for ``min_weights`` / ``max_weights`` /
            ``budget``; defaults to long-only, fully invested (D17).

    Returns:
        ``ok`` with ``weights`` (``{ticker: float}`` from skfolio), ``metrics``
        (in-sample ``mean`` / ``standard_deviation`` / ``sharpe_ratio``),
        ``assets``, ``n_observations`` and ``missing``. ``err`` on an empty
        universe, no priced assets, or too little history.
    """
    if not universe:
        return err("empty universe")

    frame, missing = load_price_frame(session, asof, universe)
    if frame.shape[1] == 0:
        return err("no priced assets in universe")
    if frame.shape[0] < 2:
        return err("insufficient price history to optimize")

    returns = prices_to_returns(frame)
    model = build_mean_risk(_resolve_config(constraints))
    model.fit(returns)
    portfolio = model.predict(returns)

    assets = [str(col) for col in returns.columns]
    weights = {a: float(w) for a, w in zip(assets, model.weights_, strict=True)}
    metrics = {
        "mean": float(portfolio.mean),
        "standard_deviation": float(portfolio.standard_deviation),
        "sharpe_ratio": float(portfolio.sharpe_ratio),
    }

    return ok(
        {
            "weights": weights,
            "metrics": metrics,
            "assets": assets,
            "n_observations": int(returns.shape[0]),
            "missing": missing,
        }
    )


__all__ = ["optimize_portfolio"]
