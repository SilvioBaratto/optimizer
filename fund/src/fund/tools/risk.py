"""``risk_check`` and ``backtest``: the risk agent's blocking-gate primitives.

Two deterministic tools the risk agent calls before an allocation is allowed
through:

* ``risk_check`` validates a weight vector against a ConstraintSet placeholder
  (``min_weights`` / ``max_weights`` / ``budget`` — the same keys ``optimize``
  honours, SPEC D17) plus the MiFID guardrails a profile carries: the ESG
  ``excluded_assets`` hard block and the ``nu1``/``nu2``/``nu3`` drawdown tiers
  compared with an ``observed_drawdown`` the caller measured. It is pure
  arithmetic on its inputs: no DB, no RNG. In-norm weights return ``passed=True``
  with an empty ``violations`` list; every breach adds a structured entry and
  flips ``passed`` to ``False`` (the ``nu1`` soft tier only adds a ``warnings``
  entry).
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
  :func:`backtest`), flagged ``drawdown_basis="compounded"``. The ``nu`` ceilings are
  judged on ``worst_rolling_loss``, the worst such decline inside any one year of the
  panel's own calendar, because the loss tolerance they derive from is a one-year
  figure; ``full_sample_max_drawdown`` is reported alongside for information.

Contract (via ``tool_envelope``): every failure — empty weights, no priced
assets, too few common return observations (see ``MIN_RETURN_OBSERVATIONS``, and
``thin_sample_error`` for the share of the median history a young listing must
reach), a bad constraint value, an invalid walk-forward window — is returned as
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

from fund.tools._base import (
    ToolResult,
    annualized_metrics,
    coerce_date,
    dropped_by_returns,
    err,
    ok,
    periods_per_year,
    thin_sample_error,
    tool_envelope,
)
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


# Windows scored per vectorised block: bounds the (windows x bars) work array of
# ``worst_window_drawdown`` on a multi-decade history.
_WINDOW_BLOCK = 256


def worst_window_drawdown(portfolio_returns: Any, window: int) -> float | None:
    """Worst peak-to-trough decline of the wealth path inside any ``window`` bars.

    The client's loss tolerance is a one-year figure, so the drawdown ceilings are
    judged on the loss a holder could have suffered within a year, not on the decline
    from the highest peak of a multi-year history to its lowest trough. Each window
    starts at its own initial wealth (a loss on its first bar is a drawdown) and its
    decline is compounded; the figure is the worst over every window of ``window``
    consecutive bars, so it is never below the plain one-year return loss and never
    above the full-sample drawdown.

    Args:
        portfolio_returns: The portfolio's linear return per bar.
        window: Bars in one window, normally a year of the panel's calendar.

    Returns:
        The decline as a non-negative fraction, or ``None`` when the history holds
        fewer than ``window`` bars or the figure is not finite: the one-year
        loss cannot be measured, and nothing is invented for it.
    """
    bars = np.asarray(portfolio_returns, dtype=float)
    if window < 1 or bars.size < window or not np.isfinite(bars).all():
        return None
    wealth = np.concatenate(([1.0], np.cumprod(1.0 + bars)))
    windows = np.lib.stride_tricks.sliding_window_view(wealth, window + 1)
    worst = 0.0
    for lo in range(0, windows.shape[0], _WINDOW_BLOCK):
        block = windows[lo : lo + _WINDOW_BLOCK]
        peak = np.maximum.accumulate(block, axis=1)
        # A wiped-out path has no peak left to fall from: nothing more to lose.
        ratio = np.divide(block, peak, out=np.ones_like(block), where=peak > 0.0)
        worst = max(worst, float(np.max(1.0 - ratio)))
    return worst if math.isfinite(worst) else None


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
            ``budget``; defaults to long-only, fully invested (D17). Also honours
            ``excluded_assets`` (tickers barred by the client's ESG block) and the
            drawdown tiers ``nu1`` / ``nu2`` / ``nu3`` with the
            ``observed_drawdown`` (a positive fraction of capital) to compare
            against them; the drawdown check runs only when ``observed_drawdown``
            is supplied.

    Returns:
        ``ok`` with ``passed`` (``True`` iff no breach), ``violations`` (a list
        of structured breach records — ``min_weight`` / ``max_weight`` /
        ``esg_exclusion`` per asset, a ``drawdown_nu2`` / ``drawdown_nu3`` tier
        breach, ``zero_loss_tolerance`` when ``nu2`` is 0 and the book has any
        drawdown, or a single ``budget`` entry) and ``warnings`` (the non-blocking
        ``drawdown_nu1`` soft tier). ``err`` on empty ``weights``; a non-numeric
        bound is caught by the envelope and returned as ``{ok: false}``.
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

    for asset in bounds.get("excluded_assets") or ():
        held = weights.get(asset)
        if held is not None and float(held) > _WEIGHT_TOL:
            violations.append(
                {"type": "esg_exclusion", "asset": asset, "value": float(held)}
            )

    warnings: list[dict[str, Any]] = []
    observed = bounds.get("observed_drawdown")
    if observed is not None:
        _check_drawdown(float(observed), bounds, violations, warnings)

    return ok(
        {"passed": not violations, "violations": violations, "warnings": warnings}
    )


def _check_drawdown(
    observed: float,
    bounds: dict[str, Any],
    violations: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> None:
    """Compare an observed drawdown with the ``nu1/nu2/nu3`` tiers, in place.

    The most severe breached tier is the one reported: past ``nu3`` the absolute
    stop, past ``nu2`` the hard ceiling (both block), past ``nu1`` only a soft
    warning. A non-finite measurement blocks, since every comparison against NaN
    is ``False`` and would otherwise wave the allocation through. A ``nu2`` of 0
    (a client who tolerates no loss) is reported as ``zero_loss_tolerance`` instead
    of as a tier breach at limit 0.
    """
    if not math.isfinite(observed):
        violations.append({"type": "drawdown_unverified", "value": observed})
        return
    hard_ceiling = bounds.get("nu2")
    if hard_ceiling is not None and float(hard_ceiling) <= 0.0 and observed > 0.0:
        # A zero tolerance fails every invested book, whatever the book does; the
        # fault is in the profile, and a bare "limit 0.0" breach hides that.
        violations.append(
            {
                "type": "zero_loss_tolerance",
                "value": observed,
                "limit": 0.0,
                "reason": (
                    "the client's one-year loss tolerance is 0, so no invested book "
                    "can meet the drawdown ceilings; revisit capacity.max_1yr_loss_pct"
                ),
            }
        )
        return
    for tier, blocking in (("nu3", True), ("nu2", True), ("nu1", False)):
        limit = bounds.get(tier)
        if limit is None or observed <= float(limit):
            continue
        record = {"type": f"drawdown_{tier}", "value": observed, "limit": float(limit)}
        (violations if blocking else warnings).append(record)
        return


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
    budget: float | None = None,
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
        budget: Total weight the proposal must sum to (``risk_check``'s ``budget``).
            When given, weights summing elsewhere (percent-style, partially
            invested, non-finite) are rejected instead of being replayed as a
            scaled or part-cash book. ``None`` evaluates the vector as given.

    Returns:
        ``ok`` with ``metrics`` (annualised ``annualized_mean`` /
        ``annualized_standard_deviation`` / ``annualized_sharpe_ratio`` plus the
        ``max_drawdown`` fraction, over the evaluated window),
        ``full_sample_max_drawdown`` (the same drawdown over every aligned return row,
        including the initial training block the walk-forward blocks skip; reported
        for information, since it can span several years),
        ``worst_rolling_loss`` (the worst compounded peak-to-trough decline inside any
        ``rolling_loss_window`` consecutive bars, one year of the panel's calendar;
        the figure to compare with a drawdown ceiling, ``None`` when the history is
        shorter than that window, so the ceilings cannot be verified),
        ``rolling_loss_window`` / ``periods_per_year`` (bars in a year, from the
        panel's own calendar; it annualises ``metrics``), ``drawdown_basis``,
        ``n_observations``
        (evaluated rows), ``n_sample_observations`` (every aligned return row),
        ``n_folds``, ``out_of_sample`` (``True`` iff at least one
        walk-forward fold was produced and every test bar falls after ``fit_end``;
        ``False`` when the panel was too short and the full IN-SAMPLE series was
        evaluated, or when the weights were fitted on the test blocks — the metrics
        are then in-sample), ``sample_start`` / ``window_start`` / ``window_end`` (ISO
        dates), and ``missing`` (zero-weight tickers absent from the panel). ``err`` on
        empty ``weights``, weights that do not sum to ``budget``, no priced assets, a
        non-zero weight on a ticker with no prices (the proposal cannot be evaluated as
        given), fewer than two return rows once the assets' histories are aligned, a
        common sample shorter than half the assets' median history (a young listing
        would otherwise hide an earlier drawdown), a bad ``window``, or metrics that
        are not finite.

        ``max_drawdown`` is the peak-to-trough decline of the **compounded** wealth
        path (``drawdown_basis == "compounded"``), the loss an investor holding the
        weights would have seen. The ``nu1`` / ``nu2`` / ``nu3`` ceilings are not
        enforced by the optimizer: ``risk_check`` compares them with
        ``worst_rolling_loss`` on this same compounded basis, because they derive from
        a one-year loss tolerance.
    """
    if not weights:
        return err("empty weights")
    if budget is not None:
        total = math.fsum(float(weight) for weight in weights.values())
        # Written as a negated ``<=`` so a NaN sum is rejected too.
        if not abs(total - budget) <= _BUDGET_TOL:
            return err(f"weights sum to {total:g}, not the budget {budget:g}")

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
    # A weighted column ``prices_to_returns`` dropped (no usable price) would otherwise
    # be evaluated as idle cash, the same understatement the ``missing`` check guards.
    if dropped := [
        ticker
        for ticker in dropped_by_returns(frame, returns)
        if float(weights[ticker]) != 0.0
    ]:
        return err(f"no prices for weighted tickers: {', '.join(dropped)}")
    if (reason := thin_sample_error(frame, returns, action="backtest")) is not None:
        return err(reason)
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

    # Compounded so max_drawdown is the wealth-path peak-to-trough decline, the basis
    # ``risk_check`` compares the nu ceilings on; ``drawdown_basis`` says so.
    portfolio = Portfolio(oos, weight_vector, compounded=True)
    full_sample_max_drawdown = (
        float(Portfolio(returns, weight_vector, compounded=True).max_drawdown)
        if splits
        else float(portfolio.max_drawdown)
    )
    # One calendar for the whole panel: the test blocks are only part of it, so their
    # own index could not say how many bars a year holds.
    periods = periods_per_year(returns.index)
    worst_rolling_loss = worst_window_drawdown(
        returns.to_numpy(dtype=float) @ weight_vector, periods
    )
    metrics = {
        **annualized_metrics(portfolio, periods),
        # skfolio yields -0.0 for a path with no drawdown; max() pins it to +0.0.
        "max_drawdown": max(0.0, float(portfolio.max_drawdown)),
    }
    # A degenerate series (zero variance) yields NaN/inf, which are not valid JSON
    # for the agent and compare False against every ceiling, so nothing is flagged.
    if not all(math.isfinite(value) for value in metrics.values()):
        return err("backtest metrics are not finite (degenerate return series)")

    return ok(
        {
            "metrics": metrics,
            "full_sample_max_drawdown": full_sample_max_drawdown,
            "worst_rolling_loss": worst_rolling_loss,
            "rolling_loss_window": periods,
            "periods_per_year": periods,
            "drawdown_basis": "compounded",
            "n_observations": int(oos.shape[0]),
            "n_sample_observations": int(returns.shape[0]),
            "n_folds": len(splits),
            "out_of_sample": out_of_sample,
            "sample_start": str(returns.index.min().date()),
            "window_start": str(oos.index.min().date()),
            "window_end": str(oos.index.max().date()),
            "missing": missing,
        }
    )


__all__ = ["backtest", "risk_check"]
