"""T3.2 — ``estimate_moments``: the second node of the allocator critical path.

Wraps ``optimizer.moments`` to turn a seeded price panel into the expected-return
vector ``mu`` and the **full covariance matrix** the optimiser needs. The default
prior is Empirical mean + Ledoit-Wolf covariance (``MomentEstimationConfig()``);
Ledoit-Wolf is a *covariance* estimator — it yields the 2-D matrix an optimiser
consumes, never the 1-D ``variance_`` a variance estimator would (SPEC D23).

The tool is a pure function of ``(session, asof, universe, config)``: same seeded
DB + same args ⇒ identical ``{mu, cov}``. Returns are computed **outside** any
pipeline via :func:`optimizer.preprocessing.prices_to_returns` (linear returns).

Contract (via :func:`fund.tools._base.tool_envelope`):

* empty ``universe`` ⇒ ``{ok: false, error}``;
* a ticker with no priced days is **flagged** in ``data["missing"]``, never raised;
* no priced assets, or fewer than two observations, ⇒ ``{ok: false, error}``;
* every other failure is caught by the envelope and returned as ``{ok: false}``.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import TYPE_CHECKING

from optimizer.moments import MomentEstimationConfig, build_prior
from optimizer.preprocessing import prices_to_returns

from fund.tools._base import ToolResult, err, ok, tool_envelope
from fund.tools.prices import load_price_frame

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


@tool_envelope
def estimate_moments(
    session: Session,
    asof: dt.date | str,
    universe: Sequence[str],
    *,
    config: MomentEstimationConfig | None = None,
) -> ToolResult:
    """Estimate ``(mu, cov)`` for ``universe`` from prices as of ``asof``.

    Args:
        session: A sync ``portopt_db`` session (D1); the tool does not own it.
        asof: Inclusive upper bound on price history (no look-ahead).
        universe: yfinance tickers to estimate moments for.
        config: Moment-estimation config. Defaults to Ledoit-Wolf covariance +
            empirical mean (a covariance estimator, D23).

    Returns:
        ``ok`` with ``assets`` (priced tickers, in order), ``mu`` (``{ticker:
        float}``), ``cov`` (row-major ``N x N`` matrix aligned to ``assets``),
        ``n_observations``, ``missing`` and ``cov_estimator``. ``err`` on an
        empty universe, no priced assets, or too little history.
    """
    if not universe:
        return err("empty universe")

    cfg: MomentEstimationConfig = (
        MomentEstimationConfig() if config is None else config
    )
    frame, missing = load_price_frame(session, asof, universe)
    if frame.shape[1] == 0:
        return err("no priced assets in universe")
    if frame.shape[0] < 2:
        return err("insufficient price history to estimate moments")

    returns = prices_to_returns(frame)
    prior = build_prior(cfg)
    prior.fit(returns)
    distribution = prior.return_distribution_

    assets = [str(col) for col in returns.columns]
    mu = {asset: float(m) for asset, m in zip(assets, distribution.mu, strict=True)}
    cov = [[float(v) for v in row] for row in distribution.covariance]

    return ok(
        {
            "assets": assets,
            "mu": mu,
            "cov": cov,
            "n_observations": int(returns.shape[0]),
            "missing": missing,
            "cov_estimator": cfg.cov_estimator.value,
        }
    )


__all__ = ["estimate_moments"]
