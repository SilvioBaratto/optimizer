"""``estimate_moments``: the second node of the allocator critical path.

Wraps ``optimizer.moments`` to turn a seeded price panel into the expected-return
vector ``mu`` and the **full covariance matrix** the optimiser needs. The default
prior is Empirical mean + Ledoit-Wolf covariance (``MomentEstimationConfig()``);
Ledoit-Wolf is a *covariance* estimator — it yields the 2-D matrix an optimiser
consumes, never the 1-D ``variance_`` a variance estimator would (SPEC D23).

The tool is a pure function of ``(session, asof, universe, config)``: same seeded
DB + same args ⇒ identical ``{mu, cov}``. Returns are computed **outside** any
pipeline via ``prices_to_returns`` (linear returns). An asset whose own history
is short for the panel is **dropped** rather than allowed to truncate every other
asset's sample down to its listing date.

Contract (via ``tool_envelope``):

* empty ``universe`` ⇒ ``{ok: false, error}``;
* a ticker with no priced days is **flagged** in ``data["missing"]``, never raised;
* a ticker with fewer than 20 return observations, or with less than half the
  panel's median depth, is **flagged** in ``data["dropped"]`` and excluded;
* no priced assets, no asset with enough history, or a surviving common sample
  shorter than twice the number of assets ⇒ ``{ok: false, error}``;
* every other failure is caught by the envelope and returned as ``{ok: false}``.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from optimizer.moments import MomentEstimationConfig, build_prior
from optimizer.preprocessing import prices_to_returns

from fund.tools._base import ToolResult, err, ok, tool_envelope
from fund.tools.prices import load_price_frame

if TYPE_CHECKING:
    import pandas as pd
    from sqlalchemy.orm import Session

# One trading month of returns: below this no asset has a history worth fitting,
# whatever the rest of the panel looks like.
_MIN_OBSERVATIONS = 20

# An asset is dropped when its own history is under this share of the panel's
# median depth. The common sample starts at the youngest kept asset's first quote,
# so a listing only slightly past the absolute floor would still cut a multi-year
# panel to a few weeks for every asset.
_MIN_DEPTH_SHARE = 0.5

# Common return rows required per kept asset. The covariance has N x N entries,
# and a sample not comfortably longer than N cannot condition it.
_OBSERVATIONS_PER_ASSET = 2


@tool_envelope
def estimate_moments(
    session: Session,
    asof: dt.date | str,
    universe: Sequence[str],
    *,
    config: MomentEstimationConfig | None = None,
    base_currency: str | None = None,
) -> ToolResult:
    """Estimate ``(mu, cov)`` for ``universe`` from prices as of ``asof``.

    Args:
        session: A sync ``portopt_db`` session (D1); the tool does not own it.
        asof: Inclusive upper bound on price history (no look-ahead).
        universe: yfinance tickers to estimate moments for.
        config: Moment-estimation config. Defaults to Ledoit-Wolf covariance +
            empirical mean (a covariance estimator, D23).
        base_currency: Mandate base currency prices are FX-converted to before
            returns are derived (``None`` keeps listing currency; see
            :func:`fund.tools.prices.load_price_frame`).

    Returns:
        ``ok`` with ``assets`` (priced tickers, in order), ``mu`` (``{ticker:
        float}``), ``cov`` (row-major ``N x N`` matrix aligned to ``assets``),
        ``n_observations``, ``missing``, ``dropped`` (priced tickers excluded
        for having fewer than 20 returns or under half the panel's median
        history) and ``cov_estimator``. ``err`` on an empty universe, no priced
        assets, or too little history, including a common sample shorter than
        twice the number of assets left after the drop.
    """
    if not universe:
        return err("empty universe")

    cfg: MomentEstimationConfig = MomentEstimationConfig() if config is None else config
    frame, missing = load_price_frame(
        session, asof, universe, base_currency=base_currency
    )
    if frame.shape[1] == 0:
        return err("no priced assets in universe")

    # ``drop_inceptions_nan=False`` keeps each asset's own history intact so a
    # short one can be dropped by name; the default would silently cut every row
    # before the youngest asset's first price for the whole panel.
    all_returns = cast(
        "pd.DataFrame", prices_to_returns(frame, drop_inceptions_nan=False)
    )
    depth = all_returns.count()
    floor = max(_MIN_OBSERVATIONS, math.ceil(_MIN_DEPTH_SHARE * float(depth.median())))
    kept = [col for col in all_returns.columns if depth[col] >= floor]
    dropped = [str(col) for col in all_returns.columns if depth[col] < floor]
    if not kept:
        return err(
            "insufficient price history to estimate moments: no asset has the "
            f"{floor} return observations needed"
        )

    returns = all_returns[kept].dropna(how="any")
    required = max(_MIN_OBSERVATIONS, _OBSERVATIONS_PER_ASSET * len(kept))
    if returns.shape[0] < required:
        kept_depth = depth[kept]
        shortest = kept_depth.idxmin()
        return err(
            "insufficient common history to estimate moments: "
            f"{returns.shape[0]} return observations shared by all {len(kept)} "
            f"assets, need at least {required}; {shortest} has only "
            f"{int(kept_depth.min())} returns, so drop it from the universe or "
            "wait for more history"
        )

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
            "dropped": dropped,
            "cov_estimator": cfg.cov_estimator.value,
        }
    )


__all__ = ["estimate_moments"]
