"""``get_regime_stats``: the numbers behind the economist's regime read.

The ``macro-regime-read`` skill tells the economist to interpret three statistics and
never to invent them. This tool computes them from a seeded price panel, as a pure
function of ``(session, asof, tickers, window)``:

* **Financial turbulence** (Kritzman & Li): the Mahalanobis distance of each daily
  return vector from the window's mean under its covariance. The latest value is
  reported with its percentile inside the window, the unit-free reading.
* **Absorption ratio** (Kritzman et al.): the share of total variance carried by the
  top fifth of the covariance eigenvectors. It is reported alongside the same
  statistic one quarter earlier, so "rising" needs no second call.
* **Two-state mixture** (Chow et al.): a full-covariance Gaussian mixture with two
  components; the high-variance component is the stress state, and the posterior
  probability of it at the latest observation is reported.

Returns are linear and computed outside any pipeline (``prices_to_returns``). Nothing
random is left unseeded, so identical data gives identical output.

Contract (via ``tool_envelope``):

* empty ``tickers``, fewer than two priced assets, or fewer than 60 returns ⇒
  ``{ok: false, error}``;
* a sample too thin for a full-covariance two-state fit still returns the other two
  statistics, with ``mixture`` set to ``None``;
* every other failure (e.g. a malformed ``asof``) is caught by the envelope.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from optimizer.preprocessing import prices_to_returns

from fund.tools._base import ToolResult, coerce_date, err, ok, tool_envelope
from fund.tools.prices import load_price_frame

if TYPE_CHECKING:
    import numpy.typing as npt
    from sqlalchemy.orm import Session

_MIN_OBSERVATIONS = 60
# A full-covariance two-state fit needs this many returns per asset to be stable.
_MIXTURE_OBS_PER_ASSET = 20
_ABSORPTION_WINDOW = 252
_ABSORPTION_LAG = 63
# Kritzman et al. fix the eigenvector set at one fifth of the assets.
_ABSORPTION_FRACTION = 0.2
_MIXTURE_SEED = 0


def _turbulence(returns: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Mahalanobis distance of every return row from the sample mean."""
    centred = returns - returns.mean(axis=0)
    precision = np.linalg.pinv(np.atleast_2d(np.cov(returns, rowvar=False)))
    return np.einsum("ij,jk,ik->i", centred, precision, centred)


def _absorption_ratio(returns: npt.NDArray[np.float64]) -> tuple[float, int]:
    """Variance share of the top ``max(1, n/5)`` covariance eigenvectors."""
    eigenvalues = np.linalg.eigvalsh(np.atleast_2d(np.cov(returns, rowvar=False)))
    n_components = max(1, round(returns.shape[1] * _ABSORPTION_FRACTION))
    top = np.sort(eigenvalues)[::-1][:n_components]
    return float(top.sum() / eigenvalues.sum()), n_components


def _mixture_block(returns: npt.NDArray[np.float64]) -> dict[str, float] | None:
    """Posterior stress probability of the latest return, or ``None`` if too thin."""
    n_obs, n_assets = returns.shape
    if n_obs < _MIXTURE_OBS_PER_ASSET * n_assets:
        return None

    from sklearn.mixture import GaussianMixture

    model = GaussianMixture(
        n_components=2,
        covariance_type="full",
        n_init=3,
        random_state=_MIXTURE_SEED,
    ).fit(returns)
    covariances, weights = model.covariances_, model.weights_
    if covariances is None or weights is None:  # set by fit; narrows the Optionals
        return None
    stress = int(np.argmax([np.trace(cov) for cov in covariances]))
    posterior = model.predict_proba(returns[-1:])[0]
    return {
        "stress_probability": float(posterior[stress]),
        "stress_weight": float(weights[stress]),
        "calm_weight": float(weights[1 - stress]),
    }


@tool_envelope
def get_regime_stats(
    session: Session,
    asof: dt.date | str,
    tickers: Sequence[str],
    *,
    window: int = 756,
) -> ToolResult:
    """Compute turbulence, absorption ratio and stress probability for ``tickers``.

    Args:
        session: A sync ``portopt_db`` session; the tool does not own it.
        asof: Inclusive upper bound on price history (no look-ahead). Accepts a
            ``date`` or an ISO ``YYYY-MM-DD`` string.
        tickers: yfinance tickers whose joint behaviour defines the regime, e.g.
            broad market proxies. At least two must be priced.
        window: Trailing returns used for every statistic (default ~3 years).

    Returns:
        ``ok`` with ``turbulence`` (``latest``, its ``percentile`` in the window and
        ``date``), ``absorption_ratio`` (``value``, ``n_components``, ``prior`` and
        ``change`` against the window ending one quarter earlier, ``None`` when the
        history is too short for it), ``mixture`` (``stress_probability``,
        ``stress_weight``, ``calm_weight``; ``None`` for a thin sample), plus
        ``asof``, ``assets``, ``missing``, ``n_observations``, ``window_start`` and
        ``window_end``. ``err`` on empty ``tickers``, fewer than two priced assets,
        or fewer than 60 returns.
    """
    if not tickers:
        return err("no tickers requested")

    frame, missing = load_price_frame(session, asof, tickers)
    if frame.shape[1] < 2:
        return err("regime statistics need at least two priced assets")

    # Rows where any asset lacks a price cannot enter a joint covariance.
    returns = cast("pd.DataFrame", prices_to_returns(frame)).dropna().tail(window)
    if returns.shape[0] < _MIN_OBSERVATIONS:
        return err(
            f"only {returns.shape[0]} joint return observations; regime statistics "
            f"need at least {_MIN_OBSERVATIONS}"
        )

    matrix = returns.to_numpy(dtype=np.float64)
    distances = _turbulence(matrix)
    ratio, n_components = _absorption_ratio(matrix[-_ABSORPTION_WINDOW:])
    prior_end = matrix.shape[0] - _ABSORPTION_LAG
    prior = None
    if prior_end >= _MIN_OBSERVATIONS:
        prior, _ = _absorption_ratio(matrix[:prior_end][-_ABSORPTION_WINDOW:])

    data: dict[str, Any] = {
        "asof": coerce_date(asof).isoformat(),
        "assets": [str(col) for col in returns.columns],
        "missing": missing,
        "n_observations": int(matrix.shape[0]),
        "window_start": str(returns.index.min().date()),
        "window_end": str(returns.index.max().date()),
        "turbulence": {
            "latest": float(distances[-1]),
            "percentile": float((distances <= distances[-1]).mean()),
            "date": str(returns.index.max().date()),
        },
        "absorption_ratio": {
            "value": ratio,
            "n_components": n_components,
            "prior": prior,
            "change": None if prior is None else ratio - prior,
        },
        "mixture": _mixture_block(matrix),
    }
    return ok(data)


__all__ = ["get_regime_stats"]
