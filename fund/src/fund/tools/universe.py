"""T3.3 — ``universe_filter``: prune a candidate universe with the optimizer's
pre-selection stack before moments/optimization run.

Wraps :func:`optimizer.pre_selection.build_preselection_pipeline` over a seeded
price panel: it loads prices as of ``asof``, converts them to linear returns
(:func:`optimizer.preprocessing.prices_to_returns`, run **outside** the pipeline),
fits the data-cleaning + selection pipeline, and returns the **surviving
tickers** as a ``list[str]`` in requested order — never the returns frame (SPEC
Fase 3).

``sector_mapping`` is injected as a plain ``dict[str, str]`` (SPEC: not queried
from the DB here), forwarded to the pipeline's ``SectorImputer``.

The tool is a pure, deterministic function of ``(session, asof, universe,
criteria, sector_mapping)``: pre-selection carries no RNG, so identical seeded
data + identical criteria ⇒ identical list.

Contract (via :func:`fund.tools._base.tool_envelope`):

* an empty ``universe``, a universe with nothing priced, or too little history ⇒
  ``{ok: true, data: []}`` — an empty result is a valid outcome, not an error;
* a bad selection criterion (e.g. an unknown ``select_k_measure`` enum) is caught
  by :class:`~optimizer.pre_selection.PreSelectionConfig` validation and returned
  as ``{ok: false, error}``, never raised;
* every other failure is caught by the envelope and returned as ``{ok: false}``.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from optimizer.pre_selection import PreSelectionConfig, build_preselection_pipeline
from optimizer.preprocessing import prices_to_returns

from fund.tools._base import ToolResult, ok, tool_envelope
from fund.tools.prices import load_price_frame

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

# ``PreSelectionConfig`` field names — criteria keys outside this set are ignored
# (placeholder for the Fase-4 ConstraintSet schema), mirroring ``optimize``.
_CONFIG_FIELDS: frozenset[str] = frozenset(
    f.name for f in dataclasses.fields(PreSelectionConfig)
)


def _resolve_config(criteria: dict[str, Any] | None) -> PreSelectionConfig:
    """Build a ``PreSelectionConfig`` from raw ``criteria``.

    Unknown keys are ignored; a *known* field carrying a bad value (e.g. an
    invalid ``select_k_measure`` enum, or an out-of-range threshold) raises inside
    ``PreSelectionConfig.__post_init__`` and is turned into ``{ok: false}`` by the
    envelope.
    """
    if not criteria:
        return PreSelectionConfig()
    overrides = {k: v for k, v in criteria.items() if k in _CONFIG_FIELDS}
    return PreSelectionConfig(**overrides)


@tool_envelope
def universe_filter(
    session: Session,
    asof: dt.date | str,
    universe: Sequence[str],
    *,
    criteria: dict[str, Any] | None = None,
    sector_mapping: dict[str, str] | None = None,
) -> ToolResult:
    """Filter ``universe`` down to the assets that survive pre-selection.

    Args:
        session: A sync ``portopt_db`` session (D1); the tool does not own it.
        asof: Inclusive upper bound on price history (no look-ahead). Accepts a
            ``date`` or an ISO ``YYYY-MM-DD`` string.
        universe: Candidate yfinance tickers, in the order the result should
            follow.
        criteria: Optional overrides mapped onto ``PreSelectionConfig`` fields
            (e.g. ``top_k``, ``correlation_threshold``, ``select_k_measure``);
            unknown keys are ignored. ``None`` uses the default pipeline.
        sector_mapping: Optional ticker → sector ``dict`` forwarded to the
            pipeline's ``SectorImputer`` (injected, not DB-queried).

    Returns:
        ``ok`` with a ``list[str]`` of surviving tickers (subset of ``universe``,
        in requested order). An empty universe, no priced assets, or too little
        history all yield ``ok([])``. ``err`` on a bad criterion.
    """
    config = _resolve_config(criteria)
    if not universe:
        return ok([])

    frame, _ = load_price_frame(session, asof, universe)
    # No priced assets, or too little history to derive returns: an empty
    # investable set is a valid answer, not an error.
    if frame.shape[1] == 0 or frame.shape[0] < 2:
        return ok([])

    returns = prices_to_returns(frame)
    pipeline = build_preselection_pipeline(config=config, sector_mapping=sector_mapping)
    selected = pipeline.fit_transform(returns)

    survivors = {str(col) for col in selected.columns}
    return ok([ticker for ticker in universe if ticker in survivors])


__all__ = ["universe_filter"]
