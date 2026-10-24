"""``universe_filter``: prune a candidate universe with the optimizer's
pre-selection stack before moments/optimization run.

Wraps ``build_preselection_pipeline`` over a seeded price panel: it loads prices
as of ``asof``, converts them to linear returns (``prices_to_returns``, run
**outside** the pipeline), fits the data-cleaning + selection pipeline, and
returns the **surviving tickers** as a ``list[str]`` in requested order — never
the returns frame.

``sector_mapping`` is injected as a plain ``dict[str, str]`` (SPEC: not queried
from the DB here), forwarded to the pipeline's ``SectorImputer``.

The tool is a pure, deterministic function of ``(session, asof, universe,
criteria, sector_mapping)``: pre-selection carries no RNG, so identical seeded
data + identical criteria ⇒ identical list.

Contract (via ``tool_envelope``):

* an empty ``universe``, a universe with nothing priced, too little history, or
  one the screens leave with a single name or none ⇒ ``{ok: true, data: [...]}``
  (``[]`` when nothing survives) — a thin result is a valid outcome, not an error;
* a bad selection criterion (e.g. an unknown ``select_k_measure`` enum) is caught
  by ``PreSelectionConfig`` validation and returned as ``{ok: false, error}``,
  never raised;
* every other failure is caught by the envelope and returned as ``{ok: false}``;
* an ``exclusions`` criterion (GICS sector values) is enforced against each
  instrument's ``TickerProfile.sector``; an instrument whose sector is unknown
  cannot be shown to sit outside the excluded sectors, so it is dropped (fails
  closed) and reported in ``excluded_by_esg``;
* a criterion the tool cannot apply is never silently dropped behind ``ok``: its
  key is listed in ``ignored_criteria`` so the caller knows it was not enforced.

``list_universe`` is the DB-side counterpart: the allocator picks its candidates from
the instruments the database holds (active, priced on the bar, outside the client's
ESG exclusions) instead of from model memory.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import re
from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING, Any

from optimizer.pre_selection import PreSelectionConfig, build_preselection_pipeline
from optimizer.preprocessing import prices_to_returns
from portopt_db.models.market_data.yfinance_data import PriceHistory, TickerProfile
from portopt_db.models.universe.universe import Instrument
from portopt_db.repositories.market_data.yfinance_repository import YFinanceRepository
from sqlalchemy import exists, func, or_, select

from fund.schemas.enums import GicsSector
from fund.tools._base import ToolResult, coerce_date, err, ok, tool_envelope
from fund.tools.prices import load_price_frame

if TYPE_CHECKING:
    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sqlalchemy import Select
    from sqlalchemy.orm import Session

# ``PreSelectionConfig`` field names; the other criterion the tool applies is
# ``exclusions``. Any other key is reported back in ``ignored_criteria``.
_CONFIG_FIELDS: frozenset[str] = frozenset(
    f.name for f in dataclasses.fields(PreSelectionConfig)
)
_EXCLUSIONS_KEY = "exclusions"
_APPLIED_CRITERIA: frozenset[str] = _CONFIG_FIELDS | {_EXCLUSIONS_KEY}

_UNCLASSIFIED = "unclassified"

# Longest run of missing bars carried forward as a stale price. It covers a venue
# closed for a holiday week (Golden Week, Lunar New Year) so a live ticker is not
# mistaken for a delisted one; a longer silence leaves trailing NaNs for
# ``SelectComplete`` to screen out.
_MAX_STALE_BARS = 5

# The one pre-selection step that compares assets pairwise: skfolio's
# ``DropCorrelated`` raises on fewer than two, where there is nothing to compare.
_PAIRWISE_STEP = "drop_correlated"

# ``TickerProfile.sector`` holds Yahoo's sector names, the same eleven the optimizer's
# sector bands use; they map one-to-one onto the GICS sectors a client can exclude.
_SECTOR_NAME_BY_GICS: dict[GicsSector, str] = {
    GicsSector.ENERGY: "Energy",
    GicsSector.MATERIALS: "Basic Materials",
    GicsSector.INDUSTRIALS: "Industrials",
    GicsSector.CONSUMER_DISCRETIONARY: "Consumer Cyclical",
    GicsSector.CONSUMER_STAPLES: "Consumer Defensive",
    GicsSector.HEALTH_CARE: "Healthcare",
    GicsSector.FINANCIALS: "Financial Services",
    GicsSector.INFORMATION_TECHNOLOGY: "Technology",
    GicsSector.COMMUNICATION_SERVICES: "Communication Services",
    GicsSector.UTILITIES: "Utilities",
    GicsSector.REAL_ESTATE: "Real Estate",
}

# The same names normalised to snake case, the form ``_gics_sector`` looks labels up
# in. A GICS-named label that is not in this table still resolves by its value.
_YAHOO_SECTOR_TO_GICS: dict[str, GicsSector] = {
    "_".join(name.lower().split()): gics for gics, name in _SECTOR_NAME_BY_GICS.items()
}

# One page never exceeds this, so a single call cannot flood the model's context
# with the whole (~9k-name) universe.
_MAX_PAGE_SIZE = 500


def _label(name: str) -> str:
    """Fold case and the space/hyphen separators of a sector name to snake case."""
    return "_".join(name.replace("-", " ").lower().split())


def _labels_of(sectors: Collection[GicsSector]) -> set[str]:
    """Every stored spelling (GICS value or Yahoo label) of the given sectors."""
    return {sector.value for sector in sectors} | {
        label for label, gics in _YAHOO_SECTOR_TO_GICS.items() if gics in sectors
    }


def _resolve_sector(name: str) -> GicsSector | None:
    """The GICS sector a caller's sector filter names, in any spelling or case."""
    label = _label(name)
    if label in _YAHOO_SECTOR_TO_GICS:
        return _YAHOO_SECTOR_TO_GICS[label]
    try:
        return GicsSector(label)
    except ValueError:
        return None


def _resolve_config(criteria: dict[str, Any] | None) -> PreSelectionConfig:
    """Build a ``PreSelectionConfig`` from raw ``criteria``.

    Keys that are not config fields are left out here (``universe_filter`` reports
    them); a *known* field carrying a bad value (e.g. an invalid
    ``select_k_measure`` enum, or an out-of-range threshold) raises inside
    ``PreSelectionConfig.__post_init__`` and is turned into ``{ok: false}`` by the
    envelope.
    """
    if not criteria:
        return PreSelectionConfig()
    overrides = {k: v for k, v in criteria.items() if k in _CONFIG_FIELDS}
    return PreSelectionConfig(**overrides)


def _resolve_exclusions(criteria: dict[str, Any] | None) -> frozenset[GicsSector]:
    """Parse the ``exclusions`` criterion into GICS sectors.

    Raises:
        ValueError: If an entry is not one of the GICS sector values.
    """
    raw = (criteria or {}).get(_EXCLUSIONS_KEY) or ()
    entries = [raw] if isinstance(raw, str) else list(raw)
    try:
        return frozenset(GicsSector(entry) for entry in entries)
    except ValueError:
        valid = sorted(sector.value for sector in GicsSector)
        raise ValueError(
            f"unknown exclusion in {entries!r}; expected GICS sectors from {valid}"
        ) from None


def _gics_sector(session: Session, ticker: str) -> GicsSector | None:
    """The GICS sector of ``ticker`` from its stored profile, or ``None`` if unknown."""
    repo = YFinanceRepository(session)
    instrument = repo.get_instrument_by_yfinance_ticker(ticker)
    profile = None if instrument is None else repo.get_profile(instrument.id)
    if profile is None or not profile.sector:
        return None
    label = re.sub(r"[^a-z]+", "_", profile.sector.lower()).strip("_")
    if label in _YAHOO_SECTOR_TO_GICS:
        return _YAHOO_SECTOR_TO_GICS[label]
    try:
        return GicsSector(label)
    except ValueError:
        return None


def _esg_barred(
    session: Session, tickers: Sequence[str], exclusions: frozenset[GicsSector]
) -> dict[str, str]:
    """Map each ticker that may not be held to the sector (or ``unclassified``)."""
    barred: dict[str, str] = {}
    for ticker in tickers:
        sector = _gics_sector(session, ticker)
        if sector is None:
            barred[ticker] = _UNCLASSIFIED
        elif sector in exclusions:
            barred[ticker] = sector.value
    return barred


def _preselect(pipeline: Pipeline, returns: Any) -> pd.DataFrame:
    """Run ``pipeline`` step by step, tolerating fewer than two assets left.

    ``Pipeline.fit_transform`` raises once the cleaning steps leave one asset (in
    the pairwise correlation screen) or none (in whichever step comes next), though
    a one-name or empty survivor set is a valid answer for this tool.
    """
    selected = returns
    for name, step in pipeline.steps:
        n_assets = selected.shape[1]
        if n_assets == 0:
            break
        if name == _PAIRWISE_STEP and n_assets < 2:
            continue
        selected = step.fit_transform(selected)
    return selected


def _result(
    survivors: list[str],
    ignored: list[str],
    barred: dict[str, str],
) -> ToolResult:
    """Envelope the survivors; report keys appear only when non-empty."""
    result = ok(survivors)
    if ignored:
        result["ignored_criteria"] = ignored
    if barred:
        result["excluded_by_esg"] = barred
    return result


@tool_envelope
def universe_filter(
    session: Session,
    asof: dt.date | str,
    universe: Sequence[str],
    *,
    criteria: dict[str, Any] | None = None,
    sector_mapping: dict[str, str] | None = None,
    base_currency: str | None = None,
) -> ToolResult:
    """Filter ``universe`` down to the assets that survive pre-selection.

    Args:
        session: A sync ``portopt_db`` session (D1); the tool does not own it.
        asof: Inclusive upper bound on price history (no look-ahead). Accepts a
            ``date`` or an ISO ``YYYY-MM-DD`` string.
        universe: Candidate yfinance tickers, in the order the result should
            follow.
        criteria: Optional overrides mapped onto ``PreSelectionConfig`` fields
            (e.g. ``top_k``, ``correlation_threshold``, ``select_k_measure``),
            plus ``exclusions``, a list of GICS sector values to bar. Any other
            key is not applied and is returned in ``ignored_criteria``. ``None``
            uses the default pipeline.
        sector_mapping: Optional ticker → sector ``dict`` forwarded to the
            pipeline's ``SectorImputer`` (injected, not DB-queried).
        base_currency: Mandate base currency prices are FX-converted to before
            returns are derived (``None`` keeps listing currency; see
            :func:`fund.tools.prices.load_price_frame`).

    Returns:
        ``ok`` with a ``list[str]`` of surviving tickers (subset of ``universe``,
        in requested order). An empty universe, no priced assets, or too little
        history all yield ``ok([])``. The envelope also carries
        ``ignored_criteria`` (keys that were not applied) and ``excluded_by_esg``
        (``{ticker: sector or "unclassified"}``) when either is non-empty.
        ``err`` on a bad criterion.
    """
    config = _resolve_config(criteria)
    exclusions = _resolve_exclusions(criteria)
    ignored = sorted(k for k in (criteria or {}) if k not in _APPLIED_CRITERIA)
    if not universe:
        return _result([], ignored, {})

    frame, _ = load_price_frame(session, asof, universe, base_currency=base_currency)
    barred = _esg_barred(session, list(frame.columns), exclusions) if exclusions else {}
    frame = frame.drop(columns=list(barred))
    # No priced assets, or too little history to derive returns: an empty
    # investable set is a valid answer, not an error.
    if frame.shape[1] == 0 or frame.shape[0] < 2:
        return _result([], ignored, barred)

    # Keep the leading NaNs of a young listing: the default would drop those rows
    # for every asset, so SelectComplete would never see the short history it exists
    # to screen out and the other statistics would run on the truncated sample. The
    # unbounded forward fill would likewise turn a stopped asset's last quote into
    # zero returns up to ``asof``, so the fill is capped instead of left to
    # ``prices_to_returns``.
    returns = prices_to_returns(
        frame.ffill(limit=_MAX_STALE_BARS),
        drop_inceptions_nan=False,
        fill_nan=False,
    )
    pipeline = build_preselection_pipeline(config=config, sector_mapping=sector_mapping)
    selected = _preselect(pipeline, returns)

    survivors = {str(col) for col in selected.columns}
    return _result(
        [ticker for ticker in universe if ticker in survivors], ignored, barred
    )


def _investable(asof: dt.date, exclusions: Collection[GicsSector | str]) -> Select[Any]:
    """Select ``(ticker, name, sector, market_cap)`` of every investable instrument.

    Investable on ``asof`` means: it has a yfinance ticker, it had not been delisted
    by that bar (a later delisting does not retroactively make it uninvestable) and
    it has a price on or before it. A non-empty ``exclusions`` additionally drops
    the excluded sectors and every instrument whose sector is unknown — a hard ESG
    block must fail closed rather than let an unclassified name through. The sector
    and market cap come from the latest ``ticker_profiles`` snapshot, not from
    ``asof``.
    """
    priced = exists().where(
        PriceHistory.instrument_id == Instrument.id, PriceHistory.date <= asof
    )
    stmt = (
        select(
            Instrument.yfinance_ticker,
            func.coalesce(Instrument.name, Instrument.short_name),
            TickerProfile.sector,
            TickerProfile.market_cap,
        )
        .outerjoin(TickerProfile, TickerProfile.instrument_id == Instrument.id)
        .where(
            Instrument.yfinance_ticker.is_not(None),
            or_(Instrument.delisted_at.is_(None), Instrument.delisted_at > asof),
            priced,
        )
    )
    if exclusions:
        label = func.lower(
            func.replace(func.replace(TickerProfile.sector, " ", "_"), "-", "_")
        )
        excluded = _labels_of({GicsSector(e) for e in exclusions})
        known = _labels_of(set(GicsSector))
        stmt = stmt.where(
            TickerProfile.sector.is_not(None),
            label.in_(known),
            label.not_in(excluded),
        )
    return stmt


@tool_envelope
def list_universe(
    session: Session,
    asof: dt.date | str,
    *,
    exclusions: Collection[GicsSector | str] = (),
    sector: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> ToolResult:
    """List one page of the investable universe as of ``asof``.

    Membership is point-in-time: listing, delisting and first price are judged on
    ``asof``. ``sector`` and ``market_cap`` are not: ``ticker_profiles`` keeps only the
    latest snapshot, so for a past ``asof`` the ordering ranks names by their current
    size and the sector is today's classification.

    Args:
        session: A sync ``portopt_db`` session; the tool does not own it.
        asof: Decision bar; delistings and prices after it are ignored.
        exclusions: The client's ESG-excluded GICS sectors, removed from the list.
        sector: Keep only this sector: a stored sector name (``Technology``) or the
            matching GICS name (``information_technology``), in any case.
        limit: Page size, clamped to ``1..500``.
        offset: Rows to skip, for paging.

    Returns:
        ``ok`` with ``{total, offset, limit, instruments}``, the instruments ordered
        by market cap (largest first, unknown last) as ``{ticker, name, sector,
        market_cap}`` rows. ``err`` naming the valid sectors when ``sector`` is not
        one of them, so a typo is never read as an empty universe.
    """
    page = max(1, min(int(limit), _MAX_PAGE_SIZE))
    start = max(0, int(offset))
    stmt = _investable(coerce_date(asof), exclusions)
    if sector and sector.strip():
        wanted = _resolve_sector(sector)
        if wanted is None:
            stored = ", ".join(sorted(_SECTOR_NAME_BY_GICS.values()))
            gics = ", ".join(g.value for g in _SECTOR_NAME_BY_GICS)
            return err(
                f"unknown sector {sector!r}; use one of: {stored} "
                f"(or the GICS names: {gics})"
            )
        label = func.lower(
            func.replace(func.replace(TickerProfile.sector, " ", "_"), "-", "_")
        )
        stmt = stmt.where(label.in_(_labels_of({wanted})))

    total = session.execute(
        select(func.count()).select_from(stmt.subquery())
    ).scalar_one()
    rows = session.execute(
        stmt.order_by(
            TickerProfile.market_cap.desc().nulls_last(), Instrument.yfinance_ticker
        )
        .limit(page)
        .offset(start)
    ).all()
    return ok(
        {
            "total": total,
            "offset": start,
            "limit": page,
            "instruments": [
                {"ticker": t, "name": n, "sector": sec, "market_cap": cap}
                for t, n, sec, cap in rows
            ],
        }
    )


__all__ = ["list_universe", "universe_filter"]
