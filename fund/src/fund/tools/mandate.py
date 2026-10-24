"""Mandate screening: ESG sector exclusions and K&E instrument restrictions.

The client's ``ConstraintSet`` carries hard restrictions the optimizer cannot model
as weight bounds — excluded GICS sectors and, for low knowledge-and-experience
clients, no complex / no leveraged instruments. They are evaluated here against the
instrument table, in code, so neither a hallucinating nor a prompt-injected model can
route a forbidden name into the allocation.

Every check **fails closed**: when a restriction is active and an instrument cannot
be classified (no ticker row, no profile sector, no instrument type), the instrument
is reported as a breach rather than waved through.

Sector names in ``ticker_profiles`` are Yahoo's, not GICS; ``YAHOO_SECTOR_BY_GICS``
is the bridge. Leverage is inferred from the instrument name of a non-stock product,
because the instrument table carries no leverage flag.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable
from typing import TYPE_CHECKING, Any

from portopt_db.repositories.market_data.yfinance_repository import YFinanceRepository

from fund.schemas.enums import GicsSector

if TYPE_CHECKING:
    from collections.abc import Mapping

    from sqlalchemy.orm import Session

    from fund.schemas import ConstraintSet

# Yahoo's sector labels, one per GICS sector (Yahoo renames five of them).
YAHOO_SECTOR_BY_GICS: dict[GicsSector, str] = {
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

# Plain instruments a low-K&E client may hold. Anything else (ETC, ETN, a missing
# type, ...) is treated as complex.
_PLAIN_INSTRUMENT_TYPES: frozenset[str] = frozenset({"STOCK", "EQUITY", "ETF"})
# Common stock is never a leveraged product; the name test would only produce false
# positives on companies called "Ultra ..." or "Bull ...".
_STOCK_TYPES: frozenset[str] = frozenset({"STOCK", "EQUITY"})
# A lookbehind rather than ``\b`` so a leading minus ("-1x") still anchors. The
# pattern errs towards excluding: a restricted client losing an "Ultra Short Duration"
# bond fund is cheaper than holding a leveraged ETF.
_LEVERAGED_NAME = re.compile(
    r"(?<!\w)(?:-?[123]x|leverage[d]?|inverse|ultra(?:pro|short)?|bull|bear)(?!\w)",
    re.IGNORECASE,
)


def _sector_key(sector: str | None) -> str:
    return (sector or "").strip().casefold()


def _is_leveraged(instrument_type: str, name: str) -> bool:
    if instrument_type in _STOCK_TYPES:
        return False
    return _LEVERAGED_NAME.search(name) is not None


def mandate_breaches(
    session: Session,
    tickers: Iterable[str],
    *,
    excluded_sectors: Collection[str] = (),
    no_complex: bool = False,
    no_leverage: bool = False,
) -> list[dict[str, Any]]:
    """Report every ``tickers`` entry that breaches the mandate restrictions.

    Args:
        session: A sync ``portopt_db`` session; the function does not own it.
        tickers: yfinance tickers to screen.
        excluded_sectors: Yahoo sector names the client refuses to hold (see
            ``YAHOO_SECTOR_BY_GICS``). Empty disables the ESG screen entirely.
        no_complex: Reject anything that is not a plain stock or ETF.
        no_leverage: Reject leveraged or inverse products.

    Returns:
        One record per breach, each with ``type`` and ``asset``. ``esg_exclusion``
        carries the ``sector``; ``unclassified_sector`` means an ESG screen is
        active but the sector is unknown. ``complex_instrument`` and
        ``leveraged_instrument`` carry ``instrument_type``. A ticker can appear
        more than once. Empty when no restriction is active, without touching the
        database.
    """
    excluded = {_sector_key(sector) for sector in excluded_sectors}
    if not excluded and not no_complex and not no_leverage:
        return []

    repo = YFinanceRepository(session)
    breaches: list[dict[str, Any]] = []
    for ticker in tickers:
        instrument = repo.get_instrument_by_yfinance_ticker(ticker)
        profile = repo.get_profile(instrument.id) if instrument is not None else None

        if excluded:
            sector = profile.sector if profile is not None else None
            if not _sector_key(sector):
                breaches.append({"type": "unclassified_sector", "asset": ticker})
            elif _sector_key(sector) in excluded:
                breaches.append(
                    {"type": "esg_exclusion", "asset": ticker, "sector": sector}
                )

        if not (no_complex or no_leverage):
            continue
        instrument_type = (
            (instrument.instrument_type or "").strip().upper() if instrument else ""
        )
        name = " ".join(
            part
            for part in (
                instrument.name if instrument is not None else None,
                profile.long_name if profile is not None else None,
            )
            if part
        )
        leveraged = _is_leveraged(instrument_type, name)
        if no_leverage and leveraged:
            breaches.append(
                {
                    "type": "leveraged_instrument",
                    "asset": ticker,
                    "instrument_type": instrument_type,
                }
            )
        if no_complex and (leveraged or instrument_type not in _PLAIN_INSTRUMENT_TYPES):
            breaches.append(
                {
                    "type": "complex_instrument",
                    "asset": ticker,
                    "instrument_type": instrument_type,
                }
            )
    return breaches


def constraint_set_breaches(
    session: Session, constraint_set: ConstraintSet, tickers: Iterable[str]
) -> list[dict[str, Any]]:
    """The ``constraint_set``'s ESG and K&E breaches among ``tickers``.

    The one translation of a ``ConstraintSet`` into :func:`mandate_breaches`
    arguments, shared by the bound toolsets and the TUI recompute so the live gate and
    the cockpit's read-out cannot disagree about what the mandate forbids.
    """
    return mandate_breaches(
        session,
        tickers,
        excluded_sectors=[
            YAHOO_SECTOR_BY_GICS[s] for s in constraint_set.esg.exclusions
        ],
        no_complex=constraint_set.universe_filters.no_complex,
        no_leverage=constraint_set.universe_filters.no_leverage,
    )


# A weight above this is a real holding; the optimizer's solver noise sits far
# below it and must not trigger a mandate screen or a cap breach.
HELD_TOL = 1e-6


def held_tickers(weights: Mapping[str, float]) -> list[str]:
    """The names ``weights`` actually holds, ignoring solver noise around zero."""
    return [ticker for ticker, weight in weights.items() if abs(weight) > HELD_TOL]


__all__ = [
    "HELD_TOL",
    "YAHOO_SECTOR_BY_GICS",
    "constraint_set_breaches",
    "held_tickers",
    "mandate_breaches",
]
