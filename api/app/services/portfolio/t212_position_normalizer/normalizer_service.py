"""T212 portfolio normalizer orchestrator (issue #777, extended by #795).

Composes `Trading212Client` + `lookup_yf_ticker` + `T212PositionMapper` +
`FxRateProvider` to produce a typed `NormalizationResult`. Runs a ±1.5%
reconciliation gate against the account's invested cash; never raises,
always returns flags.

Cycle 4 (#795) emits `FlagInstance` (via `make_flag_instance`) instead of
bare `PositionFlag` enums and adds a `stale_price_count` aggregate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.schemas.portfolio import NormalizedPosition, PositionFlag
from app.services._shared._ticker_lookup import lookup_yf_ticker
from app.services.portfolio.flag_reasons import make_flag_instance
from app.services.portfolio.t212_position_normalizer._fx_rate_provider import (
    FxRateProvider,
)
from app.services.portfolio.t212_position_normalizer._position_mapper import (
    LastTickLookup,
    T212PositionMapper,
)
from app.services.portfolio.t212_position_normalizer.freshness_config import (
    FreshnessConfig,
)
from app.services.universe.trading212.client import Trading212Client

logger = logging.getLogger(__name__)

RECONCILIATION_TOLERANCE: float = 0.015


@dataclass(frozen=True)
class NormalizationResult:
    """Output of the live-position normalization orchestrator."""

    positions: list[NormalizedPosition]
    reconciliation_ok: bool
    reconciliation_delta_pct: float | None
    base_currency: str
    unmapped_count: int
    fx_missing_count: int
    stale_price_count: int = 0
    dropped_raw_tickers: list[str] = field(default_factory=list)


def normalize_live_positions(
    client: Trading212Client,
    session: Session,
    *,
    as_of_date: date | None = None,
    fx_provider: FxRateProvider | None = None,
    mapper: T212PositionMapper | None = None,
    freshness_config: FreshnessConfig | None = None,
    last_tick_lookup: LastTickLookup | None = None,
) -> NormalizationResult:
    """Fetch, normalize, weight, and reconcile T212 live positions."""
    effective_date = as_of_date or date.today()
    base_currency, invested = _fetch_account_context(client)
    fx = fx_provider or FxRateProvider(base_currency=base_currency)
    pos_mapper = mapper or _build_default_mapper(
        fx, base_currency, freshness_config, last_tick_lookup
    )

    raw_positions = client.get_portfolio_positions()
    mapped, dropped = _map_all_positions(
        raw_positions, pos_mapper, session, effective_date
    )
    if not mapped:
        return _empty_result(base_currency, dropped)

    weighted = _assign_weights(mapped)
    reconciled, recon_ok, delta_pct = _run_reconciliation_gate(
        weighted, invested, base_currency
    )

    unmapped_count = _count_flag(reconciled, PositionFlag.UNMAPPED)
    fx_missing_count = _count_flag(reconciled, PositionFlag.FX_MISSING)
    stale_price_count = _count_flag(reconciled, PositionFlag.STALE_PRICE)
    logger.info(
        "normalize_live_positions base=%s positions=%d unmapped=%d "
        "fx_missing=%d stale_price=%d delta_pct=%s",
        base_currency,
        len(reconciled),
        unmapped_count,
        fx_missing_count,
        stale_price_count,
        delta_pct,
    )

    return NormalizationResult(
        positions=reconciled,
        reconciliation_ok=recon_ok,
        reconciliation_delta_pct=delta_pct,
        base_currency=base_currency,
        unmapped_count=unmapped_count,
        fx_missing_count=fx_missing_count,
        stale_price_count=stale_price_count,
        dropped_raw_tickers=dropped,
    )


def _build_default_mapper(
    fx: FxRateProvider,
    base_currency: str,
    freshness_config: FreshnessConfig | None,
    last_tick_lookup: LastTickLookup | None,
) -> T212PositionMapper:
    """Build the default mapper.

    Time-based stale check is opt-in: only wired when the caller passes
    `freshness_config` or `last_tick_lookup` explicitly. Default behaviour
    preserves Cycle-1 null/zero stale-price semantics, leaving the new
    time-based path to be wired in via `drift_service` (issue #796).
    """
    return T212PositionMapper(
        fx_provider=fx,
        base_currency=base_currency,
        freshness_config=freshness_config,
        last_tick_lookup=last_tick_lookup,
    )


def make_db_last_tick_lookup(session: Session) -> LastTickLookup:
    """Closure: yfinance ticker → max(price_history.date) as datetime."""
    from app.models.market_data.yfinance_data import PriceHistory
    from app.models.universe.universe import Instrument

    def _lookup(yf_ticker: str) -> datetime | None:
        if not yf_ticker:
            return None
        stmt = (
            select(PriceHistory.date)
            .join(Instrument, Instrument.id == PriceHistory.instrument_id)
            .where(Instrument.yfinance_ticker == yf_ticker)
            .order_by(PriceHistory.date.desc())
            .limit(1)
        )
        latest: date | None = session.execute(stmt).scalar_one_or_none()
        if latest is None:
            return None
        return datetime.combine(latest, datetime.min.time())

    return _lookup


def _fetch_account_context(client: Trading212Client) -> tuple[str, float]:
    info = client.get_account_info() or {}
    cash = client.get_account_cash() or {}
    base = str(info.get("currencyCode") or "EUR").upper()
    invested = float(cash.get("invested") or 0.0)
    return base, invested


def _map_all_positions(
    raw_positions: list[dict[str, object]],
    pos_mapper: T212PositionMapper,
    session: Session,
    as_of: date,
) -> tuple[list[NormalizedPosition], list[str]]:
    out: list[NormalizedPosition] = []
    dropped: list[str] = []
    for raw in raw_positions:
        raw_ticker = str(raw.get("ticker") or "")
        yf = lookup_yf_ticker(raw_ticker, session) if raw_ticker else None
        try:
            out.append(
                pos_mapper.map_position(raw, yf_ticker=yf, as_of_date=as_of)
            )
        except ValueError as exc:
            logger.warning("Skipping malformed T212 position: %s", exc)
            dropped.append(raw_ticker)
    return out, dropped


def _assign_weights(
    positions: list[NormalizedPosition],
) -> list[NormalizedPosition]:
    total = sum(p.eur_value for p in positions)
    if total <= 0.0:
        return positions
    return [
        p.model_copy(update={"eur_weight": p.eur_value / total}) for p in positions
    ]


def _run_reconciliation_gate(
    positions: list[NormalizedPosition], invested: float, base_currency: str
) -> tuple[list[NormalizedPosition], bool, float | None]:
    if base_currency != "EUR":
        logger.warning(
            "Reconciliation skipped: account base is %s, not EUR", base_currency
        )
        return positions, True, None
    if invested <= 0.0:
        return positions, True, 0.0

    total = sum(p.eur_value for p in positions)
    signed_delta = (total - invested) / invested
    delta_pct = abs(signed_delta)
    if delta_pct <= RECONCILIATION_TOLERANCE:
        return positions, True, delta_pct

    reference = _format_signed_pct(signed_delta)
    flagged = [
        _attach_reconciliation_flag(p, reference) for p in positions
    ]
    return flagged, False, delta_pct


def _format_signed_pct(signed_delta: float) -> str:
    return f"{signed_delta * 100:+.1f}%"


def _attach_reconciliation_flag(
    position: NormalizedPosition, reference: str
) -> NormalizedPosition:
    if PositionFlag.RECONCILIATION_MISMATCH in position.flags:
        return position
    new_flag = make_flag_instance(
        PositionFlag.RECONCILIATION_MISMATCH, reference=reference
    )
    return position.model_copy(update={"flags": [*position.flags, new_flag]})


def _count_flag(
    positions: list[NormalizedPosition], flag: PositionFlag
) -> int:
    return sum(1 for p in positions if flag in p.flags)


def _empty_result(
    base_currency: str, dropped: list[str] | None = None
) -> NormalizationResult:
    return NormalizationResult(
        positions=[],
        reconciliation_ok=True,
        reconciliation_delta_pct=0.0,
        base_currency=base_currency,
        unmapped_count=0,
        fx_missing_count=0,
        stale_price_count=0,
        dropped_raw_tickers=dropped or [],
    )
