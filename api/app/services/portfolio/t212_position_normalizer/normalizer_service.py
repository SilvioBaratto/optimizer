"""T212 portfolio normalizer orchestrator (issue #777).

Composes `Trading212Client` + `lookup_yf_ticker` + `T212PositionMapper` +
`FxRateProvider` to produce a typed `NormalizationResult`. Runs a ±1.5%
reconciliation gate against the account's invested cash; never raises,
always returns flags.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

from sqlalchemy.orm import Session

from app.schemas.portfolio import NormalizedPosition, PositionFlag
from app.services._shared._ticker_lookup import lookup_yf_ticker
from app.services.portfolio.t212_position_normalizer._fx_rate_provider import (
    FxRateProvider,
)
from app.services.portfolio.t212_position_normalizer._position_mapper import (
    T212PositionMapper,
)
from app.services.universe.trading212.client import Trading212Client

logger = logging.getLogger(__name__)

_RECONCILIATION_TOLERANCE: float = 0.015


@dataclass(frozen=True)
class NormalizationResult:
    """Output of the live-position normalization orchestrator."""

    positions: list[NormalizedPosition]
    reconciliation_ok: bool
    reconciliation_delta_pct: float | None
    base_currency: str
    unmapped_count: int
    fx_missing_count: int


def normalize_live_positions(
    client: Trading212Client,
    session: Session,
    *,
    as_of_date: date | None = None,
    fx_provider: FxRateProvider | None = None,
    mapper: T212PositionMapper | None = None,
) -> NormalizationResult:
    """Fetch, normalize, weight, and reconcile T212 live positions."""
    effective_date = as_of_date or date.today()
    base_currency, invested = _fetch_account_context(client)
    fx = fx_provider or FxRateProvider(base_currency=base_currency)
    pos_mapper = mapper or T212PositionMapper(
        fx_provider=fx, base_currency=base_currency
    )

    raw_positions = client.get_portfolio_positions()
    mapped = _map_all_positions(raw_positions, pos_mapper, session, effective_date)
    if not mapped:
        return _empty_result(base_currency)

    weighted = _assign_weights(mapped)
    reconciled, recon_ok, delta_pct = _run_reconciliation_gate(
        weighted, invested, base_currency
    )

    unmapped_count = _count_flag(reconciled, PositionFlag.UNMAPPED)
    fx_missing_count = _count_flag(reconciled, PositionFlag.FX_MISSING)
    logger.info(
        "normalize_live_positions base=%s positions=%d unmapped=%d "
        "fx_missing=%d delta_pct=%s",
        base_currency,
        len(reconciled),
        unmapped_count,
        fx_missing_count,
        delta_pct,
    )

    return NormalizationResult(
        positions=reconciled,
        reconciliation_ok=recon_ok,
        reconciliation_delta_pct=delta_pct,
        base_currency=base_currency,
        unmapped_count=unmapped_count,
        fx_missing_count=fx_missing_count,
    )


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
) -> list[NormalizedPosition]:
    out: list[NormalizedPosition] = []
    for raw in raw_positions:
        raw_ticker = str(raw.get("ticker") or "")
        yf = lookup_yf_ticker(raw_ticker, session) if raw_ticker else None
        try:
            out.append(
                pos_mapper.map_position(raw, yf_ticker=yf, as_of_date=as_of)
            )
        except ValueError as exc:
            logger.warning("Skipping malformed T212 position: %s", exc)
    return out


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
    delta_pct = abs(total - invested) / invested
    if delta_pct <= _RECONCILIATION_TOLERANCE:
        return positions, True, delta_pct

    flagged = [_attach_flag(p, PositionFlag.RECONCILIATION_MISMATCH) for p in positions]
    return flagged, False, delta_pct


def _attach_flag(
    position: NormalizedPosition, flag: PositionFlag
) -> NormalizedPosition:
    if flag in position.flags:
        return position
    return position.model_copy(update={"flags": [*position.flags, flag]})


def _count_flag(
    positions: list[NormalizedPosition], flag: PositionFlag
) -> int:
    return sum(1 for p in positions if flag in p.flags)


def _empty_result(base_currency: str) -> NormalizationResult:
    return NormalizationResult(
        positions=[],
        reconciliation_ok=True,
        reconciliation_delta_pct=0.0,
        base_currency=base_currency,
        unmapped_count=0,
        fx_missing_count=0,
    )
