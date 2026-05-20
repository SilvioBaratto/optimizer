"""Drift computation orchestrator — Cycle 2 (issues #780, #781).

Composes Cycle 1's `normalize_live_positions()` with target weights, broker
universe resolution, and a deployable-EUR base to produce a fully populated
`DriftResponse` (holdings, target, drift rows, trade rows, totals,
diagnostics, request_id).
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.schemas.portfolio import (
    BaseToggle,
    DiagnosticEntry,
    DriftDiagnostics,
    DriftResponse,
    DriftRow,
    DriftTotals,
    FlagInstance,
    NormalizedPosition,
    PositionFlag,
    TargetWeight,
    TradeAction,
    TradeRow,
)
from app.services._shared._ticker_lookup import lookup_yf_ticker
from app.services.portfolio.flag_reasons import make_flag_instance
from app.services.portfolio.t212_position_normalizer.normalizer_service import (
    RECONCILIATION_TOLERANCE,
    NormalizationResult,
    normalize_live_positions,
)
from app.services.universe.trading212.client import Trading212Client

_TRADE_EPSILON: float = 1e-6


def compute_drift(
    client: Trading212Client,
    session: Session,
    target_weights: dict[str, float],
    base: BaseToggle,
    *,
    request_id: int = 0,
    commission_rate: float = 0.0,
) -> DriftResponse:
    """Build a fully populated `DriftResponse` from live broker state."""
    norm = normalize_live_positions(client, session)
    universe = _build_broker_universe(client, session)
    drift = _join_holdings_and_target(norm.positions, target_weights, universe)
    deployable = _resolve_deployable_eur(client, base)
    invested_eur = _resolve_invested_eur(client)
    positions_by_ticker = {p.ticker: p for p in norm.positions}
    trades = _compute_trades(drift, deployable, positions_by_ticker, commission_rate)
    return DriftResponse(
        holdings=norm.positions,
        target=_build_target_list(target_weights),
        drift=drift,
        trades=trades,
        totals=_aggregate_totals(norm.positions, drift, deployable),
        diagnostics=_aggregate_diagnostics(norm, drift, invested_eur),
        request_id=request_id,
    )


def _resolve_invested_eur(client: Trading212Client) -> float:
    """Read `invested` cash (independent of `BaseToggle`) for diagnostics."""
    cash = client.get_account_cash() or {}
    return float(cash.get("invested", 0.0))


def _build_broker_universe(
    client: Trading212Client, session: Session
) -> frozenset[str]:
    """Resolve T212 instruments to yfinance tickers (filters unmapped)."""
    instruments = client.get_instruments() or []
    resolved = (
        lookup_yf_ticker(str(inst.get("ticker") or ""), session)
        for inst in instruments
    )
    return frozenset(yf for yf in resolved if yf)


def _join_holdings_and_target(
    positions: list[NormalizedPosition],
    target_weights: dict[str, float],
    broker_universe: frozenset[str],
) -> list[DriftRow]:
    """Full-outer-join holdings against target weights by yfinance ticker."""
    held = {p.ticker: p for p in positions}
    tickers = sorted(set(held) | set(target_weights))
    return [
        _build_row(t, held.get(t), target_weights, broker_universe) for t in tickers
    ]


def _build_row(
    ticker: str,
    position: NormalizedPosition | None,
    target_weights: dict[str, float],
    broker_universe: frozenset[str],
) -> DriftRow:
    target_weight = float(target_weights.get(ticker, 0.0))
    current_weight = position.eur_weight if position else 0.0
    eur_value = position.eur_value if position else 0.0
    flags = _row_flags(ticker, position, target_weight, broker_universe)
    return DriftRow(
        ticker=ticker,
        current_weight=current_weight,
        target_weight=target_weight,
        delta_weight=target_weight - current_weight,
        eur_value=eur_value,
        flags=flags,
    )


def _row_flags(
    ticker: str,
    position: NormalizedPosition | None,
    target_weight: float,
    broker_universe: frozenset[str],
) -> list[FlagInstance]:
    if position is not None:
        return list(position.flags)
    if target_weight > 0.0 and ticker not in broker_universe:
        return [
            make_flag_instance(
                PositionFlag.TARGET_NOT_ON_BROKER, reference=ticker
            )
        ]
    return []


def _build_target_list(target_weights: dict[str, float]) -> list[TargetWeight]:
    return [
        TargetWeight(ticker=t, weight=w)
        for t, w in sorted(target_weights.items())
    ]


def _resolve_deployable_eur(
    client: Trading212Client, base: BaseToggle
) -> float:
    """Read `invested` or `total` from `get_account_cash()` per `BaseToggle`."""
    cash = client.get_account_cash() or {}
    key = "invested" if base is BaseToggle.INVESTED else "total"
    return float(cash.get(key, 0.0))


def _compute_trades(
    drift_rows: list[DriftRow],
    deployable_eur: float,
    positions_by_ticker: dict[str, NormalizedPosition],
    commission_rate: float,
) -> list[TradeRow]:
    """Derive non-HOLD trade rows sorted by descending |delta_eur|."""
    trades = [
        _build_trade(row, deployable_eur, positions_by_ticker, commission_rate)
        for row in drift_rows
    ]
    return sorted(
        (t for t in trades if t.action is not TradeAction.HOLD),
        key=lambda t: abs(t.delta_eur),
        reverse=True,
    )


def _build_trade(
    row: DriftRow,
    deployable_eur: float,
    positions_by_ticker: dict[str, NormalizedPosition],
    commission_rate: float,
) -> TradeRow:
    delta_eur = row.delta_weight * deployable_eur
    action = _classify_action(delta_eur)
    price_eur = _price_eur(positions_by_ticker.get(row.ticker))
    est_shares = delta_eur / price_eur if price_eur > 0.0 else 0.0
    est_cost_eur = abs(delta_eur) * (1.0 + commission_rate)
    return TradeRow(
        ticker=row.ticker,
        action=action,
        delta_eur=delta_eur,
        est_shares=est_shares,
        est_cost_eur=est_cost_eur,
    )


def _classify_action(delta_eur: float) -> TradeAction:
    if delta_eur > _TRADE_EPSILON:
        return TradeAction.BUY
    if delta_eur < -_TRADE_EPSILON:
        return TradeAction.SELL
    return TradeAction.HOLD


def _price_eur(position: NormalizedPosition | None) -> float:
    if position is None or position.qty <= 0.0:
        return 0.0
    return position.eur_value / position.qty


def _aggregate_totals(
    positions: list[NormalizedPosition],
    drift_rows: list[DriftRow],
    deployable_eur: float,
) -> DriftTotals:
    total_holdings = sum(p.eur_value for p in positions)
    total_drift_abs = sum(abs(r.delta_weight) for r in drift_rows)
    buy_eur = sum(
        r.delta_weight * deployable_eur
        for r in drift_rows
        if r.delta_weight > 0.0
    )
    sell_eur = sum(
        -r.delta_weight * deployable_eur
        for r in drift_rows
        if r.delta_weight < 0.0
    )
    return DriftTotals(
        deployable_eur=deployable_eur,
        total_holdings_eur=total_holdings,
        total_drift_abs=total_drift_abs,
        buy_eur=buy_eur,
        sell_eur=sell_eur,
    )


def _aggregate_diagnostics(
    norm: NormalizationResult,
    drift_rows: list[DriftRow],
    invested_eur: float = 0.0,
) -> DriftDiagnostics:
    target_not_on_broker = sum(
        1 for row in drift_rows if PositionFlag.TARGET_NOT_ON_BROKER in row.flags
    )
    sum_eur = sum(p.eur_value for p in norm.positions)
    delta_eur = abs(sum_eur - invested_eur)
    entries = _build_entries(
        norm, drift_rows, sum_eur=sum_eur, invested_eur=invested_eur, delta_eur=delta_eur
    )
    return DriftDiagnostics(
        reconciliation_ok=norm.reconciliation_ok,
        reconciliation_delta_pct=norm.reconciliation_delta_pct,
        unmapped_count=norm.unmapped_count,
        fx_missing_count=norm.fx_missing_count,
        target_not_on_broker_count=target_not_on_broker,
        base_currency=norm.base_currency,
        sum_eur=sum_eur,
        invested_eur=invested_eur,
        delta_eur=delta_eur,
        tolerance_pct=RECONCILIATION_TOLERANCE,
        stale_price_count=norm.stale_price_count,
        entries=entries,
    )


def _build_entries(
    norm: NormalizationResult,
    drift_rows: list[DriftRow],
    *,
    sum_eur: float,
    invested_eur: float,
    delta_eur: float,
) -> list[DiagnosticEntry]:
    entries: list[DiagnosticEntry] = []
    entries.extend(_collect_row_entries(drift_rows))
    entries.extend(_unmapped_dropped_entries(norm.dropped_raw_tickers))
    if not norm.reconciliation_ok:
        entries.append(_reconciliation_entry(sum_eur, invested_eur, delta_eur))
    return entries


def _collect_row_entries(drift_rows: list[DriftRow]) -> list[DiagnosticEntry]:
    return [
        DiagnosticEntry(
            code=flag.code,
            reason=flag.reason,
            reference=flag.reference,
            ticker=row.ticker,
        )
        for row in drift_rows
        for flag in row.flags
    ]


def _unmapped_dropped_entries(dropped: list[str]) -> list[DiagnosticEntry]:
    return [_dropped_entry(ticker) for ticker in dropped]


def _dropped_entry(ticker: str) -> DiagnosticEntry:
    flag = make_flag_instance(PositionFlag.UNMAPPED, reference=ticker)
    return DiagnosticEntry(
        code=flag.code,
        reason=flag.reason,
        reference=flag.reference,
        ticker=ticker,
    )


def _reconciliation_entry(
    sum_eur: float, invested_eur: float, delta_eur: float
) -> DiagnosticEntry:
    reference = (
        f"sum_eur={sum_eur:.2f} invested={invested_eur:.2f} delta={delta_eur:.2f}"
    )
    flag = make_flag_instance(
        PositionFlag.RECONCILIATION_MISMATCH, reference=reference
    )
    return DiagnosticEntry(
        code=flag.code,
        reason=flag.reason,
        reference=reference,
        ticker=None,
    )
