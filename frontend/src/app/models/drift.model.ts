export type BaseToggle = 'invested' | 'total';

export type TradeAction = 'buy' | 'sell' | 'hold';

export type PositionFlag =
  | 'unmapped'
  | 'fx_missing'
  | 'stale_price'
  | 'reconciliation_mismatch'
  | 'target_not_on_broker';

export interface NormalizedPosition {
  readonly ticker: string;
  readonly t212_ticker: string;
  readonly qty: number;
  readonly price_native: number;
  readonly currency: string;
  readonly eur_value: number;
  readonly eur_weight: number;
  readonly ppl_eur: number | null;
  readonly flags: readonly PositionFlag[];
}

export interface TargetWeight {
  readonly ticker: string;
  readonly weight: number;
}

export interface DriftRow {
  readonly ticker: string;
  readonly current_weight: number;
  readonly target_weight: number;
  readonly delta_weight: number;
  readonly eur_value: number;
  readonly flags: readonly PositionFlag[];
}

export interface TradeRow {
  readonly ticker: string;
  readonly action: TradeAction;
  readonly delta_eur: number;
  readonly est_shares: number;
  readonly est_cost_eur: number;
}

export interface DriftTotals {
  readonly deployable_eur: number;
  readonly total_holdings_eur: number;
  readonly total_drift_abs: number;
  readonly buy_eur: number;
  readonly sell_eur: number;
}

export interface DriftDiagnostics {
  readonly reconciliation_ok: boolean;
  readonly reconciliation_delta_pct: number | null;
  readonly unmapped_count: number;
  readonly fx_missing_count: number;
  readonly target_not_on_broker_count: number;
  readonly base_currency: string;
}

export interface DriftResponse {
  readonly holdings: readonly NormalizedPosition[];
  readonly target: readonly TargetWeight[];
  readonly drift: readonly DriftRow[];
  readonly trades: readonly TradeRow[];
  readonly totals: DriftTotals;
  readonly diagnostics: DriftDiagnostics;
  readonly request_id: number;
}

export interface BuilderTrade extends TradeRow {
  readonly delta_weight: number;
}
