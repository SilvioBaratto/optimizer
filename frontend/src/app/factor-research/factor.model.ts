export type FactorGroupType =
  | 'value'
  | 'profitability'
  | 'investment'
  | 'momentum'
  | 'low_risk'
  | 'liquidity'
  | 'dividend'
  | 'sentiment'
  | 'ownership';

export type FactorType =
  | 'book_to_price'
  | 'earnings_yield'
  | 'cash_flow_yield'
  | 'sales_to_price'
  | 'ebitda_to_ev'
  | 'gross_profitability'
  | 'roe'
  | 'operating_margin'
  | 'profit_margin'
  | 'asset_growth'
  | 'momentum_12_1'
  | 'volatility'
  | 'beta'
  | 'amihud_illiquidity'
  | 'dividend_yield'
  | 'recommendation_change'
  | 'net_insider_buying';

/** Ordered, iterable list of every `FactorType` literal. Single source of truth
 * for UI dropdowns (see `exposure-constraints-panel`). */
export const FACTOR_TYPES: readonly FactorType[] = [
  'book_to_price',
  'earnings_yield',
  'cash_flow_yield',
  'sales_to_price',
  'ebitda_to_ev',
  'gross_profitability',
  'roe',
  'operating_margin',
  'profit_margin',
  'asset_growth',
  'momentum_12_1',
  'volatility',
  'beta',
  'amihud_illiquidity',
  'dividend_yield',
  'recommendation_change',
  'net_insider_buying',
] as const;

export type MacroRegime = 'expansion' | 'slowdown' | 'recession' | 'recovery';

export interface TAASignal {
  factor: FactorGroupType;
  currentWeight: number;
  tiltedWeight: number;
  tiltReason: string;
  regime: MacroRegime;
}

export interface FactorReturnSeries {
  factor: FactorType;
  group: FactorGroupType;
  points: { date: string; cumReturn: number }[];
}

export interface CMASet {
  label: string;
  horizon: string;
  assets: {
    ticker: string;
    expectedReturn: number;
    expectedVol: number;
  }[];
}

export interface ScreenerFilter {
  factor: FactorType;
  operator: 'gt' | 'lt' | 'between';
  value: number;
  value2?: number;
}

export interface FactorICReport {
  factor: FactorType;
  group: FactorGroupType;
  ic: number;
  icir: number;
  tStat: number;
  pValue: number;
  vif: number;
  significant: boolean;
}

// ── Backend API DTOs (mirror api/app/schemas/factors.py) ────────────────────

export interface FactorComputeRequest {
  tickers: string[];
  start_date: string;
  end_date: string;
  factor_config?: Record<string, unknown> | null;
}

export interface FactorComputeAsyncResponse {
  job_id: string;
  status: string;
  message: string;
}

export interface FactorComputeProgress {
  job_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  current: number;
  total: number;
  errors: string[];
  result: Record<string, unknown> | null;
  error: string | null;
}

export interface FactorValidateRequest {
  tickers: string[];
  start_date: string;
  end_date: string;
  factor_type: string;
  validation_type?: 'in_sample' | 'out_of_sample';
}

export interface FactorValidateResponse {
  report_date: string;
  factor_type: string | null;
  validation_type: string;
  ic_mean: number | null;
  ic_std: number | null;
  icir: number | null;
  t_stat: number | null;
  p_value: number | null;
  vif: number | null;
  details: Record<string, unknown> | null;
}

export type CompositeMethod =
  | 'equal_weight'
  | 'ic_weighted'
  | 'icir_weighted'
  | 'ridge_weighted'
  | 'gbt_weighted';

export interface FactorScoreRequest {
  tickers: string[];
  score_date: string;
  composite_method: CompositeMethod;
  training_start_date?: string;
  training_end_date?: string;
  group_weights?: Record<string, number>;
}

export interface FactorScoreApiResponse {
  score_date: string;
  scores: Record<string, number>;
  group_contributions: Record<string, number>;
}

// Persisted factor-score row — wire shape mirrors FactorScoreResponse in
// api/app/schemas/factors/factors.py (CamelCaseModel). Typed read-response for
// the contract-parity check; one row of GET /factors/scores.
export interface FactorScoreDto {
  id: string;
  ticker: string;
  factorType: string;
  factorGroup: string;
  scoreDate: string;
  rawScore: number;
  standardizedScore: number;
  compositeScore: number;
  createdAt: string;
  updatedAt: string;
}

export type Quintile = 1 | 2 | 3 | 4 | 5;

export interface ScoreRow {
  ticker: string;
  score: number;
  quintile: Quintile;
}

/**
 * Pure mapping from `FactorScoreApiResponse.scores` to sortable score rows.
 * Assigns the top 20% by descending rank to quintile 1 and the bottom 20%
 * to quintile 5. Ties break on the original map iteration order so callers
 * get deterministic rows. No signals, no side effects.
 */
export function computeQuintiles(
  scores: Record<string, number>,
): ScoreRow[] {
  const entries = Object.entries(scores);
  if (entries.length === 0) return [];
  const ranked = [...entries].sort((a, b) => b[1] - a[1]);
  const n = ranked.length;
  return ranked.map(([ticker, score], index) => ({
    ticker,
    score,
    quintile: quintileForRank(index, n),
  }));
}

function quintileForRank(rankIndex: number, total: number): Quintile {
  // rankIndex is 0-based from the top (highest score). Bucket size = total/5.
  const bucket = Math.min(4, Math.floor((rankIndex * 5) / total));
  return (bucket + 1) as Quintile;
}

export type SelectionMethod = 'fixed_count' | 'quantile';

export interface FactorSelectRequest {
  tickers: string[];
  start_date: string;
  end_date: string;
  current_members?: string[];
  method?: SelectionMethod;
  target_count?: number;
  target_quantile?: number;
  buffer_fraction?: number;
  sector_balance?: boolean;
}

export interface FactorSelectApiResponse {
  selected_tickers: string[];
  count: number;
  turnover: number | null;
  buffer_zone: {
    entered: string[];
    exited: string[];
  };
}

export type SelectStatus = 'Held' | 'Entered' | 'Exited' | 'Selected';

export interface SelectRow {
  ticker: string;
  status: SelectStatus;
}

/**
 * Pure derivation of per-ticker selection status from a select response and
 * optional prior members. Covers the union of `selected_tickers` ∪
 * `buffer_zone.exited` so exited tickers stay visible for audit.
 *
 * Status rules:
 *   - `Held`:     ticker ∈ selected_tickers AND ticker ∈ current_members
 *   - `Entered`:  ticker ∈ selected_tickers AND ticker ∈ buffer_zone.entered
 *   - `Exited`:   ticker ∈ buffer_zone.exited AND ticker ∉ selected_tickers
 *   - `Selected`: default for any remaining selected ticker
 *
 * Rows are sorted by status (Held → Entered → Selected → Exited) then ticker
 * ascending, which matches the panel's default-sort requirement.
 */
export function deriveSelectRows(
  response: FactorSelectApiResponse | null,
  currentMembers: readonly string[] = [],
): SelectRow[] {
  if (response === null) return [];
  const selected = new Set(response.selected_tickers);
  const entered = new Set(response.buffer_zone.entered);
  const exited = new Set(response.buffer_zone.exited);
  const members = new Set(currentMembers);

  const union = new Set<string>([...selected, ...exited]);
  const rows: SelectRow[] = [];
  for (const ticker of union) {
    rows.push({ ticker, status: classifyStatus(ticker, selected, entered, exited, members) });
  }
  return rows.sort(compareRows);
}

function classifyStatus(
  ticker: string,
  selected: Set<string>,
  entered: Set<string>,
  exited: Set<string>,
  members: Set<string>,
): SelectStatus {
  if (selected.has(ticker) && members.has(ticker)) return 'Held';
  if (selected.has(ticker) && entered.has(ticker)) return 'Entered';
  if (exited.has(ticker) && !selected.has(ticker)) return 'Exited';
  return 'Selected';
}

const STATUS_ORDER: Record<SelectStatus, number> = {
  Held: 0,
  Entered: 1,
  Selected: 2,
  Exited: 3,
};

function compareRows(a: SelectRow, b: SelectRow): number {
  const byStatus = STATUS_ORDER[a.status] - STATUS_ORDER[b.status];
  if (byStatus !== 0) return byStatus;
  return a.ticker.localeCompare(b.ticker);
}

export interface FactorExposureConstraintsRequest {
  tickers: string[];
  start_date: string;
  end_date: string;
  bounds: [number, number] | Record<string, [number, number]>;
}

export interface FactorExposureConstraintsApiResponse {
  left_inequality: number[][];
  right_inequality: number[];
}

export interface FactorQuintileSpreadRequest {
  tickers: string[];
  factor_name: string;
  start_date: string;
  end_date: string;
  n_quantiles?: number;
}

export interface FactorQuintileSpreadApiResponse {
  quintile_cumulative_returns: Record<string, number[]>;
  spread_cumulative_return: number[];
  annualized_spread: number;
}

export interface FactorRegimeTiltRequest {
  group_weights: Record<string, number>;
  enable?: boolean;
  max_tilt_multiplier?: number;
  min_post_tilt_weight?: number;
}

export interface FactorRegimeTiltApiResponse {
  regime: string;
  tilted_weights: Record<string, number>;
  tilt_multipliers: Record<string, number>;
}

export interface TradingEconomicsObservation {
  id: string;
  country: string;
  indicator_key: string;
  date: string;
  value: number | null;
  created_at: string;
  updated_at: string;
}
