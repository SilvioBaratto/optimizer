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

export type MacroRegime = 'expansion' | 'slowdown' | 'recession' | 'recovery';

export type HmmState = 'low_vol' | 'medium_vol' | 'high_vol';

export interface RegimeDetection {
  date: string;
  state: HmmState;
  probabilities: Record<HmmState, number>;
}

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

// ── Macro / regime API DTOs (from dashboard + macro_regime + macro_calibration)

export interface MacroCalibrationResponse {
  phase: string;
  delta: number;
  tau: number;
  confidence: number;
  rationale: string;
  macroSummary: string;
  blConfig: Record<string, unknown>;
}

export interface RegimeHistoryPoint {
  date: string;
  regime: string;
  bullProb: number;
  bearProb: number;
  sidewaysProb: number;
  volatileProb: number;
}

export interface RegimeHistoryApiResponse {
  points: RegimeHistoryPoint[];
  total: number;
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
