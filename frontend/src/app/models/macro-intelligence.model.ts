export type BusinessCyclePhase =
  | 'EARLY_EXPANSION'
  | 'MID_EXPANSION'
  | 'LATE_EXPANSION'
  | 'CONTRACTION';

export type MacroRegimeLabel = 'Expansionary' | 'Transitional' | 'Contractionary';

export type SectorRotationStance = 'OW' | 'N' | 'UW';

// ---------------------------------------------------------------------------
// Backend wire types — raw shapes returned by the API before service mapping
// ---------------------------------------------------------------------------

export interface BlackLittermanPriorConfig {
  mu_estimator: string;
  risk_aversion: number;
  cov_estimator: string;
}

export interface BlackLittermanBlConfig {
  views: unknown[];
  tau: number;
  prior_config: BlackLittermanPriorConfig;
}

export interface CountryMacroSummaryResponse {
  country: string;
  economic_indicators: Array<{
    id: string;
    country: string;
    last_inflation: number | null;
    inflation_6m: number | null;
    inflation_10y_avg: number | null;
    gdp_growth_6m: number | null;
    earnings_12m: number | null;
    eps_expected_12m: number | null;
    peg_ratio: number | null;
    lt_rate_forecast: number | null;
    reference_date: string | null;
    created_at: string;
    updated_at: string;
  }>;
  te_indicators: Array<{
    id: string;
    country: string;
    indicator_key: string;
    value: number | null;
    previous: number | null;
    unit: string | null;
    reference: string | null;
    raw_name: string | null;
    created_at: string;
    updated_at: string;
  }>;
  bond_yields: Array<{
    id: string;
    country: string;
    maturity: string;
    yield_value: number | null;
    day_change: number | null;
    month_change: number | null;
    year_change: number | null;
    reference_date: string | null;
    created_at: string;
    updated_at: string;
  }>;
}

export interface MacroNewsApiResponse {
  id: string;
  news_id: string;
  title: string | null;
  publisher: string | null;
  link: string | null;
  publish_time: string | null;
  source_ticker: string | null;
  source_query: string | null;
  themes: string | null;
  snippet: string | null;
  full_content: string | null;
  created_at: string;
  updated_at: string;
}

export interface FredObservationApiResponse {
  id: string;
  series_id: string;
  date: string;
  value: number | null;
  created_at: string;
  updated_at: string;
}

export interface MacroNewsThemeApiResponse {
  value: string;
  label: string;
}

// ---------------------------------------------------------------------------
// Domain interfaces — used by components and services
// ---------------------------------------------------------------------------

export interface MacroCalibrationResponse {
  phase: BusinessCyclePhase;
  delta: number;
  tau: number;
  confidence: number;
  rationale: string;
  macro_summary: string;
  timestamp: string;
  bl_config: BlackLittermanBlConfig;
}

export interface FredObservationPoint {
  date: string;
  value: number;
}

export interface MacroNewsItem {
  id: string;
  headline: string;
  snippet: string;
  url: string;
  publisher: string;
  published_at: string;
  theme: string;
}

export interface MacroNewsTheme {
  id: string;
  label: string;
  count: number;
}

export interface SectorPhaseRow {
  sector: string;
  phases: Record<BusinessCyclePhase, SectorRotationStance>;
}

export interface CountryMacroData {
  country: string;
  country_code: string;
  gdp_growth: number;
  inflation: number;
  unemployment: number;
  interest_rate: number;
  yield_10y: number;
  yield_2y: number;
  yield_spread_bps: number;
  yield_sparkline: number[];
}

export interface BondYieldCurvePoint {
  maturity: string;
  yield_pct: number;
}

export interface BondYieldSnapshot {
  country: string;
  date: string;
  curve: BondYieldCurvePoint[];
}

export interface CompositeScorePoint {
  month: string;
  /** Integer score on the [-3, +3] discrete scale. */
  score: number;
}

export interface IndicatorCardConfig {
  label: string;
  seriesId: string;
  bullThreshold: number;
  bearThreshold: number;
  bullAbove: boolean;
  unit: string;
  sparklineDays: number;
}

export interface MacroJobCreateResponse {
  job_id: string;
  status: string;
  message: string;
}

export interface MacroJobProgress {
  job_id: string;
  status: string;
  current: number;
  total: number;
  errors: string[];
  result: unknown | null;
  error: string | null;
}
