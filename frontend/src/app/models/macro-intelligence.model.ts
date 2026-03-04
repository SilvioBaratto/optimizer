export type BusinessCyclePhase =
  | 'EARLY_EXPANSION'
  | 'MID_EXPANSION'
  | 'LATE_EXPANSION'
  | 'CONTRACTION';

export type MacroRegimeLabel = 'Expansionary' | 'Transitional' | 'Contractionary';

export type SectorRotationStance = 'OW' | 'N' | 'UW';

export interface MacroCalibrationResponse {
  phase: BusinessCyclePhase;
  delta: number;
  tau: number;
  confidence: number;
  rationale: string;
  macro_summary: string;
  timestamp: string;
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

export interface IndicatorCardConfig {
  label: string;
  seriesId: string;
  bullThreshold: number;
  bearThreshold: number;
  bullAbove: boolean;
  unit: string;
  sparklineDays: number;
}
