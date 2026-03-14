import type { BusinessCyclePhase, SectorPhaseRow, CompositeScorePoint } from '../models/macro-intelligence.model';

export const PHASE_LABELS: Record<BusinessCyclePhase, string> = {
  EARLY_EXPANSION: 'Early Expansion',
  MID_EXPANSION: 'Mid Expansion',
  LATE_EXPANSION: 'Late Expansion',
  CONTRACTION: 'Recession',
};

export const PHASE_DEFAULTS: Record<BusinessCyclePhase, { delta: number; tau: number }> = {
  EARLY_EXPANSION: { delta: 2.25, tau: 0.05 },
  MID_EXPANSION: { delta: 2.75, tau: 0.025 },
  LATE_EXPANSION: { delta: 3.5, tau: 0.01 },
  CONTRACTION: { delta: 5.0, tau: 0.05 },
};

export const COUNTRY_CODE_MAP: Record<string, string> = {
  'USA': 'US',
  'United States': 'US',
  'Germany': 'DE',
  'France': 'FR',
  'UK': 'GB',
  'United Kingdom': 'GB',
};

export const COUNTRY_NAME_MAP: Record<string, string> = {
  'US': 'USA',
  'DE': 'Germany',
  'FR': 'France',
  'GB': 'UK',
};

/** Country names as stored in the database. Use these for API calls. */
export const DB_COUNTRIES = ['USA', 'Germany', 'France', 'UK'] as const;

export const COMPOSITE_SCORE_THRESHOLDS = {
  PMI_BULL: 52,
  PMI_BEAR: 48,
  YIELD_SPREAD_BULL: 1.0,
  YIELD_SPREAD_BEAR: 0,
  HY_OAS_BULL: 350,
  HY_OAS_BEAR: 500,
} as const;

export const COMPOSITE_CHART_AXIS = {
  MIN: -3,
  MAX: 3,
  BULL_THRESHOLD: 2,
  BEAR_THRESHOLD: -2,
} as const;

export const SECTOR_ROTATION_TABLE: SectorPhaseRow[] = [
  { sector: 'Energy', phases: { EARLY_EXPANSION: 'OW', MID_EXPANSION: 'OW', LATE_EXPANSION: 'N', CONTRACTION: 'UW' } },
  { sector: 'Materials', phases: { EARLY_EXPANSION: 'OW', MID_EXPANSION: 'N', LATE_EXPANSION: 'UW', CONTRACTION: 'UW' } },
  { sector: 'Industrials', phases: { EARLY_EXPANSION: 'OW', MID_EXPANSION: 'OW', LATE_EXPANSION: 'N', CONTRACTION: 'UW' } },
  { sector: 'Consumer Disc.', phases: { EARLY_EXPANSION: 'OW', MID_EXPANSION: 'N', LATE_EXPANSION: 'UW', CONTRACTION: 'UW' } },
  { sector: 'Consumer Staples', phases: { EARLY_EXPANSION: 'UW', MID_EXPANSION: 'UW', LATE_EXPANSION: 'N', CONTRACTION: 'OW' } },
  { sector: 'Health Care', phases: { EARLY_EXPANSION: 'UW', MID_EXPANSION: 'N', LATE_EXPANSION: 'OW', CONTRACTION: 'OW' } },
  { sector: 'Financials', phases: { EARLY_EXPANSION: 'OW', MID_EXPANSION: 'OW', LATE_EXPANSION: 'N', CONTRACTION: 'UW' } },
  { sector: 'Info Technology', phases: { EARLY_EXPANSION: 'OW', MID_EXPANSION: 'OW', LATE_EXPANSION: 'OW', CONTRACTION: 'N' } },
  { sector: 'Communication', phases: { EARLY_EXPANSION: 'N', MID_EXPANSION: 'OW', LATE_EXPANSION: 'N', CONTRACTION: 'UW' } },
  { sector: 'Utilities', phases: { EARLY_EXPANSION: 'UW', MID_EXPANSION: 'UW', LATE_EXPANSION: 'N', CONTRACTION: 'OW' } },
  { sector: 'Real Estate', phases: { EARLY_EXPANSION: 'OW', MID_EXPANSION: 'N', LATE_EXPANSION: 'UW', CONTRACTION: 'N' } },
];
