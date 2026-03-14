import type { BusinessCyclePhase, SectorPhaseRow } from '../models/macro-intelligence.model';

export const PHASE_LABELS: Record<BusinessCyclePhase, string> = {
  EARLY_EXPANSION: 'Early Expansion',
  MID_EXPANSION: 'Mid Expansion',
  LATE_EXPANSION: 'Late Expansion',
  RECESSION: 'Recession',
};

export const PHASE_DEFAULTS: Record<BusinessCyclePhase, { delta: number; tau: number }> = {
  EARLY_EXPANSION: { delta: 2.25, tau: 0.05 },
  MID_EXPANSION: { delta: 2.75, tau: 0.025 },
  LATE_EXPANSION: { delta: 3.5, tau: 0.01 },
  RECESSION: { delta: 5.0, tau: 0.05 },
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

export const SECTOR_ROTATION_TABLE: SectorPhaseRow[] = [
  { sector: 'Energy', phases: { EARLY_EXPANSION: 'OW', MID_EXPANSION: 'OW', LATE_EXPANSION: 'N', RECESSION: 'UW' } },
  { sector: 'Materials', phases: { EARLY_EXPANSION: 'OW', MID_EXPANSION: 'N', LATE_EXPANSION: 'UW', RECESSION: 'UW' } },
  { sector: 'Industrials', phases: { EARLY_EXPANSION: 'OW', MID_EXPANSION: 'OW', LATE_EXPANSION: 'N', RECESSION: 'UW' } },
  { sector: 'Consumer Disc.', phases: { EARLY_EXPANSION: 'OW', MID_EXPANSION: 'N', LATE_EXPANSION: 'UW', RECESSION: 'UW' } },
  { sector: 'Consumer Staples', phases: { EARLY_EXPANSION: 'UW', MID_EXPANSION: 'UW', LATE_EXPANSION: 'N', RECESSION: 'OW' } },
  { sector: 'Health Care', phases: { EARLY_EXPANSION: 'UW', MID_EXPANSION: 'N', LATE_EXPANSION: 'OW', RECESSION: 'OW' } },
  { sector: 'Financials', phases: { EARLY_EXPANSION: 'OW', MID_EXPANSION: 'OW', LATE_EXPANSION: 'N', RECESSION: 'UW' } },
  { sector: 'Info Technology', phases: { EARLY_EXPANSION: 'OW', MID_EXPANSION: 'OW', LATE_EXPANSION: 'OW', RECESSION: 'N' } },
  { sector: 'Communication', phases: { EARLY_EXPANSION: 'N', MID_EXPANSION: 'OW', LATE_EXPANSION: 'N', RECESSION: 'UW' } },
  { sector: 'Utilities', phases: { EARLY_EXPANSION: 'UW', MID_EXPANSION: 'UW', LATE_EXPANSION: 'N', RECESSION: 'OW' } },
  { sector: 'Real Estate', phases: { EARLY_EXPANSION: 'OW', MID_EXPANSION: 'N', LATE_EXPANSION: 'UW', RECESSION: 'N' } },
];
