import type { BusinessCyclePhase } from '../core/models/macro-intelligence.model';

export type { BusinessCyclePhase };

// ── Input ─────────────────────────────────────────────────────────────────

export interface AssetWithMetadata {
  sector: string;
  region: string;
  assetClass: string;
}

// ── Output ────────────────────────────────────────────────────────────────

export type DiversificationFlagRule =
  | 'region'
  | 'sector_pct'
  | 'sector_hhi'
  | 'top4'
  | 'healthcare'
  | 'technology'
  | 'absent_sector';

export type FlagSeverity = 'warning' | 'danger';

export interface DiversificationFlag {
  rule: DiversificationFlagRule;
  breached: boolean;
  severity: FlagSeverity;
}

export interface DiversificationReport {
  hhi: number;
  top4Weight: number;
  flags: DiversificationFlag[];
  sectorBreakdown: Record<string, number>;
  regionBreakdown: Record<string, number>;
  assetClassBreakdown: Record<string, number>;
}
