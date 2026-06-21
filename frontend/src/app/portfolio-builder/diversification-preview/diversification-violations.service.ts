import { Injectable } from '@angular/core';

export type ViolationKind =
  | 'region-concentration'
  | 'sector-concentration'
  | 'hhi'
  | 'top4-concentration'
  | 'healthcare'
  | 'technology'
  | 'absent-sector';

export interface ViolationFlag {
  kind: ViolationKind;
  escalated: boolean;
}

export interface SimpleProfile {
  sector: string;
  region: string;
}

// ── Thresholds ─────────────────────────────────────────────────────────────

const REGION_MAX = 0.60;
const SECTOR_MAX = 0.15;
const HHI_THRESHOLD = 0.12;
const TOP4_THRESHOLD = 0.30;
const HC_MIN = 0.08;
const HC_MAX = 0.12;
const TECH_MIN = 0.10;
const TECH_MAX = 0.12;

const MAJOR_SECTORS: ReadonlySet<string> = new Set([
  'Technology', 'Healthcare', 'Financials', 'Energy',
  'Consumer', 'Industrials', 'Materials', 'Utilities',
]);

const ESCALATED_REGIMES: ReadonlySet<string> = new Set([
  'CONTRACTION', 'stress', 'bear',
]);

// ── Pure helpers ────────────────────────────────────────────────────────────

function bucketWeights(items: string[]): Record<string, number> {
  const w = 1 / items.length;
  return items.reduce<Record<string, number>>((acc, item) => {
    acc[item] = (acc[item] ?? 0) + w;
    return acc;
  }, {});
}

function maxVal(record: Record<string, number>): number {
  const vals = Object.values(record);
  return vals.length === 0 ? 0 : Math.max(...vals);
}

function outOfRange(v: number, min: number, max: number): boolean {
  return v < min || v > max;
}

// ── Service ─────────────────────────────────────────────────────────────────

@Injectable({ providedIn: 'root' })
export class DiversificationViolationsService {
  computeViolations(profiles: SimpleProfile[], regime: string): ViolationFlag[] {
    if (profiles.length === 0) return [];

    const escalated = ESCALATED_REGIMES.has(regime);
    const n = profiles.length;
    const sectorWts = bucketWeights(profiles.map(p => p.sector));
    const regionWts = bucketWeights(profiles.map(p => p.region));

    const result: ViolationFlag[] = [];
    const flag = (kind: ViolationKind) => result.push({ kind, escalated });

    if (maxVal(regionWts) > REGION_MAX) flag('region-concentration');
    if (maxVal(sectorWts) > SECTOR_MAX) flag('sector-concentration');
    if (1 / n >= HHI_THRESHOLD) flag('hhi');
    if (Math.min(4, n) / n > TOP4_THRESHOLD) flag('top4-concentration');
    if (outOfRange(sectorWts['Healthcare'] ?? 0, HC_MIN, HC_MAX)) flag('healthcare');
    if (outOfRange(sectorWts['Technology'] ?? 0, TECH_MIN, TECH_MAX)) flag('technology');
    if ([...MAJOR_SECTORS].some(s => !(s in sectorWts))) flag('absent-sector');

    return result;
  }
}
