import type {
  AssetWithMetadata,
  BusinessCyclePhase,
  DiversificationFlag,
  DiversificationFlagRule,
  DiversificationReport,
  FlagSeverity,
} from './diversification.model';

export type {
  AssetWithMetadata,
  DiversificationFlag,
  DiversificationFlagRule,
  DiversificationReport,
  FlagSeverity,
} from './diversification.model';

// ── Thresholds ────────────────────────────────────────────────────────────

const REGION_MAX = 0.60;
const SECTOR_MAX_PCT = 0.15;
const SECTOR_MAX_HHI = 0.12;
const TOP4_MAX = 0.30;
const HC_MIN = 0.08;
const HC_MAX = 0.12;
const TECH_MIN = 0.10;
const TECH_MAX = 0.12;

// ── Sector reference tables ───────────────────────────────────────────────

const MAJOR_SECTORS: readonly string[] = [
  'Technology',
  'Healthcare',
  'Financials',
  'Industrials',
  'Consumer Discretionary',
  'Consumer Staples',
  'Energy',
  'Utilities',
];

const CYCLICAL_FLAG_RULES = new Set<DiversificationFlagRule>([
  'sector_pct',
  'sector_hhi',
  'technology',
]);

// ── Country → Region (checklist regions only) ─────────────────────────────

const COUNTRY_REGION_MAP: Readonly<Record<string, string>> = {
  'United States': 'North America',
  'Canada': 'North America',
  'United Kingdom': 'Europe',
  'Germany': 'Europe',
  'France': 'Europe',
  'Netherlands': 'Europe',
  'Switzerland': 'Europe',
  'Sweden': 'Europe',
  'Spain': 'Europe',
  'Italy': 'Europe',
  'Denmark': 'Europe',
  'Norway': 'Europe',
  'Finland': 'Europe',
  'Belgium': 'Europe',
  'Japan': 'Asia Pacific',
  'Australia': 'Asia Pacific',
  'Hong Kong': 'Asia Pacific',
  'Singapore': 'Asia Pacific',
  'South Korea': 'Asia Pacific',
  'New Zealand': 'Asia Pacific',
  'China': 'Emerging Markets',
  'India': 'Emerging Markets',
  'Brazil': 'Emerging Markets',
  'Taiwan': 'Emerging Markets',
  'South Africa': 'Emerging Markets',
  'Mexico': 'Emerging Markets',
  'Indonesia': 'Emerging Markets',
  'Thailand': 'Emerging Markets',
  'Malaysia': 'Emerging Markets',
  'Poland': 'Emerging Markets',
  'Turkey': 'Emerging Markets',
  'Saudi Arabia': 'Emerging Markets',
};

export function mapCountryToRegion(country: string | null | undefined): string {
  return COUNTRY_REGION_MAP[country ?? ''] ?? 'Other';
}

// ── Computation primitives ────────────────────────────────────────────────

function breakdown(assets: AssetWithMetadata[], key: keyof AssetWithMetadata): Record<string, number> {
  if (assets.length === 0) return {};
  const weight = 1 / assets.length;
  return assets.reduce<Record<string, number>>((acc, a) => {
    acc[a[key]] = (acc[a[key]] ?? 0) + weight;
    return acc;
  }, {});
}

function computeHhi(weights: Record<string, number>): number {
  return Object.values(weights).reduce((sum, w) => sum + w * w, 0);
}

function computeTop4Weight(n: number): number {
  return Math.min(4, n) / n;
}

function isOutOfRange(value: number, min: number, max: number): boolean {
  return value < min || value > max;
}

// ── Regime severity ───────────────────────────────────────────────────────

function severityFor(rule: DiversificationFlagRule, regime: BusinessCyclePhase | null | undefined): FlagSeverity {
  return regime === 'CONTRACTION' && CYCLICAL_FLAG_RULES.has(rule) ? 'danger' : 'warning';
}

function makeFlag(
  rule: DiversificationFlagRule,
  breached: boolean,
  regime: BusinessCyclePhase | null | undefined,
): DiversificationFlag {
  return { rule, breached, severity: breached ? severityFor(rule, regime) : 'warning' };
}

// ── Flag builders ─────────────────────────────────────────────────────────

function regionFlag(regionBd: Record<string, number>, regime: BusinessCyclePhase | null | undefined): DiversificationFlag {
  return makeFlag('region', Object.values(regionBd).some(w => w > REGION_MAX), regime);
}

function sectorPctFlag(sectorBd: Record<string, number>, regime: BusinessCyclePhase | null | undefined): DiversificationFlag {
  return makeFlag('sector_pct', Object.values(sectorBd).some(w => w > SECTOR_MAX_PCT), regime);
}

function sectorHhiFlag(hhiValue: number, regime: BusinessCyclePhase | null | undefined): DiversificationFlag {
  return makeFlag('sector_hhi', hhiValue > SECTOR_MAX_HHI, regime);
}

function top4Flag(top4: number, regime: BusinessCyclePhase | null | undefined): DiversificationFlag {
  return makeFlag('top4', top4 > TOP4_MAX, regime);
}

function healthcareFlag(sectorBd: Record<string, number>, regime: BusinessCyclePhase | null | undefined): DiversificationFlag {
  return makeFlag('healthcare', isOutOfRange(sectorBd['Healthcare'] ?? 0, HC_MIN, HC_MAX), regime);
}

function technologyFlag(sectorBd: Record<string, number>, regime: BusinessCyclePhase | null | undefined): DiversificationFlag {
  return makeFlag('technology', isOutOfRange(sectorBd['Technology'] ?? 0, TECH_MIN, TECH_MAX), regime);
}

function absentSectorFlag(sectorBd: Record<string, number>, regime: BusinessCyclePhase | null | undefined): DiversificationFlag {
  return makeFlag('absent_sector', MAJOR_SECTORS.some(s => !(s in sectorBd)), regime);
}

// ── Public API ────────────────────────────────────────────────────────────

export function computeDiversification(
  assets: AssetWithMetadata[],
  regime?: BusinessCyclePhase | null,
): DiversificationReport {
  const n = assets.length;
  const sectorBd = breakdown(assets, 'sector');
  const regionBd = breakdown(assets, 'region');
  const assetClassBd = breakdown(assets, 'assetClass');
  const hhiValue = computeHhi(sectorBd);
  const top4 = n > 0 ? computeTop4Weight(n) : 0;

  return {
    hhi: hhiValue,
    top4Weight: top4,
    sectorBreakdown: sectorBd,
    regionBreakdown: regionBd,
    assetClassBreakdown: assetClassBd,
    flags: [
      regionFlag(regionBd, regime),
      sectorPctFlag(sectorBd, regime),
      sectorHhiFlag(hhiValue, regime),
      top4Flag(top4, regime),
      healthcareFlag(sectorBd, regime),
      technologyFlag(sectorBd, regime),
      absentSectorFlag(sectorBd, regime),
    ],
  };
}
