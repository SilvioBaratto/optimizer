import {
  UniverseTicker,
  Constraint,
  IPS,
  WeightAssignment,
} from '../models/portfolio-builder.model';
import { WEIGHT_DATA, MOCK_INSTRUMENTS } from './mock-data';

// ── Universe Tickers (100+ tickers) ──

const sectorMarketCaps: Record<string, number> = {
  Technology: 3_000_000_000_000,
  'Financial Services': 600_000_000_000,
  Healthcare: 450_000_000_000,
  'Consumer Defensive': 350_000_000_000,
  'Consumer Cyclical': 800_000_000_000,
  Industrials: 300_000_000_000,
  Energy: 400_000_000_000,
  Utilities: 80_000_000_000,
  Communication: 200_000_000_000,
  'Real Estate': 50_000_000_000,
};

export const MOCK_UNIVERSE_TICKERS: UniverseTicker[] = WEIGHT_DATA.map((w) => ({
  ticker: w.ticker,
  name: w.name,
  sector: w.sector,
  marketCap: sectorMarketCaps[w.sector] ?? 100_000_000_000,
  weight: w.weight,
  selected: true,
}));

// Add extra tickers to exceed 100
const extraTickers: UniverseTicker[] = [
  { ticker: 'SHEL', name: 'Shell plc', sector: 'Energy', marketCap: 220_000_000_000, weight: 0, selected: false },
  { ticker: '7203.T', name: 'Toyota Motor Corp.', sector: 'Consumer Cyclical', marketCap: 280_000_000_000, weight: 0, selected: false },
  { ticker: 'NESN.SW', name: 'Nestle S.A.', sector: 'Consumer Defensive', marketCap: 260_000_000_000, weight: 0, selected: false },
  { ticker: 'MC.PA', name: 'LVMH Moet Hennessy', sector: 'Consumer Cyclical', marketCap: 350_000_000_000, weight: 0, selected: false },
  { ticker: 'SAP.DE', name: 'SAP SE', sector: 'Technology', marketCap: 250_000_000_000, weight: 0, selected: false },
  { ticker: 'ASML', name: 'ASML Holding', sector: 'Technology', marketCap: 290_000_000_000, weight: 0, selected: false },
  { ticker: 'NVO', name: 'Novo Nordisk', sector: 'Healthcare', marketCap: 420_000_000_000, weight: 0, selected: false },
  { ticker: 'AZN', name: 'AstraZeneca', sector: 'Healthcare', marketCap: 210_000_000_000, weight: 0, selected: false },
  { ticker: 'TM', name: 'Toyota Motor ADR', sector: 'Consumer Cyclical', marketCap: 280_000_000_000, weight: 0, selected: false },
  { ticker: 'SNY', name: 'Sanofi', sector: 'Healthcare', marketCap: 130_000_000_000, weight: 0, selected: false },
  { ticker: 'RY', name: 'Royal Bank of Canada', sector: 'Financial Services', marketCap: 150_000_000_000, weight: 0, selected: false },
  { ticker: 'BHP', name: 'BHP Group', sector: 'Industrials', marketCap: 140_000_000_000, weight: 0, selected: false },
  { ticker: 'UL', name: 'Unilever plc', sector: 'Consumer Defensive', marketCap: 120_000_000_000, weight: 0, selected: false },
  { ticker: 'TTE', name: 'TotalEnergies', sector: 'Energy', marketCap: 150_000_000_000, weight: 0, selected: false },
  { ticker: 'HSBC', name: 'HSBC Holdings', sector: 'Financial Services', marketCap: 160_000_000_000, weight: 0, selected: false },
];

export const MOCK_FULL_UNIVERSE: UniverseTicker[] = [
  ...MOCK_UNIVERSE_TICKERS,
  ...extraTickers,
];

// ── Constraint Presets ──

export const MOCK_CONSTRAINTS_CONSERVATIVE: Constraint[] = [
  { id: 'c1', type: 'weight_bounds', label: 'Max single position', min: 0, max: 0.05, enabled: true },
  { id: 'c2', type: 'sector_bounds', label: 'Max sector weight', min: 0, max: 0.25, target: 'all', enabled: true },
  { id: 'c3', type: 'cardinality', label: 'Min positions', value: 30, enabled: true },
  { id: 'c4', type: 'turnover', label: 'Max quarterly turnover', value: 0.15, enabled: true },
  { id: 'c5', type: 'tracking_error', label: 'Max tracking error', value: 0.03, enabled: true },
];

export const MOCK_CONSTRAINTS_MODERATE: Constraint[] = [
  { id: 'c1', type: 'weight_bounds', label: 'Max single position', min: 0, max: 0.08, enabled: true },
  { id: 'c2', type: 'sector_bounds', label: 'Max sector weight', min: 0, max: 0.30, target: 'all', enabled: true },
  { id: 'c3', type: 'cardinality', label: 'Min positions', value: 20, enabled: true },
  { id: 'c4', type: 'turnover', label: 'Max quarterly turnover', value: 0.25, enabled: false },
];

export const MOCK_CONSTRAINTS_AGGRESSIVE: Constraint[] = [
  { id: 'c1', type: 'weight_bounds', label: 'Max single position', min: 0, max: 0.12, enabled: true },
  { id: 'c2', type: 'sector_bounds', label: 'Max sector weight', min: 0, max: 0.40, target: 'all', enabled: true },
  { id: 'c3', type: 'cardinality', label: 'Min positions', value: 15, enabled: true },
];

// ── IPS Presets ──

export const MOCK_IPS_PRESETS: IPS[] = [
  {
    name: 'Conservative Income',
    riskProfile: 'conservative',
    targetReturn: 0.06,
    maxVolatility: 0.10,
    maxDrawdown: -0.08,
    rebalanceFrequency: 'quarterly',
    constraints: MOCK_CONSTRAINTS_CONSERVATIVE,
  },
  {
    name: 'Balanced Growth',
    riskProfile: 'moderate',
    targetReturn: 0.10,
    maxVolatility: 0.16,
    maxDrawdown: -0.15,
    rebalanceFrequency: 'quarterly',
    constraints: MOCK_CONSTRAINTS_MODERATE,
  },
  {
    name: 'Aggressive Growth',
    riskProfile: 'aggressive',
    targetReturn: 0.15,
    maxVolatility: 0.22,
    maxDrawdown: -0.25,
    rebalanceFrequency: 'monthly',
    constraints: MOCK_CONSTRAINTS_AGGRESSIVE,
  },
];

// ── Weight Assignments ──

export const MOCK_WEIGHT_ASSIGNMENTS: WeightAssignment[] = WEIGHT_DATA.slice(0, 30).map((w) => {
  const diff = (Math.random() - 0.5) * 0.01;
  return {
    ticker: w.ticker,
    name: w.name,
    sector: w.sector,
    currentWeight: Math.round((w.weight + diff) * 10000) / 10000,
    targetWeight: w.weight,
    difference: Math.round(-diff * 10000) / 10000,
  };
});
