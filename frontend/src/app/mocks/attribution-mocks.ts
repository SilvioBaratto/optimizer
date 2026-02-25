import {
  BrinsonAttribution,
  MultiLevelAttribution,
  FactorAttribution,
  HoldingsAttribution,
} from '../models/attribution.model';
import { WEIGHT_DATA } from './mock-data';

// ── Brinson-Fachler Attribution (10 sectors) ──

export const MOCK_BRINSON_ATTRIBUTION: BrinsonAttribution = {
  totalAllocation: 0.0042,
  totalSelection: 0.0128,
  totalInteraction: -0.0018,
  totalActive: 0.0152,
  sectors: [
    { sector: 'Technology', portfolioWeight: 0.205, benchmarkWeight: 0.285, portfolioReturn: 0.142, benchmarkReturn: 0.128, allocationEffect: 0.0010, selectionEffect: 0.0029, interactionEffect: -0.0011, totalEffect: 0.0028 },
    { sector: 'Financial Services', portfolioWeight: 0.125, benchmarkWeight: 0.130, portfolioReturn: 0.098, benchmarkReturn: 0.082, allocationEffect: -0.0001, selectionEffect: 0.0021, interactionEffect: 0.0001, totalEffect: 0.0021 },
    { sector: 'Healthcare', portfolioWeight: 0.140, benchmarkWeight: 0.135, portfolioReturn: 0.072, benchmarkReturn: 0.065, allocationEffect: 0.0000, selectionEffect: 0.0009, interactionEffect: 0.0000, totalEffect: 0.0010 },
    { sector: 'Consumer Defensive', portfolioWeight: 0.095, benchmarkWeight: 0.070, portfolioReturn: 0.054, benchmarkReturn: 0.048, allocationEffect: -0.0008, selectionEffect: 0.0004, interactionEffect: -0.0002, totalEffect: -0.0006 },
    { sector: 'Consumer Cyclical', portfolioWeight: 0.085, benchmarkWeight: 0.105, portfolioReturn: 0.118, benchmarkReturn: 0.108, allocationEffect: 0.0005, selectionEffect: 0.0011, interactionEffect: -0.0002, totalEffect: 0.0014 },
    { sector: 'Industrials', portfolioWeight: 0.105, benchmarkWeight: 0.085, portfolioReturn: 0.088, benchmarkReturn: 0.078, allocationEffect: 0.0002, selectionEffect: 0.0009, interactionEffect: 0.0002, totalEffect: 0.0013 },
    { sector: 'Energy', portfolioWeight: 0.065, benchmarkWeight: 0.045, portfolioReturn: 0.062, benchmarkReturn: 0.055, allocationEffect: -0.0004, selectionEffect: 0.0003, interactionEffect: -0.0001, totalEffect: -0.0002 },
    { sector: 'Utilities', portfolioWeight: 0.060, benchmarkWeight: 0.030, portfolioReturn: 0.042, benchmarkReturn: 0.038, allocationEffect: -0.0012, selectionEffect: 0.0001, interactionEffect: -0.0001, totalEffect: -0.0012 },
    { sector: 'Communication', portfolioWeight: 0.055, benchmarkWeight: 0.085, portfolioReturn: 0.105, benchmarkReturn: 0.095, allocationEffect: 0.0008, selectionEffect: 0.0009, interactionEffect: -0.0003, totalEffect: 0.0014 },
    { sector: 'Real Estate', portfolioWeight: 0.045, benchmarkWeight: 0.030, portfolioReturn: 0.035, benchmarkReturn: 0.028, allocationEffect: -0.0006, selectionEffect: 0.0002, interactionEffect: -0.0001, totalEffect: -0.0005 },
  ],
};

// ── Multi-Level Attribution ──

export const MOCK_MULTI_LEVEL_ATTRIBUTION: MultiLevelAttribution[] = [
  { level: 'Asset Class', name: 'US Equities', contribution: 0.0128, weight: 0.82, returnPct: 0.098 },
  { level: 'Asset Class', name: 'Intl Developed', contribution: 0.0018, weight: 0.12, returnPct: 0.072 },
  { level: 'Asset Class', name: 'Cash', contribution: 0.0006, weight: 0.06, returnPct: 0.042 },
  { level: 'Sector', name: 'Technology', contribution: 0.0058, weight: 0.205, returnPct: 0.142 },
  { level: 'Sector', name: 'Financial Services', contribution: 0.0024, weight: 0.125, returnPct: 0.098 },
];

// ── Factor Attribution ──

export const MOCK_FACTOR_ATTRIBUTION: FactorAttribution[] = [
  { factor: 'Market', exposure: 0.95, factorReturn: 0.082, contribution: 0.0779, cumulative: 0.0779 },
  { factor: 'Quality', exposure: 0.31, factorReturn: 0.045, contribution: 0.0140, cumulative: 0.0919 },
  { factor: 'Momentum', exposure: 0.22, factorReturn: 0.038, contribution: 0.0084, cumulative: 0.1003 },
  { factor: 'Value', exposure: 0.08, factorReturn: 0.025, contribution: 0.0020, cumulative: 0.1023 },
  { factor: 'Size', exposure: -0.15, factorReturn: 0.018, contribution: -0.0027, cumulative: 0.0996 },
  { factor: 'Residual', exposure: 1.0, factorReturn: -0.0014, contribution: -0.0014, cumulative: 0.0982 },
];

// ── Holdings Attribution (top/bottom contributors) ──

const holdingsRng = (() => {
  let s = 600;
  return () => {
    s = (s * 16807) % 2147483647;
    return (s - 1) / 2147483646;
  };
})();

export const MOCK_HOLDINGS_ATTRIBUTION: HoldingsAttribution[] = WEIGHT_DATA.slice(0, 30)
  .map((w) => {
    const returnPct = (holdingsRng() - 0.4) * 0.3;
    return {
      ticker: w.ticker,
      name: w.name,
      sector: w.sector,
      weight: w.weight,
      returnPct: Math.round(returnPct * 10000) / 10000,
      contribution: Math.round(w.weight * returnPct * 10000) / 10000,
    };
  })
  .sort((a, b) => b.contribution - a.contribution);

export const MOCK_TOP_CONTRIBUTORS: HoldingsAttribution[] = MOCK_HOLDINGS_ATTRIBUTION.slice(0, 10);
export const MOCK_BOTTOM_CONTRIBUTORS: HoldingsAttribution[] = MOCK_HOLDINGS_ATTRIBUTION.slice(-10).reverse();
