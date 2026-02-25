import {
  RegimeDetection,
  TAASignal,
  FactorReturnSeries,
  CMASet,
  FactorICReport,
  HmmState,
} from '../models/factor.model';
import { generateDailyTimeSeries, seededRandom } from './generators';

// ── HMM Regime Detection (3 states, 5Y history) ──

export const MOCK_REGIME_HISTORY: RegimeDetection[] = (() => {
  const rng = seededRandom(400);
  const entries: RegimeDetection[] = [];
  const start = new Date('2021-03-01');
  const states: HmmState[] = ['low_vol', 'medium_vol', 'high_vol'];
  let currentState = 0;

  for (let i = 0; i < 1260; i++) {
    const d = new Date(start);
    d.setDate(d.getDate() + Math.floor(i * 365 / 252));
    if (d.getDay() === 0 || d.getDay() === 6) continue;

    // Occasional regime transitions
    if (rng() < 0.02) {
      currentState = Math.floor(rng() * 3);
    }

    const probs: Record<HmmState, number> = { low_vol: 0, medium_vol: 0, high_vol: 0 };
    const mainProb = 0.6 + rng() * 0.3;
    probs[states[currentState]] = Math.round(mainProb * 100) / 100;
    const remaining = 1 - probs[states[currentState]];
    const other1 = states[(currentState + 1) % 3];
    const other2 = states[(currentState + 2) % 3];
    const split = rng() * remaining;
    probs[other1] = Math.round(split * 100) / 100;
    probs[other2] = Math.round((remaining - split) * 100) / 100;

    entries.push({
      date: d.toISOString().split('T')[0],
      state: states[currentState],
      probabilities: probs,
    });
  }

  return entries;
})();

// ── TAA Signals ──

export const MOCK_TAA_SIGNALS: TAASignal[] = [
  { factor: 'value', currentWeight: 0.15, tiltedWeight: 0.18, tiltReason: 'Expansion regime favors value recovery', regime: 'expansion' },
  { factor: 'momentum', currentWeight: 0.15, tiltedWeight: 0.12, tiltReason: 'Momentum reversal risk in late cycle', regime: 'expansion' },
  { factor: 'low_risk', currentWeight: 0.10, tiltedWeight: 0.08, tiltReason: 'Reduced defensive need in expansion', regime: 'expansion' },
  { factor: 'profitability', currentWeight: 0.15, tiltedWeight: 0.17, tiltReason: 'Quality earnings premium widening', regime: 'expansion' },
  { factor: 'dividend', currentWeight: 0.10, tiltedWeight: 0.10, tiltReason: 'Neutral — yield spread stable', regime: 'expansion' },
  { factor: 'investment', currentWeight: 0.10, tiltedWeight: 0.12, tiltReason: 'Capex growth positive in expansion', regime: 'expansion' },
];

// ── Factor Return Series (6 factors, 5Y cumulative) ──

export const MOCK_FACTOR_RETURN_SERIES: FactorReturnSeries[] = [
  {
    factor: 'book_to_price',
    group: 'value',
    points: generateDailyTimeSeries('2021-03-01', 1260, 0.06, 0.12, 100, 410).map((p) => ({
      date: p.date,
      cumReturn: (p.value - 100) / 100,
    })),
  },
  {
    factor: 'gross_profitability',
    group: 'profitability',
    points: generateDailyTimeSeries('2021-03-01', 1260, 0.08, 0.10, 100, 411).map((p) => ({
      date: p.date,
      cumReturn: (p.value - 100) / 100,
    })),
  },
  {
    factor: 'momentum_12_1',
    group: 'momentum',
    points: generateDailyTimeSeries('2021-03-01', 1260, 0.04, 0.18, 100, 412).map((p) => ({
      date: p.date,
      cumReturn: (p.value - 100) / 100,
    })),
  },
  {
    factor: 'volatility',
    group: 'low_risk',
    points: generateDailyTimeSeries('2021-03-01', 1260, 0.03, 0.08, 100, 413).map((p) => ({
      date: p.date,
      cumReturn: (p.value - 100) / 100,
    })),
  },
  {
    factor: 'dividend_yield',
    group: 'dividend',
    points: generateDailyTimeSeries('2021-03-01', 1260, 0.05, 0.09, 100, 414).map((p) => ({
      date: p.date,
      cumReturn: (p.value - 100) / 100,
    })),
  },
  {
    factor: 'asset_growth',
    group: 'investment',
    points: generateDailyTimeSeries('2021-03-01', 1260, 0.02, 0.14, 100, 415).map((p) => ({
      date: p.date,
      cumReturn: (p.value - 100) / 100,
    })),
  },
];

// ── Capital Market Assumptions (CMA) ──

export const MOCK_CMA_SETS: CMASet[] = [
  {
    label: 'Base Case',
    horizon: '5Y',
    assets: [
      { ticker: 'US Equities', expectedReturn: 0.082, expectedVol: 0.165 },
      { ticker: 'Intl Developed', expectedReturn: 0.068, expectedVol: 0.175 },
      { ticker: 'Emerging Markets', expectedReturn: 0.092, expectedVol: 0.225 },
      { ticker: 'US Bonds', expectedReturn: 0.042, expectedVol: 0.055 },
      { ticker: 'TIPS', expectedReturn: 0.035, expectedVol: 0.065 },
      { ticker: 'Real Estate', expectedReturn: 0.065, expectedVol: 0.185 },
      { ticker: 'Commodities', expectedReturn: 0.045, expectedVol: 0.195 },
    ],
  },
  {
    label: 'Optimistic',
    horizon: '5Y',
    assets: [
      { ticker: 'US Equities', expectedReturn: 0.105, expectedVol: 0.155 },
      { ticker: 'Intl Developed', expectedReturn: 0.088, expectedVol: 0.165 },
      { ticker: 'Emerging Markets', expectedReturn: 0.118, expectedVol: 0.210 },
      { ticker: 'US Bonds', expectedReturn: 0.038, expectedVol: 0.050 },
      { ticker: 'TIPS', expectedReturn: 0.032, expectedVol: 0.060 },
      { ticker: 'Real Estate', expectedReturn: 0.085, expectedVol: 0.175 },
      { ticker: 'Commodities', expectedReturn: 0.055, expectedVol: 0.180 },
    ],
  },
  {
    label: 'Pessimistic',
    horizon: '5Y',
    assets: [
      { ticker: 'US Equities', expectedReturn: 0.048, expectedVol: 0.195 },
      { ticker: 'Intl Developed', expectedReturn: 0.035, expectedVol: 0.205 },
      { ticker: 'Emerging Markets', expectedReturn: 0.058, expectedVol: 0.265 },
      { ticker: 'US Bonds', expectedReturn: 0.048, expectedVol: 0.065 },
      { ticker: 'TIPS', expectedReturn: 0.042, expectedVol: 0.072 },
      { ticker: 'Real Estate', expectedReturn: 0.032, expectedVol: 0.215 },
      { ticker: 'Commodities', expectedReturn: 0.028, expectedVol: 0.225 },
    ],
  },
];

// ── Factor IC Reports ──

export const MOCK_FACTOR_IC_REPORTS: FactorICReport[] = [
  { factor: 'book_to_price', group: 'value', ic: 0.048, icir: 0.42, tStat: 2.85, pValue: 0.004, vif: 1.82, significant: true },
  { factor: 'earnings_yield', group: 'value', ic: 0.052, icir: 0.48, tStat: 3.12, pValue: 0.002, vif: 2.14, significant: true },
  { factor: 'cash_flow_yield', group: 'value', ic: 0.035, icir: 0.31, tStat: 2.05, pValue: 0.041, vif: 1.95, significant: true },
  { factor: 'sales_to_price', group: 'value', ic: 0.028, icir: 0.24, tStat: 1.62, pValue: 0.106, vif: 1.78, significant: false },
  { factor: 'ebitda_to_ev', group: 'value', ic: 0.042, icir: 0.38, tStat: 2.51, pValue: 0.012, vif: 2.05, significant: true },
  { factor: 'gross_profitability', group: 'profitability', ic: 0.061, icir: 0.55, tStat: 3.68, pValue: 0.0003, vif: 1.45, significant: true },
  { factor: 'roe', group: 'profitability', ic: 0.044, icir: 0.40, tStat: 2.65, pValue: 0.008, vif: 1.92, significant: true },
  { factor: 'operating_margin', group: 'profitability', ic: 0.038, icir: 0.34, tStat: 2.25, pValue: 0.025, vif: 1.68, significant: true },
  { factor: 'profit_margin', group: 'profitability', ic: 0.032, icir: 0.28, tStat: 1.88, pValue: 0.061, vif: 2.35, significant: false },
  { factor: 'asset_growth', group: 'investment', ic: -0.025, icir: -0.22, tStat: -1.48, pValue: 0.140, vif: 1.32, significant: false },
  { factor: 'momentum_12_1', group: 'momentum', ic: 0.058, icir: 0.38, tStat: 2.52, pValue: 0.012, vif: 1.15, significant: true },
  { factor: 'volatility', group: 'low_risk', ic: -0.042, icir: -0.38, tStat: -2.55, pValue: 0.011, vif: 1.48, significant: true },
  { factor: 'beta', group: 'low_risk', ic: -0.035, icir: -0.32, tStat: -2.12, pValue: 0.034, vif: 1.62, significant: true },
  { factor: 'amihud_illiquidity', group: 'liquidity', ic: -0.018, icir: -0.15, tStat: -1.02, pValue: 0.308, vif: 1.28, significant: false },
  { factor: 'dividend_yield', group: 'dividend', ic: 0.022, icir: 0.20, tStat: 1.35, pValue: 0.178, vif: 1.55, significant: false },
  { factor: 'recommendation_change', group: 'sentiment', ic: 0.032, icir: 0.28, tStat: 1.85, pValue: 0.065, vif: 1.12, significant: false },
  { factor: 'net_insider_buying', group: 'ownership', ic: 0.025, icir: 0.21, tStat: 1.42, pValue: 0.156, vif: 1.08, significant: false },
];
