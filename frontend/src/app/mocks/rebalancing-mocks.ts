import {
  DriftEntry,
  RebalancingPolicy,
  TradePreview,
  TradeSummary,
  RebalancingHistoryEntry,
} from '../models/rebalancing.model';
import { WEIGHT_DATA } from './mock-data';
import { seededRandom } from './generators';

// ── Drift Table (15 assets) ──

const rng = seededRandom(500);

export const MOCK_DRIFT_TABLE: DriftEntry[] = WEIGHT_DATA.slice(0, 15).map((w) => {
  const driftPct = (rng() - 0.45) * 0.06;
  const currentWeight = w.weight + driftPct;
  return {
    ticker: w.ticker,
    name: w.name,
    sector: w.sector,
    targetWeight: w.weight,
    currentWeight: Math.round(currentWeight * 10000) / 10000,
    drift: Math.round(driftPct * 10000) / 10000,
    driftAbsolute: Math.round(Math.abs(driftPct) * 10000) / 10000,
    breached: Math.abs(driftPct) > 0.025,
  };
});

// ── Policies ──

export const MOCK_REBALANCING_POLICIES: RebalancingPolicy[] = [
  { id: 'rp1', name: 'Quarterly Calendar', trigger: 'calendar', frequency: 'quarterly', costBudgetBps: 15, active: true },
  { id: 'rp2', name: '5% Threshold', trigger: 'threshold', thresholdAbsolute: 0.05, thresholdRelative: 0.25, costBudgetBps: 10, active: false },
  { id: 'rp3', name: 'Monthly Hybrid (2.5%)', trigger: 'hybrid', frequency: 'monthly', thresholdAbsolute: 0.025, costBudgetBps: 12, active: false },
];

// ── Trade Preview ──

export const MOCK_TRADE_PREVIEW: TradePreview[] = [
  { ticker: 'NVDA', name: 'NVIDIA Corp.', action: 'sell', shares: 45, notional: 54_000, fromWeight: 0.0328, toWeight: 0.025, estimatedCost: 27 },
  { ticker: 'AAPL', name: 'Apple Inc.', action: 'sell', shares: 22, notional: 5_280, fromWeight: 0.0462, toWeight: 0.042, estimatedCost: 5 },
  { ticker: 'TSLA', name: 'Tesla Inc.', action: 'sell', shares: 18, notional: 4_500, fromWeight: 0.0165, toWeight: 0.013, estimatedCost: 5 },
  { ticker: 'JNJ', name: 'Johnson & Johnson', action: 'buy', shares: 30, notional: 4_800, fromWeight: 0.0142, toWeight: 0.018, estimatedCost: 5 },
  { ticker: 'XOM', name: 'Exxon Mobil', action: 'buy', shares: 28, notional: 3_080, fromWeight: 0.0098, toWeight: 0.013, estimatedCost: 3 },
  { ticker: 'WMT', name: 'Walmart Inc.', action: 'buy', shares: 15, notional: 2_700, fromWeight: 0.0148, toWeight: 0.017, estimatedCost: 3 },
  { ticker: 'KO', name: 'Coca-Cola Co.', action: 'buy', shares: 42, notional: 2_520, fromWeight: 0.0095, toWeight: 0.012, estimatedCost: 3 },
  { ticker: 'PEP', name: 'PepsiCo Inc.', action: 'buy', shares: 10, notional: 1_780, fromWeight: 0.0088, toWeight: 0.011, estimatedCost: 2 },
];

export const MOCK_TRADE_SUMMARY: TradeSummary = {
  totalTrades: 8,
  totalTurnover: 0.042,
  totalCost: 53,
  buys: 5,
  sells: 3,
  netCashFlow: -43_340,
};

// ── Rebalancing History (12 months) ──

export const MOCK_REBALANCING_HISTORY: RebalancingHistoryEntry[] = [
  { date: '2026-02-24', trigger: 'threshold', tradesExecuted: 8, turnover: 0.042, cost: 53, preDriftMax: 0.032 },
  { date: '2026-01-02', trigger: 'calendar', tradesExecuted: 12, turnover: 0.058, cost: 74, preDriftMax: 0.045 },
  { date: '2025-10-01', trigger: 'calendar', tradesExecuted: 10, turnover: 0.048, cost: 62, preDriftMax: 0.038 },
  { date: '2025-08-18', trigger: 'threshold', tradesExecuted: 4, turnover: 0.022, cost: 28, preDriftMax: 0.028 },
  { date: '2025-07-01', trigger: 'calendar', tradesExecuted: 14, turnover: 0.065, cost: 84, preDriftMax: 0.052 },
  { date: '2025-04-01', trigger: 'calendar', tradesExecuted: 11, turnover: 0.052, cost: 67, preDriftMax: 0.041 },
  { date: '2025-02-19', trigger: 'threshold', tradesExecuted: 6, turnover: 0.035, cost: 45, preDriftMax: 0.031 },
  { date: '2025-01-02', trigger: 'calendar', tradesExecuted: 15, turnover: 0.072, cost: 92, preDriftMax: 0.058 },
  { date: '2024-10-01', trigger: 'calendar', tradesExecuted: 9, turnover: 0.044, cost: 56, preDriftMax: 0.035 },
  { date: '2024-07-01', trigger: 'calendar', tradesExecuted: 13, turnover: 0.061, cost: 78, preDriftMax: 0.048 },
  { date: '2024-04-01', trigger: 'calendar', tradesExecuted: 8, turnover: 0.038, cost: 49, preDriftMax: 0.030 },
  { date: '2024-01-02', trigger: 'calendar', tradesExecuted: 16, turnover: 0.078, cost: 100, preDriftMax: 0.062 },
];
