import {
  VaRResult,
  StressScenario,
  CorrelationData,
  FactorExposure,
  ConcentrationMetric,
  LiquidityMetric,
  RiskLimit,
  RiskAlert,
} from '../models/risk.model';
import { generateCorrelationMatrix } from './generators';
import { WEIGHT_DATA } from './mock-data';

// ── VaR/CVaR Results ──

export const MOCK_VAR_RESULTS: VaRResult[] = [
  { method: 'historical', confidence: 0.95, horizon: 1, var: -0.018, cvar: -0.026, portfolioValue: 12_847_320, varDollar: 231_252, cvarDollar: 334_030 },
  { method: 'historical', confidence: 0.99, horizon: 1, var: -0.032, cvar: -0.041, portfolioValue: 12_847_320, varDollar: 411_114, cvarDollar: 526_740 },
  { method: 'parametric', confidence: 0.95, horizon: 1, var: -0.016, cvar: -0.024, portfolioValue: 12_847_320, varDollar: 205_557, cvarDollar: 308_336 },
  { method: 'parametric', confidence: 0.95, horizon: 10, var: -0.052, cvar: -0.074, portfolioValue: 12_847_320, varDollar: 668_061, cvarDollar: 950_702 },
  { method: 'monte_carlo', confidence: 0.95, horizon: 1, var: -0.017, cvar: -0.025, portfolioValue: 12_847_320, varDollar: 218_404, cvarDollar: 321_183 },
  { method: 'monte_carlo', confidence: 0.99, horizon: 1, var: -0.034, cvar: -0.044, portfolioValue: 12_847_320, varDollar: 436_809, cvarDollar: 565_282 },
];

// ── Stress Scenarios ──

export const MOCK_STRESS_SCENARIOS: StressScenario[] = [
  { id: 's1', name: 'COVID-19 Crash (Feb-Mar 2020)', description: 'Pandemic-driven global selloff', portfolioImpact: -0.285, benchmarkImpact: -0.338, worstAsset: 'XOM', worstAssetImpact: -0.52 },
  { id: 's2', name: 'GFC Lehman Collapse (Sep 2008)', description: 'Lehman Brothers bankruptcy', portfolioImpact: -0.382, benchmarkImpact: -0.445, worstAsset: 'BAC', worstAssetImpact: -0.68 },
  { id: 's3', name: 'Tech Bubble Burst (2000-2002)', description: 'Dot-com crash and recession', portfolioImpact: -0.312, benchmarkImpact: -0.492, worstAsset: 'INTC', worstAssetImpact: -0.78 },
  { id: 's4', name: 'Rate Shock +200bps', description: 'Sudden rate rise across the curve', portfolioImpact: -0.108, benchmarkImpact: -0.125, worstAsset: 'NEE', worstAssetImpact: -0.22 },
  { id: 's5', name: 'US-China Trade War Escalation', description: 'Full decoupling scenario', portfolioImpact: -0.165, benchmarkImpact: -0.188, worstAsset: 'AAPL', worstAssetImpact: -0.31 },
  { id: 's6', name: 'Energy Crisis', description: 'Oil price spike to $150/barrel', portfolioImpact: -0.092, benchmarkImpact: -0.118, worstAsset: 'COST', worstAssetImpact: -0.18 },
  { id: 's7', name: 'Stagflation', description: 'High inflation with recession', portfolioImpact: -0.198, benchmarkImpact: -0.225, worstAsset: 'TSLA', worstAssetImpact: -0.42 },
  { id: 's8', name: 'Flash Crash (-10% intraday)', description: 'Sudden market-wide liquidity evaporation', portfolioImpact: -0.088, benchmarkImpact: -0.100, worstAsset: 'NVDA', worstAssetImpact: -0.15 },
];

// ── Correlation Matrix (30x30) ──

const corrAssets = WEIGHT_DATA.slice(0, 30).map((w) => w.ticker);
const sectorMap: Record<string, number> = {};
let sectorIdx = 0;
WEIGHT_DATA.slice(0, 30).forEach((w) => {
  if (!(w.sector in sectorMap)) {
    sectorMap[w.sector] = sectorIdx++;
  }
});
const sectorAssignments = WEIGHT_DATA.slice(0, 30).map((w) => sectorMap[w.sector]);

export const MOCK_CORRELATION_DATA: CorrelationData = {
  assets: corrAssets,
  matrix: generateCorrelationMatrix(30, 0.70, 0.25, sectorAssignments, 300),
};

// ── Factor Exposures ──

export const MOCK_FACTOR_EXPOSURES: FactorExposure[] = [
  { factor: 'Market', exposure: 0.95, contribution: 0.72, marginalContribution: 0.012 },
  { factor: 'Size', exposure: -0.15, contribution: -0.04, marginalContribution: -0.001 },
  { factor: 'Value', exposure: 0.08, contribution: 0.02, marginalContribution: 0.001 },
  { factor: 'Momentum', exposure: 0.22, contribution: 0.08, marginalContribution: 0.003 },
  { factor: 'Quality', exposure: 0.31, contribution: 0.14, marginalContribution: 0.005 },
  { factor: 'Low Volatility', exposure: 0.12, contribution: 0.04, marginalContribution: 0.002 },
  { factor: 'Dividend Yield', exposure: 0.05, contribution: 0.01, marginalContribution: 0.001 },
  { factor: 'Liquidity', exposure: -0.08, contribution: -0.02, marginalContribution: -0.001 },
];

// ── Concentration Metrics ──

export const MOCK_CONCENTRATION: ConcentrationMetric[] = WEIGHT_DATA.slice(0, 20).map((w, i) => ({
  ticker: w.ticker,
  name: w.name,
  weight: w.weight,
  riskContribution: Math.round((w.weight * (1.2 - i * 0.02)) * 10000) / 10000,
  componentVar: Math.round(w.weight * 0.018 * 10000) / 10000,
}));

// ── Liquidity Metrics ──

const avgVolumes: Record<string, number> = {
  AAPL: 54_200_000, MSFT: 22_100_000, GOOGL: 18_500_000, AMZN: 32_400_000,
  NVDA: 42_800_000, META: 15_200_000, JPM: 9_800_000, V: 6_400_000,
  JNJ: 5_800_000, WMT: 7_200_000, PG: 6_100_000, UNH: 3_200_000,
  HD: 3_800_000, BAC: 28_500_000, XOM: 14_200_000,
};

export const MOCK_LIQUIDITY: LiquidityMetric[] = WEIGHT_DATA.slice(0, 15).map((w) => {
  const vol = avgVolumes[w.ticker] ?? 5_000_000;
  const positionValue = w.weight * 12_847_320;
  const sharePrice = 200;
  const sharesHeld = positionValue / sharePrice;
  const daysToLiquidate = Math.max(0.1, sharesHeld / (vol * 0.1));
  return {
    ticker: w.ticker,
    name: w.name,
    avgDailyVolume: vol,
    daysToLiquidate: Math.round(daysToLiquidate * 100) / 100,
    liquidityCost: Math.round(daysToLiquidate * 0.0005 * 10000) / 10000,
    weight: w.weight,
  };
});

// ── Risk Limits ──

export const MOCK_RISK_LIMITS: RiskLimit[] = [
  { id: 'rl1', name: 'Daily VaR 95%', metric: 'var_95_1d', limit: 0.025, current: 0.018, status: 'ok', lastChecked: '2026-02-25T08:00:00Z' },
  { id: 'rl2', name: 'Max Drawdown', metric: 'max_drawdown', limit: 0.15, current: 0.128, status: 'warning', lastChecked: '2026-02-25T08:00:00Z' },
  { id: 'rl3', name: 'Top-5 Concentration', metric: 'top5_weight', limit: 0.25, current: 0.164, status: 'ok', lastChecked: '2026-02-25T08:00:00Z' },
  { id: 'rl4', name: 'Single Position', metric: 'max_weight', limit: 0.05, current: 0.042, status: 'ok', lastChecked: '2026-02-25T08:00:00Z' },
  { id: 'rl5', name: 'Sector Concentration', metric: 'max_sector', limit: 0.30, current: 0.205, status: 'ok', lastChecked: '2026-02-25T08:00:00Z' },
  { id: 'rl6', name: 'Tracking Error', metric: 'tracking_error', limit: 0.05, current: 0.032, status: 'ok', lastChecked: '2026-02-25T08:00:00Z' },
  { id: 'rl7', name: 'Portfolio Beta', metric: 'beta', limit: 1.2, current: 0.95, status: 'ok', lastChecked: '2026-02-25T08:00:00Z' },
  { id: 'rl8', name: 'Liquidity (Days)', metric: 'max_days_liquidate', limit: 5, current: 2.1, status: 'ok', lastChecked: '2026-02-25T08:00:00Z' },
  { id: 'rl9', name: 'CVaR 99%', metric: 'cvar_99_1d', limit: 0.045, current: 0.041, status: 'warning', lastChecked: '2026-02-25T08:00:00Z' },
  { id: 'rl10', name: 'Turnover (Monthly)', metric: 'monthly_turnover', limit: 0.10, current: 0.042, status: 'ok', lastChecked: '2026-02-25T08:00:00Z' },
];

// ── Risk Alerts ──

export const MOCK_RISK_ALERTS: RiskAlert[] = [
  { id: 'ra1', severity: 'warning', title: 'Max drawdown approaching limit', message: 'Current drawdown -12.8% vs 15% limit', metric: 'max_drawdown', currentValue: 0.128, threshold: 0.15, timestamp: '2026-02-25T08:00:00Z', acknowledged: false },
  { id: 'ra2', severity: 'warning', title: 'CVaR 99% near limit', message: '1-day CVaR at 91% of limit', metric: 'cvar_99', currentValue: 0.041, threshold: 0.045, timestamp: '2026-02-25T08:00:00Z', acknowledged: false },
  { id: 'ra3', severity: 'info', title: 'VIX elevated', message: 'VIX at 16.8, above 30-day average of 14.2', metric: 'vix', currentValue: 16.8, threshold: 15.0, timestamp: '2026-02-24T14:00:00Z', acknowledged: true },
  { id: 'ra4', severity: 'critical', title: 'Correlation spike detected', message: 'Average pairwise correlation jumped from 0.32 to 0.48', metric: 'avg_correlation', currentValue: 0.48, threshold: 0.40, timestamp: '2026-02-23T10:00:00Z', acknowledged: false },
  { id: 'ra5', severity: 'info', title: 'Rebalancing due', message: 'Next scheduled rebalance in 3 trading days', metric: 'rebalance_due', currentValue: 3, threshold: 5, timestamp: '2026-02-24T16:00:00Z', acknowledged: true },
];
