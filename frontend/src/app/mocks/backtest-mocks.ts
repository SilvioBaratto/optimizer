import {
  BacktestConfig,
  BacktestResult,
  BacktestEquityPoint,
  BacktestMetrics,
  Drawdown,
  MonthlyReturnCell,
  RollingMetric,
  ReturnDistributionBin,
  FactorLoading,
} from '../models/backtest.model';
import { generateDailyTimeSeries, generateMonthlyGrid, seededRandom } from './generators';

// ── Config ──

export const MOCK_BACKTEST_CONFIG: BacktestConfig = {
  startDate: '2021-03-01',
  endDate: '2026-02-25',
  initialCapital: 10_000_000,
  rebalanceFrequency: 'quarterly',
  transactionCostBps: 10,
  benchmark: 'SPY',
};

// ── Equity Curve (5Y daily, ~1260 points) ──

const portfolioSeries = generateDailyTimeSeries('2021-03-01', 1260, 0.10, 0.15, 100, 200);
const benchmarkSeries = generateDailyTimeSeries('2021-03-01', 1260, 0.08, 0.17, 100, 201);

const MOCK_BACKTEST_EQUITY: BacktestEquityPoint[] = portfolioSeries.map((p, i) => ({
  date: p.date,
  portfolio: p.value,
  benchmark: benchmarkSeries[i]?.value ?? 100,
}));

// ── Metrics ──

const MOCK_BACKTEST_METRICS: BacktestMetrics = {
  totalReturn: 0.584,
  annualizedReturn: 0.098,
  annualizedVol: 0.154,
  sharpe: 0.636,
  sortino: 0.892,
  maxDrawdown: -0.148,
  calmar: 0.662,
  cvar95: -0.024,
  trackingError: 0.032,
  informationRatio: 0.562,
  winRate: 0.548,
  profitFactor: 1.32,
};

// ── Drawdowns ──

const MOCK_DRAWDOWNS: Drawdown[] = [
  { start: '2022-01-03', trough: '2022-06-16', end: '2022-11-28', depth: -0.148, duration: 115, recovery: 82 },
  { start: '2023-07-31', trough: '2023-10-27', end: '2023-12-15', depth: -0.082, duration: 63, recovery: 35 },
  { start: '2024-07-16', trough: '2024-08-05', end: '2024-09-12', depth: -0.068, duration: 15, recovery: 28 },
  { start: '2021-09-02', trough: '2021-10-04', end: '2021-10-21', depth: -0.054, duration: 23, recovery: 13 },
  { start: '2025-02-19', trough: '2025-04-08', end: '2025-05-12', depth: -0.092, duration: 35, recovery: 24 },
  { start: '2024-03-28', trough: '2024-04-19', end: '2024-05-15', depth: -0.045, duration: 16, recovery: 19 },
  { start: '2025-08-01', trough: '2025-08-18', end: '2025-09-08', depth: -0.038, duration: 12, recovery: 15 },
  { start: '2023-02-02', trough: '2023-03-13', end: '2023-04-03', depth: -0.061, duration: 27, recovery: 15 },
  { start: '2021-11-22', trough: '2021-12-03', end: '2021-12-23', depth: -0.032, duration: 8, recovery: 14 },
  { start: '2025-12-18', trough: '2026-01-13', end: null, depth: -0.028, duration: 18, recovery: null },
];

// ── Monthly Return Grid ──

const MOCK_MONTHLY_RETURNS: MonthlyReturnCell[] = generateMonthlyGrid(2021, 2026, 0.10, 0.15, 210);

// ── Rolling Metrics ──

const MOCK_ROLLING_METRICS: RollingMetric[] = (() => {
  const rng = seededRandom(220);
  const metrics: RollingMetric[] = [];
  const start = new Date('2022-03-01');
  for (let i = 0; i < 252 * 4; i++) {
    const d = new Date(start);
    d.setDate(d.getDate() + Math.floor(i * 365 / 252));
    if (d.getDay() === 0 || d.getDay() === 6) continue;
    metrics.push({
      date: d.toISOString().split('T')[0],
      sharpe: 0.5 + (rng() - 0.5) * 0.8,
      volatility: 0.12 + (rng() - 0.5) * 0.08,
      beta: 0.9 + (rng() - 0.5) * 0.3,
    });
  }
  return metrics;
})();

// ── Return Distribution ──

const MOCK_RETURN_DISTRIBUTION: ReturnDistributionBin[] = (() => {
  const rng = seededRandom(230);
  const bins: ReturnDistributionBin[] = [];
  const centers = [-0.04, -0.03, -0.02, -0.015, -0.01, -0.005, 0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04];
  const weights = [2, 5, 12, 18, 35, 55, 68, 52, 32, 16, 10, 4, 1];
  const total = weights.reduce((a, b) => a + b, 0);
  for (let i = 0; i < centers.length; i++) {
    const noise = Math.floor(rng() * 3) - 1;
    const count = weights[i] + noise;
    bins.push({
      binStart: centers[i] - 0.0025,
      binEnd: centers[i] + 0.0025,
      count,
      frequency: Math.round((count / total) * 10000) / 10000,
    });
  }
  return bins;
})();

// ── Factor Loadings ──

const MOCK_FACTOR_LOADINGS: FactorLoading[] = [
  { factor: 'Market (MKT)', loading: 0.92, tStat: 28.4, pValue: 0.0001 },
  { factor: 'Size (SMB)', loading: -0.12, tStat: -2.1, pValue: 0.036 },
  { factor: 'Value (HML)', loading: 0.08, tStat: 1.4, pValue: 0.162 },
  { factor: 'Momentum (MOM)', loading: 0.15, tStat: 3.2, pValue: 0.001 },
  { factor: 'Quality (QMJ)', loading: 0.21, tStat: 4.5, pValue: 0.0001 },
  { factor: 'Low Vol (BAB)', loading: 0.06, tStat: 1.1, pValue: 0.271 },
];

// ── Assembled Result ──

export const MOCK_BACKTEST_RESULT: BacktestResult = {
  equity: MOCK_BACKTEST_EQUITY,
  metrics: MOCK_BACKTEST_METRICS,
  drawdowns: MOCK_DRAWDOWNS,
  monthlyReturns: MOCK_MONTHLY_RETURNS,
  rollingMetrics: MOCK_ROLLING_METRICS,
  returnDistribution: MOCK_RETURN_DISTRIBUTION,
  factorLoadings: MOCK_FACTOR_LOADINGS,
};
