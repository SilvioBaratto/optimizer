import {
  DashboardKPI,
  EquityCurvePoint,
  ActivityFeedItem,
  MarketContext,
  RegimeInfo,
  AllocationNode,
  DriftEntry,
  AssetClassReturn,
} from '../models/dashboard.model';
import { generateDailyTimeSeries, seededRandom } from './generators';

// ── Equity Curve (3Y daily, ~756 points) ──

const portfolioSeries = generateDailyTimeSeries('2023-03-01', 756, 0.10, 0.15, 100, 42);
const benchmarkSeries = generateDailyTimeSeries('2023-03-01', 756, 0.08, 0.17, 100, 99);

export const MOCK_EQUITY_CURVE: EquityCurvePoint[] = portfolioSeries.map((p, i) => ({
  date: p.date,
  portfolio: p.value,
  benchmark: benchmarkSeries[i]?.value ?? 100,
}));

// ── KPIs ──

function generateSparkline(seed: number, length: number): number[] {
  const rng = seededRandom(seed);
  const values: number[] = [];
  let v = 50;
  for (let i = 0; i < length; i++) {
    v += (rng() - 0.48) * 5;
    values.push(Math.round(v * 100) / 100);
  }
  return values;
}

export const MOCK_DASHBOARD_KPIS: DashboardKPI[] = [
  {
    label: 'Total Return',
    value: 0.2847,
    format: 'percent',
    change: 0.018,
    changeLabel: 'vs last month',
    sparkline: generateSparkline(1, 30),
  },
  {
    label: 'Annualized Return',
    value: 0.0982,
    format: 'percent',
    change: 0.004,
    changeLabel: 'vs last quarter',
    sparkline: generateSparkline(2, 30),
  },
  {
    label: 'Sharpe Ratio',
    value: 0.62,
    format: 'ratio',
    change: 0.05,
    changeLabel: 'vs last month',
    sparkline: generateSparkline(3, 30),
  },
  {
    label: 'Max Drawdown',
    value: -0.128,
    format: 'percent',
    change: 0.012,
    changeLabel: 'improved from -14%',
    sparkline: generateSparkline(4, 30),
  },
  {
    label: 'Portfolio Value',
    value: 12_847_320,
    format: 'currency',
    change: 228_450,
    changeLabel: '+1.8% MTD',
    sparkline: generateSparkline(5, 30),
  },
  {
    label: 'Volatility',
    value: 0.154,
    format: 'percent',
    change: -0.008,
    changeLabel: 'vs last month',
    sparkline: generateSparkline(6, 30),
  },
  {
    label: 'CVaR 95%',
    value: -0.024,
    format: 'percent',
    change: 0.003,
    changeLabel: 'improved',
    sparkline: generateSparkline(7, 30),
  },
];

// ── Activity Feed ──

export const MOCK_ACTIVITY_FEED: ActivityFeedItem[] = [
  { id: '1', type: 'optimization', title: 'Portfolio re-optimized', description: 'Max Sharpe strategy applied to 85-asset universe', timestamp: '2026-02-25T08:30:00Z' },
  { id: '2', type: 'rebalance', title: 'Quarterly rebalance executed', description: '12 trades, 4.2% turnover, $1,240 estimated cost', timestamp: '2026-02-24T16:00:00Z' },
  { id: '3', type: 'regime_change', title: 'Regime shift detected', description: 'HMM transitioned from low-vol to medium-vol state (p=0.82)', timestamp: '2026-02-24T09:15:00Z' },
  { id: '4', type: 'alert', title: 'VaR limit warning', description: '1-day 95% VaR at 92% of limit ($245K vs $265K)', timestamp: '2026-02-23T14:30:00Z' },
  { id: '5', type: 'ai_decision', title: 'AI reduced tech exposure', description: 'Factor researcher flagged momentum reversal in tech sector', timestamp: '2026-02-23T10:00:00Z' },
  { id: '6', type: 'trade', title: 'Executed 8 trades', description: 'Reduced NVDA by 1.2%, added JNJ +0.8%, XOM +0.4%', timestamp: '2026-02-22T15:45:00Z' },
  { id: '7', type: 'optimization', title: 'Risk parity backtest completed', description: '5Y backtest: Sharpe 0.58, MaxDD -11.2%', timestamp: '2026-02-22T11:20:00Z' },
  { id: '8', type: 'alert', title: 'Concentration limit OK', description: 'Top-5 weight at 16.4% (limit: 25%)', timestamp: '2026-02-21T16:00:00Z' },
  { id: '9', type: 'regime_change', title: 'Macro regime: Expansion', description: 'GDP acceleration + yield curve steepening', timestamp: '2026-02-21T08:00:00Z' },
  { id: '10', type: 'rebalance', title: 'Drift threshold triggered', description: 'NVDA drifted +2.8% above target (threshold: 2.5%)', timestamp: '2026-02-20T14:00:00Z' },
  { id: '11', type: 'ai_decision', title: 'Risk analyst approved hedge', description: 'Tail-risk hedge via put spread on SPY approved', timestamp: '2026-02-20T10:30:00Z' },
  { id: '12', type: 'trade', title: 'Dividend reinvestment', description: 'Reinvested $4,280 in dividends across 15 positions', timestamp: '2026-02-19T16:00:00Z' },
  { id: '13', type: 'optimization', title: 'Factor tilt adjustment', description: 'Increased value tilt by 15% based on regime signal', timestamp: '2026-02-19T09:00:00Z' },
  { id: '14', type: 'alert', title: 'Tracking error within budget', description: 'TE at 3.2% vs 5% budget', timestamp: '2026-02-18T16:00:00Z' },
  { id: '15', type: 'rebalance', title: 'Sector rebalance', description: 'Financials +1.2%, Healthcare -0.8%, Energy flat', timestamp: '2026-02-18T11:00:00Z' },
  { id: '16', type: 'ai_decision', title: 'Execution agent split order', description: 'AAPL sell split into 3 tranches to minimize market impact', timestamp: '2026-02-17T15:30:00Z' },
  { id: '17', type: 'regime_change', title: 'VIX spike detected', description: 'VIX rose from 14.2 to 18.5 intraday', timestamp: '2026-02-17T10:00:00Z' },
  { id: '18', type: 'trade', title: 'Tax-loss harvesting', description: 'Sold INTC at -8.2% loss, replaced with TXN', timestamp: '2026-02-14T15:00:00Z' },
  { id: '19', type: 'optimization', title: 'Cross-validation completed', description: 'Walk-forward CV: avg Sharpe 0.55 (+-0.12)', timestamp: '2026-02-14T09:00:00Z' },
  { id: '20', type: 'alert', title: 'Liquidity check passed', description: 'All positions liquidatable within 2 days at current volumes', timestamp: '2026-02-13T16:00:00Z' },
];

// ── Market Context ──

export const MOCK_MARKET_CONTEXT: MarketContext = {
  vix: 16.8,
  vixChange: -1.2,
  sp500Return: 0.0182,
  tenYearYield: 4.22,
  yieldChange: -0.03,
  usdIndex: 103.4,
  usdChange: -0.28,
};

// ── Regime Info ──

export const MOCK_REGIME_INFO: RegimeInfo = {
  current: 'bull',
  probability: 0.72,
  since: '2025-11-15',
  hmmStates: [
    { regime: 'bull', probability: 0.72 },
    { regime: 'sideways', probability: 0.21 },
    { regime: 'bear', probability: 0.05 },
    { regime: 'volatile', probability: 0.02 },
  ],
};

// ── Allocation Sunburst ──

export const MOCK_ALLOCATION_SUNBURST: AllocationNode[] = [
  {
    name: 'Equities',
    value: 76.0,
    children: [
      { name: 'Technology', value: 20.5 },
      { name: 'Healthcare', value: 14.0 },
      { name: 'Financial Svcs', value: 12.5 },
      { name: 'Industrials', value: 10.5 },
      { name: 'Consumer Def.', value: 9.5 },
      { name: 'Consumer Cyc.', value: 8.5 },
    ],
  },
  {
    name: 'Fixed Income',
    value: 12.0,
    children: [
      { name: 'US Treasury', value: 5.0 },
      { name: 'IG Corporate', value: 4.0 },
      { name: 'TIPS', value: 3.0 },
    ],
  },
  {
    name: 'Alternatives',
    value: 8.0,
    children: [
      { name: 'Commodities', value: 3.0 },
      { name: 'Real Estate', value: 3.0 },
      { name: 'Gold', value: 2.0 },
    ],
  },
  {
    name: 'Cash',
    value: 4.0,
  },
];

// ── Drift Table ──

const rng = seededRandom(77);

export const MOCK_DRIFT_TABLE: DriftEntry[] = [
  { ticker: 'AAPL', name: 'Apple Inc.', target: 0.042, actual: 0.048, drift: 0.006, breached: true },
  { ticker: 'MSFT', name: 'Microsoft Corp.', target: 0.038, actual: 0.041, drift: 0.003, breached: false },
  { ticker: 'NVDA', name: 'NVIDIA Corp.', target: 0.025, actual: 0.032, drift: 0.007, breached: true },
  { ticker: 'GOOGL', name: 'Alphabet Inc.', target: 0.031, actual: 0.029, drift: -0.002, breached: false },
  { ticker: 'AMZN', name: 'Amazon.com Inc.', target: 0.028, actual: 0.026, drift: -0.002, breached: false },
  { ticker: 'JPM', name: 'JPMorgan Chase', target: 0.020, actual: 0.022, drift: 0.002, breached: false },
  { ticker: 'META', name: 'Meta Platforms', target: 0.022, actual: 0.019, drift: -0.003, breached: false },
  { ticker: 'JNJ', name: 'Johnson & Johnson', target: 0.018, actual: 0.016, drift: -0.002, breached: false },
  { ticker: 'XOM', name: 'Exxon Mobil', target: 0.013, actual: 0.018, drift: 0.005, breached: true },
  { ticker: 'TSLA', name: 'Tesla Inc.', target: 0.013, actual: 0.010, drift: -0.003, breached: false },
].map(d => ({ ...d, _sort: Math.abs(d.drift) }))
  .sort((a, b) => b._sort - a._sort)
  .map(({ _sort, ...rest }) => rest);

// ── Asset Class Returns ──

export const MOCK_ASSET_CLASS_RETURNS: AssetClassReturn[] = [
  { name: 'US Equities',    '1D': 0.0082, '1W': 0.0214, '1M': 0.0345, 'YTD': 0.0612 },
  { name: 'Intl Equities',  '1D': 0.0041, '1W': 0.0098, '1M': 0.0178, 'YTD': 0.0289 },
  { name: 'EM Equities',    '1D': -0.0023, '1W': -0.0051, '1M': 0.0124, 'YTD': 0.0198 },
  { name: 'Fixed Income',   '1D': 0.0008, '1W': 0.0032, '1M': -0.0045, 'YTD': -0.0112 },
  { name: 'Commodities',    '1D': -0.0045, '1W': -0.0128, '1M': 0.0256, 'YTD': 0.0489 },
  { name: 'Gold',           '1D': 0.0032, '1W': 0.0078, '1M': 0.0312, 'YTD': 0.0534 },
  { name: 'Real Estate',    '1D': -0.0012, '1W': 0.0045, '1M': -0.0089, 'YTD': -0.0234 },
  { name: 'Crypto',         '1D': 0.0234, '1W': 0.0567, '1M': 0.1245, 'YTD': 0.2156 },
];
