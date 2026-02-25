export interface DashboardKPI {
  label: string;
  value: number;
  format: 'percent' | 'currency' | 'ratio' | 'number';
  change: number;
  changeLabel: string;
  sparkline: number[];
}

export interface EquityCurvePoint {
  date: string;
  portfolio: number;
  benchmark: number;
}

export type ActivityType =
  | 'rebalance'
  | 'optimization'
  | 'alert'
  | 'trade'
  | 'regime_change'
  | 'ai_decision';

export interface ActivityFeedItem {
  id: string;
  type: ActivityType;
  title: string;
  description: string;
  timestamp: string;
}

export type MarketRegime = 'bull' | 'bear' | 'sideways' | 'volatile';

export interface MarketContext {
  vix: number;
  vixChange: number;
  sp500Return: number;
  tenYearYield: number;
  yieldChange: number;
  usdIndex: number;
  usdChange: number;
}

export interface RegimeInfo {
  current: MarketRegime;
  probability: number;
  since: string;
  hmmStates: { regime: MarketRegime; probability: number }[];
}

export interface AllocationNode {
  name: string;
  value: number;
  children?: AllocationNode[];
}

export interface DriftEntry {
  ticker: string;
  name: string;
  target: number;
  actual: number;
  drift: number;
  breached: boolean;
}

export interface AssetClassReturn {
  name: string;
  '1D': number;
  '1W': number;
  '1M': number;
  'YTD': number;
}
