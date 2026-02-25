export interface DriftEntry {
  ticker: string;
  name: string;
  sector: string;
  targetWeight: number;
  currentWeight: number;
  drift: number;
  driftAbsolute: number;
  breached: boolean;
}

export type RebalancingFrequency = 'daily' | 'weekly' | 'monthly' | 'quarterly';
export type RebalancingTrigger = 'calendar' | 'threshold' | 'hybrid';

export interface RebalancingPolicy {
  id: string;
  name: string;
  trigger: RebalancingTrigger;
  frequency?: RebalancingFrequency;
  thresholdAbsolute?: number;
  thresholdRelative?: number;
  costBudgetBps: number;
  active: boolean;
}

export type TradeAction = 'buy' | 'sell';

export interface TradePreview {
  ticker: string;
  name: string;
  action: TradeAction;
  shares: number;
  notional: number;
  fromWeight: number;
  toWeight: number;
  estimatedCost: number;
}

export interface TradeSummary {
  totalTrades: number;
  totalTurnover: number;
  totalCost: number;
  buys: number;
  sells: number;
  netCashFlow: number;
}

export interface RebalancingHistoryEntry {
  date: string;
  trigger: RebalancingTrigger;
  tradesExecuted: number;
  turnover: number;
  cost: number;
  preDriftMax: number;
}
