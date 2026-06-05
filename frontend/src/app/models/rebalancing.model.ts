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

// ── Backend API DTOs (mirror api/app/schemas/{rebalancing,dashboard}.py) ────

export type PolicyType = 'calendar' | 'threshold' | 'hybrid';

// GET /portfolio-analytics/{name}/drift
export interface DriftEntryDto {
  ticker: string;
  name: string | null;
  target: number;
  actual: number;
  drift: number;
  breached: boolean;
}

export interface DriftApiResponse {
  entries: DriftEntryDto[];
  totalDrift: number;
  breachedCount: number;
  threshold: number;
}

// GET/POST /portfolio/{name}/rebalance-policy
export interface RebalancingPolicyDto {
  id: string;
  portfolioId: string;
  name: string;
  policyType: PolicyType;
  config: Record<string, unknown>;
  isActive: boolean;
  createdAt: string;
  updatedAt: string;
}

export interface RebalancingPolicyListApiResponse {
  items: RebalancingPolicyDto[];
  total: number;
}

export interface RebalancingPolicyCreatePayload {
  name: string;
  policy_type: PolicyType;
  config: Record<string, unknown>;
}

// POST /rebalance/decide
export interface RebalanceDecideRequest {
  current_weights: Record<string, number>;
  target_weights: Record<string, number>;
  policy_type: PolicyType;
  policy_config?: Record<string, unknown>;
  current_date?: string;
  last_review_date?: string;
  transaction_costs?: number;
}

export interface RebalanceDecideApiResponse {
  shouldRebalance: boolean;
  turnover: number;
  estimatedCost: number;
  tradeWeights: Record<string, number>;
}

// GET /rebalance/preview/{portfolio_name}
export interface TradeItemDto {
  ticker: string;
  weightDelta: number;
  side: 'buy' | 'sell';
  shares: number | null;
}

export interface RebalancePreviewApiResponse {
  portfolioName: string;
  policyType: string;
  targetWeights: Record<string, number>;
  currentWeights: Record<string, number>;
  trades: TradeItemDto[];
  portfolioValue: number | null;
  // Empty-state reason ('no_active_policy' | 'no_snapshots'), null on happy path.
  status?: string | null;
}
