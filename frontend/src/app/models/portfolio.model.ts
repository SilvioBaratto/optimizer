export type StrategyType =
  | 'min_variance'
  | 'max_sharpe'
  | 'max_utility'
  | 'min_cvar'
  | 'efficient_frontier'
  | 'risk_parity'
  | 'cvar_parity'
  | 'max_diversification'
  | 'hrp'
  | 'herc'
  | 'equal_weighted';

export type StrategyCategory = 'convex' | 'hierarchical' | 'naive';

export interface StrategyInfo {
  type: StrategyType;
  name: string;
  description: string;
  category: StrategyCategory;
}

export interface PortfolioOptimizeRequest {
  strategy: StrategyType;
  backtest: boolean;
  pre_selection: boolean;
  sector_tolerance: number;
  risk_aversion?: number;
  cvar_beta?: number;
}

export interface AssetWeight {
  ticker: string;
  name: string;
  sector: string;
  weight: number;
}

export interface PerformanceMetrics {
  annualized_return: number;
  annualized_volatility: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown: number;
  cvar_95: number;
  calmar_ratio: number;
  tracking_error: number | null;
}

export interface MonthlyReturn {
  month: string;
  return_pct: number;
}

export interface SectorAllocation {
  sector: string;
  weight: number;
}

export interface BacktestPoint {
  date: string;
  cumulative_return: number;
}

export interface PortfolioResult {
  weights: AssetWeight[];
  metrics: PerformanceMetrics;
  monthly_returns: MonthlyReturn[];
  sector_allocations: SectorAllocation[];
  backtest_cumulative: BacktestPoint[] | null;
  backtest_metrics: PerformanceMetrics | null;
}

export interface PortfolioJobProgress {
  phase: string;
  pct: number;
  detail: string;
}
