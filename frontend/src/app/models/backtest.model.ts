export interface BacktestConfig {
  startDate: string;
  endDate: string;
  initialCapital: number;
  rebalanceFrequency: 'daily' | 'weekly' | 'monthly' | 'quarterly';
  transactionCostBps: number;
  benchmark: string;
}

export interface BacktestEquityPoint {
  date: string;
  portfolio: number;
  benchmark: number;
}

export interface BacktestResult {
  equity: BacktestEquityPoint[];
  metrics: BacktestMetrics;
  drawdowns: Drawdown[];
  monthlyReturns: MonthlyReturnCell[];
  rollingMetrics: RollingMetric[];
  returnDistribution: ReturnDistributionBin[];
  factorLoadings: FactorLoading[];
}

export interface BacktestMetrics {
  totalReturn: number;
  annualizedReturn: number;
  annualizedVol: number;
  sharpe: number;
  sortino: number;
  maxDrawdown: number;
  calmar: number;
  cvar95: number;
  trackingError: number;
  informationRatio: number;
  winRate: number;
  profitFactor: number;
}

export interface Drawdown {
  start: string;
  trough: string;
  end: string | null;
  depth: number;
  duration: number;
  recovery: number | null;
}

export interface MonthlyReturnCell {
  year: number;
  month: number;
  value: number;
}

export interface RollingMetric {
  date: string;
  sharpe: number;
  volatility: number;
  beta: number;
}

export interface ReturnDistributionBin {
  binStart: number;
  binEnd: number;
  count: number;
  frequency: number;
}

export interface FactorLoading {
  factor: string;
  loading: number;
  tStat: number;
  pValue: number;
}
