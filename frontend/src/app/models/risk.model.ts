export type VaRMethod = 'historical' | 'parametric' | 'monte_carlo';

export interface VaRResult {
  method: VaRMethod;
  confidence: number;
  horizon: number;
  var: number;
  cvar: number;
  portfolioValue: number;
  varDollar: number;
  cvarDollar: number;
}

export interface StressScenario {
  id: string;
  name: string;
  description: string;
  portfolioImpact: number;
  benchmarkImpact: number;
  worstAsset: string;
  worstAssetImpact: number;
}

export interface CorrelationData {
  assets: string[];
  matrix: number[][];
}

export interface FactorExposure {
  factor: string;
  exposure: number;
  contribution: number;
  marginalContribution: number;
}

export interface ConcentrationMetric {
  ticker: string;
  name: string;
  weight: number;
  riskContribution: number;
  componentVar: number;
}

export interface LiquidityMetric {
  ticker: string;
  name: string;
  avgDailyVolume: number;
  daysToLiquidate: number;
  liquidityCost: number;
  weight: number;
}

export type RiskLimitStatus = 'ok' | 'warning' | 'breached';

export interface RiskLimit {
  id: string;
  name: string;
  metric: string;
  limit: number;
  current: number;
  status: RiskLimitStatus;
  lastChecked: string;
}

export type RiskAlertSeverity = 'info' | 'warning' | 'critical';

export interface RiskAlert {
  id: string;
  severity: RiskAlertSeverity;
  title: string;
  message: string;
  metric: string;
  currentValue: number;
  threshold: number;
  timestamp: string;
  acknowledged: boolean;
}
