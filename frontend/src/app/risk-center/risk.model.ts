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

// ── Backend API DTOs ─────────────────────────────────────────────────────────
// Mirror api/app/schemas/{risk_analytics,risk,stress_scenarios,risk_budget}.py.
// CamelCaseModel on the backend means responses arrive already camelCased,
// so TypeScript interfaces use camelCase to match.

export interface VarApiResponse {
  var: Record<string, number>;
  cvar: Record<string, number>;
  method: string;
  lookback: number;
  nObservations: number;
}

export interface CorrelationApiResponse {
  assets: string[];
  matrix: number[][];
  clusterLabels: number[];
}

export interface FactorExposureApiResponse {
  exposures: Record<string, number>;
  assetExposures: Record<string, Record<string, number>>;
}

export interface ConcentrationAssetApi {
  ticker: string;
  name: string;
  weight: number;
}

export interface ConcentrationSummaryApi {
  hhi: number;
  effectiveN: number;
  topNRatio: number;
}

export interface ConcentrationApiResponse {
  assets: ConcentrationAssetApi[];
  summary: ConcentrationSummaryApi;
}

export interface LiquidityAssetApi {
  ticker: string;
  name: string;
  weight: number;
  avgDailyVolume: number | null;
  daysToLiquidate: number | null;
  liquidityCost: number | null;
}

export interface LiquiditySummaryApi {
  weightedAvgDaysToLiquidate: number;
}

export interface LiquidityApiResponse {
  assets: LiquidityAssetApi[];
  summary: LiquiditySummaryApi;
}

// ── Risk limits (CRUD) ──────────────────────────────────────────────────────

export type RiskLimitType = 'upper' | 'lower';

export interface RiskLimitCreatePayload {
  metric: string;
  limit_type: RiskLimitType;
  threshold: number;
}

export interface RiskLimitUpdatePayload {
  current_value: number;
  is_breached: boolean;
}

export interface RiskLimitDto {
  id: string;
  portfolioId: string;
  metric: string;
  limitType: RiskLimitType;
  threshold: number;
  currentValue: number | null;
  isBreached: boolean;
  lastCheckedAt: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface RiskLimitListResponse {
  items: RiskLimitDto[];
  breachCount: number;
}

// ── LLM-driven stress + budget ──────────────────────────────────────────────

export interface StressScenarioRequest {
  current_portfolio: Record<string, number>;
  macro_context: string;
  n_scenarios?: number;
}

export interface StressScenarioItemApi {
  name: string;
  description: string;
  shocks: Record<string, number>;
  probability: number;
  horizonDays: number;
  syntheticDataArgs: Record<string, unknown>;
}

export interface StressScenarioApiResponse {
  nScenarios: number;
  tickers: string[];
  scenarios: StressScenarioItemApi[];
}

