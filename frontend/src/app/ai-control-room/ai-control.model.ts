export type AgentRole =
  | 'portfolio_manager'
  | 'risk_analyst'
  | 'factor_researcher'
  | 'execution_agent';

export type AgentStatusType = 'active' | 'idle' | 'paused' | 'error';

export interface AgentStatus {
  id: string;
  role: AgentRole;
  name: string;
  status: AgentStatusType;
  lastAction: string;
  lastActionTime: string;
  decisionsToday: number;
  confidence: number;
}

export type DecisionType =
  | 'rebalance'
  | 'risk_alert'
  | 'factor_tilt'
  | 'trade'
  | 'veto'
  | 'regime_change';

export type DecisionOutcome = 'approved' | 'rejected' | 'pending' | 'executed';

export interface DecisionFeedItem {
  id: string;
  agent: AgentRole;
  type: DecisionType;
  title: string;
  reasoning: string;
  outcome: DecisionOutcome;
  timestamp: string;
  confidence: number;
  impact?: string;
}

export interface AgentConfig {
  role: AgentRole;
  name: string;
  enabled: boolean;
  autonomyLevel: 'full' | 'supervised' | 'advisory';
  maxPositionSize: number;
  maxDailyTurnover: number;
  riskBudget: number;
}

export type WorkflowStateType =
  | 'data_ingestion'
  | 'factor_analysis'
  | 'risk_assessment'
  | 'optimization'
  | 'execution'
  | 'monitoring';

export type WorkflowStepStatus = 'completed' | 'running' | 'pending' | 'skipped';

export interface WorkflowState {
  step: WorkflowStateType;
  label: string;
  status: WorkflowStepStatus;
  startedAt?: string;
  completedAt?: string;
  detail?: string;
}

// ── AI Fund Portfolio View ──

export interface AIFundHolding {
  ticker: string;
  weight: number;
  saaTarget: number;
  taaTilt: number;
  taaReasoning: string;
  sector: string;
  returnMtd: number;
  returnYtd: number;
}

export interface AIFundSignal {
  name: string;
  value: number;
  label: 'bullish' | 'neutral' | 'bearish';
  description: string;
}

export type RiskStatusLevel = 'ok' | 'warning' | 'breach';

export interface AIFundRiskStatus {
  metric: string;
  current: number;
  limit: number;
  utilization: number;
  status: RiskStatusLevel;
}

export interface AIFundSummary {
  aum: number;
  holdings: AIFundHolding[];
  signals: AIFundSignal[];
  riskStatuses: AIFundRiskStatus[];
}

// ── Extended Agent Config ──

export interface AgentConfigExtended extends AgentConfig {
  llmModel: string;
  temperature: number;
  frequency: string;
  costBudgetDaily: number;
  costUsedToday: number;
}

// ── Orchestration ──

export interface OrchestrationConfig {
  pipeline: string[];
  vetoEnabled: boolean;
  conflictResolution: 'risk_priority' | 'consensus' | 'cio_override';
  maxLatencyMs: number;
}

export interface VetoLogEntry {
  id: string;
  vetoAgent: AgentRole;
  targetAgent: AgentRole;
  action: string;
  reason: string;
  timestamp: string;
  overridden: boolean;
}

// ── Comparison ──

export interface ComparisonMetric {
  metric: string;
  user: number;
  ai: number;
  benchmark: number;
  unit: string;
}

export interface ComparisonEquityPoint {
  date: string;
  user: number;
  ai: number;
  benchmark: number;
}

export interface DecisionDivergence {
  date: string;
  asset: string;
  userAction: string;
  aiAction: string;
  userOutcome: number;
  aiOutcome: number;
  delta: number;
}

export interface ComparisonData {
  metrics: ComparisonMetric[];
  equity: ComparisonEquityPoint[];
  divergences: DecisionDivergence[];
}
