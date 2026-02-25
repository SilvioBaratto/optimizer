import {
  AgentStatus,
  DecisionFeedItem,
  AgentConfig,
  WorkflowState,
  AIFundSummary,
  AgentConfigExtended,
  OrchestrationConfig,
  VetoLogEntry,
  ComparisonData,
} from '../models/ai-control.model';

// ── Agent Statuses ──

export const MOCK_AGENT_STATUSES: AgentStatus[] = [
  {
    id: 'a1',
    role: 'portfolio_manager',
    name: 'Portfolio Manager',
    status: 'active',
    lastAction: 'Executed quarterly rebalance',
    lastActionTime: '2026-02-24T16:00:00Z',
    decisionsToday: 3,
    confidence: 0.88,
  },
  {
    id: 'a2',
    role: 'risk_analyst',
    name: 'Risk Analyst',
    status: 'active',
    lastAction: 'Updated VaR estimates',
    lastActionTime: '2026-02-25T08:00:00Z',
    decisionsToday: 5,
    confidence: 0.92,
  },
  {
    id: 'a3',
    role: 'factor_researcher',
    name: 'Factor Researcher',
    status: 'idle',
    lastAction: 'Completed factor IC analysis',
    lastActionTime: '2026-02-24T22:00:00Z',
    decisionsToday: 2,
    confidence: 0.85,
  },
  {
    id: 'a4',
    role: 'execution_agent',
    name: 'Execution Agent',
    status: 'active',
    lastAction: 'Splitting AAPL sell into 3 tranches',
    lastActionTime: '2026-02-25T09:30:00Z',
    decisionsToday: 8,
    confidence: 0.95,
  },
];

// ── Decision Feed (50 items) ──

export const MOCK_DECISION_FEED: DecisionFeedItem[] = [
  { id: 'd1', agent: 'portfolio_manager', type: 'rebalance', title: 'Quarterly rebalance triggered', reasoning: 'Calendar-based quarterly review date reached. Portfolio drift at 4.2% exceeds monitoring threshold.', outcome: 'executed', timestamp: '2026-02-24T16:00:00Z', confidence: 0.92, impact: '-$53 cost, 4.2% turnover' },
  { id: 'd2', agent: 'risk_analyst', type: 'risk_alert', title: 'CVaR approaching limit', reasoning: '99% CVaR at 91% of limit. Recommend reducing tail-risk exposure in tech sector.', outcome: 'approved', timestamp: '2026-02-25T08:00:00Z', confidence: 0.88, impact: 'Advisory only' },
  { id: 'd3', agent: 'factor_researcher', type: 'factor_tilt', title: 'Increase value tilt', reasoning: 'Expansion regime detected. Value factor IC improving (0.052 rolling 3M). Historical backtest supports +3% tilt.', outcome: 'approved', timestamp: '2026-02-24T22:00:00Z', confidence: 0.78 },
  { id: 'd4', agent: 'execution_agent', type: 'trade', title: 'AAPL sell split into tranches', reasoning: 'Order size exceeds 15% of ADV. Splitting into 3 TWAP tranches over 2 hours to minimize market impact.', outcome: 'executed', timestamp: '2026-02-25T09:30:00Z', confidence: 0.95, impact: 'Est. savings: $120 vs single fill' },
  { id: 'd5', agent: 'risk_analyst', type: 'veto', title: 'Vetoed TSLA position increase', reasoning: 'Proposed +2% TSLA weight would breach single-position concentration limit of 5% (current: 3.8%).', outcome: 'rejected', timestamp: '2026-02-24T14:00:00Z', confidence: 0.98 },
  { id: 'd6', agent: 'portfolio_manager', type: 'regime_change', title: 'Regime transition detected', reasoning: 'HMM model signals transition from low-vol to medium-vol state (p=0.82). Adjusting risk budget.', outcome: 'executed', timestamp: '2026-02-24T09:15:00Z', confidence: 0.82 },
  { id: 'd7', agent: 'factor_researcher', type: 'factor_tilt', title: 'Reduce momentum exposure', reasoning: 'Momentum factor showing reversal signals. 12-1 momentum IC dropped from 0.058 to 0.032 over 1M.', outcome: 'approved', timestamp: '2026-02-23T18:00:00Z', confidence: 0.72 },
  { id: 'd8', agent: 'risk_analyst', type: 'risk_alert', title: 'Correlation spike alert', reasoning: 'Average pairwise correlation jumped from 0.32 to 0.48. Diversification benefit reduced.', outcome: 'approved', timestamp: '2026-02-23T10:00:00Z', confidence: 0.90 },
  { id: 'd9', agent: 'execution_agent', type: 'trade', title: 'Tax-loss harvest: INTC→TXN', reasoning: 'INTC at -8.2% unrealized loss. TXN identified as correlated substitute (0.82 correlation, same sector).', outcome: 'executed', timestamp: '2026-02-22T15:00:00Z', confidence: 0.88, impact: 'Est. tax savings: $3,200' },
  { id: 'd10', agent: 'portfolio_manager', type: 'rebalance', title: 'Drift threshold triggered', reasoning: 'NVDA drifted +2.8% above target weight (threshold: 2.5%). Initiating targeted rebalance.', outcome: 'executed', timestamp: '2026-02-22T14:00:00Z', confidence: 0.94 },
  { id: 'd11', agent: 'risk_analyst', type: 'risk_alert', title: 'VaR limit at 92%', reasoning: '1-day 95% VaR at $245K vs $265K limit. Within tolerance but monitoring closely.', outcome: 'approved', timestamp: '2026-02-22T08:00:00Z', confidence: 0.85 },
  { id: 'd12', agent: 'factor_researcher', type: 'factor_tilt', title: 'Quality factor signal strengthening', reasoning: 'Gross profitability IC at 0.061, highest in 6 months. Recommend increasing quality tilt by 2%.', outcome: 'pending', timestamp: '2026-02-21T22:00:00Z', confidence: 0.80 },
  { id: 'd13', agent: 'execution_agent', type: 'trade', title: 'Dividend reinvestment', reasoning: 'Accumulated $4,280 in dividends. Reinvesting proportionally across existing positions.', outcome: 'executed', timestamp: '2026-02-21T16:00:00Z', confidence: 0.99 },
  { id: 'd14', agent: 'risk_analyst', type: 'veto', title: 'Blocked leveraged position', reasoning: 'Proposed 1.1x leverage would violate fund mandate. Max leverage ratio: 1.0x.', outcome: 'rejected', timestamp: '2026-02-21T11:00:00Z', confidence: 0.99 },
  { id: 'd15', agent: 'portfolio_manager', type: 'regime_change', title: 'Macro regime: Expansion', reasoning: 'GDP growth accelerating, yield curve steepening. Adjusting factor tilts for expansion regime.', outcome: 'executed', timestamp: '2026-02-21T08:00:00Z', confidence: 0.75 },
  { id: 'd16', agent: 'factor_researcher', type: 'factor_tilt', title: 'Dividend yield neutral', reasoning: 'Yield spread between dividend payers and market stable at historical median. No tilt recommended.', outcome: 'approved', timestamp: '2026-02-20T22:00:00Z', confidence: 0.82 },
  { id: 'd17', agent: 'execution_agent', type: 'trade', title: 'Sector rotation executed', reasoning: 'Reduced healthcare -0.8%, increased financials +1.2%, energy flat per PM directive.', outcome: 'executed', timestamp: '2026-02-20T15:00:00Z', confidence: 0.91 },
  { id: 'd18', agent: 'risk_analyst', type: 'risk_alert', title: 'Tracking error within budget', reasoning: 'TE at 3.2% vs 5% budget. Portfolio aligned with benchmark within acceptable range.', outcome: 'approved', timestamp: '2026-02-20T08:00:00Z', confidence: 0.95 },
  { id: 'd19', agent: 'portfolio_manager', type: 'rebalance', title: 'Sector rebalance completed', reasoning: '6 trades executed. Financials +1.2%, Healthcare -0.8%. Total cost: $38.', outcome: 'executed', timestamp: '2026-02-19T16:00:00Z', confidence: 0.90 },
  { id: 'd20', agent: 'factor_researcher', type: 'factor_tilt', title: 'Low-risk factor underweight', reasoning: 'Expansion regime reduces defensive benefit. Recommend -2% tilt on low-volatility factor.', outcome: 'approved', timestamp: '2026-02-19T22:00:00Z', confidence: 0.76 },
  { id: 'd21', agent: 'risk_analyst', type: 'risk_alert', title: 'Liquidity check passed', reasoning: 'All positions can be liquidated within 2 trading days at 10% ADV participation rate.', outcome: 'approved', timestamp: '2026-02-19T08:00:00Z', confidence: 0.98 },
  { id: 'd22', agent: 'execution_agent', type: 'trade', title: 'Cross-trade optimization', reasoning: 'Matched internal buy/sell for JNJ. Saved $12 in market impact costs.', outcome: 'executed', timestamp: '2026-02-18T15:30:00Z', confidence: 0.96 },
  { id: 'd23', agent: 'portfolio_manager', type: 'rebalance', title: 'Rebalance deferred', reasoning: 'Drift at 1.8%, below 2.5% threshold. No action required until next calendar date.', outcome: 'approved', timestamp: '2026-02-18T08:00:00Z', confidence: 0.88 },
  { id: 'd24', agent: 'risk_analyst', type: 'veto', title: 'Blocked emerging market add', reasoning: 'EM allocation proposal would increase portfolio beta to 1.25 (limit: 1.20).', outcome: 'rejected', timestamp: '2026-02-17T14:00:00Z', confidence: 0.92 },
  { id: 'd25', agent: 'factor_researcher', type: 'factor_tilt', title: 'Investment factor neutral', reasoning: 'Asset growth factor IC near zero (-0.01). No predictive signal detected.', outcome: 'approved', timestamp: '2026-02-17T22:00:00Z', confidence: 0.84 },
  { id: 'd26', agent: 'execution_agent', type: 'trade', title: 'Limit order filled: MSFT', reasoning: 'Passive limit order for MSFT filled at $428.50, saving $0.35/share vs market.', outcome: 'executed', timestamp: '2026-02-17T11:00:00Z', confidence: 0.97 },
  { id: 'd27', agent: 'risk_analyst', type: 'risk_alert', title: 'VIX spike detected', reasoning: 'VIX rose from 14.2 to 18.5 intraday. Monitoring for sustained elevation.', outcome: 'approved', timestamp: '2026-02-17T10:00:00Z', confidence: 0.85 },
  { id: 'd28', agent: 'portfolio_manager', type: 'regime_change', title: 'Volatility regime: medium', reasoning: 'HMM medium-vol probability exceeded 0.6 threshold. Risk budget reduced by 10%.', outcome: 'executed', timestamp: '2026-02-16T22:00:00Z', confidence: 0.78 },
  { id: 'd29', agent: 'factor_researcher', type: 'factor_tilt', title: 'Profitability tilt +2%', reasoning: 'Quality earnings premium widening. ROE factor IC at 0.044, above 75th percentile.', outcome: 'approved', timestamp: '2026-02-16T18:00:00Z', confidence: 0.81 },
  { id: 'd30', agent: 'execution_agent', type: 'trade', title: 'Rebalance trades: 10 fills', reasoning: 'All 10 rebalance orders filled within VWAP tolerance of ±5bps.', outcome: 'executed', timestamp: '2026-02-14T16:00:00Z', confidence: 0.94 },
  { id: 'd31', agent: 'risk_analyst', type: 'risk_alert', title: 'Concentration: top 5 at 16.4%', reasoning: 'Within 25% limit. Sector concentration also within bounds.', outcome: 'approved', timestamp: '2026-02-14T08:00:00Z', confidence: 0.96 },
  { id: 'd32', agent: 'portfolio_manager', type: 'rebalance', title: 'Walk-forward CV completed', reasoning: 'Average Sharpe 0.55 (±0.12) over 20 rolling windows. Model stable.', outcome: 'approved', timestamp: '2026-02-14T06:00:00Z', confidence: 0.82 },
  { id: 'd33', agent: 'factor_researcher', type: 'factor_tilt', title: 'Sentiment factor weak', reasoning: 'Recommendation change IC at 0.032, below significance threshold.', outcome: 'approved', timestamp: '2026-02-13T22:00:00Z', confidence: 0.70 },
  { id: 'd34', agent: 'execution_agent', type: 'trade', title: 'Order slippage report', reasoning: 'Weekly slippage summary: avg 2.1bps, within 5bps target. 42 orders total.', outcome: 'executed', timestamp: '2026-02-13T16:00:00Z', confidence: 0.93 },
  { id: 'd35', agent: 'risk_analyst', type: 'veto', title: 'Blocked concentrated sector bet', reasoning: 'Proposed tech weight of 32% exceeds 30% sector limit.', outcome: 'rejected', timestamp: '2026-02-13T11:00:00Z', confidence: 0.97 },
  { id: 'd36', agent: 'portfolio_manager', type: 'rebalance', title: 'Monthly review: no action', reasoning: 'Portfolio within all drift thresholds. Next scheduled rebalance in 4 weeks.', outcome: 'approved', timestamp: '2026-02-12T08:00:00Z', confidence: 0.90 },
  { id: 'd37', agent: 'factor_researcher', type: 'factor_tilt', title: 'Ownership factor weak signal', reasoning: 'Net insider buying IC at 0.025. Insufficient for active tilt.', outcome: 'approved', timestamp: '2026-02-11T22:00:00Z', confidence: 0.68 },
  { id: 'd38', agent: 'execution_agent', type: 'trade', title: 'Dividend capture trades', reasoning: 'Positioned for 4 upcoming ex-dividend dates. Expected income: $1,850.', outcome: 'executed', timestamp: '2026-02-11T14:00:00Z', confidence: 0.89 },
  { id: 'd39', agent: 'risk_analyst', type: 'risk_alert', title: 'Drawdown recovery tracking', reasoning: 'Current drawdown: -2.8%. Recovery trend positive over 5 trading days.', outcome: 'approved', timestamp: '2026-02-11T08:00:00Z', confidence: 0.86 },
  { id: 'd40', agent: 'portfolio_manager', type: 'regime_change', title: 'GDP data: expansion confirmed', reasoning: 'Q4 GDP revised up to 2.8%. Maintaining expansion regime classification.', outcome: 'executed', timestamp: '2026-02-10T09:00:00Z', confidence: 0.80 },
  { id: 'd41', agent: 'factor_researcher', type: 'factor_tilt', title: 'Liquidity factor stable', reasoning: 'Amihud illiquidity IC at -0.018. No significant signal for portfolio action.', outcome: 'approved', timestamp: '2026-02-10T22:00:00Z', confidence: 0.74 },
  { id: 'd42', agent: 'execution_agent', type: 'trade', title: 'Corporate action: AVGO split', reasoning: 'Adjusted position sizing for AVGO 10:1 stock split. No economic impact.', outcome: 'executed', timestamp: '2026-02-10T12:00:00Z', confidence: 0.99 },
  { id: 'd43', agent: 'risk_analyst', type: 'risk_alert', title: 'Stress test update', reasoning: 'Updated 8 stress scenarios with latest factor loadings. Worst case: -38.2% (GFC replay).', outcome: 'approved', timestamp: '2026-02-10T08:00:00Z', confidence: 0.91 },
  { id: 'd44', agent: 'portfolio_manager', type: 'rebalance', title: 'Tax-loss harvest review', reasoning: 'Identified 3 positions with unrealized losses >5%. Recommending swaps.', outcome: 'approved', timestamp: '2026-02-07T16:00:00Z', confidence: 0.85 },
  { id: 'd45', agent: 'factor_researcher', type: 'factor_tilt', title: 'Value spread widening', reasoning: 'Value spread at 85th percentile historically. Supports overweight stance.', outcome: 'approved', timestamp: '2026-02-07T22:00:00Z', confidence: 0.79 },
  { id: 'd46', agent: 'execution_agent', type: 'trade', title: 'Batch orders submitted', reasoning: '14 rebalance orders submitted as MOC (market-on-close) for guaranteed closing price.', outcome: 'executed', timestamp: '2026-02-07T15:45:00Z', confidence: 0.92 },
  { id: 'd47', agent: 'risk_analyst', type: 'veto', title: 'Blocked high-beta additions', reasoning: 'Adding 3 high-beta names would push portfolio beta to 1.18 (limit: 1.20, buffer: 0.05).', outcome: 'rejected', timestamp: '2026-02-06T14:00:00Z', confidence: 0.88 },
  { id: 'd48', agent: 'portfolio_manager', type: 'rebalance', title: 'Quarterly strategy review', reasoning: 'Max Sharpe continues to outperform. No strategy change recommended.', outcome: 'approved', timestamp: '2026-02-06T09:00:00Z', confidence: 0.87 },
  { id: 'd49', agent: 'factor_researcher', type: 'factor_tilt', title: 'Factor validation complete', reasoning: 'Annual OOS validation: 8/17 factors significant at 5% level. Model validated.', outcome: 'approved', timestamp: '2026-02-05T22:00:00Z', confidence: 0.83 },
  { id: 'd50', agent: 'execution_agent', type: 'trade', title: 'Algo benchmark met', reasoning: 'Monthly execution quality: 95% of trades within VWAP ±3bps. Target: 90%.', outcome: 'executed', timestamp: '2026-02-05T16:00:00Z', confidence: 0.96 },
];

// ── Agent Configs ──

export const MOCK_AGENT_CONFIGS: AgentConfig[] = [
  { role: 'portfolio_manager', name: 'Portfolio Manager', enabled: true, autonomyLevel: 'supervised', maxPositionSize: 0.05, maxDailyTurnover: 0.03, riskBudget: 0.15 },
  { role: 'risk_analyst', name: 'Risk Analyst', enabled: true, autonomyLevel: 'full', maxPositionSize: 0, maxDailyTurnover: 0, riskBudget: 0 },
  { role: 'factor_researcher', name: 'Factor Researcher', enabled: true, autonomyLevel: 'advisory', maxPositionSize: 0, maxDailyTurnover: 0, riskBudget: 0 },
  { role: 'execution_agent', name: 'Execution Agent', enabled: true, autonomyLevel: 'full', maxPositionSize: 0.05, maxDailyTurnover: 0.05, riskBudget: 0.001 },
];

// ── Workflow States ──

export const MOCK_WORKFLOW_STATES: WorkflowState[] = [
  { step: 'data_ingestion', label: 'Data Ingestion', status: 'completed', startedAt: '2026-02-25T06:00:00Z', completedAt: '2026-02-25T06:12:00Z', detail: 'Price data, fundamentals, macro indicators' },
  { step: 'factor_analysis', label: 'Factor Analysis', status: 'completed', startedAt: '2026-02-25T06:12:00Z', completedAt: '2026-02-25T06:28:00Z', detail: '17 factors computed, 8 significant' },
  { step: 'risk_assessment', label: 'Risk Assessment', status: 'completed', startedAt: '2026-02-25T06:28:00Z', completedAt: '2026-02-25T06:35:00Z', detail: 'VaR, stress tests, limits checked' },
  { step: 'optimization', label: 'Portfolio Optimization', status: 'running', startedAt: '2026-02-25T06:35:00Z', detail: 'Max Sharpe with BL views' },
  { step: 'execution', label: 'Trade Execution', status: 'pending', detail: 'Awaiting optimization output' },
  { step: 'monitoring', label: 'Continuous Monitoring', status: 'pending', detail: 'Post-trade compliance checks' },
];

// ── AI Fund Summary ──

export const MOCK_AI_FUND_SUMMARY: AIFundSummary = {
  aum: 12_450_000,
  holdings: [
    { ticker: 'AAPL', weight: 0.048, saaTarget: 0.05, taaTilt: -0.002, taaReasoning: 'Slight underweight due to valuation stretch', sector: 'Technology', returnMtd: 0.032, returnYtd: 0.078 },
    { ticker: 'MSFT', weight: 0.045, saaTarget: 0.045, taaTilt: 0, taaReasoning: 'On target weight', sector: 'Technology', returnMtd: 0.018, returnYtd: 0.062 },
    { ticker: 'NVDA', weight: 0.042, saaTarget: 0.035, taaTilt: 0.007, taaReasoning: 'Momentum tilt: strong factor IC', sector: 'Technology', returnMtd: 0.058, returnYtd: 0.142 },
    { ticker: 'GOOGL', weight: 0.038, saaTarget: 0.04, taaTilt: -0.002, taaReasoning: 'Regulatory headwinds', sector: 'Technology', returnMtd: -0.012, returnYtd: 0.035 },
    { ticker: 'AMZN', weight: 0.036, saaTarget: 0.035, taaTilt: 0.001, taaReasoning: 'AWS growth re-accelerating', sector: 'Consumer Discretionary', returnMtd: 0.025, returnYtd: 0.068 },
    { ticker: 'JPM', weight: 0.034, saaTarget: 0.03, taaTilt: 0.004, taaReasoning: 'Expansion regime: overweight financials', sector: 'Financials', returnMtd: 0.041, returnYtd: 0.095 },
    { ticker: 'JNJ', weight: 0.032, saaTarget: 0.035, taaTilt: -0.003, taaReasoning: 'Defensive tilt reduced in expansion', sector: 'Healthcare', returnMtd: -0.005, returnYtd: 0.012 },
    { ticker: 'V', weight: 0.028, saaTarget: 0.025, taaTilt: 0.003, taaReasoning: 'Quality factor overweight', sector: 'Financials', returnMtd: 0.022, returnYtd: 0.055 },
    { ticker: 'UNH', weight: 0.026, saaTarget: 0.028, taaTilt: -0.002, taaReasoning: 'Policy uncertainty drag', sector: 'Healthcare', returnMtd: -0.018, returnYtd: -0.008 },
    { ticker: 'XOM', weight: 0.024, saaTarget: 0.02, taaTilt: 0.004, taaReasoning: 'Value + dividend tilt', sector: 'Energy', returnMtd: 0.035, returnYtd: 0.072 },
    { ticker: 'PG', weight: 0.022, saaTarget: 0.025, taaTilt: -0.003, taaReasoning: 'Low-vol underweight in expansion', sector: 'Consumer Staples', returnMtd: 0.008, returnYtd: 0.018 },
    { ticker: 'HD', weight: 0.020, saaTarget: 0.02, taaTilt: 0, taaReasoning: 'On target', sector: 'Consumer Discretionary', returnMtd: 0.015, returnYtd: 0.042 },
    { ticker: 'BAC', weight: 0.018, saaTarget: 0.015, taaTilt: 0.003, taaReasoning: 'Rate sensitivity in expansion', sector: 'Financials', returnMtd: 0.038, returnYtd: 0.088 },
    { ticker: 'AVGO', weight: 0.016, saaTarget: 0.015, taaTilt: 0.001, taaReasoning: 'AI capex beneficiary', sector: 'Technology', returnMtd: 0.045, returnYtd: 0.112 },
    { ticker: 'LLY', weight: 0.015, saaTarget: 0.018, taaTilt: -0.003, taaReasoning: 'Valuation premium compression', sector: 'Healthcare', returnMtd: -0.022, returnYtd: 0.005 },
    { ticker: 'CASH', weight: 0.556, saaTarget: 0.559, taaTilt: -0.003, taaReasoning: 'Residual allocation', sector: 'Cash', returnMtd: 0.004, returnYtd: 0.012 },
  ],
  signals: [
    { name: 'Market Regime', value: 0.65, label: 'bullish', description: 'HMM expansion state probability 0.82' },
    { name: 'Factor Momentum', value: 0.32, label: 'bullish', description: 'Value and quality ICs improving' },
    { name: 'Volatility Forecast', value: -0.15, label: 'neutral', description: 'VIX at 15.2, near historical median' },
    { name: 'Credit Spread', value: -0.08, label: 'neutral', description: 'IG spreads stable at 95bps' },
    { name: 'Sentiment', value: -0.42, label: 'bearish', description: 'Retail positioning extended, contrarian signal' },
  ],
  riskStatuses: [
    { metric: '1-Day VaR (95%)', current: 245_000, limit: 265_000, utilization: 0.92, status: 'warning' },
    { metric: '1-Day CVaR (99%)', current: 380_000, limit: 420_000, utilization: 0.91, status: 'warning' },
    { metric: 'Max Position', current: 0.048, limit: 0.05, utilization: 0.96, status: 'warning' },
    { metric: 'Sector Concentration', current: 0.22, limit: 0.30, utilization: 0.73, status: 'ok' },
    { metric: 'Portfolio Beta', current: 1.08, limit: 1.20, utilization: 0.90, status: 'ok' },
    { metric: 'Tracking Error', current: 0.032, limit: 0.05, utilization: 0.64, status: 'ok' },
  ],
};

// ── Extended Agent Configs ──

export const MOCK_AGENT_CONFIGS_EXTENDED: AgentConfigExtended[] = [
  { role: 'portfolio_manager', name: 'Portfolio Manager', enabled: true, autonomyLevel: 'supervised', maxPositionSize: 0.05, maxDailyTurnover: 0.03, riskBudget: 0.15, llmModel: 'Claude 4 Opus', temperature: 0.3, frequency: 'Daily 06:00 UTC', costBudgetDaily: 5.00, costUsedToday: 2.85 },
  { role: 'risk_analyst', name: 'Risk Analyst', enabled: true, autonomyLevel: 'full', maxPositionSize: 0, maxDailyTurnover: 0, riskBudget: 0, llmModel: 'Claude 4 Sonnet', temperature: 0.1, frequency: 'Continuous', costBudgetDaily: 8.00, costUsedToday: 4.20 },
  { role: 'factor_researcher', name: 'Factor Researcher', enabled: true, autonomyLevel: 'advisory', maxPositionSize: 0, maxDailyTurnover: 0, riskBudget: 0, llmModel: 'Claude 4 Opus', temperature: 0.5, frequency: 'Daily 22:00 UTC', costBudgetDaily: 3.00, costUsedToday: 1.60 },
  { role: 'execution_agent', name: 'Execution Agent', enabled: true, autonomyLevel: 'full', maxPositionSize: 0.05, maxDailyTurnover: 0.05, riskBudget: 0.001, llmModel: 'Claude 4 Haiku', temperature: 0.0, frequency: 'Market hours', costBudgetDaily: 2.00, costUsedToday: 1.10 },
];

// ── Orchestration Config ──

export const MOCK_ORCHESTRATION_CONFIG: OrchestrationConfig = {
  pipeline: ['data_ingestion', 'factor_analysis', 'risk_assessment', 'optimization', 'execution', 'monitoring'],
  vetoEnabled: true,
  conflictResolution: 'risk_priority',
  maxLatencyMs: 30_000,
};

// ── Veto Log ──

export const MOCK_VETO_LOG: VetoLogEntry[] = [
  { id: 'v1', vetoAgent: 'risk_analyst', targetAgent: 'portfolio_manager', action: 'Increase TSLA to 6%', reason: 'Exceeds 5% single-position limit', timestamp: '2026-02-24T14:00:00Z', overridden: false },
  { id: 'v2', vetoAgent: 'risk_analyst', targetAgent: 'portfolio_manager', action: 'Apply 1.1x leverage', reason: 'Violates fund mandate max leverage 1.0x', timestamp: '2026-02-21T11:00:00Z', overridden: false },
  { id: 'v3', vetoAgent: 'risk_analyst', targetAgent: 'factor_researcher', action: 'Add EM allocation +5%', reason: 'Portfolio beta would reach 1.25 (limit 1.20)', timestamp: '2026-02-17T14:00:00Z', overridden: false },
  { id: 'v4', vetoAgent: 'risk_analyst', targetAgent: 'portfolio_manager', action: 'Tech sector to 32%', reason: 'Exceeds 30% sector concentration limit', timestamp: '2026-02-13T11:00:00Z', overridden: false },
  { id: 'v5', vetoAgent: 'risk_analyst', targetAgent: 'execution_agent', action: 'Add 3 high-beta names', reason: 'Beta would reach 1.18 (limit 1.20, buffer 0.05)', timestamp: '2026-02-06T14:00:00Z', overridden: false },
  { id: 'v6', vetoAgent: 'portfolio_manager', targetAgent: 'factor_researcher', action: 'Momentum tilt +5%', reason: 'Reversal signals detected, IC declining', timestamp: '2026-02-04T10:00:00Z', overridden: true },
  { id: 'v7', vetoAgent: 'risk_analyst', targetAgent: 'execution_agent', action: 'Market order 50K shares', reason: 'Exceeds 25% of ADV, use TWAP instead', timestamp: '2026-02-03T14:30:00Z', overridden: false },
  { id: 'v8', vetoAgent: 'portfolio_manager', targetAgent: 'factor_researcher', action: 'Remove quality tilt', reason: 'Quality premium still widening, keep tilt', timestamp: '2026-01-28T09:00:00Z', overridden: true },
];

// ── Comparison Data ──

function generateComparisonEquity(): { date: string; user: number; ai: number; benchmark: number }[] {
  const points: { date: string; user: number; ai: number; benchmark: number }[] = [];
  let user = 100, ai = 100, benchmark = 100;
  const start = new Date('2025-02-25');
  for (let i = 0; i < 252; i++) {
    const d = new Date(start);
    d.setDate(d.getDate() + i);
    if (d.getDay() === 0 || d.getDay() === 6) continue;
    user *= 1 + (Math.random() - 0.48) * 0.012;
    ai *= 1 + (Math.random() - 0.47) * 0.011;
    benchmark *= 1 + (Math.random() - 0.485) * 0.010;
    points.push({
      date: d.toISOString().split('T')[0],
      user: Math.round(user * 100) / 100,
      ai: Math.round(ai * 100) / 100,
      benchmark: Math.round(benchmark * 100) / 100,
    });
  }
  return points;
}

export const MOCK_COMPARISON_DATA: ComparisonData = {
  metrics: [
    { metric: 'Total Return', user: 0.128, ai: 0.156, benchmark: 0.112, unit: '%' },
    { metric: 'Annualized Return', user: 0.128, ai: 0.156, benchmark: 0.112, unit: '%' },
    { metric: 'Volatility', user: 0.142, ai: 0.128, benchmark: 0.135, unit: '%' },
    { metric: 'Sharpe Ratio', user: 0.62, ai: 0.88, benchmark: 0.55, unit: 'x' },
    { metric: 'Sortino Ratio', user: 0.85, ai: 1.22, benchmark: 0.72, unit: 'x' },
    { metric: 'Max Drawdown', user: -0.085, ai: -0.062, benchmark: -0.098, unit: '%' },
    { metric: 'Calmar Ratio', user: 1.51, ai: 2.52, benchmark: 1.14, unit: 'x' },
    { metric: 'Win Rate', user: 0.52, ai: 0.56, benchmark: 0.51, unit: '%' },
    { metric: 'Avg Turnover', user: 0.045, ai: 0.032, benchmark: 0.008, unit: '%' },
    { metric: 'Trading Costs', user: 1_850, ai: 1_220, benchmark: 320, unit: '$' },
    { metric: 'Tracking Error', user: 0.048, ai: 0.032, benchmark: 0, unit: '%' },
    { metric: 'Information Ratio', user: 0.33, ai: 0.82, benchmark: 0, unit: 'x' },
  ],
  equity: generateComparisonEquity(),
  divergences: [
    { date: '2026-02-24', asset: 'NVDA', userAction: 'Hold', aiAction: 'Add +0.7%', userOutcome: 0.012, aiOutcome: 0.058, delta: 0.046 },
    { date: '2026-02-20', asset: 'XOM', userAction: 'Reduce -1%', aiAction: 'Add +0.4%', userOutcome: -0.008, aiOutcome: 0.035, delta: 0.043 },
    { date: '2026-02-18', asset: 'TSLA', userAction: 'Add +2%', aiAction: 'Veto (limit)', userOutcome: -0.032, aiOutcome: 0, delta: 0.032 },
    { date: '2026-02-14', asset: 'JPM', userAction: 'Hold', aiAction: 'Add +0.4%', userOutcome: 0.015, aiOutcome: 0.041, delta: 0.026 },
    { date: '2026-02-12', asset: 'JNJ', userAction: 'Add +1%', aiAction: 'Reduce -0.3%', userOutcome: -0.005, aiOutcome: 0.008, delta: 0.013 },
    { date: '2026-02-10', asset: 'AAPL', userAction: 'Hold', aiAction: 'Trim -0.2%', userOutcome: 0.018, aiOutcome: 0.020, delta: 0.002 },
    { date: '2026-02-07', asset: 'GOOGL', userAction: 'Add +0.5%', aiAction: 'Hold', userOutcome: -0.012, aiOutcome: 0, delta: 0.012 },
    { date: '2026-02-05', asset: 'BAC', userAction: 'Hold', aiAction: 'Add +0.3%', userOutcome: 0.022, aiOutcome: 0.038, delta: 0.016 },
    { date: '2026-02-03', asset: 'MSFT', userAction: 'Reduce -0.5%', aiAction: 'Hold', userOutcome: -0.015, aiOutcome: 0.018, delta: 0.033 },
    { date: '2026-01-31', asset: 'V', userAction: 'Hold', aiAction: 'Add +0.3%', userOutcome: 0.008, aiOutcome: 0.022, delta: 0.014 },
    { date: '2026-01-28', asset: 'LLY', userAction: 'Add +1%', aiAction: 'Reduce -0.3%', userOutcome: -0.022, aiOutcome: 0.005, delta: 0.027 },
    { date: '2026-01-24', asset: 'AVGO', userAction: 'Hold', aiAction: 'Add +0.1%', userOutcome: 0.032, aiOutcome: 0.045, delta: 0.013 },
    { date: '2026-01-22', asset: 'PG', userAction: 'Add +0.5%', aiAction: 'Reduce -0.3%', userOutcome: 0.003, aiOutcome: 0.008, delta: 0.005 },
    { date: '2026-01-20', asset: 'UNH', userAction: 'Hold', aiAction: 'Reduce -0.2%', userOutcome: -0.018, aiOutcome: -0.005, delta: 0.013 },
    { date: '2026-01-15', asset: 'HD', userAction: 'Reduce -0.3%', aiAction: 'Hold', userOutcome: -0.008, aiOutcome: 0.015, delta: 0.023 },
  ],
};
