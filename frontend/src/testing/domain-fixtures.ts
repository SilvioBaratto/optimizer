// Typed factory functions over src/app/models/* DTOs. Each returns a minimal
// schema-valid object and accepts a Partial `overrides` merged last. Plural
// factories (…Metrics/…Scenarios) return a one-element array; their `overrides`
// merge into that single element so callers tweak one field without rebuilding
// the row. For object factories that embed a nested array (sectors / trades /
// entries / assets), `overrides` patches top-level keys only — to change an
// inner row, pass the whole array key (e.g. `{ entries: [...] }`).
// Keep factories < 10 lines — extend by adding a field, not a branch.

import type { PortfolioDto, SnapshotDto } from '../app/core/models/portfolio-api.model';
import type { ApiMarketSnapshotResponse } from '../app/core/models/dashboard-api.model';
import type {
  BrinsonApiResponse,
  FactorAttributionApiResponse,
} from '../app/models/attribution.model';
import type {
  ConcentrationAssetApi,
  ConcentrationMetric,
  CorrelationApiResponse,
  CorrelationData,
  LiquidityMetric,
  StressScenario,
  StressScenarioApiResponse,
  StressScenarioItemApi,
  VarApiResponse,
} from '../app/models/risk.model';
import type {
  DriftApiResponse,
  RebalanceDecideApiResponse,
  RebalancePreviewApiResponse,
} from '../app/models/rebalancing.model';
import type {
  CMASet,
  FactorICReport,
  FactorScoreApiResponse,
  FactorScoreDto,
  FactorSelectApiResponse,
  TAASignal,
} from '../app/models/factor.model';
import type { JobListResponse, JobSummary } from '../app/core/models/jobs.model';
import type { ReportJobCreateResponse } from '../app/core/models/report.model';
import type { UniverseScreenResponse } from '../app/core/models/universe.model';
import type { PriceHistory } from '../app/core/models/yfinance.model';
import type {
  EntropyPoolingResponse,
  GenerateViewsResponse,
  OpinionPoolResponse,
  OptimizationRunListResponse,
  OptimizationRunResponse,
} from '../app/models/optimization.model';
import type {
  BacktestAsyncResponse,
  BacktestProgressResponse,
  BacktestRunResponse,
} from '../app/models/backtest.model';
import type { MacroCalibrationApiResponse } from '../app/models/macro-intelligence.model';
import type {
  CreateSessionResponse,
  StepPollResponse,
} from '../app/models/pipeline-builder.model';
import type { DriftResponse as DriftResponseRich } from '../app/models/drift.model';

const ISO = '2026-01-01T00:00:00.000Z';

export function makePortfolioDto(overrides: Partial<PortfolioDto> = {}): PortfolioDto {
  return {
    id: 'pf-1',
    name: 'Test Portfolio',
    description: null,
    currency: 'EUR',
    benchmark_ticker: 'SPY',
    is_active: true,
    created_at: ISO,
    updated_at: ISO,
    ...overrides,
  };
}

export function makeSnapshotDto(overrides: Partial<SnapshotDto> = {}): SnapshotDto {
  return {
    id: 'snap-1',
    portfolio_id: 'pf-1',
    snapshot_date: '2026-01-01',
    snapshot_type: 'optimization',
    weights: { AAPL: 0.6, MSFT: 0.4 },
    sector_mapping: null,
    summary: null,
    optimizer_config: null,
    turnover: null,
    holding_count: 2,
    created_at: ISO,
    ...overrides,
  };
}

export function makeBrinsonResponse(
  overrides: Partial<BrinsonApiResponse> = {},
): BrinsonApiResponse {
  return {
    sectors: [
      {
        sector: 'Technology',
        portfolioWeight: 0.6,
        benchmarkWeight: 0.5,
        portfolioReturn: 0.1,
        benchmarkReturn: 0.08,
        allocationEffect: 0.002,
        selectionEffect: 0.001,
        interactionEffect: 0.0,
        totalEffect: 0.003,
      },
    ],
    totalAllocation: 0.002,
    totalSelection: 0.001,
    totalInteraction: 0.0,
    totalActiveReturn: 0.003,
    portfolioReturn: 0.1,
    benchmarkReturn: 0.08,
    ...overrides,
  };
}

export function makeFactorAttributionResponse(
  overrides: Partial<FactorAttributionApiResponse> = {},
): FactorAttributionApiResponse {
  return {
    factors: [
      { factorName: 'momentum', exposure: 0.5, factorReturn: 0.04, contribution: 0.02 },
    ],
    portfolioReturn: 0.1,
    explainedReturn: 0.08,
    residual: 0.02,
    ...overrides,
  };
}

export function makeVarApiResponse(overrides: Partial<VarApiResponse> = {}): VarApiResponse {
  return {
    var: { '0.95': 0.03 },
    cvar: { '0.95': 0.05 },
    method: 'historical',
    lookback: 252,
    nObservations: 252,
    ...overrides,
  };
}

export function makeCorrelationData(overrides: Partial<CorrelationData> = {}): CorrelationData {
  return {
    assets: ['AAPL', 'MSFT'],
    matrix: [
      [1, 0.5],
      [0.5, 1],
    ],
    ...overrides,
  };
}

// Wire DTO for GET /risk/correlation — adds `clusterLabels` that the client-only
// `CorrelationData` omits. Parity pins the wire shape, not the client view.
export function makeCorrelationApiResponse(
  overrides: Partial<CorrelationApiResponse> = {},
): CorrelationApiResponse {
  return {
    assets: ['AAPL', 'MSFT'],
    matrix: [
      [1, 0.5],
      [0.5, 1],
    ],
    clusterLabels: [0, 0],
    ...overrides,
  };
}

// Wire DTO row for GET /risk/concentration — the bare wire shape (ticker, name,
// weight), separate from the client-enriched `ConcentrationMetric`.
export function makeConcentrationAssetApi(
  overrides: Partial<ConcentrationAssetApi> = {},
): ConcentrationAssetApi {
  return { ticker: 'AAPL', name: 'Apple Inc.', weight: 0.6, ...overrides };
}

export function makeConcentrationMetrics(
  overrides: Partial<ConcentrationMetric> = {},
): ConcentrationMetric[] {
  return [
    {
      ticker: 'AAPL',
      name: 'Apple Inc.',
      weight: 0.6,
      riskContribution: 0.7,
      componentVar: 0.04,
      ...overrides,
    },
  ];
}

export function makeLiquidityMetrics(
  overrides: Partial<LiquidityMetric> = {},
): LiquidityMetric[] {
  return [
    {
      ticker: 'AAPL',
      name: 'Apple Inc.',
      avgDailyVolume: 1_000_000,
      daysToLiquidate: 1.5,
      liquidityCost: 0.001,
      weight: 0.6,
      ...overrides,
    },
  ];
}

export function makeStressScenarios(
  overrides: Partial<StressScenario> = {},
): StressScenario[] {
  return [
    {
      id: 'scn-1',
      name: 'Rate Shock',
      description: '+100bps parallel shift',
      portfolioImpact: -0.08,
      benchmarkImpact: -0.06,
      worstAsset: 'TLT',
      worstAssetImpact: -0.15,
      ...overrides,
    },
  ];
}

export function makeMarketSnapshotResponse(
  overrides: Partial<ApiMarketSnapshotResponse> = {},
): ApiMarketSnapshotResponse {
  return {
    vix: 18.5,
    vixChange: -0.4,
    sp500Return: 0.012,
    tenYearYield: 4.2,
    yieldChange: 0.03,
    usdIndex: 104.3,
    usdChange: -0.1,
    asOf: ISO,
    ...overrides,
  };
}

export function makeDriftResponse(overrides: Partial<DriftApiResponse> = {}): DriftApiResponse {
  return {
    entries: [
      { ticker: 'AAPL', name: 'Apple Inc.', target: 0.6, actual: 0.65, drift: 0.05, breached: false },
    ],
    totalDrift: 0.05,
    breachedCount: 0,
    threshold: 0.05,
    ...overrides,
  };
}

export function makeRebalancePreview(
  overrides: Partial<RebalancePreviewApiResponse> = {},
): RebalancePreviewApiResponse {
  return {
    portfolioName: 'Test Portfolio',
    policyType: 'threshold',
    targetWeights: { AAPL: 0.6, MSFT: 0.4 },
    currentWeights: { AAPL: 0.65, MSFT: 0.35 },
    trades: [{ ticker: 'AAPL', weightDelta: -0.05, side: 'sell', shares: 10 }],
    portfolioValue: 100_000,
    status: null,
    ...overrides,
  };
}

export function makeRebalanceDecideResponse(
  overrides: Partial<RebalanceDecideApiResponse> = {},
): RebalanceDecideApiResponse {
  return {
    shouldRebalance: true,
    turnover: 0.1,
    estimatedCost: 0.0005,
    tradeWeights: { AAPL: -0.05, MSFT: 0.05 },
    ...overrides,
  };
}

export function makeFactorICReport(overrides: Partial<FactorICReport> = {}): FactorICReport {
  return {
    factor: 'momentum_12_1',
    group: 'momentum',
    ic: 0.05,
    icir: 0.8,
    tStat: 2.5,
    pValue: 0.01,
    vif: 1.2,
    significant: true,
    ...overrides,
  };
}

export function makeCMASet(overrides: Partial<CMASet> = {}): CMASet {
  return {
    label: 'Base CMA',
    horizon: '10Y',
    assets: [{ ticker: 'AAPL', expectedReturn: 0.07, expectedVol: 0.2 }],
    ...overrides,
  };
}

export function makeTAASignal(overrides: Partial<TAASignal> = {}): TAASignal {
  return {
    factor: 'momentum',
    currentWeight: 0.2,
    tiltedWeight: 0.25,
    tiltReason: 'Expansion regime favours momentum',
    regime: 'expansion',
    ...overrides,
  };
}

// ── Batch B wire DTOs (jobs, reports, universe, scenarios, market_data) ──────
// jobs / reports / market_data serialise snake_case (plain BaseModel); scenarios
// and universe-screen serialise camelCase (CamelCaseModel). Keys mirror each
// domain's snapshot `properties` verbatim — casing follows the wire, not a rule.

export function makeJobSummary(overrides: Partial<JobSummary> = {}): JobSummary {
  return {
    id: 'job-1',
    domain: 'yfinance_fetch',
    status: 'completed',
    current: 10,
    total: 10,
    error: null,
    errors_count: 0,
    started_at: ISO,
    finished_at: ISO,
    duration_seconds: 12.5,
    ...overrides,
  };
}

export function makeJobListResponse(
  overrides: Partial<JobListResponse> = {},
): JobListResponse {
  return { jobs: [makeJobSummary()], total: 1, limit: 50, offset: 0, ...overrides };
}

export function makeReportJobCreateResponse(
  overrides: Partial<ReportJobCreateResponse> = {},
): ReportJobCreateResponse {
  return { job_id: 'report-1', status: 'pending', message: 'Report queued', ...overrides };
}

export function makeUniverseScreenResponse(
  overrides: Partial<UniverseScreenResponse> = {},
): UniverseScreenResponse {
  return {
    passingTickers: ['AAPL', 'MSFT'],
    totalScreened: 100,
    diagnostics: { AAPL: { market_cap: true } },
    ...overrides,
  };
}

export function makeStressScenarioItemApi(
  overrides: Partial<StressScenarioItemApi> = {},
): StressScenarioItemApi {
  return {
    name: 'Rate Shock',
    description: '+100bps parallel shift',
    shocks: { TLT: -0.15 },
    probability: 0.1,
    horizonDays: 21,
    syntheticDataArgs: {},
    ...overrides,
  };
}

export function makeStressScenarioApiResponse(
  overrides: Partial<StressScenarioApiResponse> = {},
): StressScenarioApiResponse {
  return {
    nScenarios: 1,
    tickers: ['TLT', 'SPY'],
    scenarios: [makeStressScenarioItemApi()],
    ...overrides,
  };
}

export function makePriceHistoryResponse(overrides: Partial<PriceHistory> = {}): PriceHistory {
  return {
    id: 'px-1',
    instrument_id: 'inst-1',
    date: '2026-01-01',
    open: 100,
    high: 105,
    low: 99,
    close: 104,
    volume: 1_000_000,
    dividends: 0,
    stock_splits: 0,
    created_at: ISO,
    updated_at: ISO,
    ...overrides,
  };
}

// ── Batch C wire DTOs (optimization, backtest, factors, views, macro,
// pipeline_builder). All camelCase except the snake_case fields that the
// backend serialises plain (factors composite/select, backtest progress). ────

export function makeOptimizationRunResponse(
  overrides: Partial<OptimizationRunResponse> = {},
): OptimizationRunResponse {
  return {
    id: 'opt-1',
    portfolioId: null,
    jobId: null,
    status: 'completed',
    optimizerType: 'mean_risk',
    universeTickers: ['AAPL', 'MSFT'],
    config: {},
    weights: { AAPL: 0.6, MSFT: 0.4 },
    metrics: { sharpe: 1.2 },
    riskContributions: { AAPL: 0.55, MSFT: 0.45 },
    efficientFrontier: null,
    errorMessage: null,
    solverLog: null,
    durationSeconds: 1.2,
    createdAt: ISO,
    updatedAt: ISO,
    ...overrides,
  };
}

export function makeOptimizationRunListResponse(
  overrides: Partial<OptimizationRunListResponse> = {},
): OptimizationRunListResponse {
  return { items: [makeOptimizationRunResponse()], total: 1, ...overrides };
}

export function makeBacktestRunResponse(
  overrides: Partial<BacktestRunResponse> = {},
): BacktestRunResponse {
  return {
    id: 'bt-1',
    portfolioId: null,
    jobId: null,
    status: 'completed',
    config: {},
    equityCurve: { '2026-01-01': 1.0 },
    drawdowns: { '2026-01-01': 0 },
    monthlyReturns: { '2026-01': 0.01 },
    yearlyReturns: { '2026': 0.12 },
    rollingMetrics: { sharpe: { '2026-01-01': 1.1 } },
    turnoverHistory: { '2026-01-01': 0.05 },
    cvFoldMetrics: null,
    summaryStats: { sharpe: 1.1 },
    errorMessage: null,
    durationSeconds: 2.0,
    createdAt: ISO,
    updatedAt: ISO,
    ...overrides,
  };
}

export function makeBacktestProgressResponse(
  overrides: Partial<BacktestProgressResponse> = {},
): BacktestProgressResponse {
  return {
    job_id: 'bt-job-1',
    status: 'running',
    current: 3,
    total: 10,
    errors: [],
    result: null,
    error: null,
    ...overrides,
  };
}

// POST /api/v1/backtest returns a raw JSONResponse with NO Pydantic
// `response_model` — the one schema-less route. Pinned by key-set, not snapshot.
export function makeBacktestAsyncResponse(
  overrides: Partial<BacktestAsyncResponse> = {},
): BacktestAsyncResponse {
  return { jobId: 'bt-job-1', runId: 'bt-run-1', status: 'pending', message: 'Queued', ...overrides };
}

export function makeFactorScoreDto(overrides: Partial<FactorScoreDto> = {}): FactorScoreDto {
  return {
    id: 'fs-1',
    ticker: 'AAPL',
    factorType: 'momentum_12_1',
    factorGroup: 'momentum',
    scoreDate: '2026-01-01',
    rawScore: 0.5,
    standardizedScore: 0.8,
    compositeScore: 0.65,
    createdAt: ISO,
    updatedAt: ISO,
    ...overrides,
  };
}

export function makeFactorCompositeResponse(
  overrides: Partial<FactorScoreApiResponse> = {},
): FactorScoreApiResponse {
  return {
    score_date: '2026-01-01',
    scores: { AAPL: 0.65 },
    group_contributions: { momentum: 0.4 },
    ...overrides,
  };
}

export function makeFactorSelectResponse(
  overrides: Partial<FactorSelectApiResponse> = {},
): FactorSelectApiResponse {
  return {
    selected_tickers: ['AAPL', 'MSFT'],
    count: 2,
    turnover: 0.1,
    buffer_zone: { entered: ['MSFT'], exited: [] },
    ...overrides,
  };
}

export function makeGenerateViewsResponse(
  overrides: Partial<GenerateViewsResponse> = {},
): GenerateViewsResponse {
  return {
    nViews: 1,
    nAssets: 2,
    viewStrings: ['AAPL == 0.05'],
    p: [[1, 0]],
    q: [0.05],
    viewConfidences: [0.8],
    idzorekAlphas: { AAPL: 0.5 },
    views: [
      { asset: 'AAPL', direction: 1, magnitudeBps: 500, confidence: 0.8, reasoning: 'momentum' },
    ],
    rationale: 'factor signals',
    tickersWithData: ['AAPL', 'MSFT'],
    tickersMissingData: [],
    ...overrides,
  };
}

export function makeOpinionPoolResponse(
  overrides: Partial<OpinionPoolResponse> = {},
): OpinionPoolResponse {
  return {
    nExperts: 2,
    tickers: ['AAPL', 'MSFT'],
    tickersWithData: ['AAPL', 'MSFT'],
    tickersMissingData: [],
    experts: [
      {
        persona: 'VALUE_INVESTOR',
        name: 'Value',
        nViews: 1,
        viewStrings: ['AAPL == 0.05'],
        idzorekAlphas: { AAPL: 0.5 },
        icWeight: 0.5,
      },
    ],
    icWeights: [0.5, 0.5],
    poolingType: 'linear',
    totalViews: 2,
    ...overrides,
  };
}

export function makeEntropyPoolingResponse(
  overrides: Partial<EntropyPoolingResponse> = {},
): EntropyPoolingResponse {
  return {
    tickers: ['AAPL', 'MSFT'],
    mu: [0.1, 0.05],
    covariance: [
      [0.04, 0.01],
      [0.01, 0.03],
    ],
    ...overrides,
  };
}

export function makeMacroCalibrationApiResponse(
  overrides: Partial<MacroCalibrationApiResponse> = {},
): MacroCalibrationApiResponse {
  return {
    phase: 'MID_EXPANSION',
    delta: 2.5,
    tau: 0.05,
    confidence: 0.8,
    rationale: 'cycle expansion',
    macroSummary: 'PMI above 55',
    blConfig: { views: [], tau: 0.05, prior_config: { mu_estimator: 'shrunk', risk_aversion: 2.5, cov_estimator: 'empirical' } },
    ...overrides,
  };
}

export function makeCreateSessionResponse(
  overrides: Partial<CreateSessionResponse> = {},
): CreateSessionResponse {
  return { sessionId: 'sess-1', ...overrides };
}

export function makeStepPollResponse(
  overrides: Partial<StepPollResponse> = {},
): StepPollResponse {
  return {
    status: 'running',
    progress: { current: 1, total: 3 },
    result: null,
    error: null,
    gateReason: null,
    ...overrides,
  };
}

// Wire shape of StepRunResponse (AsyncJobCreateResponse, snake_case `job_id`).
// The frontend `StepRunResponse` deliberately remaps this to camelCase `jobId`
// at the HTTP boundary, so parity pins the wire shape the service receives.
export function makeStepRunWireResponse(
  overrides: Partial<{ job_id: string; status: string; message: string }> = {},
): { job_id: string; status: string; message: string } {
  return { job_id: 'step-job-1', status: 'pending', message: 'Step queued', ...overrides };
}

// Portfolio rich DriftResponse (drift.model.ts) — distinct from the dashboard
// simple DriftResponse (entries[]). Asserted against portfolio.json by the
// pipeline_builder/builder-drift parity block.
export function makeDriftResponseRich(
  overrides: Partial<DriftResponseRich> = {},
): DriftResponseRich {
  return {
    holdings: [],
    target: [],
    drift: [],
    trades: [],
    totals: {
      deployable_eur: 0,
      total_holdings_eur: 100,
      total_drift_abs: 0,
      buy_eur: 0,
      sell_eur: 0,
    },
    diagnostics: {
      reconciliation_ok: true,
      reconciliation_delta_pct: 0,
      unmapped_count: 0,
      fx_missing_count: 0,
      target_not_on_broker_count: 0,
      base_currency: 'EUR',
      sum_eur: 100,
      invested_eur: 100,
      delta_eur: 0,
      tolerance_pct: 0,
      stale_price_count: 0,
      entries: [],
    },
    request_id: 1,
    ...overrides,
  };
}
