// Typed factory functions over src/app/models/* DTOs. Each returns a minimal
// schema-valid object and accepts a Partial `overrides` merged last. Plural
// factories (…Metrics/…Scenarios) return a one-element array; their `overrides`
// merge into that single element so callers tweak one field without rebuilding
// the row. For object factories that embed a nested array (sectors / trades /
// entries / assets), `overrides` patches top-level keys only — to change an
// inner row, pass the whole array key (e.g. `{ entries: [...] }`).
// Keep factories < 10 lines — extend by adding a field, not a branch.
//
// Cycle-4 contract-parity conventions (field-level assertions):
//   - Mirror wire casing verbatim: snake_case for REST responses (portfolio.json),
//     camelCase only where the backend actually emits camelCase.
//   - Reuse existing make<X>() factories as the fixture argument to assertFieldParity.
//   - Always pass the domain snapshot as the third `root` argument to assertFieldParity
//     so $ref pointers inside the schema are resolved within the same document, e.g.:
//       import portfolioSnapshot from './contract-snapshots/portfolio.json';
//       assertFieldParity(schemaOf(portfolioSnapshot, 'PortfolioResponse'), makePortfolioDto(), portfolioSnapshot);

import type { PortfolioDto, SnapshotDto } from '../app/core/models/portfolio-api.model';
import type { DatabaseStatus, HealthCheck, TableInfo, TruncateResponse } from '../app/settings/database.model';
import type { SchedulerJobStatusDto, SchedulerStatusResponse } from '../app/settings/scheduler.model';
import type { ApiMarketSnapshotResponse } from '../app/core/models/dashboard-api.model';
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
} from '../app/core/models/optimization.model';
import type {
  BacktestAsyncResponse,
  BacktestProgressResponse,
  BacktestRunResponse,
} from '../app/backtesting/backtest.model';
import type { MacroCalibrationApiResponse } from '../app/core/models/macro-intelligence.model';
import type { DriftResponse as DriftResponseRich } from '../app/core/models/drift.model';

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

// ── Settings wire DTOs (database, scheduler) — snake_case except scheduler
// which uses CamelCaseModel. Mirrors api/app/schemas/database.py and
// api/app/schemas/scheduler.py exactly. ──────────────────────────────────────

export function makeHealthCheck(overrides: Partial<HealthCheck> = {}): HealthCheck {
  return { healthy: true, latency_ms: 3.2, database_url: 'postgresql://***@localhost/db', ...overrides };
}

export function makeTableInfo(overrides: Partial<TableInfo> = {}): TableInfo {
  return { name: 'instruments', schema: 'public', row_count: 500, size_bytes: 65536, size_pretty: '64 kB', ...overrides };
}

export function makeDatabaseStatus(overrides: Partial<DatabaseStatus> = {}): DatabaseStatus {
  return { health: makeHealthCheck(), tables: [makeTableInfo()], total_size_pretty: '64 kB', ...overrides };
}

export function makeTruncateResponse(overrides: Partial<TruncateResponse> = {}): TruncateResponse {
  return { table: 'price_history', status: 'truncated', ...overrides };
}

export function makeSchedulerJobStatusDto(overrides: Partial<SchedulerJobStatusDto> = {}): SchedulerJobStatusDto {
  return {
    jobId: 'daily_pipeline',
    name: 'Daily Pipeline',
    nextRunTime: ISO,
    lastRunTime: ISO,
    lastStatus: 'completed',
    trigger: 'cron[hour=7]',
    ...overrides,
  };
}

export function makeSchedulerStatusResponse(overrides: Partial<SchedulerStatusResponse> = {}): SchedulerStatusResponse {
  return { schedulerRunning: true, jobs: [makeSchedulerJobStatusDto()], ...overrides };
}

// ── New factories for issue #968 (cross-page service field parity) ────────────
// All shapes mirror the JSON-schema snapshot verbatim. Casing follows the wire:
// camelCase for CamelCaseModel endpoints, snake_case for plain BaseModel endpoints.

// DriftResponse — dashboard.json, camelCase (simple per-holding drift table)
export function makeDriftResponse(overrides: Record<string, unknown> = {}) {
  return {
    entries: [{ ticker: 'AAPL', name: 'Apple Inc.', target: 0.5, actual: 0.55, drift: 0.05, breached: true }],
    totalDrift: 0.05,
    breachedCount: 1,
    threshold: 0.03,
    ...overrides,
  };
}

// StressScenarioItem — scenarios.json, camelCase (single stress test scenario)
export function makeStressScenarioItemApi(overrides: Record<string, unknown> = {}) {
  return {
    name: 'Market Crash',
    description: 'Severe equity sell-off',
    shocks: { AAPL: -0.3, MSFT: -0.25 },
    probability: 0.05,
    horizonDays: 20,
    syntheticDataArgs: { conditioning: { AAPL: -0.3 } },
    ...overrides,
  };
}

// StressScenarioResponse — scenarios.json, camelCase (list of scenarios)
export function makeStressScenarioApiResponse(overrides: Record<string, unknown> = {}) {
  return {
    nScenarios: 1,
    tickers: ['AAPL', 'MSFT'],
    scenarios: [makeStressScenarioItemApi()],
    ...overrides,
  };
}

// AllocationResponse — dashboard.json, camelCase
export function makeAllocationResponse() {
  return {
    nodes: [{ name: 'Technology', value: 0.6, children: [{ name: 'AAPL', value: 0.6 }] }],
    totalPositions: 1,
    totalSectors: 1,
  };
}

// AssetClassReturnsResponse — dashboard.json, camelCase (numeric period keys)
export function makeAssetClassReturnsResponse() {
  return {
    returns: [{ name: 'Technology', '1D': 0.01, '1W': 0.03, '1M': 0.05, YTD: 0.12 }],
    asOf: '2026-01-01',
  };
}

// EquityCurveResponse — dashboard.json, camelCase
export function makeEquityCurveResponse() {
  return {
    points: [{ date: '2026-01-01', portfolio: 1.0, benchmark: 1.0 }],
    portfolioTotalReturn: 0.12,
    benchmarkTotalReturn: 0.1,
  };
}

// PerformanceMetricsResponse — dashboard.json, camelCase (currency has default 'EUR', not null)
export function makePerformanceMetricsResponse() {
  return {
    kpis: [{ label: 'Sharpe', value: 1.2, format: 'ratio', change: 0.1, changeLabel: '+0.1', sparkline: [1.0, 1.1, 1.2] }],
    nav: 100_000,
    navChangePct: 0.12,
    currency: 'EUR',
  };
}

// RollingMetricsResponse — dashboard.json, camelCase (RollingMetricPoint = {date, value})
export function makeRollingMetricsResponse() {
  return {
    window: 63,
    sharpe: [{ date: '2026-01-01', value: 1.2 }],
    volatility: [{ date: '2026-01-01', value: 0.15 }],
    beta: [{ date: '2026-01-01', value: 0.9 }],
  };
}

// InstrumentListResponse — universe.json, snake_case items inside camelCase wrapper
export function makeInstrumentListResponse() {
  return {
    items: [{ id: 'inst-1', ticker: 'AAPL', short_name: 'Apple', created_at: ISO, updated_at: ISO }],
    total: 1,
  };
}

// UniverseStatsResponse — universe.json, snake_case (plain BaseModel)
export function makeUniverseStatsResponse() {
  return { exchange_count: 5, instrument_count: 100 };
}

// TickerProfileResponse — market_data.json, snake_case (plain BaseModel).
// All optional fields included as null so assertFieldParity type-matches anyOf[T, null].
export function makeTickerProfileResponse() {
  return {
    id: 'prof-1',
    instrument_id: 'inst-1',
    created_at: ISO,
    updated_at: ISO,
    symbol: null,
    short_name: null,
    long_name: null,
    currency: null,
    exchange: null,
    quote_type: null,
    isin: null,
    sector: null,
    industry: null,
    country: null,
    website: null,
    long_business_summary: null,
    current_price: null,
    previous_close: null,
    market_cap: null,
    enterprise_value: null,
    shares_outstanding: null,
    float_shares: null,
    beta: null,
    trailing_pe: null,
    forward_pe: null,
    trailing_eps: null,
    forward_eps: null,
    peg_ratio: null,
    book_value: null,
    price_to_book: null,
    dividend_rate: null,
    dividend_yield: null,
    payout_ratio: null,
    total_revenue: null,
    gross_margins: null,
    operating_margins: null,
    profit_margins: null,
    ebitda: null,
    operating_cashflow: null,
    free_cashflow: null,
    total_debt: null,
    return_on_assets: null,
    return_on_equity: null,
    revenue_growth: null,
    earnings_growth: null,
    current_ratio: null,
    debt_to_equity: null,
    enterprise_to_revenue: null,
    enterprise_to_ebitda: null,
    full_time_employees: null,
    fifty_day_average: null,
    two_hundred_day_average: null,
    fifty_two_week_high: null,
    fifty_two_week_low: null,
    average_volume: null,
    price_to_sales_trailing_12months: null,
    recommendation_key: null,
    recommendation_mean: null,
  };
}

