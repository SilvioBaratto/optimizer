import type { Page } from '@playwright/test';

import type { BacktestRunResponse } from '../../src/app/models/backtest.model';
import type {
  ApiAllocationResponse,
  ApiAssetClassReturnsResponse,
  ApiEquityCurveResponse,
  ApiPerformanceMetricsResponse,
  ApiRollingMetricsResponse,
} from '../../src/app/models/dashboard-api.model';
import type { DriftResponse, DriftDiagnostics, DriftRow, FlagInstance, PositionFlag } from '../../src/app/models/drift.model';
import type { JobSummary } from '../../src/app/models/jobs.model';
import type { PipelineStepId } from '../../src/app/models/pipeline-builder.model';
import type { EfficientFrontierPoint, OptimizationRunResponse } from '../../src/app/models/optimization.model';
import type { DriftApiResponse } from '../../src/app/models/rebalancing.model';
import type { CorrelationApiResponse, VarApiResponse } from '../../src/app/models/risk.model';
import type { PortfolioDto, PortfolioListResponseDto } from '../../src/app/models/portfolio-api.model';

// ── Optimization fixtures ─────────────────────────────────────────────────────

const FRONTIER_POINT: EfficientFrontierPoint = { risk: 0.1, return: 0.12, sharpe: 1.2 };

const BASE_OPT_RUN: OptimizationRunResponse = {
  id: 'e2e-run-opt',
  portfolioId: null,
  jobId: null,
  status: 'completed',
  optimizerType: 'mean_risk',
  universeTickers: ['AAPL', 'MSFT'],
  config: {},
  weights: { AAPL: 0.6, MSFT: 0.4 },
  // snake_case keys matching ResultsPanelComponent.statsCards lookups so all
  // four stat cards render real values (not the '—' fallback).
  metrics: {
    annualized_sharpe_ratio: 1.2,
    annualized_return: 0.12,
    annualized_volatility: 0.1,
    max_drawdown: -0.05,
  },
  riskContributions: { AAPL: 0.55, MSFT: 0.45 },
  efficientFrontier: [FRONTIER_POINT],
  errorMessage: null,
  solverLog: null,
  durationSeconds: 1.5,
  createdAt: '2026-01-01T00:00:00Z',
  updatedAt: '2026-01-01T00:00:01Z',
};

function makeOptRun(overrides: Partial<OptimizationRunResponse>): OptimizationRunResponse {
  return { ...BASE_OPT_RUN, ...overrides };
}

async function fulfillJson(route: Parameters<Parameters<Page['route']>[1]>[0], status: number, body: unknown): Promise<void> {
  await route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });
}

// ── stubOptimizeSync ──────────────────────────────────────────────────────────

export async function stubOptimizeSync(
  page: Page,
  overrides: Partial<OptimizationRunResponse> = {},
): Promise<void> {
  await page.route('**/api/v1/optimize', async (route) => {
    if (route.request().method() !== 'POST') { await route.continue(); return; }
    await fulfillJson(route, 200, makeOptRun(overrides));
  });
}

// ── stubOptimizeAsync ─────────────────────────────────────────────────────────

/**
 * Registers three routes for the async optimize flow.
 * A call-counter drives the running→completed transition deterministically.
 * routeOnce is not used because it de-registers after one call; subsequent
 * polls for the same job_id would fall through to the network. A counter
 * correctly handles N polls while keeping the route registered for the
 * entire duration of the test.
 *
 * `jobId`/`runId` are parameterised because Optimization Studio derives the
 * run id from the job id by stripping a leading `job-` (see
 * `optimization-studio.html`'s `onJobCompleted`). Callers that rely on that
 * convention pass `jobId: 'job-<runId>'` so the run-fetch route resolves.
 * Defaults preserve the original ids used by the support-layer self-tests.
 */
export async function stubOptimizeAsync(
  page: Page,
  opts: { jobId?: string; runId?: string } = {},
): Promise<void> {
  const jobId = opts.jobId ?? 'e2e-job-opt';
  const runId = opts.runId ?? 'e2e-run-opt';

  await page.route('**/api/v1/optimize', async (route) => {
    if (route.request().method() !== 'POST') { await route.continue(); return; }
    await fulfillJson(route, 202, { job_id: jobId, run_id: runId });
  });

  let pollCount = 0;

  await page.route(`**/api/v1/jobs/${jobId}`, async (route) => {
    const status: JobSummary['status'] = pollCount === 0 ? 'running' : 'completed';
    pollCount += 1;
    const summary: JobSummary = {
      id: jobId,
      domain: 'optimize',
      status,
      current: status === 'completed' ? 1 : 0,
      total: 1,
      error: null,
      errors_count: 0,
      started_at: '2026-01-01T00:00:00Z',
      finished_at: status === 'completed' ? '2026-01-01T00:00:01Z' : null,
      duration_seconds: status === 'completed' ? 1 : null,
    };
    await fulfillJson(route, 200, summary);
  });

  await page.route(`**/api/v1/optimize/${runId}`, async (route) => {
    await fulfillJson(route, 200, makeOptRun({ id: runId }));
  });
}

// ── Backtest fixtures ─────────────────────────────────────────────────────────

const BASE_BACKTEST_RUN: BacktestRunResponse = {
  id: 'e2e-run-bt',
  portfolioId: null,
  jobId: null,
  status: 'completed',
  config: {},
  equityCurve: { '2025-01-01': 1.0, '2025-06-01': 1.08 },
  drawdowns: { '2025-01-01': 0.0, '2025-03-01': -0.04 },
  monthlyReturns: {},
  yearlyReturns: {},
  rollingMetrics: {},
  turnoverHistory: {},
  cvFoldMetrics: null,
  summaryStats: { sharpe: 1.1, annualizedReturn: 0.1 },
  errorMessage: null,
  durationSeconds: 2.0,
  createdAt: '2026-01-01T00:00:00Z',
  updatedAt: '2026-01-01T00:00:02Z',
};

// ── stubBacktest ──────────────────────────────────────────────────────────────

/**
 * Backtest is an async 202 flow (BacktestAsyncResponse).
 * Mirrors the optimize pattern: POST → 202 job_id/run_id, poll job, fetch run.
 *
 * Two poll surfaces are stubbed because two callers exist:
 *  - the portfolio-builder flow polls `pollBacktest` → GET /backtest/{jobId};
 *  - the Backtesting Lab page polls via `app-job-progress-tracker` →
 *    GET /jobs/{jobId} (JobsService). Both transition running→completed.
 */
export async function stubBacktest(
  page: Page,
  opts?: { error?: { status: number; detail: string } },
): Promise<void> {
  await page.route('**/api/v1/backtest', async (route) => {
    if (route.request().method() !== 'POST') { await route.continue(); return; }
    if (opts?.error) {
      await fulfillJson(route, opts.error.status, { detail: opts.error.detail });
      return;
    }
    await fulfillJson(route, 202, { jobId: 'e2e-job-bt', runId: 'e2e-run-bt', status: 'pending', message: 'queued' });
  });

  if (opts?.error) return;

  let pollCount = 0;

  await page.route('**/api/v1/backtest/e2e-job-bt', async (route) => {
    const status = pollCount === 0 ? 'running' : 'completed';
    pollCount += 1;
    await fulfillJson(route, 200, {
      job_id: 'e2e-job-bt',
      status,
      current: status === 'completed' ? 1 : 0,
      total: 1,
      errors: [],
      result: null,
      error: null,
    });
  });

  let jobsPollCount = 0;

  await page.route('**/api/v1/jobs/e2e-job-bt', async (route) => {
    const status: JobSummary['status'] = jobsPollCount === 0 ? 'running' : 'completed';
    jobsPollCount += 1;
    const summary: JobSummary = {
      id: 'e2e-job-bt',
      domain: 'backtest',
      status,
      current: status === 'completed' ? 1 : 0,
      total: 1,
      error: null,
      errors_count: 0,
      started_at: '2026-01-01T00:00:00Z',
      finished_at: status === 'completed' ? '2026-01-01T00:00:01Z' : null,
      duration_seconds: status === 'completed' ? 1 : null,
    };
    await fulfillJson(route, 200, summary);
  });

  await page.route('**/api/v1/backtest/runs/e2e-run-bt', async (route) => {
    await fulfillJson(route, 200, BASE_BACKTEST_RUN);
  });
}

// ── Drift fixtures ────────────────────────────────────────────────────────────

type DriftFixtureKey =
  | 'base'
  | 'total'
  | 'unmapped'
  | 'fx_missing'
  | 'stale_price'
  | 'reconciliation_mismatch'
  | 'target_not_on_broker';

export type { DriftFixtureKey };

function makeFlag(code: PositionFlag): FlagInstance {
  return { code, reason: `test: ${code}`, reference: null };
}

function makeDriftRow(code: PositionFlag): DriftRow {
  return {
    ticker: 'TEST',
    current_weight: 0.5,
    target_weight: 0.5,
    delta_weight: 0.0,
    eur_value: 1000,
    flags: [makeFlag(code)],
  };
}

function makeCleanDiagnostics(): DriftDiagnostics {
  return {
    reconciliation_ok: true,
    reconciliation_delta_pct: null,
    unmapped_count: 0,
    fx_missing_count: 0,
    target_not_on_broker_count: 0,
    base_currency: 'EUR',
    sum_eur: 1000,
    invested_eur: 1000,
    delta_eur: 0,
    tolerance_pct: 0.01,
    stale_price_count: 0,
    entries: [],
  };
}

function makeDrift(partial: Partial<DriftResponse>): DriftResponse {
  return {
    holdings: [],
    target: [{ ticker: 'TEST', weight: 1.0 }],
    drift: [],
    trades: [],
    totals: {
      deployable_eur: 0,
      total_holdings_eur: 1000,
      total_drift_abs: 0,
      buy_eur: 0,
      sell_eur: 0,
    },
    diagnostics: makeCleanDiagnostics(),
    request_id: 1,
    ...partial,
  };
}

function makeFlaggedDrift(code: PositionFlag, diagOverrides: Partial<DriftDiagnostics>): DriftResponse {
  return makeDrift({
    drift: [makeDriftRow(code)],
    diagnostics: { ...makeCleanDiagnostics(), ...diagOverrides },
  });
}

export const DRIFT_FIXTURES: Record<DriftFixtureKey, DriftResponse> = {
  base: makeDrift({}),

  total: makeDrift({
    trades: [{ ticker: 'TEST', action: 'buy', delta_eur: 500, est_shares: 5, est_cost_eur: 500 }],
    totals: { deployable_eur: 500, total_holdings_eur: 1000, total_drift_abs: 0.05, buy_eur: 500, sell_eur: 0 },
  }),

  unmapped: makeFlaggedDrift('unmapped', {
    unmapped_count: 1,
    entries: [{ code: 'unmapped', reason: 'test: unmapped', reference: null, ticker: 'TEST' }],
  }),

  fx_missing: makeFlaggedDrift('fx_missing', {
    fx_missing_count: 1,
    entries: [{ code: 'fx_missing', reason: 'test: fx_missing', reference: null, ticker: 'TEST' }],
  }),

  stale_price: makeFlaggedDrift('stale_price', {
    stale_price_count: 1,
    entries: [{ code: 'stale_price', reason: 'test: stale_price', reference: null, ticker: 'TEST' }],
  }),

  reconciliation_mismatch: makeFlaggedDrift('reconciliation_mismatch', {
    reconciliation_ok: false,
    reconciliation_delta_pct: 0.05,
    entries: [{ code: 'reconciliation_mismatch', reason: 'test: reconciliation_mismatch', reference: null, ticker: 'TEST' }],
  }),

  target_not_on_broker: makeFlaggedDrift('target_not_on_broker', {
    target_not_on_broker_count: 1,
    entries: [{ code: 'target_not_on_broker', reason: 'test: target_not_on_broker', reference: null, ticker: 'TEST' }],
  }),
};

// ── stubDrift ─────────────────────────────────────────────────────────────────

// The trailing `*` matches the real `?base=invested|total` query the
// BuilderDriftService appends (a bare `**/drift` glob would miss it).
export async function stubDrift(page: Page, key: DriftFixtureKey): Promise<void> {
  await page.route('**/api/v1/portfolio/*/drift*', async (route) => {
    await fulfillJson(route, 200, DRIFT_FIXTURES[key]);
  });
}

// ── stubDriftByBase ───────────────────────────────────────────────────────────

// Clean, populated target-vs-actual rows (no flags → no diagnostics strip), with
// a different trade set per base so toggling invested→total visibly recomputes
// the trade list.
function makeCleanRow(
  ticker: string,
  current: number,
  target: number,
  eur: number,
): DriftRow {
  return {
    ticker,
    current_weight: current,
    target_weight: target,
    delta_weight: +(target - current).toFixed(4),
    eur_value: eur,
    flags: [],
  };
}

const POPULATED_ROWS: readonly DriftRow[] = [
  makeCleanRow('AAPL', 0.6, 0.5, 6000),
  makeCleanRow('MSFT', 0.4, 0.5, 4000),
];

const POPULATED_INVESTED: DriftResponse = makeDrift({
  drift: [...POPULATED_ROWS],
  trades: [{ ticker: 'MSFT', action: 'buy', delta_eur: 1000, est_shares: 3, est_cost_eur: 1000 }],
  totals: { deployable_eur: 1000, total_holdings_eur: 10000, total_drift_abs: 0.2, buy_eur: 1000, sell_eur: 0 },
});

const POPULATED_TOTAL: DriftResponse = makeDrift({
  drift: [...POPULATED_ROWS],
  trades: [
    { ticker: 'AAPL', action: 'sell', delta_eur: -800, est_shares: 2, est_cost_eur: 800 },
    { ticker: 'MSFT', action: 'buy', delta_eur: 1200, est_shares: 4, est_cost_eur: 1200 },
  ],
  totals: { deployable_eur: 2000, total_holdings_eur: 10000, total_drift_abs: 0.25, buy_eur: 1200, sell_eur: 800 },
});

/** One route that returns the total-base body for `?base=total`, else invested. */
export async function stubDriftByBase(page: Page): Promise<void> {
  await page.route('**/api/v1/portfolio/*/drift*', async (route) => {
    const base = new URL(route.request().url()).searchParams.get('base');
    await fulfillJson(route, 200, base === 'total' ? POPULATED_TOTAL : POPULATED_INVESTED);
  });
}

// ── Landing-page read stubs ───────────────────────────────────────────────────

// Dashboard / risk-center / rebalancing fire analytics GETs on first load that
// 4xx against the minimal smoke seed (no holdings/analytics rows). These 200
// stubs let the page render its landmark cleanly without masking journey logic
// (each body is minimal-but-valid so the component success path can't crash).
const LANDING_READS: ReadonlyArray<{ pattern: string; body: unknown }> = [
  {
    pattern: '**/api/v1/portfolio-analytics/*/performance-metrics*',
    body: { kpis: [], nav: 0, navChangePct: 0, currency: 'EUR' } satisfies ApiPerformanceMetricsResponse,
  },
  {
    pattern: '**/api/v1/portfolio-analytics/*/equity-curve*',
    body: { points: [], portfolioTotalReturn: 0, benchmarkTotalReturn: 0 } satisfies ApiEquityCurveResponse,
  },
  {
    pattern: '**/api/v1/portfolio-analytics/*/rolling-metrics*',
    body: { window: 63, sharpe: [], volatility: [], beta: [] } satisfies ApiRollingMetricsResponse,
  },
  {
    pattern: '**/api/v1/portfolio-analytics/*/allocation*',
    body: { nodes: [], totalPositions: 0, totalSectors: 0 } satisfies ApiAllocationResponse,
  },
  {
    pattern: '**/api/v1/portfolio-analytics/*/asset-class-returns*',
    body: { returns: [], asOf: '2025-01-01' } satisfies ApiAssetClassReturnsResponse,
  },
  {
    pattern: '**/api/v1/portfolio-analytics/*/drift*',
    body: { entries: [], totalDrift: 0, breachedCount: 0, threshold: 0.05 } satisfies DriftApiResponse,
  },
  {
    pattern: '**/api/v1/portfolio/*/risk/var*',
    body: { var: {}, cvar: {}, method: 'historical', lookback: 252, nObservations: 0 } satisfies VarApiResponse,
  },
  {
    pattern: '**/api/v1/portfolio/*/risk/correlation*',
    body: { assets: [], matrix: [], clusterLabels: [] } satisfies CorrelationApiResponse,
  },
];

/** Stub the landing-page analytics GETs that 4xx against the smoke seed. */
export async function stubLandingReads(page: Page): Promise<void> {
  for (const read of LANDING_READS) {
    await page.route(read.pattern, (route) => fulfillJson(route, 200, read.body));
  }
}

// ── Pipeline-builder fixtures ─────────────────────────────────────────────────

// Drives both the V2 builder and the legacy stepper. Step POST → 202; the first
// poll returns a terminal state (completed, or failed+gateReason for the abort
// step) so journeys never sleep — the timer(0,…) poll resolves on its first tick.

export const PB_ARTIFACT_NAMES = [
  'report.md',
  'weights.csv',
  'metrics.json',
  'checklist.json',
] as const;

export type PbArtifactName = (typeof PB_ARTIFACT_NAMES)[number];

export interface PipelineSessionStub {
  /** Session id returned by POST /sessions. */
  readonly sessionId?: string;
  /** Tickers embedded in load/screen results — feed the result-run optimize. */
  readonly tickers?: string[];
  /** Step whose poll returns {status:'failed', gateReason} (abort-gate path). */
  readonly abortAtStep?: PipelineStepId;
  /** gateReason text surfaced on the abort path. */
  readonly gateReason?: string;
}

// Disjoint by path depth: `[^/]+` cannot cross a slash, so the single-segment
// create/delete patterns never shadow the deeper steps/artifacts routes.
const PB = '/pipeline-builder/sessions';
const PB_CREATE_RE = new RegExp(`${PB}(?:\\?.*)?$`);
const PB_DELETE_RE = new RegExp(`${PB}/[^/]+(?:\\?.*)?$`);
const PB_STEP_RE = new RegExp(`${PB}/[^/]+/steps/([^/?]+)`);
const PB_ARTIFACT_RE = new RegExp(`${PB}/[^/]+/artifacts/([^/?]+)`);

const PB_ARTIFACT_BODIES: Record<PbArtifactName, string> = {
  'report.md': '# Pipeline Report\n\nAll 17 acceptance checks passed.\n',
  'weights.csv': 'ticker,weight\nAAPL,0.6\nMSFT,0.4\n',
  'metrics.json': JSON.stringify({ sharpe: 1.2, annual_return: 0.12 }),
  'checklist.json': JSON.stringify({ pass_count: 17, total: 17, passed: true }),
};

const PB_ARTIFACT_MEDIA: Record<PbArtifactName, string> = {
  'report.md': 'text/markdown',
  'weights.csv': 'text/csv',
  'metrics.json': 'application/json',
  'checklist.json': 'application/json',
};

const PB_ARTIFACT_PATHS = {
  report_md: '/tmp/e2e/report.md',
  weights_csv: '/tmp/e2e/weights.csv',
  metrics_json: '/tmp/e2e/metrics.json',
  checklist_json: '/tmp/e2e/checklist.json',
  weights_diagnostic: '/tmp/e2e/weights_diagnostic.csv',
};

function pbStepResult(step: string, tickers: string[]): Record<string, unknown> {
  switch (step) {
    case 'load':
      return {
        n_tickers: tickers.length, n_trading_days: 1260, assembly_hash: 'e2e-hash',
        base_currency: 'EUR', price_start: '2020-01-01', price_end: '2025-01-01', tickers,
      };
    case 'screen':
      return {
        n_investable: tickers.length, preset: 'developed_markets',
        band_warning: false, band_low: 15, band_high: 30, tickers,
      };
    case 'clean_returns':
      return {
        n_days: 1260, n_tickers: tickers.length, return_start: '2020-01-02',
        return_end: '2025-01-01', preprocessing_steps: [],
      };
    case 'build_history':
      return {
        succeeded_dates: 12, total_dates: 12, failed_dates: 0, n_factors: 4,
        rebalance_freq: 21, market_proxy_loaded: true,
      };
    case 'validate_is':
      return {
        ic_results: [], n_significant: 0, significant_factors: [],
        high_vif_factors: [], config: {},
      };
    case 'validate_oos':
      return { n_folds: 0, oos_results: [], config: {} };
    case 'coverage_gate':
      return {
        passing_factors: ['momentum', 'value'], is_only_factors: [],
        oos_only_factors: [], n_passing: 2, min_factors: 2,
      };
    case 'report':
      return {
        pass_count: 17, checklist_total: 17, checklist_passed: true,
        checklist_rules: [], metrics: {}, artifact_paths: PB_ARTIFACT_PATHS,
        chart_paths: [], output_dir: '/tmp/e2e',
      };
    case 'persist':
      return { persisted: true, reason: 'all checks passed', pass_count: 17, checklist_passed: true };
    default:
      return {};
  }
}

function pbSegment(url: string, re: RegExp): string {
  return decodeURIComponent(url.match(re)?.[1] ?? '');
}

/**
 * Register session-create, per-step (POST + poll), delete and artifact routes
 * for one pipeline-builder session. A single registrar keeps journeys from
 * re-declaring `page.route()` blocks.
 */
export async function stubPipelineSession(
  page: Page,
  opts: PipelineSessionStub = {},
): Promise<void> {
  const sessionId = opts.sessionId ?? 'e2e-session';
  const tickers = opts.tickers ?? ['AAPL', 'MSFT'];

  await page.route(PB_CREATE_RE, async (route) => {
    if (route.request().method() !== 'POST') { await route.continue(); return; }
    await fulfillJson(route, 201, { sessionId });
  });

  await page.route(PB_DELETE_RE, async (route) => {
    if (route.request().method() !== 'DELETE') { await route.continue(); return; }
    await route.fulfill({ status: 204, body: '' });
  });

  await page.route(PB_STEP_RE, async (route) => {
    const step = pbSegment(route.request().url(), PB_STEP_RE);
    if (route.request().method() === 'POST') {
      await fulfillJson(route, 202, { job_id: `job-${step}`, status: 'pending', message: '' });
      return;
    }
    if (step === opts.abortAtStep) {
      await fulfillJson(route, 200, {
        status: 'failed', progress: {}, result: null, error: null,
        gateReason: opts.gateReason ?? 'Only 1 factor passed IS ∩ OOS; 2 required.',
      });
      return;
    }
    await fulfillJson(route, 200, {
      status: 'completed', progress: {}, result: pbStepResult(step, tickers),
      error: null, gateReason: null,
    });
  });

  await stubPipelineArtifacts(page);
}

/** Stub the four artifact downloads with attachment dispositions + bodies. */
export async function stubPipelineArtifacts(page: Page): Promise<void> {
  await page.route(PB_ARTIFACT_RE, async (route) => {
    const name = pbSegment(route.request().url(), PB_ARTIFACT_RE) as PbArtifactName;
    await route.fulfill({
      status: 200,
      contentType: PB_ARTIFACT_MEDIA[name] ?? 'application/octet-stream',
      headers: { 'content-disposition': `attachment; filename="${name}"` },
      body: PB_ARTIFACT_BODIES[name] ?? '',
    });
  });
}

// ── Secondary-page fixtures (#859) ────────────────────────────────────────────

// Factor Research → score panel. POST /factors/score returns per-ticker scores;
// the panel maps them to a quintile table.
const FACTOR_SCORES = { AAPL: 0.82, MSFT: 0.41, GOOGL: 0.55, AMZN: -0.12, NVDA: 0.93 };

export async function stubFactorScore(page: Page): Promise<void> {
  await page.route('**/api/v1/factors/score', async (route) => {
    if (route.request().method() !== 'POST') { await route.continue(); return; }
    await fulfillJson(route, 200, { scores: FACTOR_SCORES });
  });
}

// Macro Intelligence → empty FRED/TE series so every regime score is 0 and the
// composite regime resolves to "Transitional" (deterministic, seed-independent).
export async function stubMacroTransitional(page: Page): Promise<void> {
  await page.route('**/api/v1/macro-data/fred/series*', (route) => fulfillJson(route, 200, []));
  await page.route('**/api/v1/macro-data/te-observations*', (route) => fulfillJson(route, 200, []));
}

// Risk Center → a populated correlation matrix so the heatmap renders a canvas
// (the landing-read stub returns an empty matrix).
export async function stubRiskCorrelation(page: Page): Promise<void> {
  await page.route('**/api/v1/portfolio/*/risk/correlation*', (route) =>
    fulfillJson(route, 200, {
      assets: ['AAPL', 'MSFT'],
      matrix: [[1, 0.5], [0.5, 1]],
      clusterLabels: [0, 1],
    } satisfies CorrelationApiResponse),
  );
}

// Settings → Portfolios round-trip. GET lists the seeded portfolio; after a
// POST create, the list re-fetch includes the new portfolio.
export const SEEDED_PORTFOLIO_NAME = 'e2e-portfolio';
export const CREATED_PORTFOLIO_NAME = 'journey-portfolio';

function makePortfolioDto(id: string, name: string, active: boolean): PortfolioDto {
  return {
    id,
    name,
    description: null,
    currency: 'USD',
    benchmark_ticker: 'SPY',
    is_active: active,
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:00:00Z',
  };
}

const SEEDED_PF = makePortfolioDto('pf-e2e', SEEDED_PORTFOLIO_NAME, true);
const CREATED_PF = makePortfolioDto('pf-new', CREATED_PORTFOLIO_NAME, false);

export async function stubPortfoliosRoundTrip(page: Page): Promise<void> {
  await page.route('**/api/v1/market/indices*', (route) =>
    fulfillJson(route, 200, { indices: [{ ticker: 'SPY', name: 'S&P 500' }] }),
  );

  let created = false;
  await page.route('**/api/v1/portfolio/', async (route) => {
    if (route.request().method() === 'POST') {
      created = true;
      await fulfillJson(route, 201, CREATED_PF);
      return;
    }
    const items = created ? [SEEDED_PF, CREATED_PF] : [SEEDED_PF];
    await fulfillJson(route, 200, { items, total: items.length } satisfies PortfolioListResponseDto);
  });
}
