/**
 * Source-blind contract + error-state spec for BacktestingComponent (issue #954).
 *
 * Authored from acceptance criteria and requirements §8 only.
 * No implementation source was read.
 *
 * Criteria pinned (T3-verifiable per oracle report):
 *
 * [T3] "Every page/panel shows a visible, non-blank error state when its primary
 *       API call returns an error."
 *   → After POST /backtest 4xx/5xx, a [role="alert"] element with non-empty text
 *     is rendered in the DOM.
 *
 * [T3] "Every wired page has specs covering: (a) loads data on init, (b) handles
 *       API error, and (c) reacts to portfolio context change."
 *   → (a) initial state is correct — not running, no error, no result.
 *   → (b) POST /backtest error → error state visible.
 *   → (c) portfolio change → tickers reflect new portfolio (contract only; seeding
 *         detail already covered in backtesting-ticker-seeding.spec.ts).
 *
 * Contract tests (requirements §8):
 *   → POST /backtest body contains `tickers` as a non-empty string[].
 *   → POST /backtest body contains `start_date` as an ISO-date string (YYYY-MM-DD).
 *   → POST /backtest body contains `end_date` as an ISO-date string (YYYY-MM-DD).
 *   → POST /backtest body uses snake_case keys (no camelCase leakage).
 *   → Async path: POST /backtest returns { job_id, run_id, status } → polling starts.
 *   → Polling GET /backtest/{jobId} → result delivered and polling stops.
 *   → Walk-forward POST /validate/walk-forward body contains tickers and dates.
 *   → Walk-forward error → visible error state in DOM.
 *
 * Assumption: BacktestingComponent exposes:
 *   - onRun(): void — triggers POST /backtest using current signal state
 *   - onRunWalkForward(): void — triggers POST /validate/walk-forward
 *   - onJobCompleted(runId: string): void — called by JobProgressTracker when done
 *   - onJobFailed(message: string): void — called on job failure
 *   - isRunning: WritableSignal<boolean>
 *   - runError: WritableSignal<string | null>
 *   - hasResult: Signal<boolean> (computed from runResult !== null)
 *   - isPolling: Signal<boolean> (computed from runJobId !== null && !hasResult)
 *   - tickersRaw: WritableSignal<string> — comma-separated string
 *   - tickers: Signal<string[]> — computed array from tickersRaw
 *
 * If the run trigger method is named differently (e.g. onRunBacktest), these tests
 * will fail at compile time, making the discrepancy explicit for the implementer.
 *
 * No property-based tests: none of the T3 criteria imply round-trip, idempotence,
 * never-raises, or ordering invariants at this scope.
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';
import type { HttpTestingController, TestRequest } from '@angular/common/http/testing';

import {
  assertBodyKeys,
  assertMethod,
  assertUrl,
  configureTestBed,
  injectHttp,
  installResizeObserverStub,
} from '../../testing';
import { ICON_PROVIDER } from '../icons';
import { BacktestingComponent } from './backtesting';
import { BacktestService } from './backtest.service';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { environment } from '../../environments/environment';

const BACKTEST_URL = `${environment.apiUrl}backtest`;
const WALK_FORWARD_URL = `${environment.apiUrl}validate/walk-forward`;
const PORTFOLIO_LIST_URL = `${environment.apiUrl}portfolio/`;

// Matches BacktestAsyncResponse: backend uses CamelCaseModel for /backtest responses.
const ASYNC_RESPONSE = { jobId: 'job-bt-1', runId: 'run-bt-1', status: 'pending', message: '' };
const BACKTEST_RESULT = {
  id: 'run-bt-1',
  portfolioId: null,
  jobId: null,
  status: 'completed' as const,
  config: {},
  equityCurve: { '2024-01-01': 0.0, '2024-06-01': 0.05 },
  drawdowns: {},
  monthlyReturns: {},
  yearlyReturns: {},
  rollingMetrics: {},
  turnoverHistory: {},
  cvFoldMetrics: null,
  summaryStats: {},
  errorMessage: null,
  durationSeconds: 5,
  createdAt: '2024-01-01T00:00:00Z',
  updatedAt: '2024-06-01T00:00:00Z',
};

function buildPortfolioList(portfolios: { id: string; name: string }[]) {
  return {
    items: portfolios.map((p) => ({
      id: p.id,
      name: p.name,
      description: null,
      currency: 'USD',
      benchmark_ticker: 'SPY',
    })),
    total: portfolios.length,
  };
}

describe('BacktestingComponent — /backtest contracts & error state (issue #954)', () => {
  let fixture: ComponentFixture<BacktestingComponent>;
  let comp: BacktestingComponent;
  let http: HttpTestingController;
  let ctx: PortfolioContextService;

  beforeEach(async () => {
    localStorage.clear();
    installResizeObserverStub();
    await configureTestBed({
      imports: [BacktestingComponent],
      withHttp: true,
      providers: [ICON_PROVIDER],
    });
    fixture = TestBed.createComponent(BacktestingComponent);
    comp = fixture.componentInstance;
    http = injectHttp();
    ctx = TestBed.inject(PortfolioContextService);
    fixture.detectChanges();
  });

  afterEach(() => {
    // Drain the PortfolioContextService auto-select bootstrap (GET /portfolio/)
    // that fires when no portfolio id is stored, so verify() only asserts on the
    // requests each test explicitly exercises.
    http.match((r) => r.method === 'GET' && r.url.endsWith('/portfolio/'));
    http.verify();
    localStorage.clear();
  });

  // ── T3-a: loads data on init ─────────────────────────────────────────────────

  it('when component initialises, isRunning is false', () => {
    expect(comp.isRunning()).toBe(false);
  });

  it('when component initialises, runError is null', () => {
    expect(comp.runError()).toBeNull();
  });

  it('when component initialises, hasResult is false', () => {
    expect(comp.hasResult()).toBe(false);
  });

  it('when component initialises, isPolling is false', () => {
    expect(comp.isPolling()).toBe(false);
  });

  it('when component initialises without a portfolio, tickersRaw is the default string', () => {
    // tickers must be a non-empty comma-separated string; DEFAULT_TICKERS applies.
    expect(comp.tickersRaw().trim().length).toBeGreaterThan(0);
    http.expectNone((r) => r.url.includes('snapshot'));
  });

  // ── POST /backtest — request body contract (requirements §8) ─────────────────

  it('when backtest runs, POST is sent to /backtest', () => {
    comp.onRun();
    const req = http.expectOne((r) => r.method === 'POST' && r.url === BACKTEST_URL);
    req.flush(ASYNC_RESPONSE);
    expect(req.request.url).toBe(BACKTEST_URL);
  });

  it('when backtest runs, request body contains tickers as a non-empty array', () => {
    comp.onRun();
    const req = http.expectOne(BACKTEST_URL);
    const body = req.request.body as Record<string, unknown>;
    expect(Array.isArray(body['tickers'])).toBe(true);
    expect((body['tickers'] as unknown[]).length).toBeGreaterThan(0);
    req.flush(ASYNC_RESPONSE);
  });

  it('when backtest runs, request body contains start_date as an ISO date string', () => {
    comp.onRun();
    const req = http.expectOne(BACKTEST_URL);
    const body = req.request.body as Record<string, unknown>;
    expect(typeof body['start_date']).toBe('string');
    expect(body['start_date'] as string).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    req.flush(ASYNC_RESPONSE);
  });

  it('when backtest runs, request body contains end_date as an ISO date string', () => {
    comp.onRun();
    const req = http.expectOne(BACKTEST_URL);
    const body = req.request.body as Record<string, unknown>;
    expect(typeof body['end_date']).toBe('string');
    expect(body['end_date'] as string).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    req.flush(ASYNC_RESPONSE);
  });

  it('when backtest runs, start_date precedes end_date', () => {
    comp.onRun();
    const req = http.expectOne(BACKTEST_URL);
    const body = req.request.body as Record<string, unknown>;
    const start = new Date(body['start_date'] as string);
    const end = new Date(body['end_date'] as string);
    expect(start.getTime()).toBeLessThan(end.getTime());
    req.flush(ASYNC_RESPONSE);
  });

  it('when backtest runs, request body uses snake_case keys only', () => {
    comp.onRun();
    const req = http.expectOne(BACKTEST_URL);
    const keys = Object.keys(req.request.body as object);
    const camelCaseKeys = keys.filter((k) => /[A-Z]/.test(k));
    expect(camelCaseKeys).toEqual([]);
    req.flush(ASYNC_RESPONSE);
  });

  // ── AC1: body matches BacktestRequest field-for-field, no extra keys ──────────

  it('when backtest runs, request body contains pipeline_config as an object', () => {
    comp.onRun();
    const req = http.expectOne(BACKTEST_URL);
    const body = req.request.body as Record<string, unknown>;
    expect(typeof body['pipeline_config']).toBe('object');
    expect(body['pipeline_config']).not.toBeNull();
    req.flush(ASYNC_RESPONSE);
  });

  it('when backtest runs, the body has exactly the BacktestRequest keys', () => {
    comp.onRun();
    const req = http.expectOne(BACKTEST_URL);
    expect(Object.keys(req.request.body as object).sort()).toEqual([
      'end_date',
      'pipeline_config',
      'start_date',
      'tickers',
    ]);
    req.flush(ASYNC_RESPONSE);
  });

  it('when backtest runs, pipeline_config carries the selected benchmark', () => {
    comp.onRun();
    const req = http.expectOne(BACKTEST_URL);
    const pipelineConfig = (req.request.body as {
      pipeline_config: Record<string, unknown>;
    }).pipeline_config;
    expect(pipelineConfig['benchmark']).toBe('SPY');
    req.flush(ASYNC_RESPONSE);
  });

  it('when backtest runs, request body tickers reflect the current tickersRaw signal', () => {
    comp.tickersRaw.set('TSLA, NVDA');
    comp.onRun();
    const req = http.expectOne(BACKTEST_URL);
    const body = req.request.body as Record<string, unknown>;
    expect(body['tickers']).toEqual(jasmine.arrayContaining(['TSLA', 'NVDA']));
    req.flush(ASYNC_RESPONSE);
  });

  // ── POST /backtest — async response path ─────────────────────────────────────

  it('when /backtest returns a job_id, isPolling becomes true', () => {
    comp.onRun();
    http.expectOne(BACKTEST_URL).flush(ASYNC_RESPONSE);
    expect(comp.isPolling()).toBe(true);
  });

  it('when /backtest returns a job_id, isRunning stays true while polling', () => {
    comp.onRun();
    http.expectOne(BACKTEST_URL).flush(ASYNC_RESPONSE);
    expect(comp.isRunning()).toBe(true);
  });

  it('when /backtest returns a job_id, hasResult stays false while polling', () => {
    comp.onRun();
    http.expectOne(BACKTEST_URL).flush(ASYNC_RESPONSE);
    expect(comp.hasResult()).toBe(false);
  });

  // ── Polling GET /backtest/{jobId} ─────────────────────────────────────────────

  it('when job completes, GET /backtest/{runId} is fetched and hasResult becomes true', () => {
    comp.onRun();
    http.expectOne(BACKTEST_URL).flush(ASYNC_RESPONSE);

    comp.onJobCompleted('run-bt-1');
    const pollReq = http.expectOne(
      (r) => r.method === 'GET' && r.url.includes('run-bt-1'),
    );
    pollReq.flush(BACKTEST_RESULT);

    expect(comp.hasResult()).toBe(true);
    expect(comp.isPolling()).toBe(false);
  });

  it('when job completes, isRunning becomes false after result is fetched', () => {
    comp.onRun();
    http.expectOne(BACKTEST_URL).flush(ASYNC_RESPONSE);
    comp.onJobCompleted('run-bt-1');
    http.expectOne((r) => r.method === 'GET' && r.url.includes('run-bt-1')).flush(BACKTEST_RESULT);
    expect(comp.isRunning()).toBe(false);
  });

  // ── T3-b: handles API error — visible non-blank error state ───────────────────

  it('when /backtest returns 5xx, runError is non-null', () => {
    comp.onRun();
    http
      .expectOne(BACKTEST_URL)
      .flush({ detail: 'Server error' }, { status: 500, statusText: 'Server Error' });
    expect(comp.runError()).not.toBeNull();
  });

  it('when /backtest returns 5xx, isRunning becomes false', () => {
    comp.onRun();
    http
      .expectOne(BACKTEST_URL)
      .flush({ detail: 'Server error' }, { status: 500, statusText: 'Server Error' });
    expect(comp.isRunning()).toBe(false);
  });

  it('when /backtest returns 5xx, a visible non-blank error element is rendered', () => {
    // Requirement: "visible, non-blank error state (message or alert component)"
    comp.onRun();
    http
      .expectOne(BACKTEST_URL)
      .flush({ detail: 'Backtest failed' }, { status: 500, statusText: 'Server Error' });
    fixture.detectChanges();

    const alertEl = fixture.nativeElement.querySelector('[role="alert"]');
    expect(alertEl).withContext('expected a [role="alert"] element in the DOM').toBeTruthy();
    expect(alertEl.textContent.trim().length)
      .withContext('error message must not be blank')
      .toBeGreaterThan(0);
  });

  it('when /backtest returns 4xx, a visible non-blank error element is rendered', () => {
    comp.onRun();
    http
      .expectOne(BACKTEST_URL)
      .flush({ detail: 'Bad request' }, { status: 400, statusText: 'Bad Request' });
    fixture.detectChanges();

    const alertEl = fixture.nativeElement.querySelector('[role="alert"]');
    expect(alertEl).withContext('expected a [role="alert"] element after 4xx').toBeTruthy();
    expect(alertEl.textContent.trim().length).toBeGreaterThan(0);
  });

  it('when onJobFailed is called, runError is set and a visible alert is rendered', () => {
    comp.onJobFailed('Backtest job timed out');
    fixture.detectChanges();

    expect(comp.runError()).toBe('Backtest job timed out');
    const alertEl = fixture.nativeElement.querySelector('[role="alert"]');
    expect(alertEl).withContext('expected [role="alert"] after job failure').toBeTruthy();
    expect(alertEl.textContent.trim().length).toBeGreaterThan(0);
  });

  // ── Walk-forward — contract (requirements §8) ─────────────────────────────────

  it('when walk-forward runs, POST is sent to /validate/walk-forward', () => {
    comp.onRunWalkForward();
    const req = http.expectOne(
      (r) => r.method === 'POST' && r.url === WALK_FORWARD_URL,
    );
    req.flush({ job_id: 'wf-job-1', run_id: 'wf-run-1', status: 'pending' });
    expect(req.request.url).toBe(WALK_FORWARD_URL);
  });

  it('when walk-forward runs, the body carries the full ValidateRequest contract', () => {
    comp.onRunWalkForward();
    const req = http.expectOne(WALK_FORWARD_URL);
    expect(Object.keys(req.request.body as object).sort()).toEqual([
      'cv_config',
      'cv_type',
      'end_date',
      'optimizer_config',
      'optimizer_type',
      'start_date',
      'tickers',
    ]);
    req.flush({ job_id: 'wf-job-1', run_id: 'wf-run-1', status: 'pending' });
  });

  it('when walk-forward runs, request body contains tickers as a non-empty array', () => {
    comp.onRunWalkForward();
    const req = http.expectOne(WALK_FORWARD_URL);
    const body = req.request.body as Record<string, unknown>;
    expect(Array.isArray(body['tickers'])).toBe(true);
    expect((body['tickers'] as unknown[]).length).toBeGreaterThan(0);
    req.flush({ job_id: 'wf-job-1', run_id: 'wf-run-1', status: 'pending' });
  });

  it('when walk-forward runs, request body contains start_date as an ISO date string', () => {
    comp.onRunWalkForward();
    const req = http.expectOne(WALK_FORWARD_URL);
    const body = req.request.body as Record<string, unknown>;
    expect(typeof body['start_date']).toBe('string');
    expect(body['start_date'] as string).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    req.flush({ job_id: 'wf-job-1', run_id: 'wf-run-1', status: 'pending' });
  });

  it('when walk-forward errors, a visible non-blank error element is rendered', () => {
    comp.onRunWalkForward();
    http
      .expectOne(WALK_FORWARD_URL)
      .flush({ detail: 'Walk-forward failed' }, { status: 500, statusText: 'Server Error' });
    fixture.detectChanges();

    const alertEl = fixture.nativeElement.querySelector('[role="alert"]');
    expect(alertEl).withContext('expected [role="alert"] after walk-forward error').toBeTruthy();
    expect(alertEl.textContent.trim().length).toBeGreaterThan(0);
  });

  // ── T3-c: reacts to portfolio context change ──────────────────────────────────

  it('when portfolio is selected and then backtest runs, tickers in body reflect the snapshot', () => {
    ctx.currentPortfolioId.set('pf-1');
    fixture.detectChanges();

    http
      .expectOne((r) => r.method === 'GET' && r.url === PORTFOLIO_LIST_URL)
      .flush(buildPortfolioList([{ id: 'pf-1', name: 'bt-portfolio' }]));
    fixture.detectChanges();

    // Two snapshot requests fire in parallel:
    //   • initWeightResolver() uses portfolio ID → /portfolio/pf-1/snapshots/latest
    //   • initTickerSeeding() uses portfolio name → /portfolio/bt-portfolio/snapshots/latest
    const snapshotPayload = {
      id: 'snap-1',
      portfolio_id: 'pf-1',
      snapshot_date: '2026-01-01',
      snapshot_type: 'manual',
      weights: { TSLA: 0.6, GOOGL: 0.4 },
      sector_mapping: null,
      summary: null,
      optimizer_config: null,
      turnover: null,
      holding_count: 2,
      created_at: '2026-01-01T00:00:00Z',
    };
    http.match((r) => r.url.includes('snapshots/latest')).forEach((r) => r.flush(snapshotPayload));
    fixture.detectChanges();

    // onRun() passes this.tickers() (seeded from snapshot) to onRunBacktest()
    comp.onRun();
    const req = http.expectOne(BACKTEST_URL);
    expect(req.request.body['tickers']).toEqual(jasmine.arrayContaining(['TSLA', 'GOOGL']));
    req.flush(ASYNC_RESPONSE);
  });
});

// ── Service-level request contract parity (issue #1000) ──────────────────────
// Covers every BacktestService method's method + URL (path params encoded, no
// raw token) + body, independently of the page component above.
describe('BacktestService — request contract parity (issue #1000)', () => {
  let svc: BacktestService;
  let http: HttpTestingController;
  const JOB_ID = 'job 1';
  const JOB_ENC = encodeURIComponent(JOB_ID);
  const RUN_ID = 'run 1';
  const RUN_ENC = encodeURIComponent(RUN_ID);
  const NAME_ENC = encodeURIComponent('My Port');

  beforeEach(async () => {
    await configureTestBed({ withHttp: true });
    svc = TestBed.inject(BacktestService);
    http = injectHttp();
  });

  afterEach(() => http.verify());

  function capture(url: string): TestRequest {
    const req = http.expectOne((r) => r.url === url);
    req.flush({});
    return req;
  }

  it('when runBacktest is called, a POST to /backtest carries the request body', () => {
    svc.runBacktest({ tickers: ['AAPL'], start_date: '2024-01-01', end_date: '2024-12-31' })
      .subscribe({ error: () => {} });
    const req = capture(BACKTEST_URL);
    assertMethod(req, 'POST');
    assertUrl(req, BACKTEST_URL);
    assertBodyKeys(req, ['tickers', 'start_date', 'end_date']);
  });

  it('when pollBacktest is called, a GET to the interpolated /backtest/{jobId} URL is issued', () => {
    svc.pollBacktest(JOB_ID).subscribe({ error: () => {} });
    const req = capture(`${BACKTEST_URL}/${JOB_ENC}`);
    assertMethod(req, 'GET');
    assertUrl(req, `${BACKTEST_URL}/${JOB_ENC}`);
  });

  it('when getBacktestRun is called, a GET to the interpolated /backtest/runs/{runId} URL is issued', () => {
    svc.getBacktestRun(RUN_ID).subscribe({ error: () => {} });
    const req = capture(`${BACKTEST_URL}/runs/${RUN_ENC}`);
    assertMethod(req, 'GET');
    assertUrl(req, `${BACKTEST_URL}/runs/${RUN_ENC}`);
  });

  it('when runWalkForward is called, a POST to /validate/walk-forward carries the request body', () => {
    svc.runWalkForward({ tickers: ['AAPL'], start_date: '2024-01-01', end_date: '2024-12-31' })
      .subscribe({ error: () => {} });
    const req = capture(WALK_FORWARD_URL);
    assertMethod(req, 'POST');
    assertUrl(req, WALK_FORWARD_URL);
    assertBodyKeys(req, ['tickers', 'start_date', 'end_date']);
  });

  it('when pollWalkForward is called, a GET to the interpolated walk-forward job URL is issued', () => {
    svc.pollWalkForward(JOB_ID).subscribe({ error: () => {} });
    const req = capture(`${WALK_FORWARD_URL}/${JOB_ENC}`);
    assertMethod(req, 'GET');
    assertUrl(req, `${WALK_FORWARD_URL}/${JOB_ENC}`);
  });

  it('when getEquityCurve is called, a GET to the interpolated analytics URL is issued', () => {
    svc.getEquityCurve('My Port').subscribe({ error: () => {} });
    const req = capture(`${environment.apiUrl}portfolio-analytics/${NAME_ENC}/equity-curve`);
    assertMethod(req, 'GET');
    assertUrl(req, `${environment.apiUrl}portfolio-analytics/${NAME_ENC}/equity-curve`);
  });
});
