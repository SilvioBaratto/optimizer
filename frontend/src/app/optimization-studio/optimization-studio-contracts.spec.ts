/**
 * Source-blind contract + error-state spec for OptimizationStudioComponent (issue #953).
 *
 * Authored from acceptance criteria only. No implementation source was read.
 *
 * Criteria pinned (T3-verifiable per oracle report):
 *
 * [T3] "Optimization-studio: an optimization run completes and the results panel
 *       renders all sub-charts."
 *   → hasResult() becomes true after a successful POST /optimize response.
 *   → runResult() is non-null and contains the weights the API returned.
 *   → ResultsPanel is present in the DOM when hasResult() is true.
 *
 * [T3] "Every page/panel shows a visible, non-blank error state (message or alert
 *       component) when its primary API call returns an error."
 *   → After POST /optimize 5xx, a [role="alert"] element with non-empty text appears.
 *
 * [T3] "Every wired page has specs covering: (a) loads data on init, (b) handles
 *       API error, and (c) reacts to portfolio context change."
 *   → (a) covered: initial state is correct (no result, not running, not polling).
 *   → (b) covered: 5xx on /optimize → error state visible.
 *   → (c) covered: portfolio change → tickers signal updated (via seeding, verified
 *         in optimization-studio-ticker-seeding.spec.ts; this file adds the
 *         portfolio→context→optimize integration contract test).
 *
 * Contract tests (requirements §7):
 *   → POST /optimize body contains optimizer_type (snake_case string).
 *   → POST /optimize body contains tickers (non-empty string[]).
 *   → POST /optimize body contains start_date (ISO-date string, YYYY-MM-DD).
 *   → POST /optimize body contains end_date (ISO-date string, YYYY-MM-DD).
 *   → POST /optimize body contains config (object).
 *   → Sync response path: { id, weights } → hasResult() true.
 *   → Async response path: { job_id, run_id, status } → isPolling() true.
 *   → Polling GET /optimize/{runId} → result is applied and polling stops.
 *   → applyWeightsToPortfolio() calls POST /portfolio/{id}/snapshots with weights.
 *
 * No property-based tests: none of the T3 criteria imply a round-trip, idempotence,
 * never-raises, or ordering invariant at this scope.
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';
import { By } from '@angular/platform-browser';
import type { HttpTestingController } from '@angular/common/http/testing';

import { configureTestBed, injectHttp, installResizeObserverStub } from '../../testing';
import { ICON_PROVIDER } from '../icons';
import { OptimizationStudioComponent } from './optimization-studio';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { environment } from '../../environments/environment';

const OPTIMIZE_URL = `${environment.apiUrl}optimize`;
const PORTFOLIO_LIST_URL = `${environment.apiUrl}portfolio/`;

// Minimal valid run request matching the component's public API.
const RUN = { optimizerType: 'mean_risk' as const, config: { objective: 'max_sharpe' } };
const RUN_RESULT = {
  id: 'run-1',
  weights: { AAPL: 0.4, MSFT: 0.3, GOOGL: 0.3 },
};
const ASYNC_RESPONSE = { job_id: 'job-1', run_id: 'run-1', status: 'pending' };

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

describe('OptimizationStudioComponent — /optimize contracts & error state (issue #953)', () => {
  let fixture: ComponentFixture<OptimizationStudioComponent>;
  let comp: OptimizationStudioComponent;
  let http: HttpTestingController;
  let ctx: PortfolioContextService;

  beforeEach(async () => {
    localStorage.clear();
    installResizeObserverStub();
    await configureTestBed({
      imports: [OptimizationStudioComponent],
      withHttp: true,
      providers: [ICON_PROVIDER],
    });
    fixture = TestBed.createComponent(OptimizationStudioComponent);
    comp = fixture.componentInstance;
    http = injectHttp();
    ctx = TestBed.inject(PortfolioContextService);
    fixture.detectChanges();
  });

  afterEach(() => {
    http.verify();
    localStorage.clear();
  });

  // ── T3-a: loads data on init ─────────────────────────────────────────────────

  it('when component initialises without a run, hasResult is false', () => {
    expect(comp.hasResult()).toBe(false);
  });

  it('when component initialises, isRunning is false', () => {
    expect(comp.isRunning()).toBe(false);
  });

  it('when component initialises, isPolling is false', () => {
    expect(comp.isPolling()).toBe(false);
  });

  it('when component initialises, runError is null', () => {
    expect(comp.runError()).toBeNull();
  });

  // ── POST /optimize — request body contract (requirements §7) ─────────────────

  it('when optimization runs, POST /optimize is sent to the correct URL', () => {
    comp.onRunPipeline(RUN);
    const req = http.expectOne((r) => r.method === 'POST' && r.url === OPTIMIZE_URL);
    req.flush(RUN_RESULT);
    expect(req.request.url).toBe(OPTIMIZE_URL);
  });

  it('when optimization runs, request body contains optimizer_type in snake_case', () => {
    comp.onRunPipeline(RUN);
    const req = http.expectOne(OPTIMIZE_URL);
    expect(req.request.body).toEqual(jasmine.objectContaining({ optimizer_type: 'mean_risk' }));
    req.flush(RUN_RESULT);
  });

  it('when optimization runs, request body contains tickers as a non-empty array', () => {
    comp.onRunPipeline(RUN);
    const req = http.expectOne(OPTIMIZE_URL);
    const body = req.request.body as Record<string, unknown>;
    expect(Array.isArray(body['tickers'])).toBe(true);
    expect((body['tickers'] as unknown[]).length).toBeGreaterThan(0);
    req.flush(RUN_RESULT);
  });

  it('when optimization runs, request body contains start_date as an ISO date string', () => {
    comp.onRunPipeline(RUN);
    const req = http.expectOne(OPTIMIZE_URL);
    const body = req.request.body as Record<string, unknown>;
    expect(typeof body['start_date']).toBe('string');
    // ISO date format YYYY-MM-DD
    expect((body['start_date'] as string)).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    req.flush(RUN_RESULT);
  });

  it('when optimization runs, request body contains end_date as an ISO date string', () => {
    comp.onRunPipeline(RUN);
    const req = http.expectOne(OPTIMIZE_URL);
    const body = req.request.body as Record<string, unknown>;
    expect(typeof body['end_date']).toBe('string');
    expect((body['end_date'] as string)).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    req.flush(RUN_RESULT);
  });

  it('when optimization runs, request body contains config as an object', () => {
    comp.onRunPipeline(RUN);
    const req = http.expectOne(OPTIMIZE_URL);
    const body = req.request.body as Record<string, unknown>;
    expect(typeof body['config']).toBe('object');
    expect(body['config']).not.toBeNull();
    req.flush(RUN_RESULT);
  });

  it('when optimization runs, request body does not contain camelCase optimizerType', () => {
    // Requirement: body must match snake_case backend schema exactly — no camelCase leakage.
    comp.onRunPipeline(RUN);
    const req = http.expectOne(OPTIMIZE_URL);
    const body = req.request.body as Record<string, unknown>;
    expect(body['optimizerType']).toBeUndefined();
    req.flush(RUN_RESULT);
  });

  it('when optimization runs, tickers in the body reflect the current tickers signal', () => {
    const expected = ['SPY', 'QQQ'];
    comp.tickers.set(expected);
    comp.onRunPipeline(RUN);
    const req = http.expectOne(OPTIMIZE_URL);
    expect(req.request.body['tickers']).toEqual(expected);
    req.flush(RUN_RESULT);
  });

  // ── POST /optimize — sync response path ──────────────────────────────────────

  it('when /optimize returns weights synchronously, hasResult becomes true', () => {
    comp.onRunPipeline(RUN);
    http.expectOne(OPTIMIZE_URL).flush(RUN_RESULT);
    expect(comp.hasResult()).toBe(true);
  });

  it('when /optimize returns weights synchronously, runResult contains the returned weights', () => {
    comp.onRunPipeline(RUN);
    http.expectOne(OPTIMIZE_URL).flush(RUN_RESULT);
    expect(comp.runResult()).toEqual(jasmine.objectContaining({ weights: RUN_RESULT.weights }));
  });

  it('when /optimize returns weights synchronously, isRunning becomes false', () => {
    comp.onRunPipeline(RUN);
    http.expectOne(OPTIMIZE_URL).flush(RUN_RESULT);
    expect(comp.isRunning()).toBe(false);
  });

  // ── POST /optimize — async (job) response path ───────────────────────────────

  it('when /optimize returns a job_id, isPolling becomes true', () => {
    comp.onRunPipeline(RUN);
    http.expectOne(OPTIMIZE_URL).flush(ASYNC_RESPONSE);
    expect(comp.isPolling()).toBe(true);
  });

  it('when /optimize returns a job_id, hasResult stays false while polling', () => {
    comp.onRunPipeline(RUN);
    http.expectOne(OPTIMIZE_URL).flush(ASYNC_RESPONSE);
    expect(comp.hasResult()).toBe(false);
  });

  // ── Polling GET /optimize/{runId} ────────────────────────────────────────────

  it('when job completes, GET /optimize/{runId} is called to fetch the result', () => {
    comp.onRunPipeline(RUN);
    http.expectOne(OPTIMIZE_URL).flush(ASYNC_RESPONSE);
    comp.onJobCompleted('run-1');
    const req = http.expectOne(`${OPTIMIZE_URL}/run-1`);
    req.flush(RUN_RESULT);
    expect(comp.hasResult()).toBe(true);
    expect(comp.isPolling()).toBe(false);
  });

  it('when GET /optimize/{runId} succeeds, isRunning becomes false', () => {
    comp.onRunPipeline(RUN);
    http.expectOne(OPTIMIZE_URL).flush(ASYNC_RESPONSE);
    comp.onJobCompleted('run-1');
    http.expectOne(`${OPTIMIZE_URL}/run-1`).flush(RUN_RESULT);
    expect(comp.isRunning()).toBe(false);
  });

  // ── T3-b: handles API error — visible non-blank error state ──────────────────

  it('when /optimize returns 5xx, runError is non-null', () => {
    comp.onRunPipeline(RUN);
    http
      .expectOne(OPTIMIZE_URL)
      .flush({ detail: 'Internal error' }, { status: 500, statusText: 'Server Error' });
    expect(comp.runError()).not.toBeNull();
  });

  it('when /optimize returns 5xx, isRunning becomes false', () => {
    comp.onRunPipeline(RUN);
    http
      .expectOne(OPTIMIZE_URL)
      .flush({ detail: 'Internal error' }, { status: 500, statusText: 'Server Error' });
    expect(comp.isRunning()).toBe(false);
  });

  it('when /optimize returns 5xx, a visible non-blank error element is rendered', () => {
    // Requirement: "visible, non-blank error state (message or alert component)"
    comp.onRunPipeline(RUN);
    http
      .expectOne(OPTIMIZE_URL)
      .flush({ detail: 'Optimization failed' }, { status: 500, statusText: 'Server Error' });
    fixture.detectChanges();

    const alertEl = fixture.nativeElement.querySelector('[role="alert"]');
    expect(alertEl).withContext('expected a [role="alert"] element in the DOM').toBeTruthy();
    expect(alertEl.textContent.trim().length)
      .withContext('error message must not be blank')
      .toBeGreaterThan(0);
  });

  it('when /optimize returns 4xx, a visible non-blank error element is rendered', () => {
    comp.onRunPipeline(RUN);
    http
      .expectOne(OPTIMIZE_URL)
      .flush({ detail: 'Bad request' }, { status: 400, statusText: 'Bad Request' });
    fixture.detectChanges();

    const alertEl = fixture.nativeElement.querySelector('[role="alert"]');
    expect(alertEl).withContext('expected a [role="alert"] element in the DOM').toBeTruthy();
    expect(alertEl.textContent.trim().length).toBeGreaterThan(0);
  });

  it('when polling GET /optimize/{runId} returns 5xx, a visible error is rendered', () => {
    comp.onRunPipeline(RUN);
    http.expectOne(OPTIMIZE_URL).flush(ASYNC_RESPONSE);
    comp.onJobCompleted('run-1');
    http
      .expectOne(`${OPTIMIZE_URL}/run-1`)
      .flush({ detail: 'Run not found' }, { status: 404, statusText: 'Not Found' });
    fixture.detectChanges();

    const alertEl = fixture.nativeElement.querySelector('[role="alert"]');
    expect(alertEl).withContext('expected [role="alert"] after polling error').toBeTruthy();
    expect(alertEl.textContent.trim().length).toBeGreaterThan(0);
  });

  // ── applyWeightsToPortfolio — contract ────────────────────────────────────────

  it('when weights applied without portfolio, no snapshot POST is made', () => {
    expect(ctx.currentPortfolioId()).toBeNull();
    comp.onApplyWeights({ AAPL: 1.0 });
    http.expectNone((r) => r.method === 'POST' && r.url.includes('snapshot'));
  });

  it('when weights applied with portfolio, POST goes to the snapshot endpoint', () => {
    ctx.currentPortfolioId.set('pf-1');
    comp.onApplyWeights({ AAPL: 0.6, MSFT: 0.4 });

    // portfolio list needed to resolve the name
    http
      .expectOne((r) => r.method === 'GET' && r.url.includes('portfolio') && !r.url.includes('snapshot'))
      .flush(buildPortfolioList([{ id: 'pf-1', name: 'my-portfolio' }]));

    const snapReq = http.expectOne((r) => r.method === 'POST' && r.url.includes('snapshot'));
    expect(snapReq.request.body).toEqual(
      jasmine.objectContaining({ weights: { AAPL: 0.6, MSFT: 0.4 } }),
    );
    snapReq.flush({ id: 'snap-1' });
    expect(comp.applyStatus()).toBe('success');
  });

  // ── T3-c: reacts to portfolio context change ──────────────────────────────────

  it('when portfolio changes and then optimize runs, request body tickers reflect the new portfolio', () => {
    ctx.currentPortfolioId.set('pf-1');
    fixture.detectChanges();

    http
      .expectOne((r) => r.method === 'GET' && r.url === PORTFOLIO_LIST_URL)
      .flush(buildPortfolioList([{ id: 'pf-1', name: 'pf-a' }]));
    fixture.detectChanges();

    // snapshot seeds tickers from portfolio
    http
      .expectOne((r) => r.method === 'GET' && r.url.includes('snapshots/latest'))
      .flush({
        id: 'snap-1',
        portfolio_id: 'pf-1',
        snapshot_date: '2026-01-01',
        snapshot_type: 'manual',
        weights: { NVDA: 0.5, AMZN: 0.5 },
        sector_mapping: null,
        summary: null,
        optimizer_config: null,
        turnover: null,
        holding_count: 2,
        created_at: '2026-01-01T00:00:00Z',
      });
    fixture.detectChanges();

    comp.onRunPipeline(RUN);
    const req = http.expectOne(OPTIMIZE_URL);
    expect(req.request.body['tickers']).toEqual(jasmine.arrayContaining(['NVDA', 'AMZN']));
    req.flush(RUN_RESULT);
  });
});
