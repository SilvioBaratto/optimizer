/**
 * Walk-forward and pre-run state tests for BacktestingComponent (issue #1024).
 *
 * Covers:
 *   - Pre-run state: hasLoadedResult() is false before any run (13b / KPI em-dash precondition)
 *   - onJobCompleted() → GET /backtest/runs/{runId} is called (criterion 3)
 *   - result signal is set after hydration, enabling chart effects
 *   - walkForwardError() is set on POST /validate/walk-forward error (criterion 4)
 *
 * Sibling specs handle these in full detail:
 *   backtesting-contracts.spec.ts       (walk-forward body parity + error DOM)
 *   backtesting-run-persistence.spec.ts (13a/13b em-dash KPIs and persistence)
 *   backtesting-render-coverage.spec.ts (chart stub injection + setOption assertions)
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';
import type { HttpTestingController } from '@angular/common/http/testing';

import { configureTestBed, injectHttp, installResizeObserverStub } from '../../testing';
import { ICON_PROVIDER } from '../icons';
import { BacktestingComponent } from './backtesting';
import { environment } from '../../environments/environment';

const WALK_FWD_URL = `${environment.apiUrl}validate/walk-forward`;
const BACKTEST_URL = `${environment.apiUrl}backtest`;

function runUrl(id: string): string {
  return `${environment.apiUrl}backtest/runs/${encodeURIComponent(id)}`;
}

function makeRun(id: string) {
  return {
    id,
    portfolioId: null,
    jobId: null,
    status: 'completed' as const,
    config: {},
    equityCurve: { '2024-01-01': 0.0, '2024-12-31': 0.12 },
    drawdowns: {},
    monthlyReturns: {},
    yearlyReturns: {},
    rollingMetrics: {},
    turnoverHistory: {},
    cvFoldMetrics: null,
    summaryStats: {},
    errorMessage: null,
    durationSeconds: 2,
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-12-31T00:00:00Z',
  };
}

describe('BacktestingComponent — walk-forward and pre-run state (issue #1024)', () => {
  let fixture: ComponentFixture<BacktestingComponent>;
  let comp: BacktestingComponent;
  let http: HttpTestingController;

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
    fixture.detectChanges();
  });

  afterEach(() => {
    http.match((r) => r.method === 'GET' && r.url.endsWith('/portfolio/'));
    http.verify();
    localStorage.clear();
  });

  // ── Pre-run state (criterion 13b) ─────────────────────────────────────────

  it('when no backtest has run, hasLoadedResult is false', () => {
    expect(comp.hasLoadedResult()).toBe(false);
  });

  it('when no backtest has run, walkForwardError is null', () => {
    expect(comp.walkForwardError()).toBeNull();
  });

  it('when no backtest has run, displayPct returns em-dash for totalReturn', () => {
    expect(comp.displayPct(comp.result().metrics.totalReturn)).toBe('—');
  });

  // ── onJobCompleted → GET /backtest/runs/{runId} (criterion 3) ─────────────

  it('when onJobCompleted is called with a runId, GET /backtest/runs/{runId} is issued', () => {
    comp.onJobCompleted('run-abc');
    const req = http.expectOne(runUrl('run-abc'));
    expect(req.request.method).toBe('GET');
    req.flush(makeRun('run-abc'));
  });

  it('when GET /backtest/runs returns a run, hasLoadedResult becomes true', () => {
    comp.onJobCompleted('run-abc');
    http.expectOne(runUrl('run-abc')).flush(makeRun('run-abc'));
    expect(comp.hasLoadedResult()).toBe(true);
  });

  it('when GET /backtest/runs returns a run, runResponse is set', () => {
    comp.onJobCompleted('run-abc');
    http.expectOne(runUrl('run-abc')).flush(makeRun('run-abc'));
    expect(comp.runResponse()).not.toBeNull();
  });

  it('when GET /backtest/runs returns a run, result signal equity is non-empty', () => {
    comp.onJobCompleted('run-abc');
    http.expectOne(runUrl('run-abc')).flush(makeRun('run-abc'));
    expect(comp.result().equity.length).toBeGreaterThan(0);
  });

  // ── Walk-forward error path (criterion 4) ────────────────────────────────

  it('when POST /validate/walk-forward returns 500, walkForwardError is set', () => {
    comp.onRunWalkForward();
    http
      .expectOne(WALK_FWD_URL)
      .flush({ detail: 'server error' }, { status: 500, statusText: 'Server Error' });
    expect(comp.walkForwardError()).not.toBeNull();
  });

  it('when POST /validate/walk-forward returns 422, walkForwardError is a non-blank string', () => {
    comp.onRunWalkForward();
    http
      .expectOne(WALK_FWD_URL)
      .flush({ detail: 'Unprocessable' }, { status: 422, statusText: 'Unprocessable Entity' });
    const err = comp.walkForwardError();
    expect(typeof err).toBe('string');
    expect((err ?? '').length).toBeGreaterThan(0);
  });

  // ── Walk-forward happy path ───────────────────────────────────────────────

  it('when POST /validate/walk-forward succeeds, walkForwardError remains null', () => {
    comp.onRunWalkForward();
    http
      .expectOne(WALK_FWD_URL)
      .flush({ jobId: 'wf-job-1', run_id: 'wf-run-1', status: 'pending' });
    expect(comp.walkForwardError()).toBeNull();
  });
});
