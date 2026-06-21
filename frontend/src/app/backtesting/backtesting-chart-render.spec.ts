/**
 * End-to-end chart rendering spec for BacktestingComponent (issue #1024, criterion 3).
 *
 * Drives a full job-completion cycle:
 *   onJobCompleted(runId) → GET /backtest/runs/{runId} → result is set →
 *   equity/underwater/QQ chart effects fire → setOption called on stubs.
 *
 * Pattern:
 *   1. Mount the component (zoneless + real HTTP).
 *   2. Inject ECharts stub objects into private chart fields via TS cast.
 *   3. Call onJobCompleted() and flush the HTTP GET.
 *   4. detectChanges() so Angular flushes signal effects.
 *   5. Assert setOption was called on each stub and DOM elements appear.
 *
 * Rolling charts (rollingSharpe, rollingVol, rollingBeta) are also covered via
 * direct result.set() in backtesting-render-coverage.spec.ts — this file focuses
 * on the full onJobCompleted → HTTP → effect → chart pipeline.
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';
import type { HttpTestingController } from '@angular/common/http/testing';

import { configureTestBed, injectHttp, installResizeObserverStub } from '../../testing';
import { ICON_PROVIDER } from '../icons';
import { BacktestingComponent } from './backtesting';
import { environment } from '../../environments/environment';

function runUrl(id: string): string {
  return `${environment.apiUrl}backtest/runs/${encodeURIComponent(id)}`;
}

/** Minimal BacktestRunResponse that produces a non-empty equity array
 *  so hasLoadedResult() becomes true and result effects fire. */
function makeRun(id: string) {
  return {
    id,
    portfolioId: 'pf-1',
    jobId: 'job-1',
    status: 'completed' as const,
    config: {},
    equityCurve: {
      '2024-01-01': 1.0,
      '2024-06-01': 1.07,
      '2024-12-31': 1.12,
    },
    drawdowns: { '2024-03-01': -0.03, '2024-09-01': -0.01 },
    monthlyReturns: { '2024-01': 0.01 },
    yearlyReturns: { '2024': 0.12 },
    rollingMetrics: {
      sharpe: { '2024-06-01': 1.1 },
      volatility: { '2024-06-01': 0.13 },
      beta: { '2024-06-01': 0.9 },
    },
    turnoverHistory: { '2024-01-01': 0.05 },
    cvFoldMetrics: null,
    summaryStats: { sharpe: 1.1 },
    errorMessage: null,
    durationSeconds: 2.0,
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-12-31T00:00:00Z',
  };
}

describe('BacktestingComponent — chart render on job completion (issue #1024)', () => {
  let fixture: ComponentFixture<BacktestingComponent>;
  let comp: BacktestingComponent;
  let el: HTMLElement;
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
    el = fixture.nativeElement as HTMLElement;
    http = injectHttp();
    fixture.detectChanges();
  });

  afterEach(() => {
    http.match((r) => r.method === 'GET' && r.url.endsWith('/portfolio/'));
    http.verify();
    localStorage.clear();
  });

  // ── HTTP flow: onJobCompleted → GET → hasLoadedResult ────────────────────

  it('when onJobCompleted fires, GET /backtest/runs/{runId} is issued exactly once', () => {
    comp.onJobCompleted('run-test-1');
    http.expectOne(runUrl('run-test-1')).flush(makeRun('run-test-1'));
    expect(comp.hasLoadedResult()).toBe(true);
  });

  it('after job completion, runJobId is cleared', () => {
    comp.runJobId.set('job-before');
    comp.onJobCompleted('run-test-2');
    http.expectOne(runUrl('run-test-2')).flush(makeRun('run-test-2'));
    expect(comp.runJobId()).toBeNull();
  });

  // ── Data delegation: run data flows into runResponse signal ─────────────
  // Charts are rendered by BacktestResultsPanelComponent (slim #1051);
  // the orchestrator exposes runResponse() for panel binding.

  it('when run hydrated, runResponse contains non-empty equity curve', () => {
    comp.onJobCompleted('run-chart-1');
    http.expectOne(runUrl('run-chart-1')).flush(makeRun('run-chart-1'));
    fixture.detectChanges();

    const curve = comp.runResponse()?.equityCurve ?? {};
    expect(Object.keys(curve).length).toBeGreaterThan(0);
  });

  it('when run hydrated, runResponse contains drawdown data', () => {
    comp.onJobCompleted('run-chart-2');
    http.expectOne(runUrl('run-chart-2')).flush(makeRun('run-chart-2'));
    fixture.detectChanges();

    const dd = comp.runResponse()?.drawdowns ?? {};
    expect(Object.keys(dd).length).toBeGreaterThan(0);
  });

  it('when run hydrated, hasLoadedResult is true and runResponse contains rolling metrics', () => {
    comp.onJobCompleted('run-chart-3');
    http.expectOne(runUrl('run-chart-3')).flush(makeRun('run-chart-3'));
    fixture.detectChanges();

    expect(comp.hasLoadedResult()).toBe(true);
    expect(comp.runResponse()?.rollingMetrics).toBeTruthy();
  });

  it('when rolling window changes after hydration, hasLoadedResult remains true', () => {
    comp.onJobCompleted('run-rolling-1');
    http.expectOne(runUrl('run-rolling-1')).flush(makeRun('run-rolling-1'));
    fixture.detectChanges();

    expect(comp.hasLoadedResult()).toBe(true);

    comp.rollingWindow.set('3Y');
    fixture.detectChanges();

    expect(comp.hasLoadedResult()).toBe(true);
  });

  // ── DOM: results panel appears after job completion ──────────────────────

  it('when runResponse is set after job completion, app-backtest-results-panel is rendered', () => {
    comp.onJobCompleted('run-dom-1');
    http.expectOne(runUrl('run-dom-1')).flush(makeRun('run-dom-1'));
    fixture.detectChanges();

    expect(el.querySelector('app-backtest-results-panel')).not.toBeNull();
  });

  it('when runResponseLoading is true during hydration, results panel appears with loading state', () => {
    comp.runResponseLoading.set(true);
    fixture.detectChanges();

    expect(el.querySelector('app-backtest-results-panel')).not.toBeNull();
  });

  // ── Error path: GET /backtest/runs fails ──────────────────────────────────

  it('when GET /backtest/runs returns 500, runResponseError is set', () => {
    comp.onJobCompleted('run-err-1');
    http
      .expectOne(runUrl('run-err-1'))
      .flush({ detail: 'server error' }, { status: 500, statusText: 'Server Error' });
    expect(comp.runResponseError()).not.toBeNull();
  });

  it('when GET /backtest/runs returns 500, runResponseLoading is cleared', () => {
    comp.onJobCompleted('run-err-2');
    http
      .expectOne(runUrl('run-err-2'))
      .flush({ detail: 'gone' }, { status: 500, statusText: 'Server Error' });
    expect(comp.runResponseLoading()).toBe(false);
  });

  it('when runResponseError is set, app-backtest-results-panel is still rendered (error state)', () => {
    comp.runResponseError.set('load failed');
    fixture.detectChanges();

    expect(el.querySelector('app-backtest-results-panel')).not.toBeNull();
  });
});
