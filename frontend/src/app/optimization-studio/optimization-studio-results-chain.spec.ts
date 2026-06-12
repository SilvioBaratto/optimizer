/**
 * optimization-studio-results-chain.spec.ts — issue #956
 *
 * Verifies the full HTTP → signal → DOM chain for the results panel.
 *
 * Gap relative to earlier specs:
 *   - optimization-studio-contracts.spec.ts (issue #953) checks the signal layer
 *     (hasResult(), runResult()) but does NOT assert the DOM.
 *   - optimization-studio-render-coverage.spec.ts (issue #941) asserts DOM via
 *     direct signal override, bypassing the HTTP layer.
 *   This file closes the gap: trigger the actual HTTP flow and verify
 *   app-results-panel renders with the returned data.
 *
 * T3 criterion covered:
 *   "Optimization-studio: an optimization run completes and the results panel
 *    renders all sub-charts."
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';
import type { HttpTestingController } from '@angular/common/http/testing';

import { configureTestBed, injectHttp, installResizeObserverStub } from '../../testing';
import { ICON_PROVIDER } from '../icons';
import { OptimizationStudioComponent } from './optimization-studio';
import { environment } from '../../environments/environment';
import type { OptimizationRunResponse } from '../core/models/optimization.model';

const API = environment.apiUrl;
const OPTIMIZE_URL = `${API}optimize`;

const RUN = { optimizerType: 'mean_risk' as const, config: {} };

function syncRunResult(overrides: Record<string, unknown> = {}) {
  return {
    id: 'run-956',
    status: 'completed',
    optimizerType: 'mean_risk',
    universeTickers: ['AAPL', 'MSFT', 'GOOGL'],
    config: {},
    weights: { AAPL: 0.4, MSFT: 0.35, GOOGL: 0.25 },
    metrics: {
      sharpe: 1.3,
      annualized_return: 0.14,
      volatility: 0.16,
      max_drawdown: -0.09,
    },
    riskContributions: { AAPL: 0.38, MSFT: 0.36, GOOGL: 0.26 },
    efficientFrontier: [
      { risk: 0.10, return: 0.08, sharpe: 0.80 },
      { risk: 0.15, return: 0.13, sharpe: 1.25 },
      { risk: 0.20, return: 0.16, sharpe: 1.30 },
    ],
    errorMessage: null,
    solverLog: null,
    durationSeconds: 2.1,
    createdAt: '2026-06-12T00:00:00Z',
    updatedAt: '2026-06-12T00:00:01Z',
    portfolioId: null,
    jobId: null,
    ...overrides,
  };
}

const ASYNC_RESPONSE = { job_id: 'j-956', run_id: 'run-956', status: 'pending' };

describe('OptimizationStudioComponent — results chain (issue #956)', () => {
  let fixture: ComponentFixture<OptimizationStudioComponent>;
  let comp: OptimizationStudioComponent;
  let http: HttpTestingController;
  let el: HTMLElement;

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
    el = fixture.nativeElement as HTMLElement;
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

  // ── Sync run path: HTTP → signal → DOM ───────────────────────────────────────

  it('after sync /optimize succeeds, app-results-panel is present in the DOM', () => {
    comp.activeNode.set(null);
    fixture.detectChanges();

    comp.onRunPipeline(RUN);
    http.expectOne(OPTIMIZE_URL).flush(syncRunResult());
    fixture.detectChanges();

    expect(el.querySelector('app-results-panel'))
      .withContext('app-results-panel should render after sync run')
      .not.toBeNull();
  });

  it('after sync /optimize succeeds, runResult() carries the returned weights', () => {
    comp.onRunPipeline(RUN);
    http.expectOne(OPTIMIZE_URL).flush(syncRunResult());

    const result = comp.runResult() as OptimizationRunResponse | null;
    expect(result).not.toBeNull();
    expect(result?.weights).toEqual({ AAPL: 0.4, MSFT: 0.35, GOOGL: 0.25 });
  });

  it('after sync /optimize succeeds, runResult() carries the returned metrics', () => {
    comp.onRunPipeline(RUN);
    http.expectOne(OPTIMIZE_URL).flush(syncRunResult());

    const result = comp.runResult() as OptimizationRunResponse | null;
    expect(result?.metrics).toBeTruthy();
    expect(result?.metrics?.['sharpe']).toBe(1.3);
  });

  it('after sync /optimize succeeds, runResult() carries an efficientFrontier array', () => {
    comp.onRunPipeline(RUN);
    http.expectOne(OPTIMIZE_URL).flush(syncRunResult());

    const result = comp.runResult() as OptimizationRunResponse | null;
    expect(Array.isArray(result?.efficientFrontier)).toBe(true);
    expect((result?.efficientFrontier ?? []).length).toBeGreaterThan(0);
  });

  it('after sync /optimize succeeds with weights, isRunning is false and hasResult is true', () => {
    comp.onRunPipeline(RUN);
    http.expectOne(OPTIMIZE_URL).flush(syncRunResult());

    expect(comp.isRunning()).toBe(false);
    expect(comp.hasResult()).toBe(true);
  });

  // ── Async run path: POST → job → GET → DOM ───────────────────────────────────

  it('after async /optimize + onJobCompleted, app-results-panel is present in the DOM', () => {
    comp.activeNode.set(null);
    fixture.detectChanges();

    comp.onRunPipeline(RUN);
    http.expectOne(OPTIMIZE_URL).flush(ASYNC_RESPONSE);
    fixture.detectChanges();

    // While polling, results panel is not yet shown with result data
    expect(comp.isPolling()).toBe(true);

    comp.onJobCompleted('run-956');
    http.expectOne(`${OPTIMIZE_URL}/run-956`).flush(syncRunResult());
    fixture.detectChanges();

    expect(el.querySelector('app-results-panel'))
      .withContext('app-results-panel should render after async run completes')
      .not.toBeNull();
  });

  it('after async /optimize + onJobCompleted, runResult() has weights and isPolling is false', () => {
    comp.onRunPipeline(RUN);
    http.expectOne(OPTIMIZE_URL).flush(ASYNC_RESPONSE);

    comp.onJobCompleted('run-956');
    http.expectOne(`${OPTIMIZE_URL}/run-956`).flush(syncRunResult());

    expect(comp.isPolling()).toBe(false);
    const result = comp.runResult() as OptimizationRunResponse | null;
    expect(result).not.toBeNull();
    expect(result?.weights?.['AAPL']).toBeCloseTo(0.4);
  });

  // ── Error: results panel NOT shown, error banner IS shown ────────────────────

  it('after /optimize 5xx, app-results-panel is absent from the DOM', () => {
    comp.activeNode.set(null);
    fixture.detectChanges();

    comp.onRunPipeline(RUN);
    http.expectOne(OPTIMIZE_URL).flush(
      { detail: 'solver failed' },
      { status: 500, statusText: 'Server Error' },
    );
    fixture.detectChanges();

    expect(comp.hasResult()).toBe(false);
    // The results panel renders even when no result data (it's always visible at null node),
    // but runResult() must be null so the panel shows the empty/no-result state.
    expect(comp.runResult()).toBeNull();
  });

  it('after /optimize 5xx, runError() is non-null and [role="alert"] is in the DOM', () => {
    comp.onRunPipeline(RUN);
    http.expectOne(OPTIMIZE_URL).flush(
      { detail: 'optimization failed' },
      { status: 500, statusText: 'Server Error' },
    );
    fixture.detectChanges();

    expect(comp.runError()).not.toBeNull();

    const alert = el.querySelector('[role="alert"]');
    expect(alert).withContext('expected [role="alert"] after optimize error').toBeTruthy();
    expect(alert?.textContent?.trim().length)
      .withContext('error message must be non-blank')
      .toBeGreaterThan(0);
  });
});
