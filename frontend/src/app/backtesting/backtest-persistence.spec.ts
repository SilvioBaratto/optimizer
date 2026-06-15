/**
 * Run-persistence addendum for BacktestingComponent (issue #1024).
 *
 * The canonical persistence spec is backtesting-run-persistence.spec.ts (issue #996).
 * This file adds the `loadBacktestRun(localStorage)` API contract assertion missed
 * by the main spec, and a quick guard that the storage-key string matches the
 * module export.
 *
 * Full restore / discard / em-dash scenarios are in backtesting-run-persistence.spec.ts.
 */

import { TestBed } from '@angular/core/testing';
import type { HttpTestingController } from '@angular/common/http/testing';

import { configureTestBed, injectHttp, installResizeObserverStub } from '../../testing';
import { ICON_PROVIDER } from '../icons';
import { BacktestingComponent } from './backtesting';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import {
  BACKTEST_RUN_STORAGE_KEY,
  loadBacktestRun,
} from './backtesting-run-storage';
import { environment } from '../../environments/environment';

const runUrl = (id: string) => `${environment.apiUrl}backtest/runs/${encodeURIComponent(id)}`;

function makeRun(id: string) {
  return {
    id,
    portfolioId: null,
    jobId: null,
    status: 'completed' as const,
    config: {},
    equityCurve: { '2024-01-01': 0, '2024-12-31': 0.12 },
    drawdowns: {},
    monthlyReturns: {},
    yearlyReturns: {},
    rollingMetrics: {},
    turnoverHistory: {},
    cvFoldMetrics: null,
    summaryStats: {},
    errorMessage: null,
    durationSeconds: null,
    createdAt: '2024-01-01',
    updatedAt: '2024-12-31',
  };
}

describe('BacktestingComponent — run-persistence addendum (issue #1024)', () => {
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
    http = injectHttp();
    ctx = TestBed.inject(PortfolioContextService);
  });

  afterEach(() => {
    http.match((r) => r.method === 'GET' && r.url.endsWith('/portfolio/'));
    http.verify();
    localStorage.clear();
  });

  // ── Storage key contract ──────────────────────────────────────────────────

  it('BACKTEST_RUN_STORAGE_KEY is the namespaced string used by the service', () => {
    expect(BACKTEST_RUN_STORAGE_KEY).toBe('optimizer.lastBacktestRun');
  });

  it('loadBacktestRun returns null when localStorage is empty', () => {
    expect(loadBacktestRun(localStorage)).toBeNull();
  });

  // ── Persistence write ─────────────────────────────────────────────────────

  it('when onJobCompleted fires, run id and portfolioId are written to localStorage', () => {
    ctx.currentPortfolioId.set('pf-9');
    const comp = TestBed.createComponent(BacktestingComponent).componentInstance;
    comp.onJobCompleted('run-7');
    http.expectOne(runUrl('run-7')).flush(makeRun('run-7'));
    expect(loadBacktestRun(localStorage)).toEqual({ runId: 'run-7', portfolioId: 'pf-9' });
  });

  // ── Restore with matching portfolioId ────────────────────────────────────

  it('when stored run matches current portfolio, GET /backtest/runs/{runId} is issued on construct', () => {
    ctx.currentPortfolioId.set('pf-1');
    localStorage.setItem(
      BACKTEST_RUN_STORAGE_KEY,
      JSON.stringify({ runId: 'run-1', portfolioId: 'pf-1' }),
    );
    TestBed.createComponent(BacktestingComponent);
    const req = http.expectOne(runUrl('run-1'));
    expect(req.request.method).toBe('GET');
    req.flush(makeRun('run-1'));
  });

  // ── Discard with mismatched portfolioId ──────────────────────────────────

  it('when stored run belongs to a different portfolio, no GET /backtest/runs is issued', () => {
    ctx.currentPortfolioId.set('pf-current');
    localStorage.setItem(
      BACKTEST_RUN_STORAGE_KEY,
      JSON.stringify({ runId: 'run-old', portfolioId: 'pf-other' }),
    );
    const comp = TestBed.createComponent(BacktestingComponent).componentInstance;
    http.expectNone(runUrl('run-old'));
    expect(comp.hasLoadedResult()).toBe(false);
    expect(loadBacktestRun(localStorage)).toBeNull();
  });
});
