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
import type { BacktestResult, BacktestRunResponse } from './backtest.model';
import { environment } from '../../environments/environment';

const runUrl = (id: string) => `${environment.apiUrl}backtest/runs/${id}`;

function makeRun(id: string): BacktestRunResponse {
  return {
    id,
    portfolioId: null,
    jobId: null,
    status: 'completed',
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

function loadedResult(): BacktestResult {
  return {
    equity: [{ date: '2024-01-01', portfolio: 1, benchmark: 1 }],
    metrics: {
      totalReturn: 0.12, annualizedReturn: 0.05, annualizedVol: 0.15, sharpe: 0.9,
      sortino: 1.1, maxDrawdown: -0.18, calmar: 0.27, cvar95: -0.04,
      trackingError: 0.06, informationRatio: 0.4, winRate: 0.55, profitFactor: 1.3,
    },
    drawdowns: [], monthlyReturns: [], rollingMetrics: [],
    returnDistribution: [], factorLoadings: [],
  };
}

describe('BacktestingComponent — run persistence & pre-run KPIs (issue #996)', () => {
  let http: HttpTestingController;
  let ctx: PortfolioContextService;

  beforeEach(async () => {
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
    // Drain the PortfolioContextService list bootstrap (GET /portfolio/) that
    // fires from selecting a portfolio id, so verify() asserts only on the
    // run-restore requests each test explicitly exercises.
    http.match((r) => r.method === 'GET' && r.url.endsWith('/portfolio/'));
    http.verify();
    localStorage.clear();
  });

  // ── 13b: pre-run em-dash KPIs ────────────────────────────────────────────────

  it('when no result is loaded, Total Return KPI is rendered as an em-dash', () => {
    const comp = TestBed.createComponent(BacktestingComponent).componentInstance;
    expect(comp.displayPct(comp.result().metrics.totalReturn)).toBe('—');
  });

  it('when no result is loaded, Sharpe Ratio KPI is rendered as an em-dash', () => {
    const comp = TestBed.createComponent(BacktestingComponent).componentInstance;
    expect(comp.displayRatio(comp.result().metrics.sharpe)).toBe('—');
  });

  it('when a result is loaded, the Total Return KPI is the formatted value', () => {
    const comp = TestBed.createComponent(BacktestingComponent).componentInstance;
    comp.result.set(loadedResult());
    expect(comp.displayPct(0.12)).toBe(comp.formatPct(0.12));
  });

  // ── 13a: persistence ─────────────────────────────────────────────────────────

  it('when a completed job loads its run, the run id and portfolio are persisted', () => {
    ctx.currentPortfolioId.set('pf-9');
    const comp = TestBed.createComponent(BacktestingComponent).componentInstance;
    comp.onJobCompleted('run-7');
    http.expectOne(runUrl('run-7')).flush(makeRun('run-7'));
    expect(loadBacktestRun(localStorage)).toEqual({ runId: 'run-7', portfolioId: 'pf-9' });
  });

  // ── 13a: restore on construct ────────────────────────────────────────────────

  it('when a stored run matches the current portfolio, it is restored via GET /backtest/runs/{runId}', () => {
    ctx.currentPortfolioId.set('pf-1');
    localStorage.setItem(
      BACKTEST_RUN_STORAGE_KEY,
      JSON.stringify({ runId: 'run-1', portfolioId: 'pf-1' }),
    );
    const comp = TestBed.createComponent(BacktestingComponent).componentInstance;
    const req = http.expectOne(runUrl('run-1'));
    expect(req.request.method).toBe('GET');
    req.flush(makeRun('run-1'));
    expect(comp.hasLoadedResult()).toBe(true);
  });

  it('when a stored run belongs to a different portfolio, it is discarded and no run is fetched', () => {
    ctx.currentPortfolioId.set('pf-2');
    localStorage.setItem(
      BACKTEST_RUN_STORAGE_KEY,
      JSON.stringify({ runId: 'run-1', portfolioId: 'pf-1' }),
    );
    const comp = TestBed.createComponent(BacktestingComponent).componentInstance;
    http.expectNone(runUrl('run-1'));
    expect(comp.hasLoadedResult()).toBe(false);
    expect(loadBacktestRun(localStorage)).toBeNull();
  });
});
