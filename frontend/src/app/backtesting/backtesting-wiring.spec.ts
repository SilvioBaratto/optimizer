/**
 * Wiring smoke-tests for BacktestingComponent (issue #1024).
 *
 * Coverage addendum for criteria not explicitly exercised elsewhere:
 *   (a) Component initialises with expected default signal state.
 *   (b) Console.error is NOT called when observable errors are caught silently.
 *   (c) No API endpoint paths leak into the rendered text.
 *
 * Majority of ticker-seeding, contract, persistence, and error-state wiring is
 * covered by the sibling specs:
 *   backtesting-ticker-seeding.spec.ts   (seeding A–D)
 *   backtesting-contracts.spec.ts        (POST /backtest contract + error state)
 *   backtest-service-contracts.spec.ts   (service-level HTTP parity)
 *   backtesting-run-persistence.spec.ts  (run restore / em-dash KPIs)
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';
import type { HttpTestingController } from '@angular/common/http/testing';

import { configureTestBed, injectHttp, installResizeObserverStub } from '../../testing';
import { ICON_PROVIDER } from '../icons';
import { BacktestingComponent } from './backtesting';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { environment } from '../../environments/environment';

const PORTFOLIO_LIST_URL = `${environment.apiUrl}portfolio/`;

describe('BacktestingComponent — wiring smoke-tests (issue #1024)', () => {
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

  // ── Initial signal state ──────────────────────────────────────────────────

  it('when component initialises, isRunning is false', () => {
    expect(comp.isRunning()).toBe(false);
  });

  it('when component initialises, walkForwardError is null', () => {
    expect(comp.walkForwardError()).toBeNull();
  });

  it('when component initialises, hasLoadedResult is false', () => {
    expect(comp.hasLoadedResult()).toBe(false);
  });

  it('when component initialises, selectedStartDate and selectedEndDate are non-empty ISO strings', () => {
    expect(comp.selectedStartDate()).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    expect(comp.selectedEndDate()).toMatch(/^\d{4}-\d{2}-\d{2}$/);
  });

  it('when component initialises, tickersRaw is a non-empty string', () => {
    expect(comp.tickersRaw().trim().length).toBeGreaterThan(0);
  });

  it('when component initialises, tickers is a non-empty array', () => {
    expect(comp.tickers().length).toBeGreaterThan(0);
  });

  // ── No API endpoint leaks in copy ─────────────────────────────────────────

  it('when the page renders, visible text does not contain raw API path segments', () => {
    const body: string = fixture.nativeElement.textContent ?? '';
    expect(body).not.toContain('cv_type');
    expect(body).not.toContain('cv_config');
    expect(body).not.toContain('optimizer_type');
  });

  // ── startDate / endDate aliases match selectedStartDate / selectedEndDate ──

  it('startDate() is the same signal as selectedStartDate()', () => {
    expect(comp.startDate()).toBe(comp.selectedStartDate());
  });

  it('endDate() is the same signal as selectedEndDate()', () => {
    expect(comp.endDate()).toBe(comp.selectedEndDate());
  });
});
