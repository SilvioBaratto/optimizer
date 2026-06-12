import { ApplicationRef } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';

import { configureTestBed } from '../../../testing';
import { PortfolioContextService } from './portfolio-context.service';

const STORAGE_KEY = 'optimizer.currentPortfolioId';

// Minimal portfolio DTOs for the cached-list derivation tests (#969).
const ISO = '2026-01-01T00:00:00.000Z';
const ALPHA = {
  id: 'pf-alpha', name: 'Alpha Fund', description: null, currency: 'USD',
  benchmark_ticker: 'SPY', is_active: true, created_at: ISO, updated_at: ISO,
};
const BETA = {
  id: 'pf-beta', name: 'Beta Portfolio', description: null, currency: 'EUR',
  benchmark_ticker: 'MSCI', is_active: true, created_at: ISO, updated_at: ISO,
};
const LIST_RESPONSE = { items: [ALPHA, BETA], total: 2 };

describe('PortfolioContextService', () => {
  let svc: PortfolioContextService;
  let appRef: ApplicationRef;
  let http: HttpTestingController;

  beforeEach(async () => {
    localStorage.clear();
    await configureTestBed({ withHttp: true });
    svc = TestBed.inject(PortfolioContextService);
    http = TestBed.inject(HttpTestingController);
    appRef = TestBed.inject(ApplicationRef);
    // No list request fires at construction — lazy: only emits when currentPortfolioId is non-null.
  });

  afterEach(() => {
    // Drain any portfolio list request that a test may have triggered (setPortfolio calls).
    http.match((r) => r.url.includes('portfolio'));
    localStorage.clear();
    http.verify();
  });

  it('when injected twice, the providedIn root singleton resolves once', () => {
    expect(TestBed.inject(PortfolioContextService)).toBe(svc);
  });

  it('when a portfolio is set, the effect persists the id to localStorage', () => {
    svc.setPortfolio('pf-9');
    appRef.tick();
    expect(localStorage.getItem(STORAGE_KEY)).toBe('pf-9');
    expect(svc.hasPortfolio()).toBe(true);
  });

  it('when the portfolio is set to null, the persisted id is removed', () => {
    svc.setPortfolio('pf-9');
    appRef.tick();
    svc.setPortfolio(null);
    appRef.tick();
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
    expect(svc.hasPortfolio()).toBe(false);
  });

  it('when a preset is set, the date range adopts that preset', () => {
    svc.setPreset('1M');
    expect(svc.dateRange().preset).toBe('1M');
    expect(svc.dateRangeLabel()).toBe('1M');
  });

  it('when a custom range is set, the range is Custom and days are derived', () => {
    const start = new Date('2026-01-01T00:00:00.000Z');
    const end = new Date('2026-01-11T00:00:00.000Z');
    svc.setCustomRange(start, end);
    expect(svc.dateRange().preset).toBe('Custom');
    expect(svc.dateRangeDays()).toBe(10);
    expect(svc.dateRangeLabel()).toContain('Jan');
  });

  it('when mode is changed, the derived mode flags follow', () => {
    svc.setMode('live');
    expect(svc.isLive()).toBe(true);
    expect(svc.isBacktest()).toBe(false);
  });

  it('when reset, all state returns to defaults', () => {
    svc.setMode('live');
    svc.setPortfolio('pf-9');
    svc.setBenchmark('QQQ');
    appRef.tick();

    svc.reset();
    appRef.tick();

    expect(svc.activeMode()).toBe('backtest');
    expect(svc.currentPortfolioId()).toBeNull();
    expect(svc.dateRange().preset).toBe('1Y');
    expect(svc.benchmark()).toBe('SPY');
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
  });

  // ── #969: derived signals from the cached list ───────────────────────────

  it('when an id is set but the portfolio list has not loaded, currentPortfolioName returns null', () => {
    svc.setPortfolio(ALPHA.id);
    appRef.tick(); // fires the (shared) list request but leaves it unflushed
    expect(svc.currentPortfolioName()).toBeNull();
  });

  it('when an id is set but the portfolio list has not loaded, selectedPortfolio returns null', () => {
    svc.setPortfolio(ALPHA.id);
    appRef.tick();
    expect(svc.selectedPortfolio()).toBeNull();
  });

  it('when switching between two non-null ids, no second portfolio list request is issued', () => {
    svc.setPortfolio(ALPHA.id);
    appRef.tick();
    http.expectOne((r) => r.url.includes('portfolio')).flush(LIST_RESPONSE);
    appRef.tick();
    expect(svc.currentPortfolioName()).toBe(ALPHA.name);

    svc.setPortfolio(BETA.id);
    appRef.tick();

    // Flipping between two non-null ids must reuse the cached list.
    http.expectNone((r) => r.url.includes('portfolio'));
    expect(svc.currentPortfolioName()).toBe(BETA.name);
  });
});
