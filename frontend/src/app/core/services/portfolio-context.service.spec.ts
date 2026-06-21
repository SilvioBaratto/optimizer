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

    expect(svc.activeMode()).toBe('live');
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

  // ── #1010: currentPortfolioName/selectedPortfolio signals and one-render-cycle resolution ──

  // --- Signal API surface ---
  // Criterion: PortfolioContextService exposes currentPortfolioName(): Signal<string|null>
  // and selectedPortfolio(): Signal<PortfolioDto|null>, alongside currentPortfolioId().

  it('when currentPortfolioId is accessed, it is a callable signal function', () => {
    expect(typeof svc.currentPortfolioId).toBe('function');
  });

  it('when currentPortfolioName is accessed, it is a callable signal function', () => {
    expect(typeof svc.currentPortfolioName).toBe('function');
  });

  it('when selectedPortfolio is accessed, it is a callable signal function', () => {
    expect(typeof svc.selectedPortfolio).toBe('function');
  });

  it('when currentPortfolioName() is called before any id is set, it returns null', () => {
    expect(svc.currentPortfolioName()).toBeNull();
  });

  it('when selectedPortfolio() is called before any id is set, it returns null', () => {
    expect(svc.selectedPortfolio()).toBeNull();
  });

  // --- Null-id handling ---
  // Criterion: currentPortfolioName() and selectedPortfolio() return null when the id is null.

  it('when setPortfolio(null) is called, currentPortfolioName() returns null', () => {
    svc.setPortfolio(null);
    appRef.tick();
    expect(svc.currentPortfolioName()).toBeNull();
  });

  it('when setPortfolio(null) is called, selectedPortfolio() returns null', () => {
    svc.setPortfolio(null);
    appRef.tick();
    expect(svc.selectedPortfolio()).toBeNull();
  });

  it('when setPortfolio(null) is called after a valid portfolio was active, currentPortfolioName() returns null', () => {
    svc.setPortfolio(ALPHA.id);
    appRef.tick();
    http.expectOne((r) => r.url.includes('portfolio')).flush(LIST_RESPONSE);
    appRef.tick();

    svc.setPortfolio(null);
    appRef.tick();

    expect(svc.currentPortfolioName()).toBeNull();
    expect(svc.selectedPortfolio()).toBeNull();
  });

  // --- Unknown-id handling ---
  // Criterion: currentPortfolioName() and selectedPortfolio() return null
  // when the id is not present in the list.
  //
  // Note: an empty list is used here. With a non-empty list the service's
  // auto-heal effect resets the stored id to the first item, making the
  // "id not in list → null" path unobservable (the id is no longer unknown
  // after the heal). The empty-list flush isolates the computed's contract
  // without depending on auto-heal timing.

  it('when the portfolio id is not found in the loaded list, currentPortfolioName() returns null', () => {
    svc.setPortfolio('id-that-does-not-exist');
    appRef.tick();
    http.expectOne((r) => r.url.includes('portfolio')).flush({ items: [], total: 0 });
    appRef.tick();

    expect(svc.currentPortfolioName()).toBeNull();
  });

  it('when the portfolio id is not found in the loaded list, selectedPortfolio() returns null', () => {
    svc.setPortfolio('id-that-does-not-exist');
    appRef.tick();
    http.expectOne((r) => r.url.includes('portfolio')).flush({ items: [], total: 0 });
    appRef.tick();

    expect(svc.selectedPortfolio()).toBeNull();
  });

  // --- One-render-cycle resolution ---
  // Criterion: A spec proves that after the list is loaded, calling setPortfolio(idA)
  // makes currentPortfolioName() return A's name within one appRef.tick().

  it('when list is already loaded and setPortfolio(idA) is called, currentPortfolioName() returns A name within one appRef.tick()', () => {
    // Step 1: load the list into the cache using a different id.
    svc.setPortfolio(BETA.id);
    appRef.tick();
    http.expectOne((r) => r.url.includes('portfolio')).flush(LIST_RESPONSE);
    appRef.tick();

    // Step 2: switch to ALPHA — list is cached, ONE tick must be enough.
    svc.setPortfolio(ALPHA.id);
    appRef.tick(); // exactly one render cycle
    expect(svc.currentPortfolioName()).toBe(ALPHA.name);
  });

  it('when list is already loaded and setPortfolio(idA) is called, selectedPortfolio() returns the full PortfolioDto within one appRef.tick()', () => {
    svc.setPortfolio(BETA.id);
    appRef.tick();
    http.expectOne((r) => r.url.includes('portfolio')).flush(LIST_RESPONSE);
    appRef.tick();

    svc.setPortfolio(ALPHA.id);
    appRef.tick(); // exactly one render cycle

    const portfolio = svc.selectedPortfolio() as typeof ALPHA | null;
    expect(portfolio).not.toBeNull();
    expect(portfolio!.id).toBe(ALPHA.id);
    expect(portfolio!.name).toBe(ALPHA.name);
  });

  it('when list is already loaded and setPortfolio(idB) is called, currentPortfolioName() returns B name within one appRef.tick()', () => {
    svc.setPortfolio(ALPHA.id);
    appRef.tick();
    http.expectOne((r) => r.url.includes('portfolio')).flush(LIST_RESPONSE);
    appRef.tick();

    svc.setPortfolio(BETA.id);
    appRef.tick(); // exactly one render cycle
    expect(svc.currentPortfolioName()).toBe(BETA.name);
  });

  it('when list is loaded and setPortfolio(idB) is called, selectedPortfolio() exposes the full PortfolioDto within one appRef.tick()', () => {
    svc.setPortfolio(ALPHA.id);
    appRef.tick();
    http.expectOne((r) => r.url.includes('portfolio')).flush(LIST_RESPONSE);
    appRef.tick();

    svc.setPortfolio(BETA.id);
    appRef.tick();

    const portfolio = svc.selectedPortfolio() as typeof BETA | null;
    expect(portfolio).not.toBeNull();
    expect(portfolio!.id).toBe(BETA.id);
    expect(portfolio!.name).toBe(BETA.name);
  });

  // --- Cached-list reuse: no second GET after switching ---
  // Criterion: A spec proves that switching setPortfolio(idA) → setPortfolio(idB)
  // (both in the cached list) fires no second GET /portfolio/.

  it('when switching from idA to idB after the list is cached, no second GET to the portfolio endpoint is fired', () => {
    // Load the list.
    svc.setPortfolio(ALPHA.id);
    appRef.tick();
    http.expectOne((r) => r.url.includes('portfolio')).flush(LIST_RESPONSE);
    appRef.tick();
    expect(svc.currentPortfolioName()).toBe(ALPHA.name);

    // Switch — must reuse the cached list.
    svc.setPortfolio(BETA.id);
    appRef.tick();

    http.expectNone(
      (r) => r.url.includes('portfolio'),
      // message shown on failure
    );
    expect(svc.currentPortfolioName()).toBe(BETA.name);
    expect((svc.selectedPortfolio() as typeof BETA).id).toBe(BETA.id);
  });

  it('when switching portfolios three times, only one portfolio list request is ever issued', () => {
    svc.setPortfolio(ALPHA.id);
    appRef.tick();
    http.expectOne((r) => r.url.includes('portfolio')).flush(LIST_RESPONSE);
    appRef.tick();

    svc.setPortfolio(BETA.id);
    appRef.tick();
    svc.setPortfolio(ALPHA.id);
    appRef.tick();

    // Exactly zero additional list requests after the first.
    http.expectNone((r) => r.url.includes('portfolio'));
    expect(svc.currentPortfolioName()).toBe(ALPHA.name);
  });

  it('when switching from a valid id to null, no second GET to the portfolio endpoint is made', () => {
    svc.setPortfolio(ALPHA.id);
    appRef.tick();
    http.expectOne((r) => r.url.includes('portfolio')).flush(LIST_RESPONSE);
    appRef.tick();

    svc.setPortfolio(null);
    appRef.tick();

    http.expectNone((r) => r.url.includes('portfolio'));
    expect(svc.currentPortfolioName()).toBeNull();
    expect(svc.selectedPortfolio()).toBeNull();
  });
});
