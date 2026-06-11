/**
 * Source-blind tests for the PortfolioContextService enrichment (issue #948).
 *
 * Derived from requirements.md §1 acceptance criteria:
 *   - PortfolioContextService exposes currentPortfolioName: Signal<string | null>
 *     resolved from the portfolio list.
 *   - PortfolioContextService exposes selectedPortfolio: Signal<PortfolioDto | null>
 *     holding the full Portfolio object.
 *   - After setPortfolio(id) and once the list has loaded, currentPortfolioName()
 *     returns the correct name within one render cycle; it returns null when there is
 *     no id, the id is not in the list, or the list load errors.
 *   - selectedPortfolio() returns the full DTO under the same conditions (else null).
 *   - localStorage persistence of currentPortfolioId is unchanged.
 *
 * Assumption: the service fetches the portfolio list via PortfolioApiService.list()
 * which calls GET /…/portfolio/ (or a URL containing "portfolio") once at construction.
 * Tests use a URL matcher (url.includes('portfolio')) so they are not brittle to the
 * exact base-URL prefix.
 *
 * Assumption: configureTestBed({ withHttp: true }) is the correct harness for a service
 * that makes an HTTP call at construction time. The list request is drained in each
 * beforeEach / afterEach according to the test's intended HTTP state.
 */

import { ApplicationRef } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';

import { configureTestBed } from '../../../testing';
import { PortfolioContextService } from './portfolio-context.service';

// ---------------------------------------------------------------------------
// Shared fixtures
// ---------------------------------------------------------------------------

const ISO = '2026-01-01T00:00:00.000Z';

const PORTFOLIO_ALPHA = {
  id: 'pf-alpha',
  name: 'Alpha Fund',
  description: null,
  currency: 'USD',
  benchmark_ticker: 'SPY',
  is_active: true,
  created_at: ISO,
  updated_at: ISO,
};

const PORTFOLIO_BETA = {
  id: 'pf-beta',
  name: 'Beta Portfolio',
  description: 'A beta portfolio',
  currency: 'EUR',
  benchmark_ticker: 'MSCI',
  is_active: true,
  created_at: ISO,
  updated_at: ISO,
};

const LIST_RESPONSE = { items: [PORTFOLIO_ALPHA, PORTFOLIO_BETA], total: 2 };

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Tick once to let the toObservable effect emit the current portfolioId signal
 * value (so switchMap fires the HTTP request), then flush the GET .../portfolio/
 * response, then tick again to propagate the result into the signal.
 */
function flushPortfolioList(http: HttpTestingController, appRef: ApplicationRef, payload = LIST_RESPONSE): void {
  appRef.tick();
  const req = http.expectOne((r) => r.url.includes('portfolio'));
  req.flush(payload);
  appRef.tick();
}

/** Same as flushPortfolioList but responds with HTTP 500. */
function flushPortfolioList500(http: HttpTestingController, appRef: ApplicationRef): void {
  appRef.tick();
  const req = http.expectOne((r) => r.url.includes('portfolio'));
  req.flush({}, { status: 500, statusText: 'Internal Server Error' });
  appRef.tick();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('PortfolioContextService – currentPortfolioName and selectedPortfolio signals', () => {
  let svc: PortfolioContextService;
  let http: HttpTestingController;
  let appRef: ApplicationRef;

  beforeEach(async () => {
    localStorage.clear();
    await configureTestBed({ withHttp: true });
    svc = TestBed.inject(PortfolioContextService);
    http = TestBed.inject(HttpTestingController);
    appRef = TestBed.inject(ApplicationRef);
  });

  afterEach(() => {
    localStorage.clear();
    http.verify();
  });

  // ── no id → both signals return null (no HTTP request fires) ─────────────

  it('when no portfolio id is set, currentPortfolioName returns null', () => {
    expect(svc.currentPortfolioName()).toBeNull();
  });

  it('when no portfolio id is set, selectedPortfolio returns null', () => {
    expect(svc.selectedPortfolio()).toBeNull();
  });

  // ── id present + list loaded → correct values ─────────────────────────────

  it('when portfolio id is set and the list has loaded, currentPortfolioName returns the matching name', () => {
    svc.setPortfolio(PORTFOLIO_ALPHA.id);
    flushPortfolioList(http, appRef);

    expect(svc.currentPortfolioName()).toBe(PORTFOLIO_ALPHA.name);
  });

  it('when portfolio id is set and the list has loaded, selectedPortfolio returns the full DTO', () => {
    svc.setPortfolio(PORTFOLIO_ALPHA.id);
    flushPortfolioList(http, appRef);

    expect(svc.selectedPortfolio()).toEqual(jasmine.objectContaining({
      id: PORTFOLIO_ALPHA.id,
      name: PORTFOLIO_ALPHA.name,
      currency: PORTFOLIO_ALPHA.currency,
    }));
  });

  // ── id not in list → both signals return null ─────────────────────────────

  it('when the selected id is not in the loaded list, currentPortfolioName returns null', () => {
    svc.setPortfolio('pf-does-not-exist');
    flushPortfolioList(http, appRef);

    expect(svc.currentPortfolioName()).toBeNull();
  });

  it('when the selected id is not in the loaded list, selectedPortfolio returns null', () => {
    svc.setPortfolio('pf-does-not-exist');
    flushPortfolioList(http, appRef);

    expect(svc.selectedPortfolio()).toBeNull();
  });

  // ── HTTP 500 on list → both signals return null ───────────────────────────

  it('when the portfolio list HTTP call returns 500, currentPortfolioName returns null', () => {
    svc.setPortfolio(PORTFOLIO_ALPHA.id);
    flushPortfolioList500(http, appRef);

    expect(svc.currentPortfolioName()).toBeNull();
  });

  it('when the portfolio list HTTP call returns 500, selectedPortfolio returns null', () => {
    svc.setPortfolio(PORTFOLIO_ALPHA.id);
    flushPortfolioList500(http, appRef);

    expect(svc.selectedPortfolio()).toBeNull();
  });

  // ── setPortfolio update propagates within one render cycle ────────────────

  it('when setPortfolio changes to another id that is in the list, currentPortfolioName updates to the new name', () => {
    svc.setPortfolio(PORTFOLIO_ALPHA.id);
    flushPortfolioList(http, appRef);
    expect(svc.currentPortfolioName()).toBe(PORTFOLIO_ALPHA.name);

    svc.setPortfolio(PORTFOLIO_BETA.id);
    appRef.tick();

    expect(svc.currentPortfolioName()).toBe(PORTFOLIO_BETA.name);
  });

  it('when setPortfolio changes to null, currentPortfolioName returns null', () => {
    svc.setPortfolio(PORTFOLIO_ALPHA.id);
    flushPortfolioList(http, appRef);
    expect(svc.currentPortfolioName()).toBe(PORTFOLIO_ALPHA.name);

    svc.setPortfolio(null);
    appRef.tick();

    expect(svc.currentPortfolioName()).toBeNull();
  });

  // ── localStorage persistence unchanged ───────────────────────────────────

  it('when portfolio id is set, it is still persisted to localStorage as before', () => {
    svc.setPortfolio(PORTFOLIO_ALPHA.id);
    flushPortfolioList(http, appRef);

    expect(localStorage.getItem('optimizer.currentPortfolioId')).toBe(PORTFOLIO_ALPHA.id);
  });
});
