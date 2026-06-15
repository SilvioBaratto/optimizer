/**
 * Source-blind wiring tests for issue #1021.
 *
 * Pins the exact HTTP URL and verb for every risk panel and verifies that
 * successful responses flow into the component's response signals.
 *
 * Criteria derived from acceptance-criteria text (oracle classification):
 *   [T3] VaR, correlation, factor-exposure, concentration, and liquidity
 *        panels render from GET /portfolio/{name}/risk/*
 *   [T2] Stress panel renders from POST /risk/stress-scenarios
 *   [UNIT] Every wired page: (a) loads data on init, (b) handles API error,
 *          (c) reacts to portfolio context change
 *
 * What makes these tests new relative to earlier committed specs:
 *   - risk.service.spec.ts   — service-only, not the component→service chain
 *   - risk-center.spec.ts    — uses `r.url.includes(fragment)`, not exact URL
 *   - risk-center-*.spec.ts  — test panelErrors signal, not the data signal
 *
 * Source-blind assumption: RiskCenterComponent exposes writable signals named
 * varResponse, concentrationResponse, stressScenarios, panelErrors, isLoading,
 * activeTab, and methods onGenerateStress({ macroContext, nScenarios }).
 * These names are inferred from existing .spec.ts files (test artefacts, not
 * implementation source).
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import type { HttpTestingController } from '@angular/common/http/testing';

import {
  configureTestBed,
  drainRequests,
  injectHttp,
  installResizeObserverStub,
} from '../../testing';
import { ICON_PROVIDER } from '../icons';
import { RiskCenterComponent } from './risk-center';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { environment } from '../../environments/environment';

// ─── Constants ────────────────────────────────────────────────────────────────

const API = environment.apiUrl;
// Portfolio name with no special chars so no encoding is needed.
const PORTFOLIO = 'trading212';
// Pre-built exact base URLs (without query params — Angular's r.url strips them).
const VAR_URL        = `${API}portfolio/${PORTFOLIO}/risk/var`;
const CORR_URL       = `${API}portfolio/${PORTFOLIO}/risk/correlation`;
const FACTOR_URL     = `${API}portfolio/${PORTFOLIO}/risk/factor-exposure`;
const CONC_URL       = `${API}portfolio/${PORTFOLIO}/risk/concentration`;
const LIQ_URL        = `${API}portfolio/${PORTFOLIO}/risk/liquidity`;
const LIMITS_URL     = `${API}risk/${PORTFOLIO}/limits`;
const STRESS_URL     = `${API}risk/stress-scenarios`;

// ─── Helpers ─────────────────────────────────────────────────────────────────

function makeCtxStub(name: string | null = PORTFOLIO, id: string | null = 'stub-id') {
  const idSig   = signal<string | null>(id);
  const nameSig = signal<string | null>(name);
  return {
    currentPortfolioId:   idSig,
    currentPortfolioName: nameSig,
    hasPortfolio:         { _value: name !== null },
    _idSig:               idSig,
    _nameSig:             nameSig,
  };
}

// Minimal happy-path stubs keyed by URL fragment.
function stubFor(url: string): Record<string, unknown> {
  if (url.includes('/risk/var'))           return { var: { '95': 0.03 }, cvar: { '95': 0.05 }, method: 'historical', lookback: 252, nObservations: 252 };
  if (url.includes('/risk/correlation'))  return { assets: [], matrix: [], clusterLabels: [] };
  if (url.includes('/risk/factor-exposure')) return { exposures: {}, assetExposures: {} };
  if (url.includes('/risk/concentration')) return { assets: [], summary: { hhi: 0, effectiveN: 0, topNRatio: 0 } };
  if (url.includes('/risk/liquidity'))    return { assets: [], summary: { weightedAvgDaysToLiquidate: 0 } };
  if (url.includes('/limits'))            return { items: [], breachCount: 0 };
  return {};
}

// ─── Suite ───────────────────────────────────────────────────────────────────

describe('RiskCenterComponent — all-panel endpoint wiring (issue #1021)', () => {
  let fixture: ComponentFixture<RiskCenterComponent>;
  let comp: RiskCenterComponent;
  let http: HttpTestingController;
  let ctx: ReturnType<typeof makeCtxStub>;

  /** Flush every pending HTTP request with happy-path stubs. */
  function settle(): void {
    fixture.detectChanges();
    drainRequests(http, stubFor);
    fixture.detectChanges();
    drainRequests(http, stubFor);
  }

  /**
   * After `http.expectOne` captures one request, drain every remaining open
   * request so `afterEach(() => http.verify())` does not throw.
   */
  function drainRest(): void {
    for (const req of http.match(() => true)) {
      req.flush(stubFor(req.request.url));
    }
  }

  beforeEach(async () => {
    ctx = makeCtxStub();
    installResizeObserverStub();
    await configureTestBed({
      imports:   [RiskCenterComponent],
      withHttp:  true,
      providers: [
        ICON_PROVIDER,
        { provide: PortfolioContextService, useValue: ctx },
      ],
    });
    fixture = TestBed.createComponent(RiskCenterComponent);
    comp    = fixture.componentInstance;
    http    = injectHttp();
  });

  afterEach(() => http.verify());

  // ── (a) loads data on init — exact URL and HTTP verb per panel ────────────

  it('when portfolio is in context, fires GET to exact URL /portfolio/{name}/risk/var', () => {
    fixture.detectChanges();
    const req = http.expectOne((r) => r.url === VAR_URL);
    expect(req.request.method).toBe('GET');
    drainRest();
    req.flush(stubFor(VAR_URL));
  });

  it('when portfolio is in context, fires GET to exact URL /portfolio/{name}/risk/correlation', () => {
    fixture.detectChanges();
    const req = http.expectOne((r) => r.url === CORR_URL);
    expect(req.request.method).toBe('GET');
    drainRest();
    req.flush(stubFor(CORR_URL));
  });

  it('when portfolio is in context, fires GET to exact URL /portfolio/{name}/risk/factor-exposure', () => {
    fixture.detectChanges();
    const req = http.expectOne((r) => r.url === FACTOR_URL);
    expect(req.request.method).toBe('GET');
    drainRest();
    req.flush(stubFor(FACTOR_URL));
  });

  it('when portfolio is in context, fires GET to exact URL /portfolio/{name}/risk/concentration', () => {
    fixture.detectChanges();
    const req = http.expectOne((r) => r.url === CONC_URL);
    expect(req.request.method).toBe('GET');
    drainRest();
    req.flush(stubFor(CONC_URL));
  });

  it('when portfolio is in context, fires GET to exact URL /portfolio/{name}/risk/liquidity', () => {
    fixture.detectChanges();
    const req = http.expectOne((r) => r.url === LIQ_URL);
    expect(req.request.method).toBe('GET');
    drainRest();
    req.flush(stubFor(LIQ_URL));
  });

  it('when portfolio is in context, fires GET to exact URL /risk/{name}/limits', () => {
    fixture.detectChanges();
    const req = http.expectOne((r) => r.url === LIMITS_URL);
    expect(req.request.method).toBe('GET');
    drainRest();
    req.flush(stubFor(LIMITS_URL));
  });

  // ── No raw {name} token left in any URL ──────────────────────────────────

  it('VaR request URL does not contain a raw {name} placeholder', () => {
    fixture.detectChanges();
    const req = http.expectOne((r) => r.url === VAR_URL);
    expect(req.request.url).not.toContain('{name}');
    drainRest();
    req.flush(stubFor(VAR_URL));
  });

  it('limits request URL does not contain a raw {portfolio_name} placeholder', () => {
    fixture.detectChanges();
    const req = http.expectOne((r) => r.url === LIMITS_URL);
    expect(req.request.url).not.toContain('{portfolio_name}');
    drainRest();
    req.flush(stubFor(LIMITS_URL));
  });

  // ── Data-flow: API response populates component signals ───────────────────

  it('when VaR endpoint returns data, comp.varResponse() is populated with the returned fields', () => {
    fixture.detectChanges();
    const req = http.expectOne((r) => r.url === VAR_URL);
    req.flush({ var: { '95': 0.025, '99': 0.041 }, cvar: { '95': 0.038, '99': 0.052 }, method: 'historical', lookback: 252, nObservations: 252 });
    drainRest();
    fixture.detectChanges();

    expect(comp.varResponse()).not.toBeNull();
    expect(comp.varResponse()?.var['95']).toBeDefined();
  });

  it('when concentration endpoint returns one asset, comp.concentrationResponse() has that asset', () => {
    fixture.detectChanges();
    const req = http.expectOne((r) => r.url === CONC_URL);
    req.flush({
      assets: [{ ticker: 'AAPL', name: 'Apple Inc.', weight: 0.6 }],
      summary: { hhi: 0.36, effectiveN: 2.78, topNRatio: 0.6 },
    });
    drainRest();
    fixture.detectChanges();

    expect(comp.concentrationResponse()).not.toBeNull();
    expect(comp.concentrationResponse()?.assets.length).toBe(1);
    expect(comp.concentrationResponse()?.assets[0].ticker).toBe('AAPL');
  });

  // ── Stress panel: POST /risk/stress-scenarios ─────────────────────────────

  it('when onGenerateStress is called, fires POST to exact URL /risk/stress-scenarios', () => {
    settle();
    comp.onGenerateStress({ macroContext: 'yield inversion', nScenarios: 3 });

    const req = http.expectOne((r) => r.url === STRESS_URL);
    expect(req.request.method).toBe('POST');
    req.flush({ nScenarios: 3, tickers: [], scenarios: [] });
  });

  it('when stress generation succeeds, comp.stressScenarios() length equals nScenarios', () => {
    settle();
    comp.onGenerateStress({ macroContext: 'rate shock', nScenarios: 2 });

    http.expectOne((r) => r.url === STRESS_URL).flush({
      nScenarios: 2,
      tickers: ['AAPL', 'MSFT'],
      scenarios: [
        { name: 'Rate Shock',   description: '100bp parallel shift', shocks: { AAPL: -0.08 }, probability: 0.1, horizonDays: 21, syntheticDataArgs: {} },
        { name: 'Equity Crash', description: 'broad selloff',        shocks: { MSFT: -0.25 }, probability: 0.05, horizonDays: 21, syntheticDataArgs: {} },
      ],
    });

    expect(comp.stressScenarios().length).toBe(2);
    expect(comp.stressScenarios()[0].name).toBe('Rate Shock');
  });

  // ── (b) handles API error: per-panel signal set ───────────────────────────

  it('when VaR endpoint returns 404, panelErrors["var"] is set', () => {
    fixture.detectChanges();
    http.match((r) => r.url === VAR_URL).forEach((r) =>
      r.flush({ detail: 'no snapshot' }, { status: 404, statusText: 'Not Found' }),
    );
    drainRest();
    fixture.detectChanges();

    expect(comp.panelErrors()['var']).toBeTruthy();
  });

  it('when correlation endpoint returns 500, panelErrors["correlation"] is set', () => {
    fixture.detectChanges();
    http.match((r) => r.url === CORR_URL).forEach((r) =>
      r.flush({ detail: 'server error' }, { status: 500, statusText: 'Server Error' }),
    );
    drainRest();
    fixture.detectChanges();

    expect(comp.panelErrors()['correlation']).toBeTruthy();
  });

  it('when factor-exposure endpoint returns 500, panelErrors["factor"] is set', () => {
    fixture.detectChanges();
    http.match((r) => r.url === FACTOR_URL).forEach((r) =>
      r.flush({ detail: 'server error' }, { status: 500, statusText: 'Server Error' }),
    );
    drainRest();
    fixture.detectChanges();

    expect(comp.panelErrors()['factor']).toBeTruthy();
  });

  it('when concentration endpoint returns 500, panelErrors["concentration"] is set', () => {
    fixture.detectChanges();
    http.match((r) => r.url === CONC_URL).forEach((r) =>
      r.flush({ detail: 'server error' }, { status: 500, statusText: 'Server Error' }),
    );
    drainRest();
    fixture.detectChanges();

    expect(comp.panelErrors()['concentration']).toBeTruthy();
  });

  it('when liquidity endpoint returns 500, panelErrors["liquidity"] is set', () => {
    fixture.detectChanges();
    http.match((r) => r.url === LIQ_URL).forEach((r) =>
      r.flush({ detail: 'server error' }, { status: 500, statusText: 'Server Error' }),
    );
    drainRest();
    fixture.detectChanges();

    expect(comp.panelErrors()['liquidity']).toBeTruthy();
  });

  it('when limits endpoint returns 404, panelErrors["limits"] is set', () => {
    fixture.detectChanges();
    http.match((r) => r.url === LIMITS_URL).forEach((r) =>
      r.flush({ detail: 'portfolio not found' }, { status: 404, statusText: 'Not Found' }),
    );
    drainRest();
    fixture.detectChanges();

    expect(comp.panelErrors()['limits']).toBeTruthy();
  });

  it('when stress generation returns 422, panelErrors["stress"] is set', () => {
    settle();
    comp.onGenerateStress({ macroContext: 'recession', nScenarios: 2 });

    http.expectOne((r) => r.url === STRESS_URL)
      .flush({ detail: 'validation error' }, { status: 422, statusText: 'Unprocessable Entity' });

    expect(comp.panelErrors()['stress']).toBeTruthy();
  });

  // ── (c) reacts to portfolio context change ────────────────────────────────

  it('when the portfolio name changes, new risk requests fire with the new name in the URL', () => {
    settle();
    const newName = 'new-portfolio';
    ctx._nameSig.set(newName);
    ctx._idSig.set('new-id');
    fixture.detectChanges();

    // At least one request must carry the new portfolio name
    const newRiskRequests = http.match((r) => r.url.includes(newName) && r.url.includes('/risk'));
    expect(newRiskRequests.length)
      .withContext(`Expected new risk requests for portfolio '${newName}'`)
      .toBeGreaterThan(0);

    for (const req of newRiskRequests) {
      req.flush(stubFor(req.request.url));
    }
    drainRest();
    fixture.detectChanges();
  });

  it('when the portfolio is cleared, no new risk requests fire', () => {
    settle();
    ctx._nameSig.set(null);
    ctx._idSig.set(null);
    fixture.detectChanges();

    const newRequests = http.match((r) => r.url.includes('/risk'));
    expect(newRequests.length)
      .withContext('Expected zero risk requests after portfolio cleared')
      .toBe(0);
  });
});
