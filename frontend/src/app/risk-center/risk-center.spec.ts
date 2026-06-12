import { ComponentFixture, TestBed } from '@angular/core/testing';
import type { HttpTestingController } from '@angular/common/http/testing';
import { signal } from '@angular/core';

import {
  configureTestBed,
  drainRequests,
  injectHttp,
  installResizeObserverStub,
  makePortfolioDto,
} from '../../testing';
import { ICON_PROVIDER } from '../icons';
import { RiskCenterComponent } from './risk-center';
import { RiskService } from './risk.service';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { of, throwError } from 'rxjs';

// URL-aware stub for the 5 parallel analytics + limits fetches.
function stubFor(url: string): Record<string, unknown> {
  if (url.includes('/risk/var')) return { var: { '95': 0.03 }, cvar: { '95': 0.05 }, method: 'historical', lookback: 252, nObservations: 252 };
  if (url.includes('/risk/correlation')) return { assets: [], matrix: [], clusterLabels: [] };
  if (url.includes('/risk/factor-exposure')) return { exposures: {}, assetExposures: {} };
  if (url.includes('/risk/concentration')) return { assets: [], summary: { hhi: 0, effectiveN: 0, topNRatio: 0 } };
  if (url.includes('/risk/liquidity')) return { assets: [], summary: { weightedAvgDaysToLiquidate: 0 } };
  if (url.includes('/limits')) return { items: [], breachCount: 0 };
  return {};
}

/** Minimal PortfolioContextService stub with a named portfolio. */
function makeCtxStub(name = 'Test Portfolio', id = 'test-id') {
  const idSig = signal<string | null>(id);
  const nameSig = signal<string | null>(name);
  return {
    currentPortfolioId: idSig,
    currentPortfolioName: nameSig,
    hasPortfolio: { _value: true },
    _idSig: idSig,
    _nameSig: nameSig,
  };
}

describe('RiskCenterComponent', () => {
  let fixture: ComponentFixture<RiskCenterComponent>;
  let comp: RiskCenterComponent;
  let http: HttpTestingController;
  let ctx: ReturnType<typeof makeCtxStub>;

  // Flush the effect-driven analytics + limits fetches.
  function settle(): void {
    fixture.detectChanges();
    drainRequests(http, stubFor);
    fixture.detectChanges();
  }

  beforeEach(async () => {
    ctx = makeCtxStub();
    installResizeObserverStub();
    await configureTestBed({
      imports: [RiskCenterComponent],
      withHttp: true,
      providers: [
        ICON_PROVIDER,
        { provide: PortfolioContextService, useValue: ctx },
      ],
    });
    fixture = TestBed.createComponent(RiskCenterComponent);
    comp = fixture.componentInstance;
    http = injectHttp();
  });

  afterEach(() => http.verify());

  it('when initialising with a portfolio in context, analytics requests fire immediately', () => {
    fixture.detectChanges();
    const pending = http.match(() => true);
    expect(pending.length).toBeGreaterThan(0);
    for (const req of pending) req.flush(stubFor(req.request.url));
    fixture.detectChanges();
  });

  it('when all analytics resolve, panel error signals remain null for each key', () => {
    settle();
    expect(comp.panelErrors()['var']).toBeFalsy();
    expect(comp.panelErrors()['correlation']).toBeFalsy();
  });

  it('when there are limit breaches, the Risk Limits tab carries a badge', () => {
    settle();
    comp.limitsResponse.set({ items: [], breachCount: 3 });
    const limitsTab = comp.tabs().find((t) => t.id === 'limits');
    expect(limitsTab?.badge).toBe(3);
  });

  it('when no portfolio name is in context, onGenerateStress fires no request', () => {
    ctx._nameSig.set(null);
    ctx._idSig.set(null);
    fixture.detectChanges();
    drainRequests(http, stubFor); // drain any in-flight from prior state
    comp.onGenerateStress({ macroContext: 'recession', nScenarios: 4 });
    expect(http.match((r) => r.url.includes('risk/stress-scenarios')).length).toBe(0);
  });

  it('when a portfolio is selected, onGenerateStress POSTs the scenario request', () => {
    settle();
    comp.onGenerateStress({ macroContext: 'recession', nScenarios: 4 });
    const req = http.expectOne((r) => r.url.includes('risk/stress-scenarios'));
    expect(req.request.method).toBe('POST');
    req.flush({ nScenarios: 4, tickers: [], scenarios: [] });
  });

  it('when a limit is created, it POSTs then reloads the limits', () => {
    settle();
    comp.onCreateLimit({ metric: 'max_drawdown', limit_type: 'upper', threshold: 0.2 });
    http.expectOne((r) => r.url.includes('/limits') && r.method === 'POST').flush({});
    http.expectOne((r) => r.url.includes('/limits') && r.method === 'GET')
      .flush({ items: [], breachCount: 3 });
    expect(comp.limitsResponse().breachCount).toBe(3);
  });

  it('when a limit is deleted, it DELETEs then reloads the limits', () => {
    settle();
    comp.onDeleteLimit('l1');
    http.expectOne((r) => r.url.includes('/limits/l1') && r.method === 'DELETE').flush(null, { status: 204, statusText: 'No Content' });
    http.expectOne((r) => r.url.includes('/limits') && r.method === 'GET')
      .flush({ items: [], breachCount: 1 });
    expect(comp.limitsResponse().breachCount).toBe(1);
  });

  it('when analytics fetches fail, the values are cleared and panel errors recorded', () => {
    settle();
    // Switch name to trigger re-fetch
    ctx._nameSig.set('Other');
    fixture.detectChanges();
    for (const req of http.match(() => true)) {
      req.flush({ detail: 'x' }, { status: 500, statusText: 'Server Error' });
    }
    expect(comp.varResponse()).toBeNull();
    expect(comp.panelErrors()['var']).toBeTruthy();
    expect(comp.panelErrors()['limits']).toBeTruthy();
  });

  it('when a limit is updated, it PUTs then reloads the limits', () => {
    settle();
    comp.onUpdateLimit({ id: 'l1', current: 0.5, breached: true });
    http.expectOne((r) => r.url.includes('/limits/l1') && r.method === 'PUT').flush({});
    http.expectOne((r) => r.url.includes('/limits') && r.method === 'GET')
      .flush({ items: [], breachCount: 2 });
    expect(comp.limitsResponse().breachCount).toBe(2);
  });

  it('when stress generation fails, a panel error is set and loading clears', () => {
    settle();
    comp.onGenerateStress({ macroContext: 'recession', nScenarios: 2 });
    http
      .expectOne((r) => r.url.includes('risk/stress-scenarios'))
      .flush({ detail: 'boom' }, { status: 500, statusText: 'Server Error' });
    expect(comp.panelErrors()['stress']).toBeTruthy();
    expect(comp.stressLoading()).toBe(false);
  });

  it('when creating a limit fails, a panel error is recorded', () => {
    settle();
    comp.onCreateLimit({ metric: 'max_drawdown', limit_type: 'upper', threshold: 0.2 });
    http
      .expectOne((r) => r.url.includes('/limits') && r.method === 'POST')
      .flush({ detail: 'bad' }, { status: 422, statusText: 'Unprocessable' });
    expect(comp.panelErrors()['limits']).toBeTruthy();
  });

  it('onMethodChange ignores monte_carlo but accepts parametric', () => {
    settle();
    comp.onMethodChange('monte_carlo');
    expect(comp.varMethod()).toBe('historical');
    comp.onMethodChange('parametric');
    expect(comp.varMethod()).toBe('parametric');
    fixture.detectChanges();
    drainRequests(http, stubFor);
  });

  it('onConfidenceChange updates the selected confidence', () => {
    settle();
    comp.onConfidenceChange(0.99);
    expect(comp.varConfidence()).toBe(0.99);
  });

  it('retry clears the error flag and refetches analytics', () => {
    settle();
    comp.hasError.set(true);
    comp.errorMessage.set('boom');
    comp.retry();
    expect(comp.hasError()).toBe(false);
    drainRequests(http, stubFor);
  });

  it('openReportModal opens the export modal without error', () => {
    settle();
    expect(() => comp.openReportModal()).not.toThrow();
  });

  it('when a 99% VaR result exists, the VaR 99% KPI is formatted (not an em dash)', () => {
    settle();
    comp.varResponse.set({
      var: { '99': 0.04 },
      cvar: { '99': 0.06 },
      method: 'historical',
      lookback: 252,
      nObservations: 252,
    });
    expect(comp.kpiVar99()).not.toBe('—');
  });

  it('onGenerateStress uses live concentration weights when they are available', () => {
    settle();
    comp.concentrationResponse.set({
      assets: [{ ticker: 'MSFT', name: 'Microsoft', weight: 0.5 }],
      summary: { hhi: 0, effectiveN: 0, topNRatio: 0 },
    });
    comp.onGenerateStress({ macroContext: 'x', nScenarios: 1 });
    const req = http.expectOne((r) => r.url.includes('risk/stress-scenarios'));
    expect((req.request.body as { current_portfolio: Record<string, number> }).current_portfolio)
      .toEqual({ MSFT: 0.5 });
    req.flush({ nScenarios: 1, tickers: [], scenarios: [] });
  });

  it('when updating a limit fails, a panel error is recorded', () => {
    settle();
    comp.onUpdateLimit({ id: 'l1', current: 0.5, breached: true });
    http
      .expectOne((r) => r.url.includes('/limits/l1') && r.method === 'PUT')
      .flush({ detail: 'bad' }, { status: 500, statusText: 'Server Error' });
    expect(comp.panelErrors()['limits']).toBeTruthy();
  });

  it('when deleting a limit fails, a panel error is recorded', () => {
    settle();
    comp.onDeleteLimit('l1');
    http
      .expectOne((r) => r.url.includes('/limits/l1') && r.method === 'DELETE')
      .flush({ detail: 'bad' }, { status: 500, statusText: 'Server Error' });
    expect(comp.panelErrors()['limits']).toBeTruthy();
  });
});

// The subscribe error callbacks are typed `(err: Error)`, and an
// HttpErrorResponse always carries a `.message`, so the `?? 'default'` fallback
// is only reachable when the source errors with a message-less value. A mocked
// service supplies exactly that, covering the otherwise-unreachable defaults.
describe('RiskCenterComponent — default error messages', () => {
  function boomRisk(): RiskService {
    const boom = () => throwError(() => ({}));
    return {
      getVar: boom, getCorrelation: boom, getFactorExposure: boom,
      getConcentration: boom, getLiquidity: boom, listLimits: boom,
      generateStressScenarios: boom, createLimit: boom, updateLimit: boom, deleteLimit: boom,
    } as unknown as RiskService;
  }

  it('falls back to default panel messages when analytics error without a message', async () => {
    const idSig = signal<string | null>('test-id');
    const nameSig = signal<string | null>('Test Portfolio');
    const ctxStub = { currentPortfolioId: idSig, currentPortfolioName: nameSig };

    installResizeObserverStub();
    await configureTestBed({
      imports: [RiskCenterComponent],
      withHttp: true,
      providers: [
        ICON_PROVIDER,
        { provide: RiskService, useValue: boomRisk() },
        { provide: PortfolioContextService, useValue: ctxStub },
      ],
    });
    const fx = TestBed.createComponent(RiskCenterComponent);
    fx.detectChanges();
    const c = fx.componentInstance;

    expect(c.panelErrors()['var']).toBe('Failed to load var');
    expect(c.panelErrors()['limits']).toBe('Load failed');
    c.onGenerateStress({ macroContext: 'x', nScenarios: 1 });
    expect(c.panelErrors()['stress']).toBe('Stress failed');
    c.onCreateLimit({ metric: 'm', limit_type: 'upper', threshold: 0.1 });
    expect(c.panelErrors()['limits']).toBe('Create failed');
  });
});
