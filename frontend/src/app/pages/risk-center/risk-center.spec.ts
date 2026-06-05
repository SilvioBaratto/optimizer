import { ComponentFixture, TestBed } from '@angular/core/testing';
import type { HttpTestingController } from '@angular/common/http/testing';

import {
  configureTestBed,
  drainRequests,
  injectHttp,
  installResizeObserverStub,
  makePortfolioDto,
} from '../../../testing';
import { ICON_PROVIDER } from '../../icons';
import { RiskCenterComponent } from './risk-center';

// URL-aware stub for the list + 5 parallel analytics + limits fetches.
function stubFor(url: string): Record<string, unknown> {
  if (url.includes('/risk/var')) return { var: { '95': 0.03 }, cvar: { '95': 0.05 }, method: 'historical', lookback: 252, nObservations: 252 };
  if (url.includes('/risk/correlation')) return { assets: [], matrix: [], clusterLabels: [] };
  if (url.includes('/risk/factor-exposure')) return { exposures: {}, assetExposures: {} };
  if (url.includes('/risk/concentration')) return { assets: [], summary: { hhi: 0, effectiveN: 0, topNRatio: 0 } };
  if (url.includes('/risk/liquidity')) return { assets: [], summary: { weightedAvgDaysToLiquidate: 0 } };
  if (url.includes('/limits')) return { items: [], breachCount: 0 };
  if (url.includes('portfolio')) return { items: [makePortfolioDto()], total: 1 };
  return {};
}

describe('RiskCenterComponent', () => {
  let fixture: ComponentFixture<RiskCenterComponent>;
  let comp: RiskCenterComponent;
  let http: HttpTestingController;

  // Flush the list, then the effect-driven analytics + limits fetches.
  function settle(): void {
    fixture.detectChanges();
    drainRequests(http, stubFor);
    fixture.detectChanges();
    drainRequests(http, stubFor);
  }

  beforeEach(async () => {
    installResizeObserverStub();
    await configureTestBed({ imports: [RiskCenterComponent], withHttp: true, providers: [ICON_PROVIDER] });
    fixture = TestBed.createComponent(RiskCenterComponent);
    comp = fixture.componentInstance;
    http = injectHttp();
  });

  afterEach(() => http.verify());

  it('when initialising, it loads until the portfolios resolve', () => {
    fixture.detectChanges();
    expect(comp.isLoading()).toBe(true);
    settle();
    expect(comp.isLoading()).toBe(false);
  });

  it('when the list resolves, all parallel analytics requests are flushed cleanly', () => {
    settle();
    expect(comp.portfolios().length).toBe(1);
    expect(comp.selectedPortfolio()).toBe('Test Portfolio');
    // afterEach http.verify() proves every parallel request was flushed.
  });

  it('when the list request fails, the error state is shown', () => {
    fixture.detectChanges();
    http
      .expectOne((r) => r.url.includes('portfolio'))
      .flush({ detail: 'boom' }, { status: 500, statusText: 'Server Error' });
    fixture.detectChanges();
    expect(comp.hasError()).toBe(true);
  });

  it('when there are limit breaches, the Risk Limits tab carries a badge', () => {
    settle();
    comp.limitsResponse.set({ items: [], breachCount: 3 });
    const limitsTab = comp.tabs().find((t) => t.id === 'limits');
    expect(limitsTab?.badge).toBe(3);
  });

  it('when no portfolio is selected, onGenerateStress fires no request', () => {
    settle();
    comp.selectedPortfolio.set('');
    comp.onGenerateStress({ macroContext: 'recession', nScenarios: 4 });
    http.expectNone((r) => r.url.includes('risk/stress-scenarios'));
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
      .flush({ items: [], breachCount: 0 });
  });

  it('when a limit is deleted, it DELETEs then reloads the limits', () => {
    settle();
    comp.onDeleteLimit('l1');
    http.expectOne((r) => r.url.includes('/limits/l1') && r.method === 'DELETE').flush(null, { status: 204, statusText: 'No Content' });
    http.expectOne((r) => r.url.includes('/limits') && r.method === 'GET')
      .flush({ items: [], breachCount: 0 });
  });
});
