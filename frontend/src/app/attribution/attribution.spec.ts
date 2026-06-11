import { ComponentFixture, TestBed } from '@angular/core/testing';
import type { HttpTestingController } from '@angular/common/http/testing';

import {
  configureTestBed,
  drainRequests,
  injectHttp,
  installResizeObserverStub,
  makePortfolioDto,
  makeBrinsonResponse,
  makeFactorAttributionResponse,
} from '../../testing';
import { ICON_PROVIDER } from '../icons';
import { AttributionComponent } from './attribution';
import type { BrinsonSectorRowDto } from './attribution.model';

// URL-aware stub: snapshot is matched first since its URL also contains
// "portfolio". list → one portfolio; snapshot → weights.
function stubFor(url: string): Record<string, unknown> {
  if (url.includes('snapshots/latest')) return { weights: { AAPL: 1 } };
  if (url.includes('portfolio')) return { items: [makePortfolioDto()], total: 1 };
  return {};
}

function unclassifiedSector(weight: number): BrinsonSectorRowDto {
  return {
    sector: 'Unclassified',
    portfolioWeight: weight,
    benchmarkWeight: weight,
    portfolioReturn: 0.05,
    benchmarkReturn: 0.04,
    allocationEffect: 0,
    selectionEffect: 0,
    interactionEffect: 0,
    totalEffect: 0,
  };
}

describe('AttributionComponent', () => {
  let fixture: ComponentFixture<AttributionComponent>;
  let comp: AttributionComponent;
  let http: HttpTestingController;

  // Flush the list fetch, then the effect-driven snapshot fetch.
  function settle(): void {
    fixture.detectChanges();
    drainRequests(http, stubFor);
    fixture.detectChanges();
    drainRequests(http, stubFor);
  }

  beforeEach(async () => {
    installResizeObserverStub();
    await configureTestBed({ imports: [AttributionComponent], withHttp: true, providers: [ICON_PROVIDER] });
    fixture = TestBed.createComponent(AttributionComponent);
    comp = fixture.componentInstance;
    http = injectHttp();
  });

  afterEach(() => http.verify());

  it('when initialising, it is in the loading state until the portfolios resolve', () => {
    fixture.detectChanges();
    expect(comp.isLoading()).toBe(true);
    settle();
    expect(comp.isLoading()).toBe(false);
  });

  it('when the list resolves, the portfolios populate and the first is selected', () => {
    settle();
    expect(comp.portfolios().length).toBe(1);
    expect(comp.selectedPortfolio()).toBe('Test Portfolio');
  });

  it('when a portfolio is selected, the effect-driven snapshot fetch populates the weights', () => {
    settle();
    expect(comp.portfolioWeights()['AAPL']).toBe(1);
  });

  it('when the list request fails, the error state is shown', () => {
    fixture.detectChanges();
    http
      .expectOne((r) => r.url.includes('portfolio'))
      .flush({ detail: 'boom' }, { status: 500, statusText: 'Server Error' });
    fixture.detectChanges();
    expect(comp.hasError()).toBe(true);
  });

  it('when weights sum to one, the form is valid; an empty portfolio invalidates it', () => {
    settle();
    comp.portfolioWeights.set({ AAPL: 1 });
    expect(comp.isFormValid()).toBe(true);
    comp.portfolioWeights.set({});
    expect(comp.isFormValid()).toBe(false);
  });

  it('when the form is invalid, runBrinson does not fire a request (guard)', () => {
    settle();
    comp.portfolioWeights.set({});
    comp.runBrinson();
    expect(http.match((r) => r.url.includes('attribution/brinson')).length).toBe(0);
  });

  it('when the benchmark is entirely Unclassified, benchAllUnclassified is true', () => {
    settle();
    comp.brinsonResponse.set(makeBrinsonResponse({ sectors: [unclassifiedSector(1)] }));
    expect(comp.benchAllUnclassified()).toBe(true);
    comp.brinsonResponse.set(makeBrinsonResponse());
    expect(comp.benchAllUnclassified()).toBe(false);
  });

  it('runBrinson posts and stores the response on success', () => {
    settle();
    comp.portfolioWeights.set({ AAPL: 1 });
    comp.runBrinson();
    http.expectOne((r) => r.url.includes('attribution/brinson')).flush(makeBrinsonResponse());
    expect(comp.brinsonResponse()).not.toBeNull();
    expect(comp.brinsonLoading()).toBe(false);
  });

  it('runBrinson records an error on failure', () => {
    settle();
    comp.portfolioWeights.set({ AAPL: 1 });
    comp.runBrinson();
    http
      .expectOne((r) => r.url.includes('attribution/brinson'))
      .flush({ detail: 'bad' }, { status: 422, statusText: 'Unprocessable' });
    expect(comp.brinsonError()).toBeTruthy();
    expect(comp.brinsonLoading()).toBe(false);
  });

  it('runFactor is guarded when weights do not sum to one', () => {
    settle();
    comp.portfolioWeights.set({});
    comp.runFactor();
    expect(http.match((r) => r.url.includes('attribution/factor')).length).toBe(0);
  });

  it('runFactor is guarded when the dates are not ordered', () => {
    settle();
    comp.portfolioWeights.set({ AAPL: 1 });
    comp.endDate.set('2000-01-01');
    comp.runFactor();
    expect(http.match((r) => r.url.includes('attribution/factor')).length).toBe(0);
  });

  it('runFactor posts and stores the response on success', () => {
    settle();
    comp.portfolioWeights.set({ AAPL: 1 });
    comp.runFactor();
    http
      .expectOne((r) => r.url.includes('attribution/factor'))
      .flush(makeFactorAttributionResponse());
    expect(comp.factorResponse()).not.toBeNull();
    expect(comp.factorLoading()).toBe(false);
  });

  it('runFactor records an error on failure', () => {
    settle();
    comp.portfolioWeights.set({ AAPL: 1 });
    comp.runFactor();
    http
      .expectOne((r) => r.url.includes('attribution/factor'))
      .flush({ detail: 'bad' }, { status: 500, statusText: 'Server Error' });
    expect(comp.factorError()).toBeTruthy();
  });

  it('selecting a portfolio refetches the snapshot weights', () => {
    settle();
    comp.onPortfolioSelect('Other');
    fixture.detectChanges();
    http.expectOne((r) => r.url.includes('snapshots/latest')).flush({ weights: { MSFT: 1 } });
    expect(comp.portfolioWeights()['MSFT']).toBe(1);
  });

  it('when the snapshot fetch fails, the weights reset to empty', () => {
    settle();
    comp.onPortfolioSelect('Other');
    fixture.detectChanges();
    http
      .expectOne((r) => r.url.includes('snapshots/latest'))
      .flush({ detail: 'x' }, { status: 500, statusText: 'Server Error' });
    expect(comp.portfolioWeights()).toEqual({});
  });

  it('retry clears the error and reloads', () => {
    fixture.detectChanges();
    http
      .expectOne((r) => r.url.includes('portfolio'))
      .flush({ detail: 'boom' }, { status: 500, statusText: 'Server Error' });
    fixture.detectChanges();
    expect(comp.hasError()).toBe(true);
    comp.retry();
    expect(comp.hasError()).toBe(false);
    drainRequests(http, stubFor);
    fixture.detectChanges();
    drainRequests(http, stubFor);
  });

  it('openReportModal does not throw', () => {
    settle();
    expect(() => comp.openReportModal()).not.toThrow();
  });
});
