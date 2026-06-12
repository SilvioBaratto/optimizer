import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection, signal, computed, NO_ERRORS_SCHEMA } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { HttpTestingController } from '@angular/common/http/testing';
import { of, throwError } from 'rxjs';

import {
  installResizeObserverStub,
  makeBrinsonResponse,
  makeFactorAttributionResponse,
} from '../../testing';
import { ICON_PROVIDER } from '../icons';
import { AttributionComponent } from './attribution';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { AttributionService } from './attribution.service';
import { environment } from '../../environments/environment';

import type { BrinsonSectorRowDto } from './attribution.model';

const API = environment.apiUrl;

const PORTFOLIO_NAME = 'Test Portfolio';
const PORTFOLIO_ID = 'pf-1';

function makePortfolioCtxMock(
  id: string | null = PORTFOLIO_ID,
  name: string | null = PORTFOLIO_NAME,
) {
  const idSig = signal<string | null>(id);
  const nameSig = signal<string | null>(name);
  return {
    currentPortfolioId: idSig,
    currentPortfolioName: computed(() => nameSig()),
    selectedPortfolio: computed(() => (id ? { id, name: name ?? '' } : null)),
    dateRange: signal({
      preset: '1Y' as const,
      start: new Date('2024-01-01'),
      end: new Date('2025-01-01'),
    }),
    benchmark: signal('SPY'),
    hasPortfolio: computed(() => id !== null),
    activeMode: signal('backtest' as const),
    isLive: computed(() => false),
    isBacktest: computed(() => true),
    isPaper: computed(() => false),
    dateRangeLabel: computed(() => '1Y'),
    dateRangeDays: computed(() => 365),
    setPortfolio: jasmine.createSpy('setPortfolio'),
    setMode: jasmine.createSpy('setMode'),
    setPreset: jasmine.createSpy('setPreset'),
    setCustomRange: jasmine.createSpy('setCustomRange'),
    setBenchmark: jasmine.createSpy('setBenchmark'),
    reset: jasmine.createSpy('reset'),
    _nameSig: nameSig,
    _idSig: idSig,
  };
}

function makeAttrSvcMock(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  brinsonReturn: any = of(makeBrinsonResponse()),
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  factorReturn: any = of(makeFactorAttributionResponse()),
) {
  return {
    getBrinsonAttribution: jasmine.createSpy('getBrinsonAttribution').and.returnValue(brinsonReturn),
    getFactorAttribution: jasmine.createSpy('getFactorAttribution').and.returnValue(factorReturn),
    loadAttributionData: jasmine.createSpy('loadAttributionData').and.returnValue(of({})),
    getAttribution: jasmine.createSpy('getAttribution').and.returnValue(of({})),
    // Legacy POST methods used by runBrinson/runFactor
    brinson: jasmine.createSpy('brinson').and.returnValue(of(makeBrinsonResponse())),
    factor: jasmine.createSpy('factor').and.returnValue(of(makeFactorAttributionResponse())),
  };
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
  let portfolioCtxMock: ReturnType<typeof makePortfolioCtxMock>;
  let attrSvcMock: ReturnType<typeof makeAttrSvcMock>;

  beforeEach(async () => {
    installResizeObserverStub();
    portfolioCtxMock = makePortfolioCtxMock();
    attrSvcMock = makeAttrSvcMock();

    await TestBed.configureTestingModule({
      imports: [AttributionComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        ICON_PROVIDER,
        { provide: PortfolioContextService, useValue: portfolioCtxMock },
        { provide: AttributionService, useValue: attrSvcMock },
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(AttributionComponent);
    comp = fixture.componentInstance;
    fixture.detectChanges();
  });

  afterEach(() => {
    // Drain any pending HTTP requests (e.g. from runBrinson/runFactor calls).
    const http = TestBed.inject(HttpTestingController);
    http.verify();
  });

  it('on init, attribution API is called with the portfolio name', () => {
    expect(attrSvcMock.getBrinsonAttribution).toHaveBeenCalledWith(
      PORTFOLIO_NAME,
      jasmine.any(String),
      jasmine.any(String),
    );
  });

  it('on init, attribution API is NOT called with the raw UUID', () => {
    const calls = attrSvcMock.getBrinsonAttribution.calls.allArgs();
    for (const args of calls) {
      expect(args[0]).not.toBe(PORTFOLIO_ID);
    }
  });

  it('when attribution API returns data, brinsonResponse is populated', () => {
    expect(comp.brinsonResponse()).not.toBeNull();
  });

  it('when attribution API returns data, factorResponse is populated', () => {
    expect(comp.factorResponse()).not.toBeNull();
  });

  it('when portfolio context changes, attribution API is called again', () => {
    attrSvcMock.getBrinsonAttribution.calls.reset();
    portfolioCtxMock._nameSig.set('European Bonds');
    fixture.detectChanges();

    expect(attrSvcMock.getBrinsonAttribution.calls.count()).toBeGreaterThan(0);
  });

  it('when portfolio is cleared, attribution API is not called', () => {
    attrSvcMock.getBrinsonAttribution.calls.reset();
    attrSvcMock.getFactorAttribution.calls.reset();
    portfolioCtxMock._nameSig.set(null);
    portfolioCtxMock._idSig.set(null);
    fixture.detectChanges();

    expect(attrSvcMock.getBrinsonAttribution.calls.count()).toBe(0);
    expect(attrSvcMock.getFactorAttribution.calls.count()).toBe(0);
  });

  it('retry clears the error and refetches', () => {
    comp.hasError.set(true);
    comp.errorMessage.set('some error');
    attrSvcMock.getBrinsonAttribution.calls.reset();

    comp.retry();

    expect(comp.hasError()).toBe(false);
    expect(attrSvcMock.getBrinsonAttribution.calls.count()).toBeGreaterThan(0);
  });

  it('when weights sum to one, the form is valid; an empty portfolio invalidates it', () => {
    comp.portfolioWeights.set({ AAPL: 1 });
    expect(comp.isFormValid()).toBe(true);
    comp.portfolioWeights.set({});
    expect(comp.isFormValid()).toBe(false);
  });

  it('when the form is invalid, runBrinson does not call the brinson API', () => {
    comp.portfolioWeights.set({});
    attrSvcMock.brinson.calls.reset();
    comp.runBrinson();
    expect(attrSvcMock.brinson.calls.count()).toBe(0);
  });

  it('when the benchmark is entirely Unclassified, benchAllUnclassified is true', () => {
    comp.brinsonResponse.set(makeBrinsonResponse({ sectors: [unclassifiedSector(1)] }));
    expect(comp.benchAllUnclassified()).toBe(true);
    comp.brinsonResponse.set(makeBrinsonResponse());
    expect(comp.benchAllUnclassified()).toBe(false);
  });

  it('runBrinson calls brinson service and stores the response on success', () => {
    comp.portfolioWeights.set({ AAPL: 1 });
    const brinsonResponse = makeBrinsonResponse();
    attrSvcMock.brinson.and.returnValue(of(brinsonResponse));

    comp.runBrinson();

    expect(attrSvcMock.brinson).toHaveBeenCalled();
    expect(comp.brinsonResponse()).toEqual(brinsonResponse);
    expect(comp.brinsonLoading()).toBe(false);
  });

  it('runBrinson records an error on failure', () => {
    comp.portfolioWeights.set({ AAPL: 1 });
    attrSvcMock.brinson.and.returnValue(
      throwError(() => new Error('bad request')),
    );

    comp.runBrinson();

    expect(comp.brinsonError()).toBeTruthy();
    expect(comp.brinsonLoading()).toBe(false);
  });

  it('runFactor is guarded when weights do not sum to one', () => {
    comp.portfolioWeights.set({});
    attrSvcMock.factor.calls.reset();
    comp.runFactor();
    expect(attrSvcMock.factor.calls.count()).toBe(0);
  });

  it('runFactor is guarded when the dates are not ordered', () => {
    comp.portfolioWeights.set({ AAPL: 1 });
    comp.endDate.set('2000-01-01');
    attrSvcMock.factor.calls.reset();
    comp.runFactor();
    expect(attrSvcMock.factor.calls.count()).toBe(0);
  });

  it('runFactor calls factor service and stores the response on success', () => {
    comp.portfolioWeights.set({ AAPL: 1 });
    const factorResponse = makeFactorAttributionResponse();
    attrSvcMock.factor.and.returnValue(of(factorResponse));

    comp.runFactor();

    expect(attrSvcMock.factor).toHaveBeenCalled();
    expect(comp.factorResponse()).toEqual(factorResponse);
    expect(comp.factorLoading()).toBe(false);
  });

  it('runFactor records an error on failure', () => {
    comp.portfolioWeights.set({ AAPL: 1 });
    attrSvcMock.factor.and.returnValue(
      throwError(() => new Error('factor failed')),
    );

    comp.runFactor();

    expect(comp.factorError()).toBeTruthy();
  });

  it('openReportModal does not throw', () => {
    expect(() => comp.openReportModal()).not.toThrow();
  });

  it('onPortfolioSelect updates the selectedPortfolio signal', () => {
    comp.onPortfolioSelect('Other Portfolio');
    expect(comp.selectedPortfolio()).toBe('Other Portfolio');
  });
});

describe('AttributionComponent — error state', () => {
  let fixture: ComponentFixture<AttributionComponent>;

  beforeEach(async () => {
    installResizeObserverStub();
    const errorCtx = makePortfolioCtxMock();
    const errorSvc = makeAttrSvcMock(
      throwError(() => new Error('500 Internal Server Error')),
      throwError(() => new Error('500 Internal Server Error')),
    );

    await TestBed.configureTestingModule({
      imports: [AttributionComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        ICON_PROVIDER,
        { provide: PortfolioContextService, useValue: errorCtx },
        { provide: AttributionService, useValue: errorSvc },
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(AttributionComponent);
    fixture.detectChanges();
  });

  it('when attribution API errors, hasError is set to true', () => {
    expect(fixture.componentInstance.hasError()).toBe(true);
  });
});
