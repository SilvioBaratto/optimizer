import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection, signal, computed, NO_ERRORS_SCHEMA } from '@angular/core';
import { HttpErrorResponse, provideHttpClient } from '@angular/common/http';
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
import { PortfolioApiService } from '../core/services/portfolio-api.service';
import { AttributionService } from './attribution.service';

import type { BrinsonSectorRowDto } from './attribution.model';
import type { SnapshotDto } from '../core/models/portfolio-api.model';

const PORTFOLIO_NAME = 'Test Portfolio';
const PORTFOLIO_ID = 'pf-1';

// Weights the mocked latest snapshot resolves to — sum to 1 so the run-form is valid.
const SNAPSHOT_WEIGHTS: Record<string, number> = { AAPL: 0.6, MSFT: 0.4 };

function makePortfolioApiMock(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  snapshotReturn: any = of({ weights: SNAPSHOT_WEIGHTS } as unknown as SnapshotDto),
) {
  return {
    getLatestSnapshot: jasmine.createSpy('getLatestSnapshot').and.returnValue(snapshotReturn),
  };
}

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

// The service exposes only the POST methods brinson()/factor(); the params feed
// those spies so tests can inject success or error streams.
function makeAttrSvcMock(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  brinsonReturn: any = of(makeBrinsonResponse()),
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  factorReturn: any = of(makeFactorAttributionResponse()),
) {
  return {
    brinson: jasmine.createSpy('brinson').and.returnValue(brinsonReturn),
    factor: jasmine.createSpy('factor').and.returnValue(factorReturn),
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
  let portfolioApiMock: ReturnType<typeof makePortfolioApiMock>;

  beforeEach(async () => {
    installResizeObserverStub();
    portfolioCtxMock = makePortfolioCtxMock();
    attrSvcMock = makeAttrSvcMock();
    portfolioApiMock = makePortfolioApiMock();

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
        { provide: PortfolioApiService, useValue: portfolioApiMock },
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

  it('on init, the latest snapshot is fetched with the portfolio name', () => {
    expect(portfolioApiMock.getLatestSnapshot).toHaveBeenCalledWith(PORTFOLIO_NAME);
  });

  it('on init, the snapshot is NOT fetched with the raw UUID', () => {
    for (const args of portfolioApiMock.getLatestSnapshot.calls.allArgs()) {
      expect(args[0]).not.toBe(PORTFOLIO_ID);
    }
  });

  it('when a snapshot loads, portfolioWeights is populated from snapshot.weights', () => {
    expect(comp.portfolioWeights()).toEqual(SNAPSHOT_WEIGHTS);
  });

  it('when attribution API returns data, brinsonResponse is populated', () => {
    expect(comp.brinsonResponse()).not.toBeNull();
  });

  it('when attribution API returns data, factorResponse is populated', () => {
    expect(comp.factorResponse()).not.toBeNull();
  });

  it('when portfolio context changes, the snapshot is fetched again', () => {
    portfolioApiMock.getLatestSnapshot.calls.reset();
    portfolioCtxMock._nameSig.set('European Bonds');
    fixture.detectChanges();

    expect(portfolioApiMock.getLatestSnapshot.calls.count()).toBeGreaterThan(0);
  });

  it('when portfolio context changes, the snapshot is fetched with the new name', () => {
    portfolioApiMock.getLatestSnapshot.calls.reset();
    portfolioCtxMock._nameSig.set('European Bonds');
    fixture.detectChanges();

    expect(portfolioApiMock.getLatestSnapshot).toHaveBeenCalledWith('European Bonds');
  });

  it('when portfolio is cleared, the snapshot is not fetched', () => {
    portfolioApiMock.getLatestSnapshot.calls.reset();
    portfolioCtxMock._nameSig.set(null);
    portfolioCtxMock._idSig.set(null);
    fixture.detectChanges();

    expect(portfolioApiMock.getLatestSnapshot.calls.count()).toBe(0);
  });

  it('retry clears the error and refetches the snapshot', () => {
    comp.hasError.set(true);
    comp.errorMessage.set('some error');
    portfolioApiMock.getLatestSnapshot.calls.reset();

    comp.retry();

    expect(comp.hasError()).toBe(false);
    expect(portfolioApiMock.getLatestSnapshot.calls.count()).toBeGreaterThan(0);
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

  // ── issue #997, 13c: 404 → not-available empty-state vs other → error ──────

  it('when factor() returns a 404, factorNotAvailable is true and factorError stays null', () => {
    comp.portfolioWeights.set({ AAPL: 1 });
    attrSvcMock.factor.and.returnValue(
      throwError(() => new HttpErrorResponse({ status: 404, statusText: 'Not Found' })),
    );

    comp.runFactor();

    expect(comp.factorNotAvailable()).toBe(true);
    expect(comp.factorError()).toBeNull();
  });

  it('when factor() returns a non-404 error, factorError is set and factorNotAvailable is false', () => {
    comp.portfolioWeights.set({ AAPL: 1 });
    attrSvcMock.factor.and.returnValue(
      throwError(() => new HttpErrorResponse({ status: 500, statusText: 'Server Error' })),
    );

    comp.runFactor();

    expect(comp.factorError()).toBeTruthy();
    expect(comp.factorNotAvailable()).toBe(false);
  });

  it('when a factor run later succeeds, factorNotAvailable is cleared', () => {
    comp.portfolioWeights.set({ AAPL: 1 });
    attrSvcMock.factor.and.returnValue(
      throwError(() => new HttpErrorResponse({ status: 404, statusText: 'Not Found' })),
    );
    comp.runFactor();
    expect(comp.factorNotAvailable()).toBe(true);

    attrSvcMock.factor.and.returnValue(of(makeFactorAttributionResponse()));
    comp.runFactor();

    expect(comp.factorNotAvailable()).toBe(false);
  });

  it('when a portfolio is selected, the form hint shows the portfolio name and not "(none)"', () => {
    const text = (fixture.nativeElement as HTMLElement).textContent ?? '';
    expect(text).toContain(PORTFOLIO_NAME);
    expect(text).not.toContain('(none)');
  });

  it('openReportModal does not throw', () => {
    expect(() => comp.openReportModal()).not.toThrow();
  });
});

describe('AttributionComponent — error state', () => {
  let fixture: ComponentFixture<AttributionComponent>;

  beforeEach(async () => {
    installResizeObserverStub();
    const errorCtx = makePortfolioCtxMock();
    const errorSvc = makeAttrSvcMock();
    // The snapshot fetch is the primary call; a 500 here drives the error state.
    const errorApi = makePortfolioApiMock(
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
        { provide: PortfolioApiService, useValue: errorApi },
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(AttributionComponent);
    fixture.detectChanges();
  });

  it('when the snapshot fetch errors, hasError is set to true', () => {
    expect(fixture.componentInstance.hasError()).toBe(true);
  });
});

// AC2: a factor() failure during the auto-load must surface a visible state
// (factorError), not silently blank — the prior `catchError(() => EMPTY)` hid it.
describe('AttributionComponent — factor auto-load surfaces a visible error (issue #983)', () => {
  it('when the auto-loaded factor() call fails, factorError is set (not a silent blank)', async () => {
    installResizeObserverStub();
    const ctx = makePortfolioCtxMock();
    const svc = makeAttrSvcMock(
      of(makeBrinsonResponse()),
      throwError(() => new Error('Factor scores not available')),
    );
    const api = makePortfolioApiMock();

    await TestBed.configureTestingModule({
      imports: [AttributionComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        ICON_PROVIDER,
        { provide: PortfolioContextService, useValue: ctx },
        { provide: AttributionService, useValue: svc },
        { provide: PortfolioApiService, useValue: api },
      ],
    }).compileComponents();

    const fx = TestBed.createComponent(AttributionComponent);
    fx.detectChanges();

    expect(fx.componentInstance.factorError()).toBeTruthy();
  });
});

// issue #997, 13f: the form hint must read an explicit copy when no portfolio
// is selected — not the stale "(none)" placeholder.
describe('AttributionComponent — no-portfolio form hint (issue #997)', () => {
  it('when no portfolio is selected, the hint reads the exact no-portfolio copy', async () => {
    installResizeObserverStub();
    const ctx = makePortfolioCtxMock(null, null);
    const svc = makeAttrSvcMock();
    const api = makePortfolioApiMock();

    await TestBed.configureTestingModule({
      imports: [AttributionComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        ICON_PROVIDER,
        { provide: PortfolioContextService, useValue: ctx },
        { provide: AttributionService, useValue: svc },
        { provide: PortfolioApiService, useValue: api },
      ],
    }).compileComponents();

    const fx = TestBed.createComponent(AttributionComponent);
    fx.detectChanges();

    const text = (fx.nativeElement as HTMLElement).textContent ?? '';
    expect(text).toContain('No portfolio selected. Select one in the navigation bar.');
  });
});
