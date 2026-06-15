// Source-blind spec — authored from acceptance criteria only (Red phase).
// Criterion (UNIT): `portfolioWeights` signal populates from `getLatestSnapshot(name)`.
// Complements attribution-context-wiring.spec.ts (which tests service call args);
// this file tests the resulting *signal value* on the component instance.

import { NO_ERRORS_SCHEMA, computed, provideZonelessChangeDetection, signal } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { HttpErrorResponse } from '@angular/common/http';
import { of, throwError } from 'rxjs';

import { AttributionComponent } from './attribution';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { PortfolioApiService } from '../core/services/portfolio-api.service';
import { AttributionService } from './attribution.service';
import { ICON_PROVIDER } from '../icons';

// ─── Shared fixtures ─────────────────────────────────────────────────────────

const PORTFOLIO_ID   = 'stub-uuid-attr-1';
const PORTFOLIO_NAME = 'Global Equity Fund';
const SNAPSHOT_WEIGHTS: Record<string, number> = { AAPL: 0.5, MSFT: 0.3, GOOG: 0.2 };

function makeCtx(nameSig: ReturnType<typeof signal<string | null>>) {
  const idSig = signal<string | null>(nameSig() ? PORTFOLIO_ID : null);
  return {
    currentPortfolioId:   idSig,
    currentPortfolioName: computed(() => nameSig()),
    selectedPortfolio:    computed(() => nameSig() ? { id: PORTFOLIO_ID, name: nameSig()! } : null),
    hasPortfolio:         computed(() => nameSig() !== null),
    dateRange: signal({ preset: '1Y' as const, start: new Date('2024-01-01'), end: new Date('2025-01-01') }),
    benchmark:      signal('SPY'),
    activeMode:     signal('backtest' as const),
    isLive:         computed(() => false),
    isBacktest:     computed(() => true),
    isPaper:        computed(() => false),
    dateRangeLabel: computed(() => '1Y'),
    dateRangeDays:  computed(() => 365),
    setPortfolio:   jasmine.createSpy('setPortfolio'),
    setMode:        jasmine.createSpy('setMode'),
    setPreset:      jasmine.createSpy('setPreset'),
    setCustomRange: jasmine.createSpy('setCustomRange'),
    setBenchmark:   jasmine.createSpy('setBenchmark'),
    reset:          jasmine.createSpy('reset'),
    _nameSig: nameSig,
    _idSig:   idSig,
  };
}

function makeAttrSvc() {
  return {
    brinson: jasmine.createSpy('brinson').and.returnValue(of({})),
    factor:  jasmine.createSpy('factor').and.returnValue(of({})),
  };
}

function setup(
  apiReturn: ReturnType<typeof of>,
  nameSig: ReturnType<typeof signal<string | null>>,
): ComponentFixture<AttributionComponent> {
  const portfolioApiMock = {
    getLatestSnapshot: jasmine.createSpy('getLatestSnapshot').and.returnValue(apiReturn),
  };

  TestBed.configureTestingModule({
    imports: [AttributionComponent],
    schemas: [NO_ERRORS_SCHEMA],
    providers: [
      provideZonelessChangeDetection(),
      provideHttpClient(),
      provideHttpClientTesting(),
      ICON_PROVIDER,
      { provide: PortfolioContextService, useValue: makeCtx(nameSig) },
      { provide: PortfolioApiService,     useValue: portfolioApiMock },
      { provide: AttributionService,      useValue: makeAttrSvc() },
    ],
  });

  const f = TestBed.createComponent(AttributionComponent);
  f.detectChanges();
  return f;
}

// ─── Suite 1: signal value — portfolioWeights is populated ───────────────────

describe('AttributionComponent — portfolioWeights() signal (issue #1020)', () => {
  it('when portfolio name is resolved, portfolioWeights() equals the snapshot weights', () => {
    const nameSig = signal<string | null>(PORTFOLIO_NAME);
    const fixture = setup(
      of({ weights: SNAPSHOT_WEIGHTS }) as ReturnType<typeof of>,
      nameSig,
    );

    expect(fixture.componentInstance.portfolioWeights()).toEqual(SNAPSHOT_WEIGHTS);
  });

  it('when getLatestSnapshot errors, portfolioWeights() does not throw and no console.error fires', () => {
    const consoleSpy = spyOn(console, 'error');
    const nameSig = signal<string | null>(PORTFOLIO_NAME);

    expect(() =>
      setup(
        throwError(() => new HttpErrorResponse({ status: 500 })) as ReturnType<typeof of>,
        nameSig,
      ),
    ).not.toThrow();

    expect(consoleSpy).not.toHaveBeenCalled();
  });

  it('when portfolio name changes, portfolioWeights() reflects the new snapshot weights', () => {
    const newWeights: Record<string, number> = { TSLA: 0.7, AMZN: 0.3 };
    const nameSig = signal<string | null>(PORTFOLIO_NAME);

    const portfolioApiMock = {
      getLatestSnapshot: jasmine
        .createSpy('getLatestSnapshot')
        .withArgs(PORTFOLIO_NAME).and.returnValue(of({ weights: SNAPSHOT_WEIGHTS }))
        .withArgs('Other Fund').and.returnValue(of({ weights: newWeights })),
    };

    const ctx = makeCtx(nameSig);
    TestBed.configureTestingModule({
      imports: [AttributionComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        ICON_PROVIDER,
        { provide: PortfolioContextService, useValue: ctx },
        { provide: PortfolioApiService,     useValue: portfolioApiMock },
        { provide: AttributionService,      useValue: makeAttrSvc() },
      ],
    });

    const fixture = TestBed.createComponent(AttributionComponent);
    fixture.detectChanges();
    expect(fixture.componentInstance.portfolioWeights()).toEqual(SNAPSHOT_WEIGHTS);

    nameSig.set('Other Fund');
    ctx._idSig.set('other-uuid');
    fixture.detectChanges();

    expect(fixture.componentInstance.portfolioWeights()).toEqual(newWeights);
  });
});
