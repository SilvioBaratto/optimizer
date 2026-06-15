/**
 * Acceptance criterion [T3]:
 * A 404 from POST /attribution/factor renders an inline
 * "Factor scores not available" message in the factor attribution panel,
 * NOT a generic error toast.  Any other 4xx/5xx still surfaces as a visible
 * error state.
 *
 * Tests at two layers:
 *  1. FactorAttributionPanelComponent — UI contract (notAvailable input → message text)
 *  2. AttributionComponent — state contract (404 response sets factorNotAvailable signal)
 */
import { ComponentFixture, TestBed } from '@angular/core/testing';
import {
  NO_ERRORS_SCHEMA,
  provideZonelessChangeDetection,
  signal,
  computed,
} from '@angular/core';
import { HttpErrorResponse } from '@angular/common/http';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting, HttpTestingController } from '@angular/common/http/testing';
import { of, throwError } from 'rxjs';

import { FactorAttributionPanelComponent } from './factor-attribution-panel/factor-attribution-panel';
import { AttributionComponent } from './attribution';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { PortfolioApiService } from '../core/services/portfolio-api.service';
import { AttributionService } from './attribution.service';
import { ICON_PROVIDER } from '../icons';

// ─── Layer 1: FactorAttributionPanelComponent UI contract ────────────────────

describe('FactorAttributionPanelComponent — notAvailable=true shows "Factor scores not available"', () => {
  let fixture: ComponentFixture<FactorAttributionPanelComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [FactorAttributionPanelComponent],
      providers: [provideZonelessChangeDetection(), ICON_PROVIDER],
      schemas: [NO_ERRORS_SCHEMA],
    }).compileComponents();

    fixture = TestBed.createComponent(FactorAttributionPanelComponent);
  });

  it('when notAvailable input is true, "Factor scores not available" text appears in the DOM', () => {
    fixture.componentRef.setInput('notAvailable', true);
    fixture.detectChanges();

    const text = (fixture.nativeElement as HTMLElement).textContent ?? '';
    expect(text.toLowerCase()).toContain('factor scores not available');
  });

  it('when notAvailable input is false and no response, "Factor scores not available" text is absent', () => {
    fixture.componentRef.setInput('notAvailable', false);
    fixture.componentRef.setInput('response', null);
    fixture.detectChanges();

    const text = (fixture.nativeElement as HTMLElement).textContent ?? '';
    expect(text.toLowerCase()).not.toContain('factor scores not available');
  });
});

// ─── Layer 2: AttributionComponent state after factor 404 ────────────────────

function makeCtx(name = 't212', id = 'stub-uuid-attr') {
  const $name = signal<string | null>(name);
  const $id   = signal<string | null>(id);
  return {
    currentPortfolioId:   $id,
    currentPortfolioName: computed(() => $name()),
    selectedPortfolio:    computed(() => ($name() ? { id: $id() ?? '', name: $name()! } : null)),
    hasPortfolio:         computed(() => $id() !== null),
    dateRange:            signal({ preset: '1Y' as const, start: new Date('2024-01-01'), end: new Date('2025-01-01') }),
    benchmark:            signal('SPY'),
    activeMode:           signal('backtest' as const),
    isLive:               computed(() => false),
    isBacktest:           computed(() => true),
    isPaper:              computed(() => false),
    dateRangeLabel:       computed(() => '1Y'),
    dateRangeDays:        computed(() => 365),
    setPortfolio:         jasmine.createSpy('setPortfolio'),
    setMode:              jasmine.createSpy('setMode'),
    setPreset:            jasmine.createSpy('setPreset'),
    setCustomRange:       jasmine.createSpy('setCustomRange'),
    setBenchmark:         jasmine.createSpy('setBenchmark'),
    reset:                jasmine.createSpy('reset'),
  };
}

const STUB_SNAPSHOT = { id: '1', portfolioName: 't212', date: '2025-01-01', weights: { AAPL: 0.5, MSFT: 0.5 } };
const STUB_BRINSON  = { totalActiveReturn: 0.01, totalAllocation: 0.005, totalSelection: 0.005, sectors: [] };

describe('AttributionComponent — factor 404 sets factorNotAvailable signal (issue #1016)', () => {
  let fixture: ComponentFixture<AttributionComponent>;
  let comp: AttributionComponent;
  let http: HttpTestingController;

  function flushAll(body: Record<string, unknown> = {}): void {
    http.match(() => true).forEach(r => r.flush(body));
  }

  function flushSnapshot(): void {
    http
      .match(r => r.url.includes('/snapshots/latest'))
      .forEach(r => r.flush(STUB_SNAPSHOT));
  }

  function flushBrinson(status = 200): void {
    http
      .match(r => r.url.includes('/attribution/brinson') && r.method === 'POST')
      .forEach(r =>
        status === 200
          ? r.flush(STUB_BRINSON)
          : r.flush({ detail: 'err' }, { status, statusText: 'Error' }),
      );
  }

  function flushFactor(status: number): void {
    http
      .match(r => r.url.includes('/attribution/factor') && r.method === 'POST')
      .forEach(r => {
        if (status === 200) {
          r.flush({ portfolioReturn: 0.1, explainedReturn: 0.09, residual: 0.01, factors: [] });
        } else {
          r.flush({ detail: 'not found' }, { status, statusText: 'Error' });
        }
      });
  }

  beforeEach(async () => {
    const ctx = makeCtx();
    await TestBed.configureTestingModule({
      imports: [AttributionComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        ICON_PROVIDER,
        { provide: PortfolioContextService, useValue: ctx },
      ],
    }).compileComponents();

    http = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(AttributionComponent);
    comp    = fixture.componentInstance;
  });

  afterEach(() => http.verify());

  it('when factor endpoint returns 404, factorNotAvailable signal becomes true', () => {
    fixture.detectChanges();
    flushSnapshot();
    flushBrinson();
    flushFactor(404);
    flushAll();
    fixture.detectChanges();

    expect(comp.factorNotAvailable()).toBeTrue();
  });

  it('when factor endpoint returns 404, factorError signal remains null', () => {
    fixture.detectChanges();
    flushSnapshot();
    flushBrinson();
    flushFactor(404);
    flushAll();
    fixture.detectChanges();

    expect(comp.factorError()).toBeNull();
  });

  it('when factor endpoint returns 500, factorNotAvailable is false and factorError is set', () => {
    fixture.detectChanges();
    flushSnapshot();
    flushBrinson();
    flushFactor(500);
    flushAll();
    fixture.detectChanges();

    expect(comp.factorNotAvailable()).toBeFalse();
    expect(comp.factorError()).toBeTruthy();
  });

  it('when factor endpoint returns 200, factorNotAvailable is false', () => {
    fixture.detectChanges();
    flushSnapshot();
    flushBrinson();
    flushFactor(200);
    flushAll();
    fixture.detectChanges();

    expect(comp.factorNotAvailable()).toBeFalse();
  });
});
