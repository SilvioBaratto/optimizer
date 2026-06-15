/**
 * Source-blind spec — authored from acceptance criteria only (Red phase).
 *
 * Criterion (UNIT / issue #1020):
 *   No API endpoint or parameter names appear in any user-facing UI copy.
 *
 * The walk-forward description ("Runs a background job via POST /validate/walk-forward
 * with cv_type=walk_forward") was flagged as an example of leaked implementation detail
 * (requirements §13e). This suite verifies the Attribution page does not commit a similar
 * mistake by rendering raw endpoint paths or HTTP-method prefixes in its template.
 */

import {
  NO_ERRORS_SCHEMA,
  computed,
  provideZonelessChangeDetection,
  signal,
} from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { of } from 'rxjs';

import { AttributionComponent } from './attribution';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { PortfolioApiService } from '../core/services/portfolio-api.service';
import { AttributionService } from './attribution.service';
import { ICON_PROVIDER } from '../icons';

// Strings that must never appear as literal text in the rendered template.
// Each entry is a raw endpoint fragment or HTTP-detail string that exposes
// backend implementation to users.
const FORBIDDEN_COPY = [
  '/attribution/brinson',
  '/attribution/factor',
  'POST /attribution',
  'GET /attribution',
  'cv_type=',
  'portfolio_weights',  // Pydantic field name
  'start_date',         // Pydantic field name
  'end_date',           // Pydantic field name
];

function makeNullCtx() {
  return {
    currentPortfolioId:   signal<string | null>(null),
    currentPortfolioName: computed(() => null),
    selectedPortfolio:    computed(() => null),
    hasPortfolio:         computed(() => false),
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
  };
}

describe('AttributionComponent — no API endpoint names in UI copy (issue #1020)', () => {
  let el: HTMLElement;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AttributionComponent],
      schemas: [NO_ERRORS_SCHEMA],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        ICON_PROVIDER,
        { provide: PortfolioContextService, useValue: makeNullCtx() },
        {
          provide: PortfolioApiService,
          useValue: { getLatestSnapshot: jasmine.createSpy().and.returnValue(of({ weights: {} })) },
        },
        {
          provide: AttributionService,
          useValue: {
            brinson: jasmine.createSpy('brinson').and.returnValue(of({})),
            factor:  jasmine.createSpy('factor').and.returnValue(of({})),
          },
        },
      ],
    }).compileComponents();

    const fixture = TestBed.createComponent(AttributionComponent);
    fixture.detectChanges();
    el = fixture.nativeElement as HTMLElement;
  });

  FORBIDDEN_COPY.forEach((pattern) => {
    it(`UI copy must not contain the implementation detail "${pattern}"`, () => {
      // textContent strips HTML tags and gives the visible text users see
      expect(el.textContent).not.toContain(
        pattern,
        `User-facing text must not expose the backend detail "${pattern}"`,
      );
    });
  });
});
