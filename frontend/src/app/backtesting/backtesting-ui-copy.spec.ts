/**
 * Source-blind unit tests — UI copy contract for BacktestingComponent
 * (R4 / issue #1027).
 *
 * Criterion [UNIT]: No API endpoint or parameter names appear in any
 * user-facing UI copy.
 *
 * Specifically, the walk-forward section previously read:
 *   "Runs a background job via POST /validate/walk-forward with
 *    cv_type=walk_forward."
 * This text leaks internal implementation details and must be replaced with
 * a user-facing description of what walk-forward validation does.
 *
 * Assumption: BacktestingComponent is at ./backtesting (imported without the
 * .component suffix, matching the existing backtesting-contracts.spec.ts).
 * The testing helper is at ../../testing (shared configureTestBed).
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { BacktestingComponent } from './backtesting';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { ICON_PROVIDER } from '../icons';
import { signal } from '@angular/core';

function makeCtxStub() {
  return {
    currentPortfolioId: signal<string | null>(null),
    currentPortfolioName: signal<string | null>(null),
    selectedPortfolio: signal<unknown>(null),
    dateRange: signal({
      preset: '1Y' as const,
      start: new Date('2024-01-01'),
      end: new Date('2025-01-01'),
    }),
  };
}

describe('BacktestingComponent — walk-forward UI copy contract (issues #1027 / #1030)', () => {
  let fixture: ComponentFixture<BacktestingComponent>;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [BacktestingComponent],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        { provide: PortfolioContextService, useValue: makeCtxStub() },
        ICON_PROVIDER,
      ],
    });
    fixture = TestBed.createComponent(BacktestingComponent);
    http = TestBed.inject(HttpTestingController);
    fixture.detectChanges();
    // Drain any bootstrap requests (e.g. portfolio list auto-fetch).
    http.match(() => true);
  });

  afterEach(() => http.verify());

  it('walk-forward section does not expose the internal endpoint path /validate/walk-forward', () => {
    const text: string = fixture.nativeElement.textContent ?? '';
    expect(text)
      .withContext(
        'User-facing text must not contain the internal API path ' +
          '"/validate/walk-forward" — replace it with a user-facing description'
      )
      .not.toContain('/validate/walk-forward');
  });

  it('walk-forward section does not expose the internal parameter name cv_type', () => {
    const text: string = fixture.nativeElement.textContent ?? '';
    expect(text)
      .withContext(
        'User-facing text must not contain the internal parameter name "cv_type" — ' +
          'use plain language instead'
      )
      .not.toContain('cv_type');
  });

  it('walk-forward section does not start a sentence with "POST /"', () => {
    const text: string = fixture.nativeElement.textContent ?? '';
    expect(text)
      .withContext(
        'User-facing text must not expose raw HTTP method + path combinations ' +
          'such as "POST /..." — replace with a user-facing description'
      )
      .not.toMatch(/\bPOST \/[a-z]/);
  });

  /**
   * Positive assertion (issue #1030, criterion 13):
   * The walk-forward description must contain at least one user-facing term
   * that describes out-of-sample validation in plain language.
   * Assumption: the replacement copy references one of: "out-of-sample",
   * "rolling", "look-ahead", "validation", or "window".
   */
  it('walk-forward section contains user-facing copy describing the validation technique', () => {
    const section: HTMLElement =
      fixture.nativeElement.querySelector('[data-testid="walk-forward-section"]') ??
      fixture.nativeElement;
    const text: string = (section.textContent ?? '').toLowerCase();
    const hasUserFacingTerm =
      text.includes('out-of-sample') ||
      text.includes('rolling') ||
      text.includes('look-ahead') ||
      text.includes('bias') ||
      text.includes('validation') ||
      text.includes('window');
    expect(hasUserFacingTerm)
      .withContext(
        'Walk-forward copy must describe the technique in user-facing terms, ' +
          'not as raw API invocations'
      )
      .toBeTrue();
  });
});

// ─── Global copy guard: no raw API strings anywhere in the component ──────────

describe('BacktestingComponent — no raw API implementation details in any UI text (issue #1030, global criterion)', () => {
  let fixture: ComponentFixture<BacktestingComponent>;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [BacktestingComponent],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        { provide: PortfolioContextService, useValue: makeCtxStub() },
        ICON_PROVIDER,
      ],
    });
    fixture = TestBed.createComponent(BacktestingComponent);
    http = TestBed.inject(HttpTestingController);
    fixture.detectChanges();
    http.match(() => true);
  });

  afterEach(() => http.verify());

  it('the rendered component text does not contain the walk-forward parameter value "walk_forward"', () => {
    // "walk_forward" as a raw snake_case parameter value (cv_type=walk_forward) must not appear in copy.
    const text: string = fixture.nativeElement.textContent ?? '';
    expect(text).not.toContain('cv_type=walk_forward');
  });
});
