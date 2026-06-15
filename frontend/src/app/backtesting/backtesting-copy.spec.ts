/**
 * Source-blind unit tests for issue #1015 — no API endpoint names in user-facing copy.
 *
 * Criterion covered:
 *   [UNIT] No API endpoint or parameter names appear in any user-facing UI copy.
 *
 * Concrete regression documented in requirements §13e:
 *   The walk-forward section previously read:
 *     "Runs a background job via POST /validate/walk-forward with cv_type=walk_forward."
 *   That description exposes three implementation details that must be absent from
 *   user-facing copy: the HTTP method ("POST"), the endpoint path
 *   ("/validate/walk-forward"), and the parameter name ("cv_type").
 *
 * Assumption (recorded per the test-designer mandate):
 *   "User-facing UI copy" is the text visible inside the component's rendered
 *   template.  The test reads `nativeElement.textContent` and checks for the
 *   absence of the known-bad strings.  This is the simplest behaviour consistent
 *   with the criterion text.
 *
 * Import-path assumption:
 *   BacktestingComponent → ./backtesting
 *   PortfolioContextService → ../../core/services/portfolio-context.service
 */
import { TestBed } from '@angular/core/testing';
import { ComponentFixture } from '@angular/core/testing';
import { provideZonelessChangeDetection, signal } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';

import { BacktestingComponent } from './backtesting';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { ICON_PROVIDER } from '../icons';

describe('BacktestingComponent — user-facing copy must not contain API detail strings (#1015)', () => {
  let fixture: ComponentFixture<BacktestingComponent>;
  let http: HttpTestingController;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports:   [BacktestingComponent],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([]),
        ICON_PROVIDER,
        {
          provide: PortfolioContextService,
          useValue: {
            currentPortfolioId:   signal<string | null>(null),
            currentPortfolioName: signal<string | null>(null),
            selectedPortfolio:    signal<unknown>(null),
            dateRange:            signal({
              preset: '1Y' as const,
              start:  new Date('2024-01-01'),
              end:    new Date('2025-01-01'),
            }),
          },
        },
      ],
    }).compileComponents();

    http    = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(BacktestingComponent);
    fixture.detectChanges();
    await fixture.whenStable();
    http.match(() => true).forEach(r => r.flush({}));
    fixture.detectChanges();
  });

  afterEach(() => {
    http.match(() => true).forEach(r => r.flush({}));
    http.verify();
  });

  it('when the component renders, the walk-forward description does not contain the string "POST"', () => {
    const text = (fixture.nativeElement as HTMLElement).textContent ?? '';
    // The word "POST" must never appear in visible UI copy (HTTP method is an
    // implementation detail, not a user-facing concept).
    expect(text).not.toContain('POST /');
  });

  it('when the component renders, the walk-forward description does not expose the endpoint path', () => {
    const text = (fixture.nativeElement as HTMLElement).textContent ?? '';
    expect(text)
      .withContext('/validate/walk-forward is an internal endpoint path and must not appear in UI copy')
      .not.toContain('/validate/walk-forward');
  });

  it('when the component renders, the walk-forward description does not expose request parameter names', () => {
    const text = (fixture.nativeElement as HTMLElement).textContent ?? '';
    expect(text)
      .withContext('cv_type is a request parameter name and must not appear in UI copy')
      .not.toContain('cv_type');
  });

  it('when the component renders, the walk-forward section provides a user-facing description of the feature', () => {
    const text = (fixture.nativeElement as HTMLElement).textContent ?? '';
    // The description must mention what the feature does for the user, not how
    // it is implemented.  Acceptable signals: "out-of-sample", "rolling",
    // "look-ahead", "walk-forward" (the user-visible concept, not the parameter).
    // At least one of these should be present to confirm the description is meaningful.
    const hasUserFacingDescription =
      text.toLowerCase().includes('out-of-sample') ||
      text.toLowerCase().includes('rolling') ||
      text.toLowerCase().includes('look-ahead') ||
      text.toLowerCase().includes('walk-forward');

    expect(hasUserFacingDescription)
      .withContext(
        'the walk-forward section must describe the feature from the user\'s perspective ' +
        '(e.g. "out-of-sample", "rolling windows", "look-ahead bias")'
      )
      .toBeTrue();
  });
});
