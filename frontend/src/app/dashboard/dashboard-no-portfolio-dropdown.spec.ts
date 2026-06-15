/**
 * Source-blind unit test for issue #1015 — no local portfolio dropdown in the dashboard.
 *
 * Criterion covered:
 *   [UNIT] The header <select> is the benchmark selector only; there is no portfolio dropdown.
 *
 * Assumption (recorded per the test-designer mandate):
 *   The requirement for this issue (#1015) is that local per-page portfolio dropdowns are
 *   removed from the dashboard (and other migrated pages).  The criterion text says "The
 *   header <select> is the benchmark selector only", meaning any <select> element on the
 *   dashboard page must not be for portfolio selection — it should only be used for things
 *   like benchmark selection.
 *
 *   Concretely: the dashboard component's rendered template must not contain a <select>
 *   whose visible text, option values, or [name] attribute indicate portfolio selection.
 *   "Portfolio selection" is identified by the substring "portfolio" (case-insensitive)
 *   in the element's label, name, or option texts.
 *
 * Import-path assumption:
 *   DashboardComponent      → ./dashboard
 *   PortfolioContextService → ../../core/services/portfolio-context.service
 */
import { TestBed } from '@angular/core/testing';
import { ComponentFixture } from '@angular/core/testing';
import { provideZonelessChangeDetection, signal } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';

import { DashboardComponent } from './dashboard';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { ICON_PROVIDER } from '../icons';

/** Returns an appropriate stub body for each analytics endpoint URL. */
function stubBody(url: string): Record<string, unknown> {
  if (url.includes('/market/indices'))        return { indices: [], total: 0 };
  if (url.includes('/market/snapshot'))       return { vix: 0, vixChange: 0, sp500Return: 0, tenYearYield: 0, yieldChange: 0, usdIndex: 0, usdChange: 0, asOf: '2026-01-01T00:00:00Z' };
  if (url.includes('/equity-curve'))          return { points: [], portfolioTotalReturn: 0, benchmarkTotalReturn: 0 };
  if (url.includes('/performance-metrics'))   return { kpis: [], nav: 0, navChangePct: 0, currency: 'EUR' };
  if (url.includes('/allocation'))            return { nodes: [], totalPositions: 0, totalSectors: 0 };
  if (url.includes('/asset-class-returns'))   return { returns: [] };
  if (url.includes('/rolling-metrics'))       return { window: 63, sharpe: [], volatility: [], beta: [] };
  return {};
}

describe('DashboardComponent — no local portfolio-selection dropdown (#1015)', () => {
  let fixture: ComponentFixture<DashboardComponent>;
  let http: HttpTestingController;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports:   [DashboardComponent],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([]),
        ICON_PROVIDER,
        {
          provide: PortfolioContextService,
          useValue: {
            currentPortfolioId:   signal<string | null>('port-1'),
            currentPortfolioName: signal<string | null>('ACME Fund'),
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
    fixture = TestBed.createComponent(DashboardComponent);
    fixture.detectChanges();
    await fixture.whenStable();
    http.match(() => true).forEach(r => r.flush(stubBody(r.request.url)));
    fixture.detectChanges();
  });

  afterEach(() => {
    http.match(() => true).forEach(r => r.flush(stubBody(r.request.url)));
    http.verify();
  });

  it('when the dashboard renders, it does not contain a portfolio-selection <select> element', () => {
    const nativeEl = fixture.nativeElement as HTMLElement;
    const allSelects = Array.from(nativeEl.querySelectorAll('select'));

    /** Returns true if a <select> element appears to be for portfolio selection. */
    function isPortfolioSelect(sel: HTMLSelectElement): boolean {
      const name      = (sel.name  ?? '').toLowerCase();
      const id        = (sel.id    ?? '').toLowerCase();
      const ariaLabel = (sel.getAttribute('aria-label') ?? '').toLowerCase();
      const optionTexts = Array.from(sel.options).map(o => o.text.toLowerCase()).join(' ');

      const label = sel.id
        ? (nativeEl.querySelector(`label[for="${sel.id}"]`)?.textContent ?? '').toLowerCase()
        : '';

      return [name, id, ariaLabel, optionTexts, label].some(s => s.includes('portfolio'));
    }

    const portfolioSelects = allSelects.filter(sel => isPortfolioSelect(sel as HTMLSelectElement));

    expect(portfolioSelects.length)
      .withContext(
        'DashboardComponent must not render a local portfolio-selection dropdown — ' +
        'portfolio selection is managed by the global context service only'
      )
      .toBe(0);
  });

  it('when the dashboard renders, any <select> present is for benchmark or other non-portfolio purpose', () => {
    const nativeEl  = fixture.nativeElement as HTMLElement;
    const allSelects = Array.from(nativeEl.querySelectorAll('select'));

    const unknownSelects = allSelects.filter(sel => {
      const combined = [
        sel.name,
        sel.id,
        sel.getAttribute('aria-label') ?? '',
        sel.getAttribute('data-testid') ?? '',
        Array.from((sel as HTMLSelectElement).options).map(o => o.text).join(' '),
      ].join(' ').toLowerCase();

      return combined.includes('portfolio');
    });

    expect(unknownSelects.length)
      .withContext('No <select> in the dashboard may be used for portfolio selection')
      .toBe(0);
  });
});
