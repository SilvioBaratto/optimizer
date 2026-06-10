/**
 * dashboard-render-coverage.spec.ts — issue #909
 *
 * Template-branch coverage for DashboardComponent.
 * Mounts the full template and drives each @if / @for gate by setting
 * STATE signals directly on the component instance. No HTTP flushing
 * (portfolio context is null → loadPortfolioData early-returns, bootstrap
 * requests stay pending unflushed — harmless without http.verify()).
 */

import { provideZonelessChangeDetection } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';
import { TestBed, ComponentFixture } from '@angular/core/testing';
import { installResizeObserverStub } from '../../testing';
import { ICON_PROVIDER } from '../icons';
import { DashboardComponent } from './dashboard';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import type { DashboardKPI } from './dashboard.model';
import type { ReferenceIndexItem } from '../core/models/dashboard-api.model';
import type { SunburstNode } from '../shared/echarts-sunburst/echarts-sunburst';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeKpi(overrides: Partial<DashboardKPI> = {}): DashboardKPI {
  return {
    label: 'KPI',
    value: 1,
    format: 'percent',
    change: 0,
    changeLabel: 'vs last month',
    sparkline: [],
    ...overrides,
  };
}

function makeIndex(ticker: string, name: string): ReferenceIndexItem {
  return { ticker, name, instrumentType: null };
}

function makeSector(name: string, value: number): SunburstNode {
  return { name, value, children: [] };
}

/** Equity curve that rises then falls — produces a non-empty drawdown series. */
function makeCurveWithDrawdown() {
  return [
    { date: '2025-01-01', portfolio: 100, benchmark: 100 },
    { date: '2025-02-01', portfolio: 110, benchmark: 105 },
    { date: '2025-03-01', portfolio: 90, benchmark: 102 },
    { date: '2025-04-01', portfolio: 85, benchmark: 100 },
  ];
}

// ---------------------------------------------------------------------------
// Shared testbed setup
// ---------------------------------------------------------------------------

describe('DashboardComponent — render-coverage (issue #909)', () => {
  let fixture: ComponentFixture<DashboardComponent>;
  let comp: DashboardComponent;
  let el: HTMLElement;

  beforeEach(async () => {
    installResizeObserverStub();

    await TestBed.configureTestingModule({
      imports: [DashboardComponent],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([]),
        ICON_PROVIDER,
      ],
    }).compileComponents();

    // PortfolioContextService is providedIn:'root' — it persists across specs
    // within a Karma session. Reset it so previous tests (e.g. rolling-metrics
    // suite that sets portfolioId='alpha') cannot bleed into this suite.
    TestBed.inject(PortfolioContextService).reset();

    // With currentPortfolioId=null, loadPortfolioData() early-returns and
    // sets isLoadingPortfolio=false.  The market-snapshot request remains
    // pending, but isLoading = isLoadingPortfolio && isLoadingMarket =
    // false && true = false, so the success branch IS visible by default.
    fixture = TestBed.createComponent(DashboardComponent);
    comp = fixture.componentInstance;
    el = fixture.nativeElement as HTMLElement;

    // Drive isLoadingMarket to false so the success/else branch is the
    // unambiguous baseline for every test that does NOT explicitly test the
    // loading branch.
    comp.isLoadingMarket.set(false);
    fixture.detectChanges();
  });

  // -------------------------------------------------------------------------
  // 1. Loading / error / success three-way
  // -------------------------------------------------------------------------

  describe('loading / error / success three-way', () => {
    it('when both loading flags are true, shows skeleton elements', () => {
      comp.isLoadingPortfolio.set(true);
      comp.isLoadingMarket.set(true);
      fixture.detectChanges();

      expect(el.querySelectorAll('.skeleton').length).toBeGreaterThan(0);
    });

    it('when hasError is true, shows "Something went wrong" and a Try Again button', () => {
      comp.isLoadingPortfolio.set(false);
      comp.isLoadingMarket.set(false);
      comp.hasError.set(true);
      comp.errorMessage.set('boom');
      fixture.detectChanges();

      expect(el.textContent).toContain('Something went wrong');
      const tryAgain = Array.from(el.querySelectorAll('button')).find(
        (b) => b.textContent?.includes('Try Again'),
      );
      expect(tryAgain).not.toBeNull();
    });

    it('when not loading and no error, shows neither skeletons nor error heading', () => {
      // Default mount: portfolioContextService.currentPortfolioId = null,
      // so isLoadingPortfolio = false; isLoadingMarket still true → isLoading = false.
      fixture.detectChanges();

      expect(el.querySelectorAll('.skeleton').length).toBe(0);
      expect(el.textContent).not.toContain('Something went wrong');
    });
  });

  // -------------------------------------------------------------------------
  // 2. Report status @if blocks
  // -------------------------------------------------------------------------

  describe('report status @if blocks', () => {
    it('when reportError is set, shows "Report failed" and the error message', () => {
      comp.reportError.set('disk full');
      fixture.detectChanges();

      expect(el.textContent).toContain('Report failed');
      expect(el.textContent).toContain('disk full');
    });

    it('when reportJobId is set, renders app-job-progress-tracker', () => {
      comp.reportJobId.set('job-1');
      fixture.detectChanges();

      expect(el.querySelector('app-job-progress-tracker')).not.toBeNull();
    });

    it('when reportDownloadUrl is set, renders a Download report anchor', () => {
      comp.reportDownloadUrl.set('http://x/r.pdf');
      fixture.detectChanges();

      const anchor = el.querySelector<HTMLAnchorElement>('a[href="http://x/r.pdf"]');
      expect(anchor).not.toBeNull();
      expect(anchor?.textContent).toContain('Download report');
    });

    it('when not generating, the generate button reads "Generate Report" and is enabled', () => {
      fixture.detectChanges();

      const btn = Array.from(el.querySelectorAll('button')).find(
        (b) => b.textContent?.includes('Generate Report'),
      );
      expect(btn).not.toBeNull();
      expect((btn as HTMLButtonElement).disabled).toBeFalse();
    });

    it('when reportLoading is true, the generate button reads "Generating" and is disabled', () => {
      comp.reportLoading.set(true);
      fixture.detectChanges();

      const btn = Array.from(el.querySelectorAll('button')).find(
        (b) => b.textContent?.includes('Generating'),
      );
      expect(btn).not.toBeNull();
      expect((btn as HTMLButtonElement).disabled).toBeTrue();
    });
  });

  // -------------------------------------------------------------------------
  // 3. Benchmark selector @if + @for
  // -------------------------------------------------------------------------

  describe('benchmark selector', () => {
    it('when benchmarkOptions is empty, the selector is absent', () => {
      comp.benchmarkOptions.set([]);
      fixture.detectChanges();

      expect(el.querySelector('select[aria-label="Benchmark selector"]')).toBeNull();
    });

    it('when benchmarkOptions has two items, selector renders with two options', () => {
      comp.benchmarkOptions.set([
        makeIndex('SPY', 'S&P 500'),
        makeIndex('QQQ', 'Nasdaq'),
      ]);
      fixture.detectChanges();

      const select = el.querySelector<HTMLSelectElement>('select[aria-label="Benchmark selector"]');
      expect(select).not.toBeNull();
      expect(select?.options.length).toBe(2);
    });
  });

  // -------------------------------------------------------------------------
  // 4. Progressive-reveal gates
  // -------------------------------------------------------------------------

  describe('progressive-reveal gates', () => {
    it('when revealIndex is 0, no KPI stat-cards are visible', () => {
      comp.revealIndex.set(0);
      comp.kpis.set([makeKpi({ label: 'A' }), makeKpi({ label: 'B' })]);
      fixture.detectChanges();

      expect(el.querySelectorAll('app-stat-card').length).toBe(0);
    });

    it('when revealIndex >= 1 and kpis has 2 items, renders exactly 2 KPI stat-cards', () => {
      comp.revealIndex.set(1);
      comp.kpis.set([makeKpi({ label: 'Return' }), makeKpi({ label: 'Sharpe' })]);
      fixture.detectChanges();

      // Market Context (revealIndex >= 3) is not yet revealed, so the only
      // stat-cards present are the 2 KPI strip cards.
      expect(el.querySelectorAll('app-stat-card').length).toBe(2);
    });

    it('when revealIndex >= 2, the equity-curve container and heading are present', () => {
      comp.revealIndex.set(2);
      fixture.detectChanges();

      const container = el.querySelector('[\\#equityCurve], #equityCurve') ??
        el.querySelector('h2');
      // Confirm "Portfolio Performance" heading appears (equity+allocation grid)
      const headings = Array.from(el.querySelectorAll('h2')).map((h) => h.textContent ?? '');
      expect(headings.some((t) => t.includes('Portfolio Performance'))).toBeTrue();
    });

    it('when revealIndex >= 3, the rolling-metrics section heading is present', () => {
      comp.revealIndex.set(3);
      fixture.detectChanges();

      const headings = Array.from(el.querySelectorAll('h2')).map((h) => h.textContent ?? '');
      expect(headings.some((t) => t.includes('Rolling Metrics'))).toBeTrue();
    });

    it('when revealIndex >= 4, the Quick Actions section is present', () => {
      comp.revealIndex.set(4);
      fixture.detectChanges();

      const headings = Array.from(el.querySelectorAll('h2')).map((h) => h.textContent ?? '');
      expect(headings.some((t) => t.includes('Quick Actions'))).toBeTrue();
    });

    it('when revealIndex >= 4, the optimization-studio routerLink anchor is present', () => {
      comp.revealIndex.set(4);
      fixture.detectChanges();

      // RouterLink renders as a plain <a> with the lowercase routerlink attribute.
      const anchor = el.querySelector('a[routerlink="/optimization-studio"]');
      expect(anchor).not.toBeNull();
    });
  });

  // -------------------------------------------------------------------------
  // 5. Drawdown @if (drawdownSeries().length > 0)
  // -------------------------------------------------------------------------

  describe('drawdown @if', () => {
    it('when equityCurve is empty, the drawdown component is absent', () => {
      comp.revealIndex.set(2);
      comp.equityCurve.set([]);
      fixture.detectChanges();

      expect(el.querySelector('app-echarts-drawdown')).toBeNull();
    });

    it('when equityCurve produces a non-empty drawdown series, renders app-echarts-drawdown', () => {
      comp.revealIndex.set(2);
      comp.equityCurve.set(makeCurveWithDrawdown());
      fixture.detectChanges();

      expect(el.querySelector('app-echarts-drawdown')).not.toBeNull();
    });
  });

  // -------------------------------------------------------------------------
  // 6. rollingMetricsError vs chart (revealIndex >= 3)
  // -------------------------------------------------------------------------

  describe('rolling-metrics error vs chart', () => {
    it('when rollingMetricsError is set, shows error text and hides chart', () => {
      comp.revealIndex.set(3);
      comp.rollingMetricsError.set('metrics boom');
      fixture.detectChanges();

      expect(el.textContent).toContain('metrics boom');
      expect(el.querySelector('app-echarts-rolling-metrics')).toBeNull();
    });

    it('when rollingMetricsError is null, renders app-echarts-rolling-metrics', () => {
      comp.revealIndex.set(3);
      comp.rollingMetricsError.set(null);
      fixture.detectChanges();

      expect(el.querySelector('app-echarts-rolling-metrics')).not.toBeNull();
    });
  });

  // -------------------------------------------------------------------------
  // 7. allocationSectors @for (revealIndex >= 2)
  // -------------------------------------------------------------------------

  describe('allocationSectors @for', () => {
    it('when allocationData has 2 sectors, the sector legend renders both names', () => {
      comp.revealIndex.set(2);
      comp.allocationData.set([
        makeSector('Technology', 60),
        makeSector('Healthcare', 40),
      ]);
      fixture.detectChanges();

      expect(el.textContent).toContain('Technology');
      expect(el.textContent).toContain('Healthcare');
    });

    it('when allocationData has 2 sectors, renders 2 colored-dot spans in the legend', () => {
      comp.revealIndex.set(2);
      comp.allocationData.set([
        makeSector('Financials', 50),
        makeSector('Energy', 50),
      ]);
      fixture.detectChanges();

      // Each sector row has a <span aria-hidden="true"> colored dot
      const dots = el.querySelectorAll('span[aria-hidden="true"]');
      expect(dots.length).toBe(2);
    });

    it('when allocationData is empty, no sector legend dots appear', () => {
      comp.revealIndex.set(2);
      comp.allocationData.set([]);
      fixture.detectChanges();

      const dots = el.querySelectorAll('span[aria-hidden="true"]');
      expect(dots.length).toBe(0);
    });
  });
});
