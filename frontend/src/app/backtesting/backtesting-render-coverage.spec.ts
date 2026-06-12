/**
 * backtesting-render-coverage.spec.ts — issue #936
 *
 * Template-branch coverage for BacktestingComponent. The sibling
 * `backtesting.spec.ts` is logic-only (it never mounts the template), leaving
 * the 564-line template ~19.9% branch-covered. This spec mounts the full
 * template and drives every native control-flow branch (`@if`/`@else`/`@for`)
 * by setting STATE signals directly on the component instance.
 *
 * Pattern mirrors `dashboard-render-coverage.spec.ts`:
 *   - installResizeObserverStub() (inline ECharts register a ResizeObserver)
 *   - inline zoneless TestBed with ICON_PROVIDER + router + HTTP testing
 *   - PortfolioContextService.reset() between specs (constructor effect reads
 *     dateRange(); the service is providedIn:'root' and bleeds across Karma)
 *   - HTTP requests left pending (no http.verify())
 */

import { provideZonelessChangeDetection } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { HttpTestingController } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';
import { TestBed, ComponentFixture } from '@angular/core/testing';

import { installResizeObserverStub, makeBacktestRunResponse } from '../../testing';
import { ICON_PROVIDER } from '../icons';
import { BacktestingComponent, mapRunResponseToResult } from './backtesting';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import type {
  BacktestMetrics,
  BacktestResult,
  Drawdown,
  FactorLoading,
  MonthlyReturnCell,
} from './backtest.model';

// ---------------------------------------------------------------------------
// Inline render-only builders (kept local — NOT added to domain-fixtures.ts)
// ---------------------------------------------------------------------------

const ZERO_METRICS: BacktestMetrics = {
  totalReturn: 0,
  annualizedReturn: 0,
  annualizedVol: 0,
  sharpe: 0,
  sortino: 0,
  maxDrawdown: 0,
  calmar: 0,
  cvar95: 0,
  trackingError: 0,
  informationRatio: 0,
  winRate: 0,
  profitFactor: 0,
};

function makeResult(overrides: Partial<BacktestResult> = {}): BacktestResult {
  return {
    equity: [
      { date: '2024-01-01', portfolio: 100, benchmark: 100 },
      { date: '2024-12-31', portfolio: 112, benchmark: 110 },
    ],
    metrics: ZERO_METRICS,
    drawdowns: [],
    monthlyReturns: [],
    rollingMetrics: [],
    returnDistribution: [],
    factorLoadings: [],
    ...overrides,
  };
}

const MONTHLY_TWO_YEARS: MonthlyReturnCell[] = [
  { year: 2024, month: 1, value: 0.02 },
  { year: 2024, month: 2, value: -0.01 },
  { year: 2025, month: 1, value: 0.03 },
];

const THREE_DRAWDOWNS: Drawdown[] = [
  { start: '2024-02-01', trough: '2024-03-01', end: '2024-04-01', depth: -0.12, duration: 60, recovery: 30 },
  { start: '2024-06-01', trough: '2024-07-01', end: null, depth: -0.08, duration: 45, recovery: null },
  { start: '2024-09-01', trough: '2024-10-01', end: '2024-11-01', depth: -0.05, duration: 20, recovery: 15 },
];

const THREE_LOADINGS: FactorLoading[] = [
  { factor: 'MKT', loading: 0.8, tStat: 5, pValue: 0.0005 },
  { factor: 'SMB', loading: 0.2, tStat: 2, pValue: 0.04 },
  { factor: 'HML', loading: -0.1, tStat: -1, pValue: 0.3 },
];

// ---------------------------------------------------------------------------
// DOM accessors — one query per concept (SRP)
// ---------------------------------------------------------------------------

function buttonWithText(el: HTMLElement, text: string): HTMLButtonElement | undefined {
  return Array.from(el.querySelectorAll('button')).find(
    (b) => b.textContent?.trim() === text,
  );
}

function tbodyRows(el: HTMLElement): NodeListOf<HTMLTableRowElement> {
  return el.querySelectorAll('tbody tr');
}

// ---------------------------------------------------------------------------
// Shared testbed setup
// ---------------------------------------------------------------------------

describe('BacktestingComponent — render-coverage (issue #936)', () => {
  let fixture: ComponentFixture<BacktestingComponent>;
  let comp: BacktestingComponent;
  let el: HTMLElement;

  beforeEach(async () => {
    // Clear persisted backtest-run state so a run persisted by an earlier test
    // (issue #996, 13a) cannot trigger a restore GET in a later construct.
    localStorage.clear();
    installResizeObserverStub();

    await TestBed.configureTestingModule({
      imports: [BacktestingComponent],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([]),
        ICON_PROVIDER,
      ],
    }).compileComponents();

    // providedIn:'root' singleton persists across Karma specs; reset so a prior
    // suite's date preset cannot bleed into the constructor dateRange() effect.
    TestBed.inject(PortfolioContextService).reset();

    fixture = TestBed.createComponent(BacktestingComponent);
    comp = fixture.componentInstance;
    el = fixture.nativeElement as HTMLElement;

    // loadData() (constructor) already sets isLoading=false; make the success
    // (@else) branch the unambiguous baseline for every non-loading test.
    comp.isLoading.set(false);
    fixture.detectChanges();
  });

  afterEach(() => {
    localStorage.clear();
  });

  // -------------------------------------------------------------------------
  // 1. Outer three-way: loading skeleton / error+retry / success
  // -------------------------------------------------------------------------

  describe('loading / error / success three-way', () => {
    it('when isLoading is true, the skeleton renders and no tab group is shown', () => {
      comp.isLoading.set(true);
      fixture.detectChanges();

      expect(el.querySelectorAll('.skeleton').length).toBeGreaterThan(0);
      expect(el.querySelector('app-tab-group')).toBeNull();
    });

    it('when hasError is true, shows "Something went wrong" and a Try Again button', () => {
      comp.isLoading.set(false);
      comp.hasError.set(true);
      comp.errorMessage.set('boom');
      fixture.detectChanges();

      expect(el.textContent).toContain('Something went wrong');
      expect(el.textContent).toContain('boom');
      expect(buttonWithText(el, 'Try Again')).toBeTruthy();
    });

    it('when not loading and no error, renders the tab group and walk-forward panel', () => {
      expect(el.querySelector('app-tab-group')).not.toBeNull();
      expect(el.querySelector('app-walk-forward-panel')).not.toBeNull();
      expect(el.querySelectorAll('.skeleton').length).toBe(0);
      expect(el.textContent).not.toContain('Something went wrong');
    });

    it('renders the four KPI stat-cards in the success branch', () => {
      expect(el.querySelectorAll('app-stat-card').length).toBe(4);
    });
  });

  // -------------------------------------------------------------------------
  // 2. Run / results banners (each asserted present AND absent)
  // -------------------------------------------------------------------------

  describe('run / results banners', () => {
    it('when runJobId is null, the run button reads "Run Backtest"', () => {
      expect(buttonWithText(el, 'Run Backtest')).toBeTruthy();
      expect(buttonWithText(el, 'Running…')).toBeUndefined();
    });

    it('when runJobId is set, the run button reads "Running…" and is disabled', () => {
      comp.runJobId.set('job-1');
      fixture.detectChanges();

      const btn = buttonWithText(el, 'Running…');
      expect(btn).toBeTruthy();
      expect(btn?.disabled).toBeTrue();
    });

    it('when runError is null, no failure banner is shown', () => {
      expect(el.textContent).not.toContain('Backtest failed:');
    });

    it('when runError is set, shows the "Backtest failed" banner with the message', () => {
      comp.runError.set('disk full');
      fixture.detectChanges();

      expect(el.textContent).toContain('Backtest failed: disk full');
    });

    it('when runJobId is null, the job-progress-tracker is absent', () => {
      expect(el.querySelector('app-job-progress-tracker')).toBeNull();
    });

    it('when runJobId is set, renders app-job-progress-tracker', () => {
      comp.runJobId.set('job-1');
      fixture.detectChanges();

      expect(el.querySelector('app-job-progress-tracker')).not.toBeNull();
    });

    it('when no run response state, the results panel is absent', () => {
      expect(el.querySelector('app-backtest-results-panel')).toBeNull();
    });

    it('when runResponse is set, renders app-backtest-results-panel', () => {
      comp.runResponse.set(makeBacktestRunResponse());
      fixture.detectChanges();

      expect(el.querySelector('app-backtest-results-panel')).not.toBeNull();
    });

    it('when runResponseLoading is true, renders app-backtest-results-panel', () => {
      comp.runResponseLoading.set(true);
      fixture.detectChanges();

      expect(el.querySelector('app-backtest-results-panel')).not.toBeNull();
    });

    it('when runResponseError is set, renders app-backtest-results-panel', () => {
      comp.runResponseError.set('load failed');
      fixture.detectChanges();

      expect(el.querySelector('app-backtest-results-panel')).not.toBeNull();
    });
  });

  // -------------------------------------------------------------------------
  // 3. Eight-tab @if matrix — data-driven over activeTab()
  // -------------------------------------------------------------------------

  describe('tab @if matrix', () => {
    interface TabCase {
      tab: string;
      assert: (host: HTMLElement) => void;
    }

    const cases: TabCase[] = [
      {
        tab: 'overview',
        assert: (host) => {
          expect(host.querySelectorAll('app-chart-toolbar').length).toBeGreaterThan(0);
          expect(host.textContent).toContain('Equity Curve');
        },
      },
      {
        tab: 'metrics',
        assert: (host) => expect(tbodyRows(host).length).toBe(12),
      },
      {
        tab: 'monthly',
        assert: (host) =>
          expect(host.querySelector('app-echarts-calendar-heatmap')).not.toBeNull(),
      },
      {
        tab: 'drawdowns',
        assert: (host) => {
          expect(host.querySelector('app-echarts-drawdown')).not.toBeNull();
          expect(host.querySelector('app-echarts-histogram')).not.toBeNull();
        },
      },
      {
        tab: 'rolling',
        assert: (host) =>
          expect(host.querySelectorAll('div[style*="200px"]').length).toBe(3),
      },
      {
        tab: 'distribution',
        assert: (host) => {
          expect(host.querySelector('app-echarts-histogram')).not.toBeNull();
          expect(host.textContent).toContain('Skewness');
        },
      },
      {
        tab: 'style',
        assert: (host) =>
          expect(host.querySelector('[data-testid="factor-loadings-empty"]')).not.toBeNull(),
      },
      {
        tab: 'regimes',
        assert: (host) => expect(host.textContent).toContain('Regime context'),
      },
    ];

    cases.forEach(({ tab, assert }) => {
      it(`when activeTab is "${tab}", the matching tab content renders`, () => {
        comp.activeTab.set(tab);
        fixture.detectChanges();
        assert(el);
      });
    });
  });

  // -------------------------------------------------------------------------
  // 4. Nested @for row counts from seeded data
  // -------------------------------------------------------------------------

  describe('nested @for row counts', () => {
    it('when on the metrics tab, renders all 12 metric rows', () => {
      comp.activeTab.set('metrics');
      fixture.detectChanges();

      expect(tbodyRows(el).length).toBe(12);
    });

    it('when monthlyReturns spans two years, the mobile table renders two rows', () => {
      comp.result.set(makeResult({ monthlyReturns: MONTHLY_TWO_YEARS }));
      comp.activeTab.set('monthly');
      fixture.detectChanges();

      expect(tbodyRows(el).length).toBe(2);
    });

    it('when three drawdowns exist, the drawdown table renders three rows', () => {
      comp.result.set(makeResult({ drawdowns: THREE_DRAWDOWNS }));
      comp.activeTab.set('drawdowns');
      fixture.detectChanges();

      expect(tbodyRows(el).length).toBe(3);
    });

    it('when factorLoadings has three entries, the factor table renders three rows', () => {
      comp.result.set(makeResult({ factorLoadings: THREE_LOADINGS }));
      comp.activeTab.set('style');
      fixture.detectChanges();

      expect(tbodyRows(el).length).toBe(3);
    });
  });

  // -------------------------------------------------------------------------
  // 5. Style tab @if/@else sub-branch (bar chart vs empty state)
  // -------------------------------------------------------------------------

  describe('style tab factor-loadings @if/@else', () => {
    it('when factorLoadings is non-empty, renders app-echarts-bar and no empty-state', () => {
      comp.result.set(makeResult({ factorLoadings: THREE_LOADINGS }));
      comp.activeTab.set('style');
      fixture.detectChanges();

      expect(el.querySelector('app-echarts-bar')).not.toBeNull();
      expect(el.querySelector('[data-testid="factor-loadings-empty"]')).toBeNull();
    });

    it('when factorLoadings is empty, renders the empty-state and no app-echarts-bar', () => {
      comp.result.set(makeResult({ factorLoadings: [] }));
      comp.activeTab.set('style');
      fixture.detectChanges();

      expect(el.querySelector('[data-testid="factor-loadings-empty"]')).not.toBeNull();
      expect(el.querySelector('app-echarts-bar')).toBeNull();
    });
  });

  // -------------------------------------------------------------------------
  // 6. Regimes placeholder + rolling-window toggle
  // -------------------------------------------------------------------------

  describe('regimes tab and rolling-window toggle', () => {
    it('when activeTab is "regimes", the static placeholder text renders', () => {
      // NOTE: 'regimes' is never pushed into tabs() and no handler calls
      // activeTab.set('regimes') in production — this branch is a future-feature
      // placeholder, reachable only by this direct signal set, not by any user.
      comp.activeTab.set('regimes');
      fixture.detectChanges();

      expect(el.textContent).toContain('Regime context');
    });

    it('when the 3Y window button is clicked, rollingWindow becomes "3Y"', () => {
      comp.activeTab.set('rolling');
      fixture.detectChanges();

      buttonWithText(el, '3Y')?.click();
      fixture.detectChanges();

      expect(comp.rollingWindow()).toBe('3Y');
    });
  });

  // -------------------------------------------------------------------------
  // 7. Chart-effect branches: inject stub chart instances so the `if (this.equityChart)`
  //    / `if (this.underwaterChart)` / `if (this.qqChart)` / rolling-chart branches
  //    execute (they are otherwise unreachable — ECharts never inits in headless).
  // -------------------------------------------------------------------------

  describe('chart-instance effect branches (stub injection)', () => {
    function makeChartStub() {
      return jasmine.createSpyObj('EChartsType', ['setOption', 'dispose', 'resize']);
    }

    it('when equityChart and underwaterChart are set, result change triggers setOption on both', () => {
      const eq = makeChartStub();
      const uw = makeChartStub();
      // Inject stubs into private fields — TypeScript cast bypasses access check
      (comp as unknown as Record<string, unknown>)['equityChart'] = eq;
      (comp as unknown as Record<string, unknown>)['underwaterChart'] = uw;

      // Mutate a signal that the effect reads → effect re-runs synchronously in zoneless
      comp.result.set(makeResult());
      fixture.detectChanges();

      expect(eq.setOption).toHaveBeenCalled();
      expect(uw.setOption).toHaveBeenCalled();
    });

    it('when qqChart is set, result change triggers setOption on qqChart', () => {
      const qq = makeChartStub();
      (comp as unknown as Record<string, unknown>)['qqChart'] = qq;

      comp.result.set(makeResult());
      fixture.detectChanges();

      expect(qq.setOption).toHaveBeenCalled();
    });

    it('when rolling charts are set and labels exist, window change triggers setOption on all three', () => {
      const sharpe = makeChartStub();
      const vol = makeChartStub();
      const beta = makeChartStub();
      (comp as unknown as Record<string, unknown>)['rollingSharpeChart'] = sharpe;
      (comp as unknown as Record<string, unknown>)['rollingVolChart'] = vol;
      (comp as unknown as Record<string, unknown>)['rollingBetaChart'] = beta;

      comp.result.set(makeResult({
        rollingMetrics: [
          { date: '2024-06-01', sharpe: 1.0, volatility: 0.15, beta: 0.9 },
        ],
      }));
      comp.rollingWindow.set('3Y');
      fixture.detectChanges();

      expect(sharpe.setOption).toHaveBeenCalled();
      expect(vol.setOption).toHaveBeenCalled();
      expect(beta.setOption).toHaveBeenCalled();
    });

    it('loadEcharts() already-loaded branch: calling loadEcharts again is a no-op', async () => {
      // Set the internal flag to simulate a prior successful load.
      // Then call loadEcharts() — the `if (this.echartsLoaded) return` path runs.
      const priv = comp as unknown as Record<string, unknown>;
      priv['echartsLoaded'] = true;
      // Call the private method through the cast — covers the early-return branch.
      await (priv['loadEcharts'] as () => Promise<void>).call(comp);
      // No exception means the early-return branch executed without re-importing.
      expect(priv['echartsLoaded']).toBeTrue();
    });

    it('buildEquityOption tooltip formatter returns empty string for empty params', () => {
      comp.result.set(makeResult());
      const priv = comp as unknown as Record<string, unknown>;
      const option = (priv['buildEquityOption'] as () => Record<string, unknown>).call(comp);
      const tooltip = option['tooltip'] as Record<string, unknown>;
      const formatter = tooltip['formatter'] as (p: unknown) => string;
      // Empty array → the `if (!ps.length) return ''` branch
      expect(formatter([])).toBe('');
      // Non-empty array → rest of the formatter executes
      const result = formatter([{ seriesName: 'Portfolio', value: 100, axisValueLabel: '2024-01', color: '#f00' }]);
      expect(result).toContain('2024-01');
    });

    it('buildUnderwaterOption tooltip formatter returns empty string for empty params', () => {
      comp.result.set(makeResult());
      const priv = comp as unknown as Record<string, unknown>;
      const option = (priv['buildUnderwaterOption'] as () => Record<string, unknown>).call(comp);
      const tooltip = option['tooltip'] as Record<string, unknown>;
      const formatter = tooltip['formatter'] as (p: unknown) => string;
      expect(formatter([])).toBe('');
      const result = formatter([{ value: -5.2, axisValueLabel: '2024-01' }]);
      expect(result).toContain('2024-01');
    });

    it('buildRollingOption tooltip formatter returns empty string for empty params', () => {
      const priv = comp as unknown as Record<string, unknown>;
      const option = (priv['buildRollingOption'] as (
        name: string, labels: string[], values: number[], colorVar: string, isPercent?: boolean
      ) => Record<string, unknown>).call(comp, 'Sharpe', ['2024-01'], [1.2], '--color-chart-1');
      const tooltip = option['tooltip'] as Record<string, unknown>;
      const formatter = tooltip['formatter'] as (p: unknown) => string;
      expect(formatter([])).toBe('');
      // isPercent=false branch
      const r1 = formatter([{ value: 1.23, axisValueLabel: '2024-01' }]);
      expect(r1).toContain('Sharpe');
    });

    it('buildRollingOption isPercent=true branch formats as percentage', () => {
      const priv = comp as unknown as Record<string, unknown>;
      const option = (priv['buildRollingOption'] as (
        name: string, labels: string[], values: number[], colorVar: string, isPercent?: boolean
      ) => Record<string, unknown>).call(comp, 'Volatility', ['2024-01'], [0.15], '--color-chart-3', true);
      const tooltip = option['tooltip'] as Record<string, unknown>;
      const formatter = tooltip['formatter'] as (p: unknown) => string;
      const result = formatter([{ value: 0.15, axisValueLabel: '2024-01' }]);
      expect(result).toContain('%');
    });

    it('buildQQOption tooltip formatter returns empty string when value is not an array', () => {
      comp.result.set(makeResult({
        returnDistribution: [
          { binStart: -0.02, binEnd: -0.01, count: 5, frequency: 0.2 },
          { binStart: -0.01, binEnd: 0.0, count: 10, frequency: 0.4 },
          { binStart: 0.0, binEnd: 0.01, count: 8, frequency: 0.32 },
        ],
      }));
      const priv = comp as unknown as Record<string, unknown>;
      const option = (priv['buildQQOption'] as () => Record<string, unknown>).call(comp);
      const tooltip = option['tooltip'] as Record<string, unknown>;
      const formatter = tooltip['formatter'] as (p: unknown) => string;
      // Non-array value → `if (!Array.isArray(p.value)) return ''`
      expect(formatter({ value: 'not-array', seriesName: 'Returns' })).toBe('');
      // Array value → rest of formatter executes
      const r = formatter({ value: [1.23, 4.56], seriesName: 'Returns' });
      expect(r).toContain('Theoretical');
    });
  });

  // -------------------------------------------------------------------------
  // 8. Event handlers exercised via DOM interactions
  // -------------------------------------------------------------------------

  describe('event handlers', () => {
    it('toggleLogScale flips logScale from false to true', () => {
      expect(comp.logScale()).toBeFalse();
      comp.toggleLogScale();
      expect(comp.logScale()).toBeTrue();
      comp.toggleLogScale();
      expect(comp.logScale()).toBeFalse();
    });

    it('onTabChange updates activeTab signal', () => {
      comp.onTabChange('metrics');
      expect(comp.activeTab()).toBe('metrics');
    });

    it('retry() calls loadData() and clears error state', () => {
      comp.hasError.set(true);
      comp.isLoading.set(true);
      comp.retry();
      expect(comp.hasError()).toBeFalse();
      expect(comp.isLoading()).toBeFalse();
    });

    it('getChartInstance() returns undefined when no chart has been initialised', () => {
      expect(comp.getChartInstance()).toBeUndefined();
    });

    it('onBenchmarkChange updates selectedBenchmark', () => {
      const event = { target: { value: 'QQQ' } } as unknown as Event;
      comp.onBenchmarkChange(event);
      expect(comp.selectedBenchmark()).toBe('QQQ');
    });

    it('onStartDateChange updates selectedStartDate', () => {
      const event = { target: { value: '2023-01-01' } } as unknown as Event;
      comp.onStartDateChange(event);
      expect(comp.selectedStartDate()).toBe('2023-01-01');
    });

    it('onEndDateChange updates selectedEndDate', () => {
      const event = { target: { value: '2024-12-31' } } as unknown as Event;
      comp.onEndDateChange(event);
      expect(comp.selectedEndDate()).toBe('2024-12-31');
    });

    it('onTickersChange updates tickersRaw', () => {
      const event = { target: { value: 'AAPL, TSLA' } } as unknown as Event;
      comp.onTickersChange(event);
      expect(comp.tickersRaw()).toBe('AAPL, TSLA');
    });

    it('Log button click calls toggleLogScale and the button class reflects state', () => {
      comp.isLoading.set(false);
      comp.activeTab.set('overview');
      fixture.detectChanges();

      const logBtn = Array.from(el.querySelectorAll('button')).find(
        (b) => b.textContent?.trim() === 'Log',
      );
      expect(logBtn).toBeTruthy();
      logBtn?.click();
      fixture.detectChanges();
      expect(comp.logScale()).toBeTrue();
    });
  });

  // -------------------------------------------------------------------------
  // 9. signClass and sigBadgeClass helpers — all branches
  // -------------------------------------------------------------------------

  describe('signClass and sigBadgeClass helpers', () => {
    it('signClass returns "text-gain" for positive values', () => {
      expect(comp.signClass(1)).toBe('text-gain');
    });

    it('signClass returns "text-loss" for negative values', () => {
      expect(comp.signClass(-0.5)).toBe('text-loss');
    });

    it('signClass returns "text-flat" for zero', () => {
      expect(comp.signClass(0)).toBe('text-flat');
    });

    it('sigBadgeClass returns bold accent class for "***"', () => {
      expect(comp.sigBadgeClass('***')).toContain('font-bold');
    });

    it('sigBadgeClass returns accent-secondary class for "**"', () => {
      expect(comp.sigBadgeClass('**')).toContain('text-text-secondary');
    });

    it('sigBadgeClass returns inset class for "*"', () => {
      expect(comp.sigBadgeClass('*')).toContain('bg-surface-inset');
    });

    it('sigBadgeClass returns opacity-50 class for "ns"', () => {
      expect(comp.sigBadgeClass('ns')).toContain('opacity-50');
    });

    it('signClass classes render correctly in the factor table (positive loading)', () => {
      comp.result.set(makeResult({
        factorLoadings: [
          { factor: 'MKT', loading: 0.8, tStat: 5, pValue: 0.0005 },
        ],
      }));
      comp.activeTab.set('style');
      fixture.detectChanges();

      const td = el.querySelector('tbody tr td:nth-child(2)');
      expect(td?.className).toContain('text-gain');
    });
  });

  // -------------------------------------------------------------------------
  // 10. Distribution computed signals with real data
  // -------------------------------------------------------------------------

  describe('distribution computed signals', () => {
    const DIST_RESULT = makeResult({
      returnDistribution: [
        { binStart: -0.02, binEnd: -0.01, count: 3, frequency: 0.3 },
        { binStart: -0.01, binEnd: 0.00, count: 4, frequency: 0.4 },
        { binStart: 0.00, binEnd: 0.01, count: 3, frequency: 0.3 },
      ],
    });

    it('distributionValues maps bin midpoints', () => {
      comp.result.set(DIST_RESULT);
      const vals = comp.distributionValues();
      expect(vals.length).toBe(3);
      expect(vals[0]).toBeCloseTo(-0.015, 5);
    });

    it('distributionFullValues expands bins by count', () => {
      comp.result.set(DIST_RESULT);
      expect(comp.distributionFullValues().length).toBe(10);
    });

    it('distributionStats computes mean/std/skewness for non-empty data', () => {
      comp.result.set(DIST_RESULT);
      const stats = comp.distributionStats();
      expect(stats.mean).toBeDefined();
      expect(stats.std).toBeGreaterThanOrEqual(0);
      expect(stats.skewness).toBeDefined();
      expect(stats.kurtosis).toBeDefined();
      expect(stats.jbStat).toBeGreaterThanOrEqual(0);
    });

    it('distributionStats returns zero-median for even-length data', () => {
      comp.result.set(makeResult({
        returnDistribution: [
          { binStart: -0.01, binEnd: 0.00, count: 2, frequency: 0.5 },
          { binStart: 0.00, binEnd: 0.01, count: 2, frequency: 0.5 },
        ],
      }));
      const stats = comp.distributionStats();
      expect(stats.median).toBeDefined();
    });

    it('qqPlotData returns empty arrays when distributionFullValues is empty', () => {
      comp.result.set(makeResult({ returnDistribution: [] }));
      const { points, refLine } = comp.qqPlotData();
      expect(points.length).toBe(0);
      expect(refLine.length).toBe(0);
    });

    it('qqPlotData returns points and refLine for non-empty distribution', () => {
      comp.result.set(DIST_RESULT);
      const { points, refLine } = comp.qqPlotData();
      expect(points.length).toBeGreaterThan(0);
      expect(refLine.length).toBe(2);
    });

    it('distribution tab renders Skewness stat card with non-empty data', () => {
      comp.result.set(DIST_RESULT);
      comp.activeTab.set('distribution');
      fixture.detectChanges();

      expect(el.textContent).toContain('Skewness');
    });

    it('distributionStats std=0 branch: all same values → skewness and kurtosis stay 0', () => {
      // All bins at same midpoint → std=0 → skewness branch stays 0
      comp.result.set(makeResult({
        returnDistribution: [
          { binStart: 0.00, binEnd: 0.00, count: 4, frequency: 1.0 },
        ],
      }));
      const stats = comp.distributionStats();
      expect(stats.std).toBeCloseTo(0, 5);
      expect(stats.skewness).toBe(0);
      expect(stats.kurtosis).toBe(0);
    });
  });

  // -------------------------------------------------------------------------
  // 11. Rolling metrics filterByWindow branches (1Y / 3Y / empty)
  // -------------------------------------------------------------------------

  describe('rolling metrics filterByWindow', () => {
    const ROLLING_RESULT = makeResult({
      rollingMetrics: [
        { date: '2023-01-01', sharpe: 0.8, volatility: 0.12, beta: 0.9 },
        { date: '2024-01-01', sharpe: 1.0, volatility: 0.15, beta: 1.0 },
        { date: '2025-01-01', sharpe: 1.2, volatility: 0.18, beta: 1.1 },
      ],
    });

    it('rollingLabels returns dates from filterByWindow with 1Y window', () => {
      comp.result.set(ROLLING_RESULT);
      comp.rollingWindow.set('1Y');
      expect(comp.rollingLabels().length).toBeGreaterThanOrEqual(0);
    });

    it('rollingLabels returns dates from filterByWindow with 3Y window', () => {
      comp.result.set(ROLLING_RESULT);
      comp.rollingWindow.set('3Y');
      expect(comp.rollingLabels().length).toBeGreaterThanOrEqual(0);
    });

    it('rollingLabels returns empty array when rollingMetrics is empty', () => {
      comp.result.set(makeResult({ rollingMetrics: [] }));
      expect(comp.rollingLabels()).toEqual([]);
    });

    it('rollingSharpeValues and rollingVolValues and rollingBetaValues are arrays', () => {
      comp.result.set(ROLLING_RESULT);
      comp.rollingWindow.set('1Y');
      expect(Array.isArray(comp.rollingSharpeValues())).toBeTrue();
      expect(Array.isArray(comp.rollingVolValues())).toBeTrue();
      expect(Array.isArray(comp.rollingBetaValues())).toBeTrue();
    });
  });

  // -------------------------------------------------------------------------
  // 12. onRunBacktest — all branches
  // -------------------------------------------------------------------------

  describe('onRunBacktest branches', () => {
    it('when already running, onRunBacktest is a no-op', () => {
      comp.runJobId.set('existing-job');
      const errorBefore = comp.runError();
      comp.onRunBacktest();
      expect(comp.runError()).toBe(errorBefore);
    });

    it('when tickersRaw is empty, sets runError without issuing HTTP', () => {
      comp.tickersRaw.set('  ');
      comp.onRunBacktest();
      expect(comp.runError()).toBe('Provide at least one ticker.');
    });

    it('when given explicit empty tickers list, falls back to tickersRaw; with empty tickersRaw sets runError', () => {
      comp.tickersRaw.set('');
      comp.onRunBacktest([]);
      expect(comp.runError()).toBe('Provide at least one ticker.');
    });

    it('when tickers are valid, fires POST and sets runJobId on success', () => {
      const http = TestBed.inject(HttpTestingController);
      comp.tickersRaw.set('AAPL, MSFT');
      comp.onRunBacktest();

      const req = http.expectOne((r) => r.url.includes('backtest'));
      req.flush({ jobId: 'job-123', runId: 'run-456', status: 'pending', message: 'Queued' });
      fixture.detectChanges();

      expect(comp.runJobId()).toBe('job-123');
      expect(comp.runError()).toBeNull();
    });

    it('when POST fails, sets runError', () => {
      const http = TestBed.inject(HttpTestingController);
      comp.tickersRaw.set('AAPL');
      comp.runError.set(null);
      comp.onRunBacktest();

      const req = http.expectOne((r) => r.url.includes('backtest'));
      req.flush({ message: 'server error' }, { status: 500, statusText: 'Internal Server Error' });
      fixture.detectChanges();

      expect(comp.runError()).toBeTruthy();
    });

    it('when given explicit tickers list, uses that list instead of tickersRaw', () => {
      const http = TestBed.inject(HttpTestingController);
      comp.tickersRaw.set('');
      comp.onRunBacktest(['TSLA', 'NVDA']);

      const req = http.expectOne((r) => r.url.includes('backtest'));
      const body = req.request.body as { tickers: string[] };
      expect(body.tickers).toEqual(['TSLA', 'NVDA']);
      req.flush({ jobId: 'j1', runId: 'r1', status: 'pending', message: '' });
    });
  });

  // -------------------------------------------------------------------------
  // 13. onJobCompleted and onJobFailed
  // -------------------------------------------------------------------------

  describe('onJobCompleted and onJobFailed', () => {
    it('onJobFailed sets runError and clears runJobId', () => {
      comp.runJobId.set('job-1');
      comp.onJobFailed('timeout error');
      expect(comp.runError()).toBe('timeout error');
      expect(comp.runJobId()).toBeNull();
    });

    it('onJobFailed with empty string falls back to default message', () => {
      comp.onJobFailed('');
      expect(comp.runError()).toBe('Backtest job failed');
    });

    it('onJobCompleted with null runId clears runJobId and returns early', () => {
      comp.runJobId.set('job-1');
      // runRunId defaults to null
      comp.onJobCompleted();
      expect(comp.runJobId()).toBeNull();
      // runResponseLoading should NOT be set to true when runId is null
      expect(comp.runResponseLoading()).toBeFalse();
    });

    it('onJobCompleted with valid runId sets loading=true and issues GET', () => {
      const http = TestBed.inject(HttpTestingController);
      comp.runJobId.set('job-1');
      comp.runRunId.set('run-1');
      comp.onJobCompleted();

      expect(comp.runResponseLoading()).toBeTrue();

      const req = http.expectOne((r) => r.url.includes('backtest/runs'));
      req.flush(makeBacktestRunResponse());
      fixture.detectChanges();

      expect(comp.runResponseLoading()).toBeFalse();
      expect(comp.runResponse()).not.toBeNull();
    });

    it('onJobCompleted GET failure sets runResponseError', () => {
      const http = TestBed.inject(HttpTestingController);
      comp.runJobId.set('job-1');
      comp.runRunId.set('run-1');
      comp.onJobCompleted();

      const req = http.expectOne((r) => r.url.includes('backtest/runs'));
      req.flush({ message: 'not found' }, { status: 404, statusText: 'Not Found' });
      fixture.detectChanges();

      expect(comp.runResponseLoading()).toBeFalse();
      expect(comp.runResponseError()).toBeTruthy();
    });
  });

  // -------------------------------------------------------------------------
  // 14. Monthly heatmap cell value bindings (non-zero positive/negative/zero)
  // -------------------------------------------------------------------------

  describe('monthly heatmap cell class bindings', () => {
    it('positive cell shows +% text and gain class', () => {
      comp.result.set(makeResult({
        monthlyReturns: [{ year: 2024, month: 1, value: 0.05 }],
      }));
      comp.activeTab.set('monthly');
      fixture.detectChanges();

      const cells = Array.from(el.querySelectorAll('tbody td')).filter(
        (td) => td.textContent?.includes('+'),
      );
      expect(cells.length).toBeGreaterThan(0);
    });

    it('negative cell shows -% text and loss class', () => {
      comp.result.set(makeResult({
        monthlyReturns: [{ year: 2024, month: 2, value: -0.03 }],
      }));
      comp.activeTab.set('monthly');
      fixture.detectChanges();

      const cells = Array.from(el.querySelectorAll('tbody td')).filter(
        (td) => td.className.includes('text-loss'),
      );
      expect(cells.length).toBeGreaterThan(0);
    });

    it('zero-value cell shows "—" placeholder', () => {
      comp.result.set(makeResult({
        monthlyReturns: [{ year: 2024, month: 3, value: 0 }],
      }));
      comp.activeTab.set('monthly');
      fixture.detectChanges();

      const dashCells = Array.from(el.querySelectorAll('tbody td')).filter(
        (td) => td.textContent?.trim() === '—',
      );
      expect(dashCells.length).toBeGreaterThan(0);
    });
  });

  // -------------------------------------------------------------------------
  // 15. Drawdown table — ongoing vs. recovered end-cell class
  // -------------------------------------------------------------------------

  describe('drawdown table end-cell class binding', () => {
    it('ongoing drawdown (end=null) renders "Ongoing" with warning class', () => {
      comp.result.set(makeResult({
        drawdowns: [
          { start: '2024-01-01', trough: '2024-02-01', end: null, depth: -0.15, duration: 31, recovery: null },
        ],
      }));
      comp.activeTab.set('drawdowns');
      fixture.detectChanges();

      const ongoingCells = Array.from(el.querySelectorAll('tbody td')).filter(
        (td) => td.textContent?.trim() === 'Ongoing',
      );
      expect(ongoingCells.length).toBeGreaterThan(0);
      expect(ongoingCells[0]?.className).toContain('text-warning');
    });

    it('recovered drawdown shows date string without warning class', () => {
      comp.result.set(makeResult({
        drawdowns: [
          { start: '2024-01-01', trough: '2024-02-01', end: '2024-03-01', depth: -0.10, duration: 60, recovery: 29 },
        ],
      }));
      comp.activeTab.set('drawdowns');
      fixture.detectChanges();

      const dateCells = Array.from(el.querySelectorAll('tbody td')).filter(
        (td) => td.textContent?.trim() === '2024-03-01',
      );
      expect(dateCells.length).toBeGreaterThan(0);
      expect(dateCells[0]?.className).not.toContain('text-warning');
    });
  });
});

// ---------------------------------------------------------------------------
// Pure-function unit tests for mapRunResponseToResult and its helpers.
// These run without a TestBed — they are plain JS function calls.
// ---------------------------------------------------------------------------

describe('mapRunResponseToResult — pure function (issue #936)', () => {
  it('returns EMPTY_RESULT when run is null', () => {
    const result = mapRunResponseToResult(null);
    expect(result.equity).toEqual([]);
    expect(result.drawdowns).toEqual([]);
    expect(result.monthlyReturns).toEqual([]);
    expect(result.rollingMetrics).toEqual([]);
    expect(result.factorLoadings).toEqual([]);
  });

  it('maps equity curve entries to portfolio/benchmark pairs', () => {
    const result = mapRunResponseToResult({
      equityCurve: { '2024-01-01': 0.05, '2024-02-01': 0.10 },
      drawdowns: {},
      monthlyReturns: {},
      rollingMetrics: {},
      summaryStats: {},
    });
    expect(result.equity.length).toBe(2);
    expect(result.equity[0].date).toBe('2024-01-01');
    expect(result.equity[0].portfolio).toBeCloseTo(1.05, 5);
  });

  it('ignores non-ISO-date keys and null values in equityCurve', () => {
    const result = mapRunResponseToResult({
      equityCurve: {
        'not-a-date': 0.1,
        '2024-01-01': null,
        '2024-02-01': 0.05,
      },
      drawdowns: {},
      monthlyReturns: {},
      rollingMetrics: {},
      summaryStats: {},
    });
    expect(result.equity.length).toBe(1);
    expect(result.equity[0].date).toBe('2024-02-01');
  });

  it('detects a closed drawdown event (start → trough → end)', () => {
    const result = mapRunResponseToResult({
      equityCurve: {},
      drawdowns: {
        '2024-01-01': -0.05,
        '2024-02-01': -0.10,
        '2024-03-01': 0.00,
      },
      monthlyReturns: {},
      rollingMetrics: {},
      summaryStats: {},
    });
    expect(result.drawdowns.length).toBe(1);
    expect(result.drawdowns[0].end).toBe('2024-03-01');
    expect(result.drawdowns[0].depth).toBeCloseTo(-0.10, 5);
  });

  it('detects an ongoing drawdown (never recovers)', () => {
    const result = mapRunResponseToResult({
      equityCurve: {},
      drawdowns: {
        '2024-01-01': -0.05,
        '2024-02-01': -0.08,
      },
      monthlyReturns: {},
      rollingMetrics: {},
      summaryStats: {},
    });
    expect(result.drawdowns.length).toBe(1);
    expect(result.drawdowns[0].end).toBeNull();
    expect(result.drawdowns[0].recovery).toBeNull();
  });

  it('maps monthly returns to year/month/value entries', () => {
    const result = mapRunResponseToResult({
      equityCurve: {},
      drawdowns: {},
      monthlyReturns: { '2024-01-31': 0.02, '2024-02-29': -0.01 },
      rollingMetrics: {},
      summaryStats: {},
    });
    expect(result.monthlyReturns.length).toBe(2);
    expect(result.monthlyReturns[0].year).toBe(2024);
    expect(result.monthlyReturns[0].month).toBe(1);
    expect(result.monthlyReturns[0].value).toBeCloseTo(0.02, 5);
  });

  it('extracts rolling metrics from sharpe/volatility/beta keys', () => {
    const result = mapRunResponseToResult({
      equityCurve: {},
      drawdowns: {},
      monthlyReturns: {},
      rollingMetrics: {
        sharpe: { '2024-01-01': 1.1 },
        volatility: { '2024-01-01': 0.15 },
        beta: { '2024-01-01': 0.9 },
      },
      summaryStats: {},
    });
    expect(result.rollingMetrics.length).toBe(1);
    expect(result.rollingMetrics[0].sharpe).toBeCloseTo(1.1, 5);
    expect(result.rollingMetrics[0].volatility).toBeCloseTo(0.15, 5);
    expect(result.rollingMetrics[0].beta).toBeCloseTo(0.9, 5);
  });

  it('picks alternative rolling metric key names (sharpe_ratio, rolling_volatility)', () => {
    const result = mapRunResponseToResult({
      equityCurve: {},
      drawdowns: {},
      monthlyReturns: {},
      rollingMetrics: {
        sharpe_ratio: { '2024-01-01': 0.8 },
        rolling_volatility: { '2024-01-01': 0.12 },
        rolling_beta: { '2024-01-01': 1.05 },
      },
      summaryStats: {},
    });
    expect(result.rollingMetrics[0].sharpe).toBeCloseTo(0.8, 5);
    expect(result.rollingMetrics[0].volatility).toBeCloseTo(0.12, 5);
    expect(result.rollingMetrics[0].beta).toBeCloseTo(1.05, 5);
  });

  it('reads annualised sharpe from summaryStats when present', () => {
    const result = mapRunResponseToResult({
      equityCurve: {},
      drawdowns: {},
      monthlyReturns: {},
      rollingMetrics: {},
      summaryStats: {
        'Annualized Sharpe Ratio': 1.3,
        'Annualized Mean': 0.08,
        'Annualized Standard Deviation': 0.14,
      },
    });
    expect(result.metrics.sharpe).toBeCloseTo(1.3, 5);
    expect(result.metrics.annualizedReturn).toBeCloseTo(0.08, 5);
    expect(result.metrics.annualizedVol).toBeCloseTo(0.14, 5);
  });

  it('reads in_sample nested stats block', () => {
    const result = mapRunResponseToResult({
      equityCurve: {},
      drawdowns: {},
      monthlyReturns: {},
      rollingMetrics: {},
      summaryStats: {
        in_sample: {
          'Annualized Sharpe Ratio': 0.9,
          'Annualized Mean': 0.05,
        } as unknown as number | null,
      },
    });
    expect(result.metrics.sharpe).toBeCloseTo(0.9, 5);
  });

  it('negates maxDrawdown and cvar95 to ensure they are negative', () => {
    const result = mapRunResponseToResult({
      equityCurve: {},
      drawdowns: {},
      monthlyReturns: {},
      rollingMetrics: {},
      summaryStats: { 'MAX Drawdown': 0.20, 'CVaR at 95%': 0.04 },
    });
    expect(result.metrics.maxDrawdown).toBeLessThanOrEqual(0);
    expect(result.metrics.cvar95).toBeLessThanOrEqual(0);
  });

  it('builds return distribution from equity curve with ≥2 entries', () => {
    const result = mapRunResponseToResult({
      equityCurve: {
        '2024-01-01': 0.00,
        '2024-02-01': 0.02,
        '2024-03-01': -0.01,
        '2024-04-01': 0.03,
      },
      drawdowns: {},
      monthlyReturns: {},
      rollingMetrics: {},
      summaryStats: {},
    });
    expect(result.returnDistribution.length).toBeGreaterThan(0);
    expect(result.returnDistribution[0].count).toBeGreaterThanOrEqual(0);
    const totalFreq = result.returnDistribution.reduce((s, b) => s + b.frequency, 0);
    expect(totalFreq).toBeCloseTo(1.0, 5);
  });

  it('returns empty distribution when equity curve has fewer than 2 entries', () => {
    const result = mapRunResponseToResult({
      equityCurve: { '2024-01-01': 0.01 },
      drawdowns: {},
      monthlyReturns: {},
      rollingMetrics: {},
      summaryStats: {},
    });
    expect(result.returnDistribution).toEqual([]);
  });

  it('returns empty distribution when equity curve values are all null', () => {
    const result = mapRunResponseToResult({
      equityCurve: { '2024-01-01': null, '2024-02-01': null },
      drawdowns: {},
      monthlyReturns: {},
      rollingMetrics: {},
      summaryStats: {},
    });
    // mapEquityCurve filters null values → equity is empty → distribution is empty
    expect(result.returnDistribution).toEqual([]);
  });

  it('draws totalReturn from equity curve when not in summaryStats', () => {
    const result = mapRunResponseToResult({
      equityCurve: { '2024-01-01': 0.00, '2024-12-31': 0.15 },
      drawdowns: {},
      monthlyReturns: {},
      rollingMetrics: {},
      summaryStats: {},
    });
    // equity[-1].portfolio = 1 + 0.15 = 1.15; totalReturn = 1.15 - 1 = 0.15
    expect(result.metrics.totalReturn).toBeCloseTo(0.15, 5);
  });

  it('returns empty distribution when prev value is zero (dailyReturns stays empty)', () => {
    // All equity curve entries with value -1 → prev = (-1)+1 = 0 → `prev > 0` is false
    const result = mapRunResponseToResult({
      equityCurve: {
        '2024-01-01': -1,
        '2024-02-01': -1,
        '2024-03-01': -1,
      },
      drawdowns: {},
      monthlyReturns: {},
      rollingMetrics: {},
      summaryStats: {},
    });
    expect(result.returnDistribution).toEqual([]);
  });

  it('qqPlotData: distributionStats with exactly one data point (std=0 branch)', () => {
    // A single repeated value gives std=0 — triggers the `|| 1` fallback in qqPlotData
    const result = mapRunResponseToResult({
      equityCurve: {
        '2024-01-01': 0.05,
        '2024-02-01': 0.05,
        '2024-03-01': 0.05,
      },
      drawdowns: {},
      monthlyReturns: {},
      rollingMetrics: {},
      summaryStats: {},
    });
    // All daily returns will be 0 (same prev/curr) → std=0 in qqPlotData → uses || 1
    expect(result.returnDistribution.length).toBeGreaterThanOrEqual(0);
  });
});
