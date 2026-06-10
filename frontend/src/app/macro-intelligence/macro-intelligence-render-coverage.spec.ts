/**
 * macro-intelligence-render-coverage.spec.ts — issue #937
 *
 * Template-branch coverage for MacroIntelligenceComponent. The only existing
 * component spec (`macro-intelligence.spec.ts`) unit-tests `parseMacroSummary`
 * and never mounts the template, leaving it ~7.9% branch-covered. This spec
 * mounts the full template and drives every native control-flow branch
 * (`@if`/`@else`/`@for`/`@let`) by setting STATE signals directly.
 *
 * Conventions (mirror dashboard/backtesting render-coverage specs):
 *   - installResizeObserverStub() (the bare chart div registers a ResizeObserver)
 *   - inline zoneless TestBed with ICON_PROVIDER + HTTP testing (no router)
 *   - constructor loadData() fires HTTP; isLoading clears only on the news emit,
 *     so `enterSuccessBranch()` flushes ONLY `macro-data/news` to reach @else
 *   - HTTP left pending; afterEach drains all in-flight requests (no leak)
 */

import { provideZonelessChangeDetection } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import type { HttpTestingController } from '@angular/common/http/testing';
import { TestBed, ComponentFixture } from '@angular/core/testing';
import { of, throwError } from 'rxjs';

import { installResizeObserverStub, injectHttp } from '../../testing';
import { ICON_PROVIDER } from '../icons';
import { MacroIntelligenceComponent } from './macro-intelligence';
import { MacroIntelligenceService } from './macro-intelligence.service';
import {
  SECTOR_ROTATION_TABLE,
  COMPOSITE_SCORE_THRESHOLDS,
  COMPOSITE_CHART_AXIS,
} from './macro-intelligence.constants';
import { environment } from '../../environments/environment';
import type {
  CountryMacroData,
  MacroCalibrationResponse,
  MacroNewsItem,
  MacroNewsSummaryResponse,
  MacroNewsTheme,
} from '../core/models/macro-intelligence.model';

const API = environment.apiUrl;
const T = COMPOSITE_SCORE_THRESHOLDS;

// ---------------------------------------------------------------------------
// Inline domain builders (kept local — no factories exist for these shapes)
// ---------------------------------------------------------------------------

function makeCountry(over: Partial<CountryMacroData> = {}): CountryMacroData {
  return {
    country: 'United States',
    country_code: 'US',
    gdp_growth: 2.5,
    inflation: 3,
    unemployment: 4,
    interest_rate: 5,
    yield_10y: 4,
    yield_2y: 3,
    yield_spread_bps: 100,
    yield_sparkline: [1, 2, 3],
    ...over,
  };
}

function makeNews(over: Partial<MacroNewsItem> = {}): MacroNewsItem {
  return {
    id: 'n1',
    headline: 'Headline',
    snippet: 'Snippet',
    url: 'https://example.com',
    publisher: 'Pub',
    published_at: '2026-01-01T00:00:00Z',
    theme: 'inflation',
    ...over,
  };
}

function makeTheme(over: Partial<MacroNewsTheme> = {}): MacroNewsTheme {
  return { id: 'inflation', label: 'Inflation', count: 2, ...over };
}

function makeSummary(over: Partial<MacroNewsSummaryResponse> = {}): MacroNewsSummaryResponse {
  return {
    id: 's1',
    country: 'USA',
    summary_date: '2026-01-01',
    summary: 'Country summary text',
    sentiment: 'BULLISH',
    sentiment_score: 0.5,
    article_count: 3,
    created_at: '',
    updated_at: '',
    ...over,
  };
}

function makeCalibration(over: Partial<MacroCalibrationResponse> = {}): MacroCalibrationResponse {
  return {
    phase: 'MID_EXPANSION',
    delta: 2.75,
    tau: 0.025,
    confidence: 0.8,
    rationale: 'AI-generated rationale',
    macro_summary: 'PMI: 51 | HY: 350bp',
    timestamp: '2026-01-01T00:00:00Z',
    bl_config: {
      views: [],
      tau: 0.025,
      prior_config: { mu_estimator: 'equilibrium', risk_aversion: 2.75, cov_estimator: 'ledoit_wolf' },
    },
    ...over,
  };
}

// ---------------------------------------------------------------------------
// Private-member access for chart-builder coverage (no `any`)
// ---------------------------------------------------------------------------

interface ChartStub {
  setOption: jasmine.Spy;
  resize: () => void;
  dispose: () => void;
}

interface PrivateView {
  chart?: ChartStub;
  chartInitialized: boolean;
  updateChart: () => void;
  buildYieldCurveOption: () => Record<string, unknown>;
  buildCreditSpreadOption: () => Record<string, unknown>;
  buildCompositeHistoryOption: () => Record<string, unknown>;
}

function priv(c: MacroIntelligenceComponent): PrivateView {
  return c as unknown as PrivateView;
}

function makeChartStub(): ChartStub {
  return { setOption: jasmine.createSpy('setOption'), resize: () => {}, dispose: () => {} };
}

// ---------------------------------------------------------------------------
// Shared testbed setup
// ---------------------------------------------------------------------------

describe('MacroIntelligenceComponent — render-coverage (issue #937)', () => {
  let fixture: ComponentFixture<MacroIntelligenceComponent>;
  let comp: MacroIntelligenceComponent;
  let el: HTMLElement;
  let http: HttpTestingController;

  beforeEach(async () => {
    installResizeObserverStub();

    await TestBed.configureTestingModule({
      imports: [MacroIntelligenceComponent],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        ICON_PROVIDER,
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(MacroIntelligenceComponent);
    comp = fixture.componentInstance;
    el = fixture.nativeElement as HTMLElement;
    http = injectHttp();
    // No detectChanges() here — leave isLoading=true as the baseline.
  });

  afterEach(() => {
    // Drain every in-flight request fired by loadData(); flushing {} is safe
    // because each service mapper is wrapped in catchError(of([])/of(null)).
    for (const req of http.match(() => true)) req.flush({});
    http.verify();
  });

  /** Flush only the news request (clears isLoading) to reach the @else branch. */
  function enterSuccessBranch(): void {
    for (const req of http.match((r) => r.url === `${API}macro-data/news`)) {
      req.flush([]);
    }
    fixture.detectChanges();
  }

  /** Drive the three composite sub-scores via the latest point of each series. */
  function setScores(pmi: number, spread: number, hy: number): void {
    comp.fredPmi.set([{ date: '2026-01-01', value: pmi }]);
    comp.fredYieldSpread.set([{ date: '2026-01-01', value: spread }]);
    comp.fredHyOas.set([{ date: '2026-01-01', value: hy }]);
  }

  // -------------------------------------------------------------------------
  // 1. Outer three-way + refresh button
  // -------------------------------------------------------------------------

  describe('loading / error / success three-way + refresh button', () => {
    it('when isLoading is true, the skeleton renders and no regime dot is shown', () => {
      fixture.detectChanges();

      expect(el.querySelectorAll('.skeleton').length).toBeGreaterThan(0);
      expect(el.querySelector('[data-region="regime-dot"]')).toBeNull();
    });

    it('when hasError is true, shows "Something went wrong" and a Try Again button', () => {
      comp.isLoading.set(false);
      comp.hasError.set(true);
      comp.errorMessage.set('boom');
      fixture.detectChanges();

      expect(el.textContent).toContain('Something went wrong');
      expect(el.textContent).toContain('boom');
      const tryAgain = Array.from(el.querySelectorAll('button')).find(
        (b) => b.textContent?.trim() === 'Try Again',
      );
      expect(tryAgain).toBeTruthy();
    });

    it('when news resolves, the success branch renders the chart container', () => {
      enterSuccessBranch();

      expect(el.querySelector('[aria-label="Macro time-series chart"]')).not.toBeNull();
      expect(el.querySelector('[data-region="regime-dot"]')).not.toBeNull();
    });

    it('when not refreshing, the refresh button reads "Refresh Data"', () => {
      fixture.detectChanges();

      expect(el.textContent).toContain('Refresh Data');
      expect(el.textContent).not.toContain('Refreshing...');
    });

    it('when refreshing, the refresh button reads "Refreshing..." and is disabled', () => {
      comp.isRefreshing.set(true);
      fixture.detectChanges();

      expect(el.textContent).toContain('Refreshing...');
      const btn = el.querySelector<HTMLButtonElement>('button[aria-label="Refreshing data..."]');
      expect(btn?.disabled).toBeTrue();
    });
  });

  // -------------------------------------------------------------------------
  // 2. Band C — chart tabs + yield-curve country toggles
  // -------------------------------------------------------------------------

  describe('Band C — chart tabs and yield-curve toggles', () => {
    beforeEach(() => enterSuccessBranch());

    it('when activeChartTab is yield-curve, a toggle renders per country', () => {
      comp.countryData.set([makeCountry({ country_code: 'US' }), makeCountry({ country_code: 'DE' })]);
      comp.activeChartTab.set('yield-curve');
      fixture.detectChanges();

      const container = el.querySelector('div.flex.items-center.gap-1.mb-3');
      expect(container?.querySelectorAll('button').length).toBe(2);
    });

    it('when activeChartTab is credit-spreads, the yield-curve toggle container is absent', () => {
      comp.countryData.set([makeCountry({ country_code: 'US' })]);
      comp.activeChartTab.set('credit-spreads');
      fixture.detectChanges();

      expect(el.querySelector('div.flex.items-center.gap-1.mb-3')).toBeNull();
    });
  });

  // -------------------------------------------------------------------------
  // 3. Band D — country cards + sparkline @if
  // -------------------------------------------------------------------------

  describe('Band D — country cards and yield sparkline', () => {
    beforeEach(() => enterSuccessBranch());

    it('when two countries exist, two country cards render', () => {
      comp.countryData.set([makeCountry({ country_code: 'US' }), makeCountry({ country_code: 'DE' })]);
      fixture.detectChanges();

      expect(el.querySelectorAll('button[aria-label^="Select "]').length).toBe(2);
    });

    it('when yield_sparkline has >1 point, the sparkline path renders', () => {
      comp.countryData.set([makeCountry({ yield_sparkline: [1, 2, 3] })]);
      fixture.detectChanges();

      // Use the sparkline SVG's unique viewBox to avoid matching Lucide icon paths.
      expect(el.querySelector('svg[viewBox="0 0 100 24"] path')).not.toBeNull();
    });

    it('when yield_sparkline has <=1 point, no sparkline path renders', () => {
      comp.countryData.set([makeCountry({ yield_sparkline: [1] })]);
      fixture.detectChanges();

      // Use the sparkline SVG's unique viewBox to avoid matching Lucide icon paths.
      expect(el.querySelector('svg[viewBox="0 0 100 24"] path')).toBeNull();
    });
  });

  // -------------------------------------------------------------------------
  // 4. Band D2 — sector rotation matrix (static 11 rows)
  // -------------------------------------------------------------------------

  describe('Band D2 — sector rotation matrix', () => {
    beforeEach(() => {
      enterSuccessBranch();
      // Switch off the macro-brief tab so its source-data <table> does not add
      // its own thead/tbody rows to the unscoped queries below.
      comp.activeIntelTab.set('themed-news');
      fixture.detectChanges();
    });

    it('renders one row per sector in the rotation table', () => {
      const rows = el.querySelectorAll('tbody tr');
      expect(rows.length).toBe(SECTOR_ROTATION_TABLE.length);
    });

    it('renders the four phase columns in the header', () => {
      // 1 "Sector" th + 4 phase th = 5 header cells.
      expect(el.querySelectorAll('thead th').length).toBe(5);
    });
  });

  // -------------------------------------------------------------------------
  // 5. Band E — Macro Brief (LLM warning @if + source-data @for)
  // -------------------------------------------------------------------------

  describe('Band E — macro-brief tab', () => {
    beforeEach(() => {
      enterSuccessBranch();
      comp.activeIntelTab.set('macro-brief');
    });

    it('when macroCalibration is null, the LLM-unavailable warning renders', () => {
      comp.macroCalibration.set(null);
      fixture.detectChanges();

      expect(el.textContent).toContain('LLM unavailable');
    });

    it('when macroCalibration is set, the LLM-unavailable warning is absent', () => {
      comp.macroCalibration.set(makeCalibration());
      fixture.detectChanges();

      expect(el.textContent).not.toContain('LLM unavailable');
    });

    it('when macro_summary has two segments, the source-data table renders two rows', () => {
      comp.macroCalibration.set(makeCalibration({ macro_summary: 'PMI: 51 | HY: 350' }));
      fixture.detectChanges();

      expect(el.querySelectorAll('details tbody tr').length).toBe(2);
    });
  });

  // -------------------------------------------------------------------------
  // 6. Band E — Themed News (@for + empty @if + theme filter)
  // -------------------------------------------------------------------------

  describe('Band E — themed-news tab', () => {
    beforeEach(() => {
      enterSuccessBranch();
      comp.activeIntelTab.set('themed-news');
    });

    it('when news items exist, an article anchor renders per item', () => {
      comp.newsItems.set([makeNews({ id: 'a' }), makeNews({ id: 'b' })]);
      comp.newsThemes.set([makeTheme()]);
      fixture.detectChanges();

      expect(el.querySelectorAll('a[target="_blank"]').length).toBe(2);
    });

    it('when filteredNews is empty, the "No articles" empty-state renders', () => {
      comp.newsItems.set([]);
      fixture.detectChanges();

      expect(el.textContent).toContain('No articles for this theme.');
    });

    it('when a theme with no matching items is selected, the empty-state renders', () => {
      comp.newsItems.set([makeNews({ theme: 'inflation' })]);
      comp.selectedNewsTheme.set('growth');
      fixture.detectChanges();

      expect(comp.filteredNews().length).toBe(0);
      expect(el.textContent).toContain('No articles for this theme.');
    });
  });

  // -------------------------------------------------------------------------
  // 7. Band E — Country Summary (empty @if/@else + @let s @if/@else)
  // -------------------------------------------------------------------------

  describe('Band E — country-summary tab', () => {
    beforeEach(() => {
      enterSuccessBranch();
      comp.activeIntelTab.set('country-summary');
    });

    it('when there are no summaries, the empty-state renders', () => {
      comp.countrySummaries.set([]);
      fixture.detectChanges();

      expect(el.textContent).toContain('No country summaries available');
    });

    it('when a country has a matching summary, its sentiment badge and summary render', () => {
      comp.countryData.set([makeCountry({ country_code: 'US' })]);
      comp.countrySummaries.set([makeSummary({ country: 'USA', summary: 'US outlook' })]);
      fixture.detectChanges();

      expect(el.textContent).toContain('US outlook');
      expect(el.textContent).toContain('BULLISH');
    });

    it('when a country has no matching summary, the no-summary placeholder renders', () => {
      comp.countryData.set([makeCountry({ country_code: 'DE' })]);
      comp.countrySummaries.set([makeSummary({ country: 'USA' })]);
      fixture.detectChanges();

      expect(el.textContent).toContain('No summary for this country.');
    });
  });

  // -------------------------------------------------------------------------
  // 8. Computed score branches (3/3/3/3 arms + fallback + 4 phases)
  // -------------------------------------------------------------------------

  describe('composite score computed branches', () => {
    it('when latest PMI is above the bull threshold, pmiScore is +1', () => {
      setScores(T.PMI_BULL + 1, 0.5, 425);
      expect(comp.pmiScore()).toBe(1);
    });

    it('when latest PMI is below the bear threshold, pmiScore is -1', () => {
      setScores(T.PMI_BEAR - 1, 0.5, 425);
      expect(comp.pmiScore()).toBe(-1);
    });

    it('when latest PMI is between thresholds, pmiScore is 0', () => {
      setScores(50, 0.5, 425);
      expect(comp.pmiScore()).toBe(0);
    });

    it('when latest spread is above the bull threshold, spreadScore is +1', () => {
      setScores(50, T.YIELD_SPREAD_BULL + 0.5, 425);
      expect(comp.spreadScore()).toBe(1);
    });

    it('when latest spread is below the bear threshold, spreadScore is -1', () => {
      setScores(50, T.YIELD_SPREAD_BEAR - 0.5, 425);
      expect(comp.spreadScore()).toBe(-1);
    });

    it('when latest spread is between thresholds, spreadScore is 0', () => {
      setScores(50, 0.5, 425);
      expect(comp.spreadScore()).toBe(0);
    });

    it('when latest HY OAS is below the bull threshold, hyScore is +1', () => {
      setScores(50, 0.5, T.HY_OAS_BULL - 50);
      expect(comp.hyScore()).toBe(1);
    });

    it('when latest HY OAS is above the bear threshold, hyScore is -1', () => {
      setScores(50, 0.5, T.HY_OAS_BEAR + 50);
      expect(comp.hyScore()).toBe(-1);
    });

    it('when latest HY OAS is between thresholds, hyScore is 0', () => {
      setScores(50, 0.5, 425);
      expect(comp.hyScore()).toBe(0);
    });

    it('when all sub-scores are empty, every score is 0', () => {
      comp.fredPmi.set([]);
      comp.fredYieldSpread.set([]);
      comp.fredHyOas.set([]);
      expect(comp.compositeScore()).toBe(0);
    });
  });

  describe('regimeDotClass + effectiveCalibration + phase mapping', () => {
    it('when composite score is bullish, regimeDotClass is bg-gain', () => {
      setScores(T.PMI_BULL + 1, T.YIELD_SPREAD_BULL + 0.5, T.HY_OAS_BULL - 50);
      expect(comp.regimeDotClass()).toBe('bg-gain');
    });

    it('when composite score is bearish, regimeDotClass is bg-loss', () => {
      setScores(T.PMI_BEAR - 1, T.YIELD_SPREAD_BEAR - 0.5, T.HY_OAS_BEAR + 50);
      expect(comp.regimeDotClass()).toBe('bg-loss');
    });

    it('when composite score is neutral, regimeDotClass is bg-warning', () => {
      setScores(50, 0.5, 425);
      expect(comp.regimeDotClass()).toBe('bg-warning');
    });

    it('when macroCalibration is null, effectiveCalibration falls back to phase defaults', () => {
      comp.macroCalibration.set(null);
      expect(comp.effectiveCalibration().rationale).toContain('LLM unavailable');
    });

    it('when macroCalibration is set, effectiveCalibration passes it through', () => {
      const cal = makeCalibration({ rationale: 'real' });
      comp.macroCalibration.set(cal);
      expect(comp.effectiveCalibration()).toBe(cal);
    });

    it('when composite score >= bull threshold, currentPhase is EARLY_EXPANSION', () => {
      comp.macroCalibration.set(null);
      setScores(T.PMI_BULL + 1, T.YIELD_SPREAD_BULL + 0.5, T.HY_OAS_BULL - 50);
      expect(comp.currentPhase()).toBe('EARLY_EXPANSION');
    });

    it('when composite score is non-negative but sub-bull, currentPhase is MID_EXPANSION', () => {
      comp.macroCalibration.set(null);
      setScores(T.PMI_BULL + 1, 0.5, 425); // +1
      expect(comp.currentPhase()).toBe('MID_EXPANSION');
    });

    it('when composite score is mildly negative, currentPhase is LATE_EXPANSION', () => {
      comp.macroCalibration.set(null);
      setScores(T.PMI_BEAR - 1, 0.5, 425); // -1
      expect(comp.currentPhase()).toBe('LATE_EXPANSION');
    });

    it('when composite score <= bear threshold, currentPhase is CONTRACTION', () => {
      comp.macroCalibration.set(null);
      setScores(T.PMI_BEAR - 1, T.YIELD_SPREAD_BEAR - 0.5, T.HY_OAS_BEAR + 50); // -3
      expect(comp.currentPhase()).toBe('CONTRACTION');
    });
  });

  // -------------------------------------------------------------------------
  // 9. refreshData() guard / success / error
  // -------------------------------------------------------------------------

  describe('refreshData()', () => {
    /**
     * Drain the constructor's loadData() requests so that subsequent spied
     * calls to triggerRefresh (which emit synchronously and call loadData
     * again) do not cancel already-pending requests — which would cause
     * afterEach's flush loop to throw "Cannot flush a cancelled request".
     */
    function drainInitialRequests(): void {
      for (const req of http.match(() => true)) req.flush([]);
    }

    it('when already refreshing, triggerRefresh is not called again', () => {
      const svc = TestBed.inject(MacroIntelligenceService);
      const spy = spyOn(svc, 'triggerRefresh').and.returnValue(of(undefined));
      comp.isRefreshing.set(true);

      comp.refreshData();

      expect(spy).not.toHaveBeenCalled();
    });

    it('when refresh succeeds, triggerRefresh runs and isRefreshing resets to false', () => {
      drainInitialRequests();
      const svc = TestBed.inject(MacroIntelligenceService);
      const spy = spyOn(svc, 'triggerRefresh').and.returnValue(of(undefined));
      comp.isRefreshing.set(false);

      comp.refreshData();
      // triggerRefresh emits synchronously → loadData(false, true) fires new requests
      // → drain those too so afterEach verify() succeeds.
      for (const req of http.match(() => true)) req.flush([]);

      expect(spy).toHaveBeenCalled();
      expect(comp.isRefreshing()).toBeFalse();
    });

    it('when refresh errors, isRefreshing still resets to false', () => {
      drainInitialRequests();
      const svc = TestBed.inject(MacroIntelligenceService);
      spyOn(svc, 'triggerRefresh').and.returnValue(throwError(() => new Error('boom')));
      spyOn(console, 'error');
      comp.isRefreshing.set(false);

      comp.refreshData();
      // error path also calls loadData(false, true) → drain those requests.
      for (const req of http.match(() => true)) req.flush([]);

      expect(comp.isRefreshing()).toBeFalse();
    });
  });

  // -------------------------------------------------------------------------
  // 10. Chart option builders (private — exercised via updateChart)
  // -------------------------------------------------------------------------

  describe('chart option builders', () => {
    beforeEach(() => {
      priv(comp).chart = makeChartStub();
      priv(comp).chartInitialized = true;
    });

    it('when the active tab is yield-curve, updateChart builds the yield-curve option', () => {
      comp.bondYieldsToday.set([
        { country: 'US', date: '', curve: [{ maturity: '2Y', yield_pct: 3 }, { maturity: '10Y', yield_pct: 4 }] },
      ]);
      comp.bondYields1YAgo.set([{ country: 'US', date: '', curve: [{ maturity: '2Y', yield_pct: 2 }] }]);
      comp.selectedCountry.set('US');
      comp.activeChartTab.set('yield-curve');

      priv(comp).updateChart();

      expect(priv(comp).chart!.setOption).toHaveBeenCalled();
    });

    it('when the active tab is credit-spreads, updateChart builds the credit-spread option', () => {
      comp.fredHyOas.set([{ date: '2026-01-01', value: 400 }]);
      comp.fredIgOas.set([{ date: '2026-01-01', value: 120 }]);
      comp.activeChartTab.set('credit-spreads');

      priv(comp).updateChart();

      expect(priv(comp).chart!.setOption).toHaveBeenCalled();
    });

    it('when the active tab is composite-history, updateChart builds the history option', () => {
      comp.compositeHistory.set([
        { month: '2026-01', score: COMPOSITE_CHART_AXIS.BULL_THRESHOLD },
        { month: '2026-02', score: COMPOSITE_CHART_AXIS.BEAR_THRESHOLD },
        { month: '2026-03', score: 0 },
      ]);
      comp.activeChartTab.set('composite-history');

      priv(comp).updateChart();

      expect(priv(comp).chart!.setOption).toHaveBeenCalled();
    });

    it('when the yield-curve axis formatter runs, it appends a percent sign', () => {
      const opt = priv(comp).buildYieldCurveOption();
      const yAxis = opt['yAxis'] as { axisLabel: { formatter: (v: number) => string } };
      expect(yAxis.axisLabel.formatter(4.25)).toBe('4.3%');
    });

    it('when the credit-spread axis formatter runs, it renders the raw basis points', () => {
      const opt = priv(comp).buildCreditSpreadOption();
      const yAxes = opt['yAxis'] as Array<{ axisLabel: { formatter: (v: number) => string } }>;
      expect(yAxes[0].axisLabel.formatter(300)).toBe('300');
    });

    it('when the composite-history tooltip formatter runs, it signs the score', () => {
      const opt = priv(comp).buildCompositeHistoryOption();
      const tooltip = opt['tooltip'] as { formatter: (p: unknown) => string };
      expect(tooltip.formatter([{ name: '2026-01', value: 2 }])).toContain('+2');
      expect(tooltip.formatter([{ name: '2026-02', value: -1 }])).toContain('-1');
    });
  });

  // -------------------------------------------------------------------------
  // 11. Small presentation helpers (branch arms)
  // -------------------------------------------------------------------------

  describe('presentation helpers', () => {
    it('getStanceClass maps OW / UW / N to distinct classes', () => {
      expect(comp.getStanceClass('OW')).toContain('text-gain');
      expect(comp.getStanceClass('UW')).toContain('text-loss');
      expect(comp.getStanceClass('N')).toContain('text-text-tertiary');
    });

    it('getScoreSign prefixes positive scores with +', () => {
      expect(comp.getScoreSign(2)).toBe('+2');
      expect(comp.getScoreSign(-1)).toBe('-1');
    });

    it('getScoreClass maps sign to gain / loss / neutral', () => {
      expect(comp.getScoreClass(1)).toBe('text-gain');
      expect(comp.getScoreClass(-1)).toBe('text-loss');
      expect(comp.getScoreClass(0)).toBe('text-text-tertiary');
    });

    it('getTrend maps delta to up / down / flat', () => {
      expect(comp.getTrend(0.5)).toBe('up');
      expect(comp.getTrend(-0.5)).toBe('down');
      expect(comp.getTrend(0)).toBe('flat');
    });

    it('getSentimentClass maps each sentiment to its palette', () => {
      expect(comp.getSentimentClass('BULLISH')).toContain('text-gain');
      expect(comp.getSentimentClass('BEARISH')).toContain('text-loss');
      expect(comp.getSentimentClass('MIXED')).toContain('text-warning');
      expect(comp.getSentimentClass(null)).toContain('text-text-tertiary');
    });

    it('getSentimentLabel falls back to "No data" when null', () => {
      expect(comp.getSentimentLabel('BULLISH')).toBe('BULLISH');
      expect(comp.getSentimentLabel(null)).toBe('No data');
    });

    it('relativeTime formats minute / hour / day deltas', () => {
      const now = Date.now();
      expect(comp.relativeTime(new Date(now).toISOString())).toBe('just now');
      expect(comp.relativeTime(new Date(now - 5 * 60000).toISOString())).toBe('5m ago');
      expect(comp.relativeTime(new Date(now - 3 * 3600000).toISOString())).toBe('3h ago');
      expect(comp.relativeTime(new Date(now - 2 * 86400000).toISOString())).toBe('2d ago');
    });

    it('buildMiniSparkline returns empty for <2 points and a path otherwise', () => {
      expect(comp.buildMiniSparkline([1])).toBe('');
      expect(comp.buildMiniSparkline([1, 2, 3]).startsWith('M')).toBeTrue();
    });
  });

  // -------------------------------------------------------------------------
  // 12. Tab + selection delegation handlers
  // -------------------------------------------------------------------------

  describe('tab and selection handlers', () => {
    it('onChartTabChange updates activeChartTab', () => {
      comp.onChartTabChange('credit-spreads');
      expect(comp.activeChartTab()).toBe('credit-spreads');
    });

    it('onIntelTabChange updates activeIntelTab', () => {
      comp.onIntelTabChange('country-summary');
      expect(comp.activeIntelTab()).toBe('country-summary');
    });

    it('selectCountry updates selectedCountry', () => {
      comp.selectCountry('DE');
      expect(comp.selectedCountry()).toBe('DE');
    });

    it('selectNewsTheme updates selectedNewsTheme', () => {
      comp.selectNewsTheme('inflation');
      expect(comp.selectedNewsTheme()).toBe('inflation');
    });

    it('getPhaseLabel maps a phase to its label', () => {
      expect(comp.getPhaseLabel('EARLY_EXPANSION')).toBe('Early Expansion');
    });
  });
});
