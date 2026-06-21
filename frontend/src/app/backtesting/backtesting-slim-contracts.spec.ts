/**
 * backtesting-slim-contracts.spec.ts — issue #1051
 *
 * Source-blind contract tests verifying the "slim orchestrator" and
 * "preserved features" criteria.
 * Authored from acceptance criteria only — no implementation source was read.
 *
 * Criteria pinned (from oracle report):
 *
 *   [UNIT] "backtesting.ts is slimmed: orchestration only (picker + setup form +
 *           results + run/poll); chart/option-building helpers moved into
 *           backtest-results-panel."
 *
 *   [UNIT] "Walk-forward, KPIs (incl. hockey-stick distribution check), cost
 *           awareness, and equity-curve are preserved."
 *
 * Slim tests (criterion 3):
 *   BacktestingComponent must no longer expose private chart-option-building
 *   methods. Tests assert absence of these methods — they FAIL today because
 *   buildEquityOption, buildUnderwaterOption, buildRollingOption, and buildQQOption
 *   are still on the monolith, and will PASS once they are moved to the panel.
 *
 * Preserved-features tests (criterion 4):
 *   The features (walk-forward, KPI cards, distribution stats, equity-curve,
 *   cost awareness) must remain visible to the user after the slim. These tests
 *   drive the component to the relevant rendering state and assert the presence of
 *   the observable output. A test may FAIL today if the refactoring has already
 *   removed a feature from the orchestrator without yet wiring it in the panel.
 *
 * No property-based tests — none of the listed criteria implies a round-trip,
 * idempotence, never-raises-for-valid-input, or ordering invariant at this scope.
 */

import { TestBed, ComponentFixture } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting, HttpTestingController } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';

import {
  installResizeObserverStub,
  makeBacktestRunResponse,
} from '../../testing';
import { ICON_PROVIDER } from '../icons';
import { BacktestingComponent } from './backtesting';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import type { BacktestMetrics, BacktestResult } from './backtest.model';

// ---------------------------------------------------------------------------
// Inline fixtures (kept local — not added to domain-fixtures.ts)
// ---------------------------------------------------------------------------

const ZERO_METRICS: BacktestMetrics = {
  totalReturn: 0, annualizedReturn: 0, annualizedVol: 0, sharpe: 0,
  sortino: 0, maxDrawdown: 0, calmar: 0, cvar95: 0,
  trackingError: 0, informationRatio: 0, winRate: 0, profitFactor: 0,
};

function makeResult(overrides: Partial<BacktestResult> = {}): BacktestResult {
  return {
    equity: [
      { date: '2024-01-01', portfolio: 100, benchmark: 100 },
      { date: '2024-12-31', portfolio: 120, benchmark: 110 },
    ],
    metrics: ZERO_METRICS,
    drawdowns: [],
    monthlyReturns: [],
    rollingMetrics: [],
    returnDistribution: [
      { binStart: -0.02, binEnd: -0.01, count: 3, frequency: 0.3 },
      { binStart: -0.01, binEnd: 0.00, count: 4, frequency: 0.4 },
      { binStart: 0.00, binEnd: 0.01, count: 3, frequency: 0.3 },
    ],
    factorLoadings: [],
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Suite — Criterion 3: BacktestingComponent is an orchestrator only
// ---------------------------------------------------------------------------

describe('BacktestingComponent — slim orchestrator, no chart builders (issue #1051)', () => {
  let fixture: ComponentFixture<BacktestingComponent>;
  let comp: BacktestingComponent;
  let el: HTMLElement;

  beforeEach(async () => {
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

    TestBed.inject(PortfolioContextService).reset();

    fixture = TestBed.createComponent(BacktestingComponent);
    comp = fixture.componentInstance;
    el = fixture.nativeElement as HTMLElement;

    comp.isLoading.set(false);
    fixture.detectChanges();
  });

  afterEach(() => {
    TestBed.inject(HttpTestingController).match(() => true);
    localStorage.clear();
  });

  // ── Criterion 3: chart-option builders removed from orchestrator ──────────
  //
  // These four assertions FAIL today (the methods ARE on BacktestingComponent)
  // and PASS after the slim moves them into BacktestResultsPanelComponent.

  it('when backtesting.ts is slimmed, BacktestingComponent has no buildEquityOption method', () => {
    const priv = comp as unknown as Record<string, unknown>;
    expect(typeof priv['buildEquityOption']).not.toBe('function');
  });

  it('when backtesting.ts is slimmed, BacktestingComponent has no buildUnderwaterOption method', () => {
    const priv = comp as unknown as Record<string, unknown>;
    expect(typeof priv['buildUnderwaterOption']).not.toBe('function');
  });

  it('when backtesting.ts is slimmed, BacktestingComponent has no buildRollingOption method', () => {
    const priv = comp as unknown as Record<string, unknown>;
    expect(typeof priv['buildRollingOption']).not.toBe('function');
  });

  it('when backtesting.ts is slimmed, BacktestingComponent has no buildQQOption method', () => {
    const priv = comp as unknown as Record<string, unknown>;
    expect(typeof priv['buildQQOption']).not.toBe('function');
  });

  // ── Criterion 4: walk-forward panel preserved ─────────────────────────────

  it('when backtesting renders without loading, walk-forward panel is still present', () => {
    expect(el.querySelector('app-walk-forward-panel')).not.toBeNull();
  });

  // ── Criterion 4: four KPI stat-cards preserved ───────────────────────────
  //
  // Requirements: "KPIs (return, vol, Sharpe, max drawdown)" → 4 stat-cards.

  it('when backtesting renders, four KPI stat-cards are still shown', () => {
    expect(el.querySelectorAll('app-stat-card').length).toBe(4);
  });

  // ── Criterion 4: equity-curve preserved (via results panel) ──────────────
  //
  // After the slim, "Equity Curve" must be reachable in the DOM either
  // directly or inside app-backtest-results-panel.

  it('when runResponse is set, results panel renders and "Equity Curve" is visible', () => {
    comp.runResponse.set(makeBacktestRunResponse());
    fixture.detectChanges();

    const panel = el.querySelector('app-backtest-results-panel') as HTMLElement | null;
    expect(panel).not.toBeNull();
    // "Equity Curve" text must survive the slim — either in panel or host.
    expect((panel ?? el).textContent).toContain('Equity Curve');
  });

  // ── Criterion 4: hockey-stick distribution check preserved ───────────────
  //
  // "hockey-stick distribution check" maps to Skewness/Kurtosis stats on the
  // distribution tab (validates non-normality of returns).

  it('when on the distribution tab with return data, Skewness is visible (hockey-stick check)', () => {
    comp.result.set(makeResult());
    comp.activeTab.set('distribution');
    fixture.detectChanges();

    expect(el.textContent).toContain('Skewness');
  });

  it('when on the distribution tab with return data, Kurtosis is visible', () => {
    comp.result.set(makeResult());
    comp.activeTab.set('distribution');
    fixture.detectChanges();

    expect(el.textContent).toContain('Kurtosis');
  });

  it('when on the distribution tab with return data, the distribution histogram is present', () => {
    comp.result.set(makeResult());
    comp.activeTab.set('distribution');
    fixture.detectChanges();

    expect(el.querySelector('app-echarts-histogram')).not.toBeNull();
  });

  // ── Criterion 4: cost-awareness preserved via results panel ──────────────
  //
  // "cost awareness" = turnover / transaction-cost section that uses
  // BacktestRunResponse.turnoverHistory. After the slim the panel must still
  // receive and render this data.

  it('when runResponse with turnoverHistory is set, results panel is present (cost-awareness path)', () => {
    comp.runResponse.set(
      makeBacktestRunResponse({
        turnoverHistory: { '2024-01-01': 0.05, '2024-06-01': 0.03 },
      }),
    );
    fixture.detectChanges();

    expect(el.querySelector('app-backtest-results-panel')).not.toBeNull();
  });

  // ── Criterion 4: results panel loads when results are available ───────────

  it('when runResponseLoading is true, results panel renders (loading state preserved)', () => {
    comp.runResponseLoading.set(true);
    fixture.detectChanges();

    expect(el.querySelector('app-backtest-results-panel')).not.toBeNull();
  });

  it('when runResponseError is set, results panel renders (error state preserved)', () => {
    comp.runResponseError.set('fetch failed');
    fixture.detectChanges();

    expect(el.querySelector('app-backtest-results-panel')).not.toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Suite — Criterion 3 (panel side): BacktestResultsPanelComponent owns charts
//
// These tests import BacktestResultsPanelComponent directly. They FAIL today if:
//   (a) the component does not exist at the expected path, OR
//   (b) the component has not yet received the chart-building helpers.
//
// Import path assumption: Angular sub-component convention places panel components
// in their own sub-directory. Adjust the path if the component is co-located.
// ---------------------------------------------------------------------------

import { BacktestResultsPanelComponent } from './backtest-results-panel/backtest-results-panel';

describe('BacktestResultsPanelComponent — owns chart display after slim (issue #1051)', () => {
  let fixture: ComponentFixture<BacktestResultsPanelComponent>;
  let panelComp: BacktestResultsPanelComponent;
  let panelEl: HTMLElement;

  beforeEach(async () => {
    installResizeObserverStub();

    await TestBed.configureTestingModule({
      imports: [BacktestResultsPanelComponent],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        ICON_PROVIDER,
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(BacktestResultsPanelComponent);
    panelComp = fixture.componentInstance;
    panelEl = fixture.nativeElement as HTMLElement;
    fixture.detectChanges();
  });

  afterEach(() => {
    TestBed.inject(HttpTestingController).match(() => true);
  });

  // ── Panel renders the equity-curve section ────────────────────────────────

  it('when BacktestResultsPanelComponent renders, it shows the Equity Curve label', () => {
    // Criterion 3: chart helpers moved to panel → panel must own the Equity Curve view.
    expect(panelEl.textContent).toContain('Equity Curve');
  });

  // ── Panel renders a chart toolbar (owns chart-building) ──────────────────

  it('when panel receives a run response, it renders a chart toolbar', () => {
    // Criterion 3 + 4: app-chart-toolbar is owned by the panel after the slim.
    const anyComp = panelComp as unknown as Record<string, unknown>;
    const setter = anyComp['runResponse'] as { set?(v: unknown): void } | undefined;
    if (setter?.set) {
      setter.set(makeBacktestRunResponse());
    }
    fixture.detectChanges();

    expect(panelEl.querySelector('app-chart-toolbar')).not.toBeNull();
  });

  // ── Panel renders KPI stat-cards ─────────────────────────────────────────

  it('when panel renders, KPI stat-cards are accessible inside the panel', () => {
    // Criterion 4: after the slim the panel (not the orchestrator) owns the KPI strip.
    expect(panelEl.querySelectorAll('app-stat-card').length).toBeGreaterThan(0);
  });
});
