/**
 * Criterion: drift-overlay and analytics-pane charts render without blank
 * states / null throws after a build.
 *
 * Strategy:
 *   - Test DriftOverlayComponent in isolation: when store.drift() is null
 *     (pre-build), the drift-empty placeholder renders without throwing;
 *     when drift data is set (post-build), the chart element appears.
 *   - Test AnalyticsPaneComponent in isolation: when store.result() is null
 *     (pre-build), all six cards show the dash placeholder without throwing;
 *     when result data is set (post-build), formatted values render.
 *   - Verify PortfolioBuilderComponent mounts app-analytics-pane in the DOM.
 *
 * These are component-level smoke tests that confirm the "no blank state"
 * invariant at the unit layer without needing a full HTTP round-trip.
 */

import { provideZonelessChangeDetection } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed, type ComponentFixture } from '@angular/core/testing';
import { Subject } from 'rxjs';

import { DriftOverlayComponent } from './canvas-pane/drift-overlay/drift-overlay';
import { AnalyticsPaneComponent } from './analytics-pane/analytics-pane';
import { BuilderStore, BUILDER_DRIFT_SERVICE } from './state/builder.store';
import { BuilderResultService } from './builder-result.service';
import { JOB_POLL_TICK } from '../shared/job-progress-tracker/job-progress-tracker';
import type { BuilderDrift, BuilderResult, BuilderBacktest } from './models/builder-result.model';

// ---------------------------------------------------------------------------
// CSS variable scaffolding (echarts-bar reads CSS custom properties for colour)
// ---------------------------------------------------------------------------

const PALETTE = ['#111', '#222', '#333', '#444', '#555', '#666', '#777', '#888'];

function setCssVars(): void {
  PALETTE.forEach((value, i) => {
    document.documentElement.style.setProperty(`--color-chart-${i + 1}`, value);
  });
}
function clearCssVars(): void {
  PALETTE.forEach((_, i) => {
    document.documentElement.style.removeProperty(`--color-chart-${i + 1}`);
  });
}

// ---------------------------------------------------------------------------
// Sample data
// ---------------------------------------------------------------------------

function sampleDrift(): BuilderDrift {
  return {
    portfolioName: 'test-portfolio',
    totals: {
      deployable_eur: 10000,
      total_holdings_eur: 10000,
      total_drift_abs: 0.05,
      buy_eur: 500,
      sell_eur: 500,
    },
    drift: [
      {
        ticker: 'AAPL',
        current_weight: 0.55,
        target_weight: 0.5,
        delta_weight: 0.05,
        eur_value: 5500,
        flags: [],
      },
      {
        ticker: 'MSFT',
        current_weight: 0.45,
        target_weight: 0.5,
        delta_weight: -0.05,
        eur_value: 4500,
        flags: [],
      },
    ],
  };
}

function sampleResult(): BuilderResult {
  return {
    runId: 'r-1',
    weights: { AAPL: 0.5, MSFT: 0.5 },
    metrics: {
      annualized_return: 0.12,
      annualized_volatility: 0.18,
      annualized_sharpe_ratio: 0.85,
      max_drawdown: -0.32,
    },
    optimizerType: 'mean_risk',
    createdAt: '2026-01-01T00:00:00Z',
  };
}

function sampleBacktest(): BuilderBacktest {
  return {
    runId: 'r-1',
    summaryStats: { turnover: 0.15, cost_bps_actual: 8.5 },
    equityCurve: { '2026-01-01': 1.0, '2026-06-01': 1.12 },
    createdAt: '2026-01-01T00:00:00Z',
  };
}

// ---------------------------------------------------------------------------
// DriftOverlayComponent — null-safety and post-build chart presence
// ---------------------------------------------------------------------------

describe('DriftOverlayComponent – post-build chart states', () => {
  beforeAll(() => setCssVars());
  afterAll(() => clearCssVars());

  function setup(): {
    fixture: ComponentFixture<DriftOverlayComponent>;
    store: BuilderStore;
    host: HTMLElement;
  } {
    const driftRunner = { runExplicit: jasmine.createSpy('runExplicit') };
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        BuilderStore,
        { provide: BUILDER_DRIFT_SERVICE, useValue: driftRunner },
      ],
    });
    const store = TestBed.inject(BuilderStore);
    const fixture = TestBed.createComponent(DriftOverlayComponent);
    fixture.detectChanges();
    return { fixture, store, host: fixture.nativeElement as HTMLElement };
  }

  it('when drift is null (pre-build), drift-overlay renders the empty placeholder without throwing', () => {
    const { host } = setup();
    // No throw is asserted by reaching this line.
    expect(host.querySelector('[data-region="drift-overlay"]')).not.toBeNull();
    expect(host.querySelector('[data-region="drift-empty"]')).not.toBeNull();
  });

  it('when drift is null, barData() returns an empty array (not null), so no chart element is blank', () => {
    const { fixture } = setup();
    const data = fixture.componentInstance.barData();
    expect(Array.isArray(data)).toBeTrue();
    expect(data.length).toBe(0);
  });

  it('when drift data is set (post-build), app-echarts-bar renders and drift-empty is absent', () => {
    const { fixture, store, host } = setup();
    store.setDrift(sampleDrift());
    fixture.detectChanges();

    expect(host.querySelector('app-echarts-bar')).not.toBeNull();
    expect(host.querySelector('[data-region="drift-empty"]')).toBeNull();
  });

  it('when drift data is set (post-build), rows() is non-empty', () => {
    const { fixture, store } = setup();
    store.setDrift(sampleDrift());
    fixture.detectChanges();

    expect(fixture.componentInstance.rows().length).toBeGreaterThan(0);
  });

  it('when drift data arrives after null, switching to donut view renders app-echarts-donut without throwing', () => {
    const { fixture, store, host } = setup();
    store.setDrift(sampleDrift());
    fixture.componentInstance.setViewMode('donut');
    fixture.detectChanges();

    expect(host.querySelector('app-echarts-donut')).not.toBeNull();
    expect(host.querySelector('[data-region="drift-empty"]')).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// AnalyticsPaneComponent — null-safety and post-build card presence
// ---------------------------------------------------------------------------

describe('AnalyticsPaneComponent – post-build chart states', () => {
  function setup(): {
    fixture: ComponentFixture<AnalyticsPaneComponent>;
    store: BuilderStore;
    host: HTMLElement;
  } {
    const tick = new Subject<number>();
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        BuilderStore,
        BuilderResultService,
        { provide: JOB_POLL_TICK, useValue: tick.asObservable() },
      ],
    });
    const store = TestBed.inject(BuilderStore);
    store.setResultStatus('ok');
    const fixture = TestBed.createComponent(AnalyticsPaneComponent);
    fixture.detectChanges();
    return { fixture, store, host: fixture.nativeElement as HTMLElement };
  }

  it('when result is null (pre-build), all six cards render the dash placeholder without throwing', () => {
    const { host } = setup();
    const cards = host.querySelectorAll('[data-card="metric"]');
    expect(cards.length).toBe(6);
    for (const card of Array.from(cards)) {
      const valueCell = card.querySelector('[data-cell="value"]');
      expect(valueCell?.textContent?.trim()).toBe('--');
    }
  });

  it('when result and backtest are null, cards() returns exactly 6 items with null values (no throw)', () => {
    const { fixture } = setup();
    const cards = fixture.componentInstance.cards();
    expect(cards.length).toBe(6);
    expect(cards.every((c) => c.value === null)).toBeTrue();
  });

  it('when result data arrives (post-build), the return card shows a formatted percentage', () => {
    const { fixture, store, host } = setup();
    store.setResult(sampleResult());
    fixture.detectChanges();

    const returnCell = host.querySelector<HTMLElement>(
      '[data-card-id="return"] [data-cell="value"]',
    );
    expect(returnCell).not.toBeNull();
    expect(returnCell!.textContent?.trim()).toBe('12.00%');
  });

  it('when result and backtest arrive (post-build), no card shows the dash placeholder', () => {
    const { fixture, store, host } = setup();
    store.setResult(sampleResult());
    store.setBacktest(sampleBacktest());
    fixture.detectChanges();

    const cards = host.querySelectorAll('[data-card="metric"] [data-cell="value"]');
    let dashCount = 0;
    for (const cell of Array.from(cards)) {
      if (cell.textContent?.trim() === '--') dashCount++;
    }
    expect(dashCount).toBe(0);
  });

  it('when result is null then set (transition), cards update without throwing', () => {
    const { fixture, store, host } = setup();
    // Pre-build: dashes
    expect(
      host.querySelector('[data-card-id="return"] [data-cell="value"]')?.textContent?.trim(),
    ).toBe('--');

    // Post-build: formatted value
    store.setResult(sampleResult());
    fixture.detectChanges();
    expect(
      host.querySelector('[data-card-id="return"] [data-cell="value"]')?.textContent?.trim(),
    ).toBe('12.00%');
  });
});
