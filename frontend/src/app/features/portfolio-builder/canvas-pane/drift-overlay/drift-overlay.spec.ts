import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { TestBed, type ComponentFixture } from '@angular/core/testing';

import { DriftOverlayComponent } from './drift-overlay';
import { BuilderStore } from '../../builder.store';
import type {
  BuilderDrift,
  BuilderDriftDiagnostics,
} from '../../builder-result.model';
import { BUILDER_DRIFT_SERVICE } from '../../builder.store';

const PALETTE = [
  '#111111',
  '#222222',
  '#333333',
  '#444444',
  '#555555',
  '#666666',
  '#777777',
  '#888888',
];

function setCssVars(): void {
  PALETTE.forEach((value, index) => {
    document.documentElement.style.setProperty(
      `--color-chart-${index + 1}`,
      value,
    );
  });
}

function clearCssVars(): void {
  PALETTE.forEach((_, index) => {
    document.documentElement.style.removeProperty(`--color-chart-${index + 1}`);
  });
}

function sampleDrift(): BuilderDrift {
  return {
    portfolioName: 'core',
    totals: {
      deployable_eur: 10000,
      total_holdings_eur: 10000,
      total_drift_abs: 0.1,
      buy_eur: 500,
      sell_eur: 500,
    },
    drift: [
      {
        ticker: 'AAPL',
        current_weight: 0.6,
        target_weight: 0.5,
        delta_weight: 0.1,
        eur_value: 6000,
        flags: [],
      },
      {
        ticker: 'MSFT',
        current_weight: 0.3,
        target_weight: 0.5,
        delta_weight: -0.2,
        eur_value: 3000,
        flags: [],
      },
      {
        ticker: 'GOOG',
        current_weight: 0.1,
        target_weight: 0.0,
        delta_weight: 0.05,
        eur_value: 1000,
        flags: [],
      },
    ],
  };
}

function setup(): {
  fixture: ComponentFixture<DriftOverlayComponent>;
  store: BuilderStore;
  host: HTMLElement;
  driftRunner: { runExplicit: jasmine.Spy };
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
  return {
    fixture,
    store,
    host: fixture.nativeElement as HTMLElement,
    driftRunner,
  };
}

function diagnosticsFor(
  entries: ReadonlyArray<{
    code:
      | 'unmapped'
      | 'fx_missing'
      | 'stale_price'
      | 'reconciliation_mismatch'
      | 'target_not_on_broker';
    reason?: string;
    reference?: string | null;
    ticker?: string | null;
  }>,
): BuilderDriftDiagnostics {
  return {
    requestId: 1,
    diagnostics: {
      reconciliation_ok: true,
      reconciliation_delta_pct: 0,
      unmapped_count: 0,
      fx_missing_count: 0,
      target_not_on_broker_count: 0,
      base_currency: 'EUR',
      sum_eur: 10000,
      invested_eur: 10000,
      delta_eur: 0,
      tolerance_pct: 0.015,
      stale_price_count: 0,
      entries: entries.map((e) => ({
        code: e.code,
        reason: e.reason ?? '',
        reference: e.reference ?? null,
        ticker: e.ticker ?? null,
      })),
    },
  };
}

describe('DriftOverlayComponent', () => {
  beforeAll(() => setCssVars());
  afterAll(() => clearCssVars());

  it('when freshly mounted with no drift, drift-overlay region renders with empty placeholder', () => {
    const { host } = setup();
    expect(host.querySelector('[data-region="drift-overlay"]')).not.toBeNull();
    expect(host.querySelector('[data-region="drift-header"]')).not.toBeNull();
    expect(host.querySelector('[data-region="drift-chart"]')).not.toBeNull();
    expect(host.querySelector('[data-region="drift-empty"]')).not.toBeNull();
  });

  it('when store.drift is null, app-echarts-bar and app-echarts-donut are not rendered', () => {
    const { host } = setup();
    expect(host.querySelector('app-echarts-bar')).toBeNull();
    expect(host.querySelector('app-echarts-donut')).toBeNull();
  });

  it('when drift is populated, default view-mode renders app-echarts-bar', () => {
    const { fixture, store, host } = setup();
    store.setDrift(sampleDrift());
    fixture.detectChanges();
    expect(host.querySelector('app-echarts-bar')).not.toBeNull();
    expect(host.querySelector('app-echarts-donut')).toBeNull();
    expect(host.querySelector('[data-region="drift-empty"]')).toBeNull();
  });

  it('when view-mode is toggled to donut, app-echarts-donut renders and app-echarts-bar is removed', () => {
    const { fixture, store, host } = setup();
    store.setDrift(sampleDrift());
    fixture.detectChanges();
    fixture.componentInstance.setViewMode('donut');
    fixture.detectChanges();
    expect(host.querySelector('app-echarts-donut')).not.toBeNull();
    expect(host.querySelector('app-echarts-bar')).toBeNull();
  });

  it('when view-mode button "donut" is clicked, viewMode signal becomes "donut"', () => {
    const { fixture, host } = setup();
    const donutBtn = host.querySelector<HTMLButtonElement>(
      '[data-mode-btn][data-mode-id="donut"]',
    );
    expect(donutBtn).not.toBeNull();
    donutBtn!.click();
    fixture.detectChanges();
    expect(fixture.componentInstance.viewMode()).toBe('donut');
  });

  it('when base toggle button "total" is clicked, store.deployableBase becomes "total"', () => {
    const { fixture, store, host } = setup();
    const totalBtn = host.querySelector<HTMLButtonElement>(
      '[data-base-btn][data-base-id="total"]',
    );
    expect(totalBtn).not.toBeNull();
    totalBtn!.click();
    fixture.detectChanges();
    expect(store.deployableBase()).toBe('total');
  });

  it('when base toggle is "invested", the invested button carries the active ring class', () => {
    const { host } = setup();
    const invested = host.querySelector<HTMLButtonElement>(
      '[data-base-btn][data-base-id="invested"]',
    );
    const total = host.querySelector<HTMLButtonElement>(
      '[data-base-btn][data-base-id="total"]',
    );
    expect(invested!.classList.contains('ring-2')).toBe(true);
    expect(total!.classList.contains('ring-2')).toBe(false);
  });

  it('when drift is populated, barSeries rows are sorted by |delta_weight| descending', () => {
    const { fixture, store } = setup();
    store.setDrift(sampleDrift());
    fixture.detectChanges();
    const tickers = fixture.componentInstance.barSeries().map((r) => r.ticker);
    expect(tickers).toEqual(['MSFT', 'AAPL', 'GOOG']);
  });

  it('when drift is populated, palette colors are keyed by tickers sorted by |current_weight| desc', () => {
    const { fixture, store } = setup();
    store.setDrift(sampleDrift());
    fixture.detectChanges();
    const byTicker = new Map(
      fixture.componentInstance
        .barSeries()
        .map((r) => [r.ticker, r.color] as const),
    );
    expect(byTicker.get('AAPL')).toBe(PALETTE[0]);
    expect(byTicker.get('MSFT')).toBe(PALETTE[1]);
    expect(byTicker.get('GOOG')).toBe(PALETTE[2]);
  });

  it('when drift is populated, barData passed to app-echarts-bar carries color props from the palette', () => {
    const { fixture, store } = setup();
    store.setDrift(sampleDrift());
    fixture.detectChanges();
    const data = fixture.componentInstance.barData();
    expect(data.length).toBe(3);
    for (const row of data) {
      expect(typeof row.color).toBe('string');
      expect(row.color!.length).toBeGreaterThan(0);
    }
  });

  it('when drift is populated, donutSegments expose current weights with palette colors', () => {
    const { fixture, store } = setup();
    store.setDrift(sampleDrift());
    fixture.detectChanges();
    const segs = fixture.componentInstance.donutSegments();
    const aapl = segs.find((s) => s.label === 'AAPL')!;
    expect(aapl.value).toBe(0.6);
    expect(aapl.color).toBe(PALETTE[0]);
  });

  it('when DriftDiagnostics.entries carries a flag for a ticker, drift-flag-row renders the badge', () => {
    const { fixture, store, host } = setup();
    store.setDrift(sampleDrift());
    store.setDriftDiagnostics(
      diagnosticsFor([
        { code: 'stale_price', reason: 'tick old', reference: 'AAPL', ticker: 'AAPL' },
      ]),
    );
    fixture.detectChanges();

    const row = host.querySelector('[data-region="drift-flag-row"][data-row-ticker="AAPL"]');
    expect(row).not.toBeNull();
    expect(row!.querySelector('app-diagnostic-badge')).not.toBeNull();
  });

  it('when ticker has UNMAPPED flag, the flag row is greyed (opacity-50)', () => {
    const { fixture, store, host } = setup();
    store.setDrift(sampleDrift());
    store.setDriftDiagnostics(
      diagnosticsFor([
        { code: 'unmapped', reason: 'no mapping', reference: 'AAPL', ticker: 'AAPL' },
      ]),
    );
    fixture.detectChanges();

    const row = host.querySelector('[data-region="drift-flag-row"][data-row-ticker="AAPL"]');
    expect(row).not.toBeNull();
    expect(row!.getAttribute('data-row-greyed')).toBe('true');
    expect(row!.className).toContain('opacity-50');
  });

  it('when ticker has TARGET_NOT_ON_BROKER flag, the flag row is greyed', () => {
    const { fixture, store, host } = setup();
    store.setDrift(sampleDrift());
    store.setDriftDiagnostics(
      diagnosticsFor([
        {
          code: 'target_not_on_broker',
          reason: 'not on broker',
          reference: 'MSFT',
          ticker: 'MSFT',
        },
      ]),
    );
    fixture.detectChanges();

    const row = host.querySelector('[data-region="drift-flag-row"][data-row-ticker="MSFT"]');
    expect(row).not.toBeNull();
    expect(row!.getAttribute('data-row-greyed')).toBe('true');
    expect(row!.className).toContain('opacity-50');
  });

  it('when ticker has STALE_PRICE flag only, the flag row is NOT greyed', () => {
    const { fixture, store, host } = setup();
    store.setDrift(sampleDrift());
    store.setDriftDiagnostics(
      diagnosticsFor([
        { code: 'stale_price', reason: 'tick old', reference: 'AAPL', ticker: 'AAPL' },
      ]),
    );
    fixture.detectChanges();

    const row = host.querySelector('[data-region="drift-flag-row"][data-row-ticker="AAPL"]');
    expect(row).not.toBeNull();
    expect(row!.getAttribute('data-row-greyed')).toBe('false');
    expect(row!.className).not.toContain('opacity-50');
  });

  it('when no DriftRow carries flags, drift-flag-list region is not rendered', () => {
    const { fixture, store, host } = setup();
    store.setDrift(sampleDrift());
    fixture.detectChanges();
    expect(host.querySelector('[data-region="drift-flag-list"]')).toBeNull();
  });

  it('when ticker has multiple flags, all badges render for that row', () => {
    const { fixture, store, host } = setup();
    store.setDrift(sampleDrift());
    store.setDriftDiagnostics(
      diagnosticsFor([
        { code: 'stale_price', ticker: 'AAPL' },
        { code: 'fx_missing', ticker: 'AAPL' },
      ]),
    );
    fixture.detectChanges();

    const badges = host.querySelectorAll(
      '[data-row-ticker="AAPL"] app-diagnostic-badge',
    );
    expect(badges.length).toBe(2);
  });

  it('when ticker has UNMAPPED in donut view, flag row still renders', () => {
    const { fixture, store, host } = setup();
    store.setDrift(sampleDrift());
    store.setDriftDiagnostics(
      diagnosticsFor([{ code: 'unmapped', ticker: 'AAPL' }]),
    );
    fixture.componentInstance.setViewMode('donut');
    fixture.detectChanges();
    expect(host.querySelector('app-echarts-donut')).not.toBeNull();
    expect(
      host.querySelector('[data-region="drift-flag-row"][data-row-ticker="AAPL"]'),
    ).not.toBeNull();
  });

  it('when refresh button is clicked, store.triggerDriftRun fires and BUILDER_DRIFT_SERVICE.runExplicit is invoked', () => {
    const { fixture, host, driftRunner } = setup();
    const refresh = host.querySelector<HTMLButtonElement>(
      '[data-region="drift-refresh"]',
    );
    expect(refresh).not.toBeNull();
    refresh!.click();
    fixture.detectChanges();
    expect(driftRunner.runExplicit).toHaveBeenCalledTimes(1);
  });
});
