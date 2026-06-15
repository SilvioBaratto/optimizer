/**
 * Render-coverage gap closure for DriftOverlayComponent — GitHub issue #904.
 *
 * PURPOSE
 * -------
 * The sibling drift-overlay.spec.ts already covers:
 *   - rows/empty branch, @switch viewMode bars|donut, flagged rows,
 *     greyed [class] binding, and the base-toggle ring-2 binding.
 *
 * The ONE residual uncovered gap is the **view-mode toggle buttons'**
 * `[class.ring-2]` / `[class.ring-accent]` bindings.  This spec closes
 * that micro-gap only.
 *
 * canvas-state-overlay NOTE (AC)
 * --------------------------------
 * `canvas-state-overlay.spec.ts` already provides complete branch coverage
 * for all 5 BuilderResultStatus values (running / error / idle / stale / ok)
 * plus the retry callback.  NO duplicate spec is created here.
 */

import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { TestBed, type ComponentFixture } from '@angular/core/testing';

import { DriftOverlayComponent } from './drift-overlay';
import { BuilderStore, BUILDER_DRIFT_SERVICE } from '../../state/builder.store';

// ---------------------------------------------------------------------------
// CSS-variable scaffolding — copied verbatim from drift-overlay.spec.ts
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Setup helper — copied verbatim from drift-overlay.spec.ts
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Specs — view-mode toggle [class.ring-2] / [class.ring-accent] bindings
// ---------------------------------------------------------------------------

describe('DriftOverlayComponent — view-mode toggle ring bindings (render-coverage)', () => {
  beforeAll(() => setCssVars());
  afterAll(() => clearCssVars());

  it('when viewMode is bars (default), the bars mode button carries ring-2 and donut does not', () => {
    const { host } = setup();

    const barsBtn = host.querySelector('[data-mode-btn][data-mode-id="bars"]');
    const donutBtn = host.querySelector('[data-mode-btn][data-mode-id="donut"]');

    expect(barsBtn).not.toBeNull();
    expect(donutBtn).not.toBeNull();

    expect(barsBtn!.classList.contains('ring-2')).toBe(true);
    expect(barsBtn!.classList.contains('ring-accent')).toBe(true);
    expect(donutBtn!.classList.contains('ring-2')).toBe(false);
  });

  it('when viewMode is set to donut, the donut mode button carries ring-2 and bars does not', () => {
    const { fixture, host } = setup();

    fixture.componentInstance.setViewMode('donut');
    fixture.detectChanges();

    const barsBtn = host.querySelector('[data-mode-btn][data-mode-id="bars"]');
    const donutBtn = host.querySelector('[data-mode-btn][data-mode-id="donut"]');

    expect(donutBtn).not.toBeNull();
    expect(barsBtn).not.toBeNull();

    expect(donutBtn!.classList.contains('ring-2')).toBe(true);
    expect(barsBtn!.classList.contains('ring-2')).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Previously-uncovered branches
//
// Branch 1: buildFlagMap() — when DiagnosticEntry.ticker is null, the entry
//           is skipped and does not appear in the flag map.
//
// Branch 2: combineFlags() — when the diagnostics flag-map has no entry for
//           a ticker but the DriftRow itself carries flags, those row-level
//           flags are used instead.
// ---------------------------------------------------------------------------

describe('DriftOverlayComponent — uncovered branches (render-coverage)', () => {
  beforeAll(() => setCssVars());
  afterAll(() => clearCssVars());

  function sampleDrift() {
    return {
      portfolioName: 'core',
      totals: {
        deployable_eur: 10000,
        total_holdings_eur: 10000,
        total_drift_abs: 0.05,
        buy_eur: 0,
        sell_eur: 500,
      },
      drift: [
        {
          ticker: 'AAPL',
          current_weight: 0.6,
          target_weight: 0.5,
          delta_weight: 0.1,
          eur_value: 6000,
          flags: [] as readonly import('../../../core/models/drift.model').FlagInstance[],
        },
      ],
    };
  }

  // Branch 1: null-ticker diagnostics entry is skipped
  it('when a DiagnosticEntry has ticker === null, that entry does not contribute flags to any row', () => {
    const { fixture, store } = setup();
    store.setDrift(sampleDrift());
    store.setDriftDiagnostics({
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
        entries: [
          // Entry with null ticker — must be ignored by buildFlagMap()
          { code: 'stale_price', reason: 'tick old', reference: null, ticker: null },
        ],
      },
    });
    fixture.detectChanges();

    // The null-ticker entry must not bleed into AAPL or any other row
    const aapl = fixture.componentInstance.barSeries().find((r) => r.ticker === 'AAPL');
    expect(aapl).not.toBeUndefined();
    expect(aapl!.flags.length).toBe(0);

    // No flag-list region should appear (null-ticker entries are not counted)
    expect(
      fixture.nativeElement.querySelector('[data-region="drift-flag-list"]'),
    ).toBeNull();
  });

  // Branch 2: DriftRow.flags are used when no diagnostics entry exists for that ticker
  it('when DriftRow.flags are non-empty and no diagnostics entry exists for that ticker, barSeries exposes those row-level flags', () => {
    const { fixture, store, host } = setup();
    const driftWithFlags = {
      ...sampleDrift(),
      drift: [
        {
          ticker: 'AAPL',
          current_weight: 0.6,
          target_weight: 0.5,
          delta_weight: 0.1,
          eur_value: 6000,
          // Flags come from the DriftRow itself, NOT from diagnostics entries
          flags: [
            { code: 'stale_price' as const, reason: 'price stale', reference: 'AAPL' },
          ],
        },
      ],
    };
    store.setDrift(driftWithFlags);
    // No driftDiagnostics set: flagMap will be empty for every ticker
    fixture.detectChanges();

    const aapl = fixture.componentInstance.barSeries().find((r) => r.ticker === 'AAPL');
    expect(aapl).not.toBeUndefined();
    // combineFlags() falls back to r.flags when flagMap has no entry
    expect(aapl!.flags.length).toBe(1);
    expect(aapl!.flags[0].code).toBe('stale_price');

    // The flag-list section renders because the row has flags
    const flagList = host.querySelector('[data-region="drift-flag-list"]');
    expect(flagList).not.toBeNull();
  });
});
