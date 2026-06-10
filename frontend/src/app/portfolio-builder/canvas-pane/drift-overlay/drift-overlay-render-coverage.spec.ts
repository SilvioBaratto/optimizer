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
