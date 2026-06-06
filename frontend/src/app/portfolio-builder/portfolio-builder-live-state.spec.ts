import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed, type ComponentFixture } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { Subject } from 'rxjs';

import { PortfolioBuilderComponent } from './portfolio-builder';
import { BuilderStore, type CanvasView } from './state/builder.store';
import { BuilderResultService } from './builder-result.service';
import { JOB_POLL_TICK } from '../shared/job-progress-tracker/job-progress-tracker';

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

let activeFixture: ComponentFixture<PortfolioBuilderComponent> | null = null;

function setup(): {
  fixture: ComponentFixture<PortfolioBuilderComponent>;
  store: BuilderStore;
  host: HTMLElement;
  runExplicitSpy: jasmine.Spy;
} {
  const tick = new Subject<number>();
  const runExplicitSpy = spyOn(
    BuilderResultService.prototype,
    'runExplicit',
  ).and.callThrough();
  TestBed.configureTestingModule({
    providers: [
      provideZonelessChangeDetection(),
      provideHttpClient(),
      provideHttpClientTesting(),
      { provide: JOB_POLL_TICK, useValue: tick.asObservable() },
    ],
  });
  const fixture = TestBed.createComponent(PortfolioBuilderComponent);
  fixture.detectChanges();
  activeFixture = fixture;
  const store = fixture.debugElement.injector.get(BuilderStore);
  return {
    fixture,
    store,
    host: fixture.nativeElement as HTMLElement,
    runExplicitSpy,
  };
}

function canvasPane(host: HTMLElement): HTMLElement {
  const el = host.querySelector<HTMLElement>('[data-region="canvas-pane"]');
  if (!el) throw new Error('canvas-pane not mounted');
  return el;
}

function analyticsPane(host: HTMLElement): HTMLElement {
  const el = host.querySelector<HTMLElement>('[data-region="analytics-pane"]');
  if (!el) throw new Error('analytics-pane not mounted');
  return el;
}

function chartSelectorFor(view: CanvasView): string {
  switch (view) {
    case 'donut':
      return 'app-allocation-donut';
    case 'bars':
      return 'app-weight-bars';
    case 'frontier':
      return 'app-efficient-frontier';
    case 'backtest':
      return 'app-backtest-sparkline';
    case 'drift':
      return 'app-drift-overlay';
  }
}

function primeOkState(
  fixture: ComponentFixture<PortfolioBuilderComponent>,
  store: BuilderStore,
): void {
  // Drive status only — chart components mount their `app-*` selector in
  // either empty-state or data-state. Avoiding result/frontier/backtest
  // seeding keeps ECharts dynamic-import async work out of the test loop.
  store.setResultStatus('ok');
  fixture.detectChanges();
}

describe('PortfolioBuilder live-state integration', () => {
  beforeAll(() => setCssVars());
  afterAll(() => clearCssVars());
  afterEach(() => {
    activeFixture?.destroy();
    activeFixture = null;
  });

  describe('idle path', () => {
    it('when mounted with no config and no session, both panes render [data-region="overlay-idle"]', () => {
      const { fixture, store, host } = setup();
      store.setResultStatus('idle');
      fixture.detectChanges();

      expect(
        canvasPane(host).querySelector('[data-region="overlay-idle"]'),
      ).not.toBeNull();
      expect(
        analyticsPane(host).querySelector('[data-region="overlay-idle"]'),
      ).not.toBeNull();
    });
  });

  describe('running path', () => {
    it('when resultStatus is "running", both panes render [data-region="overlay-loading"]', () => {
      const { fixture, store, host } = setup();
      store.setResultStatus('running');
      fixture.detectChanges();

      expect(
        canvasPane(host).querySelector('[data-region="overlay-loading"]'),
      ).not.toBeNull();
      expect(
        analyticsPane(host).querySelector('[data-region="overlay-loading"]'),
      ).not.toBeNull();
    });

    it('when resultStatus is "running", no chart component and no [data-card="metric"] is in the DOM', () => {
      const { fixture, store, host } = setup();
      store.setResultStatus('running');
      fixture.detectChanges();

      const charts = [
        'app-allocation-donut',
        'app-weight-bars',
        'app-efficient-frontier',
        'app-backtest-sparkline',
      ];
      for (const sel of charts) {
        expect(host.querySelector(sel)).toBeNull();
      }
      expect(host.querySelectorAll('[data-card="metric"]').length).toBe(0);
    });
  });

  describe('ok path with view swaps', () => {
    it('when resultStatus is "ok" and currentView iterates over all four views, only the matching chart is mounted in canvas-pane and analytics-pane keeps six metric cards', () => {
      const { fixture, store, host } = setup();
      primeOkState(fixture, store);

      const views: readonly CanvasView[] = [
        'donut',
        'bars',
        'frontier',
        'backtest',
      ];
      for (const view of views) {
        store.setCurrentView(view);
        fixture.detectChanges();

        const pane = canvasPane(host);
        const activeSel = chartSelectorFor(view);
        expect(pane.querySelector(activeSel)).not.toBeNull();
        for (const other of views) {
          if (other === view) continue;
          expect(pane.querySelector(chartSelectorFor(other))).toBeNull();
        }
        expect(
          analyticsPane(host).querySelectorAll('[data-card="metric"]').length,
        ).toBe(6);
      }
    });
  });

  describe('currentView persists across stage navigation', () => {
    it('when currentView is set to "frontier" and the user navigates rebalance → universe, currentView remains "frontier" and app-efficient-frontier is rendered', () => {
      const { fixture, store, host } = setup();
      primeOkState(fixture, store);
      store.setCurrentView('frontier');
      fixture.detectChanges();

      store.setStage('rebalance');
      fixture.detectChanges();
      store.setStage('universe');
      fixture.detectChanges();

      expect(store.currentView()).toBe('frontier');
      expect(
        canvasPane(host).querySelector('app-efficient-frontier'),
      ).not.toBeNull();
    });
  });

  describe('stale path', () => {
    it('when resultStatus transitions ok → stale, both panes render [data-region="overlay-stale-badge"] and the prior chart + metric cards remain in the DOM', () => {
      const { fixture, store, host } = setup();
      primeOkState(fixture, store);
      store.setCurrentView('donut');
      fixture.detectChanges();

      store.setResultStatus('stale');
      fixture.detectChanges();

      expect(
        canvasPane(host).querySelector('[data-region="overlay-stale-badge"]'),
      ).not.toBeNull();
      expect(
        analyticsPane(host).querySelector('[data-region="overlay-stale-badge"]'),
      ).not.toBeNull();
      expect(canvasPane(host).querySelector('app-allocation-donut')).not.toBeNull();
      expect(
        analyticsPane(host).querySelectorAll('[data-card="metric"]').length,
      ).toBe(6);
    });
  });

  describe('error-retry path', () => {
    it('when resultStatus is "error", [data-action="retry"] is in canvas-pane and clicking it invokes BuilderResultService.runExplicit exactly once', () => {
      const { fixture, store, host, runExplicitSpy } = setup();
      store.setResultStatus('error');
      fixture.detectChanges();

      const btn = canvasPane(host).querySelector<HTMLButtonElement>(
        '[data-action="retry"]',
      );
      expect(btn).not.toBeNull();
      runExplicitSpy.calls.reset();
      btn!.click();

      expect(runExplicitSpy).toHaveBeenCalledTimes(1);
    });

    it('when resultStatus is "error" then forced through "running" → "ok", the error overlay is cleared from both panes', () => {
      const { fixture, store, host } = setup();
      store.setResultStatus('error');
      fixture.detectChanges();
      expect(
        canvasPane(host).querySelector('[data-region="overlay-error"]'),
      ).not.toBeNull();

      store.setResultStatus('running');
      fixture.detectChanges();
      store.setResultStatus('ok');
      fixture.detectChanges();

      expect(
        canvasPane(host).querySelector('[data-region="overlay-error"]'),
      ).toBeNull();
      expect(
        analyticsPane(host).querySelector('[data-region="overlay-error"]'),
      ).toBeNull();
    });
  });
});
