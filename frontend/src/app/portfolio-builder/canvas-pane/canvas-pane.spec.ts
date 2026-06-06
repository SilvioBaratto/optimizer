import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { TestBed, type ComponentFixture } from '@angular/core/testing';

import { CanvasPaneComponent } from './canvas-pane';
import { BuilderStore, type CanvasView } from '../state/builder.store';
import { BuilderResultService } from '../builder-result.service';
import type { WeightItem } from '../../models/pipeline-builder.model';

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

function setup(): {
  fixture: ComponentFixture<CanvasPaneComponent>;
  store: BuilderStore;
  host: HTMLElement;
} {
  TestBed.configureTestingModule({
    providers: [
      provideZonelessChangeDetection(),
      provideHttpClient(),
      provideHttpClientTesting(),
      BuilderStore,
      BuilderResultService,
    ],
  });
  const store = TestBed.inject(BuilderStore);
  store.setResultStatus('ok');
  const fixture = TestBed.createComponent(CanvasPaneComponent);
  fixture.detectChanges();
  return { fixture, store, host: fixture.nativeElement as HTMLElement };
}

const CHART_SELECTORS = [
  'app-allocation-donut',
  'app-weight-bars',
  'app-efficient-frontier',
  'app-backtest-sparkline',
] as const;

function anyChartInDom(host: HTMLElement): boolean {
  return CHART_SELECTORS.some((sel) => host.querySelector(sel) !== null);
}

function viewButtons(host: HTMLElement): HTMLButtonElement[] {
  return Array.from(host.querySelectorAll<HTMLButtonElement>('[data-view-btn]'));
}

describe('CanvasPaneComponent', () => {
  beforeAll(() => setCssVars());
  afterAll(() => clearCssVars());

  it('when mounted, [data-region="canvas-pane"] container is present', () => {
    const { host } = setup();
    expect(host.querySelector('[data-region="canvas-pane"]')).not.toBeNull();
  });

  it('when mounted, [data-region="canvas-view-toggle"] segmented control row is present', () => {
    const { host } = setup();
    expect(
      host.querySelector('[data-region="canvas-view-toggle"]'),
    ).not.toBeNull();
  });

  it('when mounted, exactly five [data-view-btn] buttons are rendered', () => {
    const { host } = setup();
    expect(viewButtons(host).length).toBe(5);
  });

  it('when mounted, [data-region="canvas-view-content"] content region is present', () => {
    const { host } = setup();
    expect(
      host.querySelector('[data-region="canvas-view-content"]'),
    ).not.toBeNull();
  });

  it('when mounted, no chart element (<canvas> or <echarts>) is in the DOM', () => {
    const { host } = setup();
    expect(host.querySelector('canvas')).toBeNull();
    expect(host.querySelector('echarts')).toBeNull();
    expect(host.querySelector('[data-region="echarts"]')).toBeNull();
  });

  it('when mounted with default state, app-allocation-donut is rendered (donut is the default view)', () => {
    const { host } = setup();
    expect(host.querySelector('app-allocation-donut')).not.toBeNull();
  });

  it('when mounted, the donut button carries the active ring class while other buttons do not', () => {
    const { host } = setup();
    const buttons = viewButtons(host);
    expect(buttons[0].getAttribute('data-view-id')).toBe('donut');
    expect(buttons[0].classList.contains('ring-2')).toBe(true);
    for (let i = 1; i < buttons.length; i++) {
      expect(buttons[i].classList.contains('ring-2')).toBe(false);
    }
  });

  it('when store.setCurrentView is called with each of the five views, the active ring class moves to the matching button', () => {
    const { fixture, store, host } = setup();
    const views: readonly CanvasView[] = [
      'donut',
      'bars',
      'frontier',
      'backtest',
      'drift',
    ];
    for (const view of views) {
      store.setCurrentView(view);
      fixture.detectChanges();
      const buttons = viewButtons(host);
      for (const btn of buttons) {
        const isActive = btn.getAttribute('data-view-id') === view;
        expect(btn.classList.contains('ring-2')).toBe(isActive);
      }
    }
  });

  it('when a non-donut button is clicked, store.currentView updates to that view id', () => {
    const { fixture, store, host } = setup();
    const buttons = viewButtons(host);
    const barsBtn = buttons.find(
      (b) => b.getAttribute('data-view-id') === 'bars',
    );
    expect(barsBtn).toBeDefined();
    barsBtn!.click();
    fixture.detectChanges();
    expect(store.currentView()).toBe('bars');
  });

  it('when current view is "bars", app-weight-bars is mounted in the content region', () => {
    const { fixture, store, host } = setup();
    store.setCurrentView('bars');
    fixture.detectChanges();
    expect(host.querySelector('app-weight-bars')).not.toBeNull();
    expect(host.querySelector('app-allocation-donut')).toBeNull();
  });

  it('when current view is "frontier", app-efficient-frontier is mounted in the content region', () => {
    const { fixture, store, host } = setup();
    store.setCurrentView('frontier');
    fixture.detectChanges();
    expect(host.querySelector('app-efficient-frontier')).not.toBeNull();
    expect(host.querySelector('app-allocation-donut')).toBeNull();
  });

  it('when current view is "backtest", app-backtest-sparkline is mounted in the content region', () => {
    const { fixture, store, host } = setup();
    store.setCurrentView('backtest');
    fixture.detectChanges();
    expect(host.querySelector('app-backtest-sparkline')).not.toBeNull();
    expect(host.querySelector('app-allocation-donut')).toBeNull();
  });

  it('when Drift button is clicked, store.currentView transitions to "drift"', () => {
    const { fixture, store, host } = setup();
    const buttons = viewButtons(host);
    const driftBtn = buttons.find(
      (b) => b.getAttribute('data-view-id') === 'drift',
    );
    expect(driftBtn).toBeDefined();
    driftBtn!.click();
    fixture.detectChanges();
    expect(store.currentView()).toBe('drift');
  });

  it('when current view is "drift", both app-drift-overlay and app-trade-list mount in the content region', () => {
    const { fixture, store, host } = setup();
    store.setCurrentView('drift');
    fixture.detectChanges();
    expect(host.querySelector('app-drift-overlay')).not.toBeNull();
    expect(host.querySelector('app-trade-list')).not.toBeNull();
    expect(host.querySelector('[data-region="drift-split"]')).not.toBeNull();
  });

  it('when current view is "drift", the four legacy chart components are absent from the DOM', () => {
    const { fixture, store, host } = setup();
    store.setCurrentView('drift');
    fixture.detectChanges();
    for (const sel of CHART_SELECTORS) {
      expect(host.querySelector(sel)).toBeNull();
    }
  });

  it('when current view is not "drift", neither app-drift-overlay nor app-trade-list is in the DOM', () => {
    const { fixture, store, host } = setup();
    const nonDrift: readonly CanvasView[] = [
      'donut',
      'bars',
      'frontier',
      'backtest',
    ];
    for (const view of nonDrift) {
      store.setCurrentView(view);
      fixture.detectChanges();
      expect(host.querySelector('app-drift-overlay')).toBeNull();
      expect(host.querySelector('app-trade-list')).toBeNull();
    }
  });

  it('when weights input is set, the component accepts it without throwing', () => {
    const { fixture } = setup();
    const weights: WeightItem[] = [
      { ticker: 'AAPL', weight: 0.4 },
      { ticker: 'MSFT', weight: 0.6 },
    ];
    expect(() =>
      fixture.componentRef.setInput('weights', weights),
    ).not.toThrow();
    fixture.detectChanges();
    expect(fixture.componentInstance.weights()).toEqual(weights);
  });

  it('when frontier input is set with a list of points, the component accepts it without throwing', () => {
    const { fixture } = setup();
    const frontier: Record<string, number>[] = [
      { risk: 0.1, return: 0.05 },
      { risk: 0.2, return: 0.09 },
    ];
    expect(() =>
      fixture.componentRef.setInput('frontier', frontier),
    ).not.toThrow();
    fixture.detectChanges();
    expect(fixture.componentInstance.frontier()).toEqual(frontier);
  });

  it('when sessionId input is set to a string, the component accepts it without throwing', () => {
    const { fixture } = setup();
    expect(() =>
      fixture.componentRef.setInput('sessionId', 'sid-42'),
    ).not.toThrow();
    fixture.detectChanges();
    expect(fixture.componentInstance.sessionId()).toBe('sid-42');
  });

  it('when sessionId input is set to null, the component accepts it without throwing', () => {
    const { fixture } = setup();
    expect(() =>
      fixture.componentRef.setInput('sessionId', null),
    ).not.toThrow();
    fixture.detectChanges();
    expect(fixture.componentInstance.sessionId()).toBeNull();
  });

  it('when freshly constructed, all three legacy inputs default to undefined', () => {
    const { fixture } = setup();
    expect(fixture.componentInstance.weights()).toBeUndefined();
    expect(fixture.componentInstance.frontier()).toBeUndefined();
    expect(fixture.componentInstance.sessionId()).toBeUndefined();
  });

  describe('live-state overlay', () => {
    it('when resultStatus is "running", overlay-loading is present and no chart component is in the DOM', () => {
      const { fixture, store, host } = setup();
      store.setResultStatus('running');
      fixture.detectChanges();

      expect(host.querySelector('[data-region="overlay-loading"]')).not.toBeNull();
      expect(anyChartInDom(host)).toBe(false);
    });

    it('when resultStatus is "error", overlay-error is present and clicking [data-action="retry"] invokes BuilderResultService.runExplicit exactly once', () => {
      const { fixture, store, host } = setup();
      const resultService = TestBed.inject(BuilderResultService);
      const runSpy = spyOn(resultService, 'runExplicit');

      store.setResultStatus('error');
      fixture.detectChanges();

      expect(host.querySelector('[data-region="overlay-error"]')).not.toBeNull();
      const btn = host.querySelector<HTMLButtonElement>('[data-action="retry"]');
      expect(btn).not.toBeNull();
      btn!.click();

      expect(runSpy).toHaveBeenCalledTimes(1);
    });

    it('when resultStatus is "idle", overlay-idle is present and no chart component is in the DOM', () => {
      const { fixture, store, host } = setup();
      store.setResultStatus('idle');
      fixture.detectChanges();

      expect(host.querySelector('[data-region="overlay-idle"]')).not.toBeNull();
      expect(anyChartInDom(host)).toBe(false);
    });

    it('when resultStatus is "stale", the active-view chart IS in the DOM and overlay-stale-badge is present', () => {
      const { fixture, store, host } = setup();
      store.setResultStatus('stale');
      fixture.detectChanges();

      expect(host.querySelector('app-allocation-donut')).not.toBeNull();
      expect(host.querySelector('[data-region="overlay-stale-badge"]')).not.toBeNull();
    });

    it('when resultStatus is "ok", only the active-view chart is in the DOM and no overlay region is present', () => {
      const { fixture, store, host } = setup();
      store.setResultStatus('ok');
      fixture.detectChanges();

      expect(host.querySelector('app-allocation-donut')).not.toBeNull();
      expect(host.querySelector('[data-region="overlay-loading"]')).toBeNull();
      expect(host.querySelector('[data-region="overlay-error"]')).toBeNull();
      expect(host.querySelector('[data-region="overlay-idle"]')).toBeNull();
      expect(host.querySelector('[data-region="overlay-stale-badge"]')).toBeNull();
    });
  });
});
