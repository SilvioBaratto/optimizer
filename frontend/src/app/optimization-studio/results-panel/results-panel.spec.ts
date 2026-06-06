import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { ResultsPanelComponent } from './results-panel';
import type {
  EfficientFrontierPoint,
  OptimizationRunResponse,
} from '../../core/models/optimization.model';

function makeFrontier(): EfficientFrontierPoint[] {
  return [
    { risk: 0.05, return: 0.03, sharpe: 0.6, label: 'low' },
    { risk: 0.12, return: 0.1, sharpe: 0.833 },
    { risk: 0.08, return: 0.07, sharpe: 0.875 },
  ];
}

function makeRun(
  frontier: unknown[] = makeFrontier(),
  overrides: Partial<OptimizationRunResponse> = {},
): OptimizationRunResponse {
  return {
    id: 'run-1',
    status: 'completed',
    optimizerType: 'mean_variance',
    universeTickers: ['AAPL', 'MSFT'],
    durationSeconds: 1.23,
    weights: { AAPL: 0.5, MSFT: 0.5 },
    riskContributions: { AAPL: 0.5, MSFT: 0.5 },
    metrics: {},
    efficientFrontier: frontier as EfficientFrontierPoint[],
    ...overrides,
  } as OptimizationRunResponse;
}

describe('ResultsPanelComponent — efficient-frontier wiring (#452)', () => {
  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [ResultsPanelComponent],
      providers: [provideZonelessChangeDetection()],
    });
  });

  function render(
    run: OptimizationRunResponse | null,
    hasResult: boolean,
  ): HTMLElement {
    const fixture = TestBed.createComponent(ResultsPanelComponent);
    fixture.componentRef.setInput('result', run);
    fixture.componentRef.setInput('hasResult', hasResult);
    fixture.detectChanges();
    return fixture.nativeElement as HTMLElement;
  }

  it('does NOT render <app-echarts-efficient-frontier> when hasResult() is false', () => {
    const el = render(null, false);
    expect(el.querySelectorAll('app-echarts-efficient-frontier').length).toBe(0);
  });

  it('renders <app-echarts-efficient-frontier> exactly once when the run has a non-empty efficient frontier', () => {
    const el = render(makeRun(), true);
    expect(el.querySelectorAll('app-echarts-efficient-frontier').length).toBe(1);
  });

  it('does NOT render the frontier element when efficientFrontier is empty', () => {
    const el = render(makeRun([]), true);
    expect(el.querySelectorAll('app-echarts-efficient-frontier').length).toBe(0);
  });

  it('does NOT render the frontier element when efficientFrontier is malformed (non-array)', () => {
    const run = makeRun([]);
    // Intentional malformed shape: the type expects an array; we override to a non-array.
    (run as unknown as { efficientFrontier: unknown }).efficientFrontier = {
      bogus: 1,
    };

    let el!: HTMLElement;
    expect(() => {
      el = render(run, true);
    }).not.toThrow();

    expect(el.querySelectorAll('app-echarts-efficient-frontier').length).toBe(0);
  });

  it('does NOT emit an EchartsScatterComponent anywhere in the template', () => {
    const el = render(makeRun(), true);
    expect(el.querySelectorAll('app-echarts-scatter').length).toBe(0);
  });

  it('frontier() returns the raw EfficientFrontierPoint[] in source order', () => {
    const fixture = TestBed.createComponent(ResultsPanelComponent);
    fixture.componentRef.setInput('result', makeRun());
    fixture.componentRef.setInput('hasResult', true);
    fixture.detectChanges();

    const frontier = fixture.componentInstance.frontier();
    expect(frontier.length).toBe(3);
    expect(frontier[0]).toEqual(jasmine.objectContaining({ risk: 0.05, return: 0.03, label: 'low' }));
    // No ScatterPoint adapter shape (x/y) present anywhere.
    for (const p of frontier) {
      expect((p as unknown as { x?: unknown }).x).toBeUndefined();
      expect((p as unknown as { y?: unknown }).y).toBeUndefined();
    }
  });

  it('optimal() picks the max-Sharpe EfficientFrontierPoint', () => {
    const fixture = TestBed.createComponent(ResultsPanelComponent);
    fixture.componentRef.setInput('result', makeRun());
    fixture.componentRef.setInput('hasResult', true);
    fixture.detectChanges();

    const optimal = fixture.componentInstance.optimal();
    expect(optimal).not.toBeNull();
    // Highest sharpe in fixture is 0.875 at (risk=0.08, return=0.07)
    expect(optimal!.sharpe).toBeCloseTo(0.875, 6);
    expect(optimal!.risk).toBeCloseTo(0.08, 6);
    expect(optimal!.return).toBeCloseTo(0.07, 6);
  });

  it('optimal() returns null when the frontier is empty', () => {
    const fixture = TestBed.createComponent(ResultsPanelComponent);
    fixture.componentRef.setInput('result', makeRun([]));
    fixture.componentRef.setInput('hasResult', true);
    fixture.detectChanges();

    expect(fixture.componentInstance.optimal()).toBeNull();
  });

  it('assetMarkers() has exactly one entry — the minimum-risk point — when the frontier is non-empty', () => {
    const fixture = TestBed.createComponent(ResultsPanelComponent);
    fixture.componentRef.setInput('result', makeRun());
    fixture.componentRef.setInput('hasResult', true);
    fixture.detectChanges();

    const markers = fixture.componentInstance.assetMarkers();
    expect(markers.length).toBe(1);
    // Minimum risk is 0.05 (the first fixture point).
    expect(markers[0].risk).toBeCloseTo(0.05, 6);
    expect(markers[0].return).toBeCloseTo(0.03, 6);
  });

  it('assetMarkers() is [] when the frontier is empty', () => {
    const fixture = TestBed.createComponent(ResultsPanelComponent);
    fixture.componentRef.setInput('result', makeRun([]));
    fixture.componentRef.setInput('hasResult', true);
    fixture.detectChanges();

    expect(fixture.componentInstance.assetMarkers()).toEqual([]);
  });
});
