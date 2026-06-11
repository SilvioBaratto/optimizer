/**
 * results-panel-render-coverage.spec.ts — issue #942
 *
 * Template-branch coverage for ResultsPanelComponent. Drives the empty-state vs
 * result body, the frontier-present vs not-available arms, the stat-card trend
 * branches, and the export / apply-weights outputs.
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, installResizeObserverStub, makeOptimizationRunResponse } from '../../../testing';
import { ICON_PROVIDER } from '../../icons';
import { ResultsPanelComponent } from './results-panel';
import type { OptimizationRunResponse } from '../../core/models/optimization.model';

function withFrontier(): OptimizationRunResponse {
  return makeOptimizationRunResponse({
    metrics: { annualized_sharpe_ratio: 1.2, annualized_return: 0.08, annualized_volatility: 0.15, max_drawdown: -0.2 },
    efficientFrontier: [
      { risk: 0.1, return: 0.08, sharpe: 0.8 },
      { risk: 0.05, return: 0.03, sharpe: 0.6 },
    ],
  });
}

describe('ResultsPanelComponent — render-coverage (#942)', () => {
  let fixture: ComponentFixture<ResultsPanelComponent>;
  let comp: ResultsPanelComponent;
  let el: HTMLElement;

  beforeEach(async () => {
    installResizeObserverStub();
    await configureTestBed({ imports: [ResultsPanelComponent], withHttp: false, providers: [ICON_PROVIDER] });
    fixture = TestBed.createComponent(ResultsPanelComponent);
    comp = fixture.componentInstance;
    el = fixture.nativeElement as HTMLElement;
  });

  function button(text: string): HTMLButtonElement | undefined {
    return Array.from(el.querySelectorAll('button')).find((b) => b.textContent?.includes(text)) as
      | HTMLButtonElement
      | undefined;
  }

  it('when there is no result, the empty state is shown', () => {
    fixture.componentRef.setInput('hasResult', false);
    fixture.detectChanges();
    expect(el.textContent).toContain('No optimization run yet');
  });

  it('when a result with a frontier is present, the chart and tables render', () => {
    fixture.componentRef.setInput('result', withFrontier());
    fixture.componentRef.setInput('hasResult', true);
    fixture.detectChanges();
    expect(el.querySelector('app-echarts-efficient-frontier')).not.toBeNull();
    expect(el.querySelectorAll('app-stat-card').length).toBe(4);
    expect(el.querySelectorAll('app-data-table').length).toBe(2);
  });

  it('when a result has no frontier, the not-available message is shown', () => {
    fixture.componentRef.setInput('result', makeOptimizationRunResponse({ efficientFrontier: null }));
    fixture.componentRef.setInput('hasResult', true);
    fixture.detectChanges();
    expect(el.querySelector('app-echarts-efficient-frontier')).toBeNull();
    expect(el.textContent).toContain('Efficient frontier not available');
  });

  it('the stat cards reflect up/down/flat trends from the metrics', () => {
    fixture.componentRef.setInput('result', makeOptimizationRunResponse({
      metrics: { annualized_sharpe_ratio: 1.2, max_drawdown: -0.2 }, // annual_return/vol missing → flat
    }));
    fixture.componentRef.setInput('hasResult', true);
    fixture.detectChanges();
    const cards = comp.statsCards();
    expect(cards.find((c) => c.label === 'Sharpe Ratio')?.trend).toBe('up');
    expect(cards.find((c) => c.label === 'Max Drawdown')?.trend).toBe('down');
    expect(cards.find((c) => c.label === 'Annual Return')?.trend).toBe('flat');
  });

  it('optimal and assetMarkers computeds pick the tangency and min-risk points', () => {
    fixture.componentRef.setInput('result', withFrontier());
    fixture.componentRef.setInput('hasResult', true);
    fixture.detectChanges();
    expect(comp.optimal()?.sharpe).toBe(0.8);
    expect(comp.assetMarkers()[0].risk).toBe(0.05);
  });

  it('when there is no frontier, optimal is null and assetMarkers is empty', () => {
    fixture.componentRef.setInput('result', makeOptimizationRunResponse({ efficientFrontier: null }));
    fixture.componentRef.setInput('hasResult', true);
    fixture.detectChanges();
    expect(comp.optimal()).toBeNull();
    expect(comp.assetMarkers()).toEqual([]);
  });

  it('clicking Apply Weights emits the run weights', () => {
    fixture.componentRef.setInput('result', makeOptimizationRunResponse());
    fixture.componentRef.setInput('hasResult', true);
    fixture.detectChanges();
    let emitted: Record<string, number> | undefined;
    comp.applyWeights.subscribe((w) => (emitted = w));
    button('Apply Weights')!.click();
    expect(emitted).toEqual({ AAPL: 0.6, MSFT: 0.4 });
  });

  it('onApplyWeights does not emit when the run has no weights', () => {
    fixture.componentRef.setInput('result', makeOptimizationRunResponse({ weights: undefined as unknown as Record<string, number> }));
    fixture.componentRef.setInput('hasResult', true);
    fixture.detectChanges();
    let emitted = false;
    comp.applyWeights.subscribe(() => (emitted = true));
    comp.onApplyWeights();
    expect(emitted).toBe(false);
  });

  it('clicking Export Results triggers a JSON download', () => {
    fixture.componentRef.setInput('result', makeOptimizationRunResponse());
    fixture.componentRef.setInput('hasResult', true);
    fixture.detectChanges();
    const createSpy = spyOn(URL, 'createObjectURL').and.returnValue('blob:results');
    spyOn(URL, 'revokeObjectURL');
    button('Export Results')!.click();
    expect(createSpy).toHaveBeenCalled();
  });

  it('exportResults is a no-op when there is no run', () => {
    const createSpy = spyOn(URL, 'createObjectURL');
    comp.exportResults();
    expect(createSpy).not.toHaveBeenCalled();
  });
});
