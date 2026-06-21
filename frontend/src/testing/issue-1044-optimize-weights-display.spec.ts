/**
 * Issue #1044 — Criterion 1 [UNIT]
 *
 * "After Run, results show a weights donut + bars sourced from OptimizeResult.weights."
 *
 * Tests ResultsPanelComponent directly (it owns the donut + bars charts).
 */

import { CUSTOM_ELEMENTS_SCHEMA } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { By } from '@angular/platform-browser';
import { provideNoopAnimations } from '@angular/platform-browser/animations';
import { installResizeObserverStub } from './test-globals';
import { ResultsPanelComponent } from '../app/optimization-studio/results-panel/results-panel';

function makeWeightResult(weights: Record<string, number>) {
  return {
    id: 'run-1044',
    status: 'completed',
    optimizerType: 'mean_risk',
    universeTickers: Object.keys(weights),
    config: {},
    weights,
    metrics: {
      annualized_return: 0.12,
      annualized_volatility: 0.14,
      annualized_sharpe_ratio: 0.86,
      annualized_sortino_ratio: 1.1,
      max_drawdown: -0.08,
    },
    riskContributions: Object.fromEntries(Object.keys(weights).map((t) => [t, 0.1])),
    efficientFrontier: [],
    errorMessage: null,
    solverLog: null,
    durationSeconds: 1.0,
    createdAt: '2026-01-01T00:00:00Z',
    updatedAt: '2026-01-01T00:00:01Z',
    portfolioId: null,
    jobId: null,
  };
}

describe('ResultsPanelComponent — weights display (issue #1044 criterion 1)', () => {
  let fixture: ComponentFixture<ResultsPanelComponent>;

  beforeEach(async () => {
    installResizeObserverStub();
    await TestBed.configureTestingModule({
      imports: [ResultsPanelComponent],
      providers: [provideNoopAnimations()],
      schemas: [CUSTOM_ELEMENTS_SCHEMA],
    }).compileComponents();
    fixture = TestBed.createComponent(ResultsPanelComponent);
  });

  // ── Presence checks ───────────────────────────────────────────────────────

  it('when result weights are available, app-echarts-donut is rendered', () => {
    const result = makeWeightResult({ AAPL: 0.6, MSFT: 0.4 });
    fixture.componentRef.setInput('result', result);
    fixture.componentRef.setInput('hasResult', true);
    fixture.detectChanges();

    expect(fixture.debugElement.query(By.css('app-echarts-donut')))
      .withContext('app-echarts-donut must be present when result has weights')
      .not.toBeNull();
  });

  it('when result weights are available, app-echarts-bar is rendered', () => {
    const result = makeWeightResult({ AAPL: 0.6, MSFT: 0.4 });
    fixture.componentRef.setInput('result', result);
    fixture.componentRef.setInput('hasResult', true);
    fixture.detectChanges();

    expect(fixture.debugElement.query(By.css('app-echarts-bar')))
      .withContext('app-echarts-bar must be present when result has weights')
      .not.toBeNull();
  });

  // ── Pre-run absence ───────────────────────────────────────────────────────

  it('when no result is set, app-echarts-donut is absent from the DOM', () => {
    fixture.componentRef.setInput('result', null);
    fixture.componentRef.setInput('hasResult', false);
    fixture.detectChanges();

    expect(fixture.debugElement.query(By.css('app-echarts-donut'))).toBeNull();
  });

  it('when no result is set, app-echarts-bar is absent from the DOM', () => {
    fixture.componentRef.setInput('result', null);
    fixture.componentRef.setInput('hasResult', false);
    fixture.detectChanges();

    expect(fixture.debugElement.query(By.css('app-echarts-bar'))).toBeNull();
  });

  // ── Invariant: any non-empty weights renders both charts ──────────────────

  const weightSamples: Array<[string, Record<string, number>]> = [
    ['single asset (100%)', { AAPL: 1.0 }],
    ['two assets equal', { AAPL: 0.5, MSFT: 0.5 }],
    ['four assets varying', { AAPL: 0.4, MSFT: 0.3, GOOGL: 0.2, AMZN: 0.1 }],
    ['ten assets equal-weight', Object.fromEntries(
      Array.from({ length: 10 }, (_, i) => [`T${i}`, 0.1]),
    )],
  ];

  weightSamples.forEach(([label, weights]) => {
    it(`when weights are [${label}], both donut and bar are rendered`, () => {
      fixture.componentRef.setInput('result', makeWeightResult(weights));
      fixture.componentRef.setInput('hasResult', true);
      fixture.detectChanges();

      expect(fixture.debugElement.query(By.css('app-echarts-donut')))
        .withContext(`donut missing for "${label}"`)
        .not.toBeNull();
      expect(fixture.debugElement.query(By.css('app-echarts-bar')))
        .withContext(`bar missing for "${label}"`)
        .not.toBeNull();
    });
  });
});
