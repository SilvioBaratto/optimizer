/**
 * Issue #1044 — Criterion 3 [T3]
 *
 * "The echarts-efficient-frontier widget renders from the result."
 *
 * Tests ResultsPanelComponent directly (it already had the frontier wired at
 * results-panel.ts:80-92; this confirms it renders when data is present and
 * is absent when no result is set).
 */

import { CUSTOM_ELEMENTS_SCHEMA } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { By } from '@angular/platform-browser';
import { provideNoopAnimations } from '@angular/platform-browser/animations';
import { installResizeObserverStub } from './test-globals';
import { ResultsPanelComponent } from '../app/optimization-studio/results-panel/results-panel';

function makeResultWithFrontier(weights: Record<string, number>) {
  return {
    id: 'run-frontier',
    status: 'completed',
    optimizerType: 'mean_risk',
    universeTickers: Object.keys(weights),
    config: {},
    weights,
    metrics: {
      annualized_return: 0.12,
      annualized_volatility: 0.14,
      annualized_sharpe_ratio: 1.1,
      annualized_sortino_ratio: 1.3,
      max_drawdown: -0.07,
    },
    riskContributions: {},
    efficientFrontier: [
      { risk: 0.10, return: 0.08, sharpe: 0.80 },
      { risk: 0.15, return: 0.13, sharpe: 1.10 },
      { risk: 0.20, return: 0.16, sharpe: 1.30 },
    ],
    errorMessage: null,
    solverLog: null,
    durationSeconds: 1.0,
    createdAt: '2026-01-01T00:00:00Z',
    updatedAt: '2026-01-01T00:00:01Z',
    portfolioId: null,
    jobId: null,
  };
}

describe('ResultsPanelComponent — efficient-frontier widget (issue #1044 criterion 3)', () => {
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

  it('when result with frontier data is received, app-echarts-efficient-frontier is rendered', () => {
    fixture.componentRef.setInput('result', makeResultWithFrontier({ AAPL: 0.6, MSFT: 0.4 }));
    fixture.componentRef.setInput('hasResult', true);
    fixture.detectChanges();

    expect(fixture.debugElement.query(By.css('app-echarts-efficient-frontier')))
      .withContext('app-echarts-efficient-frontier must be present when result has frontier data')
      .not.toBeNull();
  });

  it('when no result is set, app-echarts-efficient-frontier is absent from the DOM', () => {
    fixture.componentRef.setInput('result', null);
    fixture.componentRef.setInput('hasResult', false);
    fixture.detectChanges();

    expect(fixture.debugElement.query(By.css('app-echarts-efficient-frontier'))).toBeNull();
  });

  const samples: Array<[string, Record<string, number>]> = [
    ['two assets', { AAPL: 0.5, MSFT: 0.5 }],
    ['five assets', { A: 0.2, B: 0.2, C: 0.2, D: 0.2, E: 0.2 }],
  ];

  samples.forEach(([label, weights]) => {
    it(`when result has [${label}] and frontier data, app-echarts-efficient-frontier is rendered`, () => {
      fixture.componentRef.setInput('result', makeResultWithFrontier(weights));
      fixture.componentRef.setInput('hasResult', true);
      fixture.detectChanges();

      expect(fixture.debugElement.query(By.css('app-echarts-efficient-frontier')))
        .withContext(`efficient frontier missing for "${label}"`)
        .not.toBeNull();
    });
  });
});
