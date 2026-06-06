import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { TestBed, type ComponentFixture } from '@angular/core/testing';
import { By } from '@angular/platform-browser';

import { EfficientFrontierComponent } from './efficient-frontier';
import { EchartsEfficientFrontierComponent } from '../../../shared/echarts-efficient-frontier/echarts-efficient-frontier';
import { BuilderStore } from '../../state/builder.store';
import type {
  BuilderFrontier,
  BuilderResult,
} from '../../models/builder-result.model';
import type { EfficientFrontierPoint } from '../../../core/models/optimization.model';

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

function samplePoints(): EfficientFrontierPoint[] {
  return [
    { risk: 0.08, return: 0.04, sharpe: 0.5 },
    { risk: 0.12, return: 0.07, sharpe: 0.58 },
    { risk: 0.18, return: 0.1, sharpe: 0.56 },
  ];
}

function makeFrontier(): BuilderFrontier {
  return { points: samplePoints() };
}

function makeResult(metrics: Record<string, unknown>): BuilderResult {
  return {
    runId: 'r-1',
    weights: { AAPL: 1 },
    metrics,
    optimizerType: 'mean_risk',
    createdAt: '2026-01-01T00:00:00Z',
  };
}

function setup(): {
  fixture: ComponentFixture<EfficientFrontierComponent>;
  store: BuilderStore;
  host: HTMLElement;
} {
  TestBed.configureTestingModule({
    providers: [
      provideZonelessChangeDetection(),
      provideHttpClient(),
      provideHttpClientTesting(),
      BuilderStore,
    ],
  });
  const store = TestBed.inject(BuilderStore);
  const fixture = TestBed.createComponent(EfficientFrontierComponent);
  fixture.detectChanges();
  return { fixture, store, host: fixture.nativeElement as HTMLElement };
}

describe('EfficientFrontierComponent', () => {
  beforeAll(() => setCssVars());
  afterAll(() => clearCssVars());

  it('when frontier is null, the empty-state placeholder renders and the wrapper is suppressed', () => {
    const { host } = setup();
    expect(
      host.querySelector('[data-region="efficient-frontier-empty"]'),
    ).not.toBeNull();
    expect(host.querySelector('app-echarts-efficient-frontier')).toBeNull();
  });

  it('when frontier has points, the echarts efficient-frontier wrapper is mounted', () => {
    const { fixture, store, host } = setup();
    store.setFrontier(makeFrontier());
    fixture.detectChanges();
    expect(host.querySelector('app-echarts-efficient-frontier')).not.toBeNull();
    expect(
      host.querySelector('[data-region="efficient-frontier-empty"]'),
    ).toBeNull();
  });

  it('when frontier has points, frontierPoints() exposes the underlying points array', () => {
    const { fixture, store } = setup();
    const frontier = makeFrontier();
    store.setFrontier(frontier);
    fixture.detectChanges();
    expect(fixture.componentInstance.frontierPoints()).toEqual(frontier.points);
  });

  it('when metrics carry all three canonical finite keys, currentPortfolioMarker() returns the composed point with label "You"', () => {
    const { fixture, store } = setup();
    store.setResult(
      makeResult({
        annualized_volatility: 0.15,
        annualized_return: 0.09,
        annualized_sharpe_ratio: 0.6,
      }),
    );
    fixture.detectChanges();
    expect(fixture.componentInstance.currentPortfolioMarker()).toEqual({
      risk: 0.15,
      return: 0.09,
      sharpe: 0.6,
      label: 'You',
    });
  });

  it('when result is null, currentPortfolioMarker() returns null', () => {
    const { fixture } = setup();
    expect(fixture.componentInstance.currentPortfolioMarker()).toBeNull();
  });

  it('when annualized_volatility is missing, currentPortfolioMarker() returns null', () => {
    const { fixture, store } = setup();
    store.setResult(
      makeResult({
        annualized_return: 0.09,
        annualized_sharpe_ratio: 0.6,
      }),
    );
    fixture.detectChanges();
    expect(fixture.componentInstance.currentPortfolioMarker()).toBeNull();
  });

  it('when annualized_return is NaN, currentPortfolioMarker() returns null', () => {
    const { fixture, store } = setup();
    store.setResult(
      makeResult({
        annualized_volatility: 0.15,
        annualized_return: Number.NaN,
        annualized_sharpe_ratio: 0.6,
      }),
    );
    fixture.detectChanges();
    expect(fixture.componentInstance.currentPortfolioMarker()).toBeNull();
  });

  it('when annualized_sharpe_ratio is Infinity, currentPortfolioMarker() returns null', () => {
    const { fixture, store } = setup();
    store.setResult(
      makeResult({
        annualized_volatility: 0.15,
        annualized_return: 0.09,
        annualized_sharpe_ratio: Number.POSITIVE_INFINITY,
      }),
    );
    fixture.detectChanges();
    expect(fixture.componentInstance.currentPortfolioMarker()).toBeNull();
  });

  it('when a metric value is a non-number type, currentPortfolioMarker() returns null', () => {
    const { fixture, store } = setup();
    store.setResult(
      makeResult({
        annualized_volatility: '0.15',
        annualized_return: 0.09,
        annualized_sharpe_ratio: 0.6,
      }),
    );
    fixture.detectChanges();
    expect(fixture.componentInstance.currentPortfolioMarker()).toBeNull();
  });

  it('when frontier and full metrics are set, the wrapper [optimal] input equals currentPortfolioMarker()', () => {
    const { fixture, store } = setup();
    store.setFrontier(makeFrontier());
    store.setResult(
      makeResult({
        annualized_volatility: 0.12,
        annualized_return: 0.07,
        annualized_sharpe_ratio: 0.58,
      }),
    );
    fixture.detectChanges();
    const wrapperDe = fixture.debugElement.query(
      By.directive(EchartsEfficientFrontierComponent),
    );
    expect(wrapperDe).not.toBeNull();
    const wrapper =
      wrapperDe.componentInstance as EchartsEfficientFrontierComponent;
    expect(wrapper.optimal()).toEqual(
      fixture.componentInstance.currentPortfolioMarker(),
    );
  });

  it('when frontier has points but metrics are missing, the wrapper [optimal] input is null while the chart still renders', () => {
    const { fixture, store, host } = setup();
    store.setFrontier(makeFrontier());
    fixture.detectChanges();
    const wrapperDe = fixture.debugElement.query(
      By.directive(EchartsEfficientFrontierComponent),
    );
    expect(wrapperDe).not.toBeNull();
    expect(
      (wrapperDe.componentInstance as EchartsEfficientFrontierComponent).optimal(),
    ).toBeNull();
    expect(host.querySelector('app-echarts-efficient-frontier')).not.toBeNull();
  });

  it('when wrapper is mounted, its height input is 280', () => {
    const { fixture, store } = setup();
    store.setFrontier(makeFrontier());
    fixture.detectChanges();
    const wrapperDe = fixture.debugElement.query(
      By.directive(EchartsEfficientFrontierComponent),
    );
    expect(
      (wrapperDe.componentInstance as EchartsEfficientFrontierComponent).height(),
    ).toBe(280);
  });
});
