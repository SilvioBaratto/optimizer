import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { TestBed, type ComponentFixture } from '@angular/core/testing';

import { AllocationDonutComponent } from './allocation-donut';
import { BuilderStore } from '../../state/builder.store';
import type { BuilderResult } from '../../models/builder-result.model';

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
    document.documentElement.style.setProperty(`--color-chart-${index + 1}`, value);
  });
}

function clearCssVars(): void {
  PALETTE.forEach((_, index) => {
    document.documentElement.style.removeProperty(`--color-chart-${index + 1}`);
  });
}

function makeResult(weights: Record<string, number>): BuilderResult {
  return {
    runId: 'r-1',
    weights,
    metrics: {},
    optimizerType: 'mean_risk',
    createdAt: '2026-01-01T00:00:00Z',
  };
}

function setup(): {
  fixture: ComponentFixture<AllocationDonutComponent>;
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
  const fixture = TestBed.createComponent(AllocationDonutComponent);
  fixture.detectChanges();
  return { fixture, store, host: fixture.nativeElement as HTMLElement };
}

describe('AllocationDonutComponent', () => {
  beforeAll(() => setCssVars());
  afterAll(() => clearCssVars());

  it('when result is null, the donut is suppressed and an empty-state placeholder renders', () => {
    const { host } = setup();
    expect(host.querySelector('app-echarts-donut')).toBeNull();
    expect(host.querySelector('[data-region="allocation-donut-empty"]')).not.toBeNull();
  });

  it('when result has weights, the donut renders with one segment per ticker', () => {
    const { fixture, store, host } = setup();
    store.setResult(makeResult({ AAPL: 0.4, MSFT: 0.35, GOOG: 0.25 }));
    fixture.detectChanges();
    expect(host.querySelector('app-echarts-donut')).not.toBeNull();
    expect(fixture.componentInstance.segments().length).toBe(3);
  });

  it('when result has weights, segments are sorted descending by absolute weight', () => {
    const { fixture, store } = setup();
    store.setResult(makeResult({ A: 0.1, B: 0.5, C: 0.3, D: -0.4 }));
    fixture.detectChanges();
    expect(fixture.componentInstance.segments().map((s) => s.label)).toEqual([
      'B',
      'D',
      'C',
      'A',
    ]);
  });

  it('when result has weights, each segment receives a color from the ticker palette', () => {
    const { fixture, store } = setup();
    store.setResult(makeResult({ AAPL: 0.6, MSFT: 0.4 }));
    fixture.detectChanges();
    const segs = fixture.componentInstance.segments();
    expect(segs[0]).toEqual({ label: 'AAPL', value: 0.6, color: PALETTE[0] });
    expect(segs[1]).toEqual({ label: 'MSFT', value: 0.4, color: PALETTE[1] });
  });

  it('when result is updated twice with the same ticker set, palette colors remain identical for each ticker', () => {
    const { fixture, store } = setup();
    store.setResult(makeResult({ AAPL: 0.5, MSFT: 0.3, GOOG: 0.2 }));
    fixture.detectChanges();
    const first = new Map(
      fixture.componentInstance.segments().map((s) => [s.label, s.color]),
    );

    store.setResult(makeResult({ AAPL: 0.4, MSFT: 0.35, GOOG: 0.25 }));
    fixture.detectChanges();
    const second = new Map(
      fixture.componentInstance.segments().map((s) => [s.label, s.color]),
    );

    expect(second.get('AAPL')).toBe(first.get('AAPL'));
    expect(second.get('MSFT')).toBe(first.get('MSFT'));
    expect(second.get('GOOG')).toBe(first.get('GOOG'));
  });
});
