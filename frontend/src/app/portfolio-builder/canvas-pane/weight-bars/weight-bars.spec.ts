import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { TestBed, type ComponentFixture } from '@angular/core/testing';

import { WeightBarsComponent } from './weight-bars';
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
  fixture: ComponentFixture<WeightBarsComponent>;
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
  const fixture = TestBed.createComponent(WeightBarsComponent);
  fixture.detectChanges();
  return { fixture, store, host: fixture.nativeElement as HTMLElement };
}

function rowElements(host: HTMLElement): HTMLElement[] {
  return Array.from(host.querySelectorAll<HTMLElement>('li[data-row="weight-bar"]'));
}

describe('WeightBarsComponent', () => {
  beforeAll(() => setCssVars());
  afterAll(() => clearCssVars());

  it('when result is null, no rows render and an empty-state placeholder is shown', () => {
    const { host } = setup();
    expect(rowElements(host).length).toBe(0);
    expect(host.querySelector('[data-region="weight-bars-empty"]')).not.toBeNull();
  });

  it('when result has weights, rows are sorted descending by absolute weight', () => {
    const { fixture, store } = setup();
    store.setResult(makeResult({ A: 0.1, B: 0.5, C: 0.3, D: -0.4 }));
    fixture.detectChanges();
    expect(fixture.componentInstance.rows().map((r) => r.ticker)).toEqual([
      'B',
      'D',
      'C',
      'A',
    ]);
  });

  it('when result has more than twenty weights, only the top twenty rows render', () => {
    const weights: Record<string, number> = {};
    for (let i = 0; i < 25; i++) {
      weights[`T${i.toString().padStart(2, '0')}`] = (25 - i) / 100;
    }
    const { fixture, store, host } = setup();
    store.setResult(makeResult(weights));
    fixture.detectChanges();
    expect(fixture.componentInstance.rows().length).toBe(20);
    expect(rowElements(host).length).toBe(20);
  });

  it('when a row weight is negative, the weight cell uses the loss color class', () => {
    const { fixture, store, host } = setup();
    store.setResult(makeResult({ SHORT: -0.2, LONG: 0.5 }));
    fixture.detectChanges();
    const rows = rowElements(host);
    const shortCell = rows
      .find((li) => li.textContent?.includes('SHORT'))!
      .querySelector<HTMLElement>('[data-cell="weight"]')!;
    const longCell = rows
      .find((li) => li.textContent?.includes('LONG'))!
      .querySelector<HTMLElement>('[data-cell="weight"]')!;
    expect(shortCell.classList).toContain('text-loss');
    expect(longCell.classList).toContain('text-gain');
  });

  it('when result is updated twice with the same ticker set, swatch colors remain identical per ticker', () => {
    const { fixture, store } = setup();
    store.setResult(makeResult({ AAPL: 0.5, MSFT: 0.3, GOOG: 0.2 }));
    fixture.detectChanges();
    const first = new Map(
      fixture.componentInstance.rows().map((r) => [r.ticker, r.color]),
    );
    store.setResult(makeResult({ AAPL: 0.4, MSFT: 0.35, GOOG: 0.25 }));
    fixture.detectChanges();
    const second = new Map(
      fixture.componentInstance.rows().map((r) => [r.ticker, r.color]),
    );
    expect(second.get('AAPL')).toBe(first.get('AAPL'));
    expect(second.get('MSFT')).toBe(first.get('MSFT'));
    expect(second.get('GOOG')).toBe(first.get('GOOG'));
  });

  it('when rendered, each row swatch background-color comes from the ticker palette', () => {
    const { fixture, store, host } = setup();
    store.setResult(makeResult({ AAPL: 0.6, MSFT: 0.4 }));
    fixture.detectChanges();
    const swatches = host.querySelectorAll<HTMLElement>('[data-cell="swatch"]');
    expect(swatches[0].style.backgroundColor).toBe('rgb(17, 17, 17)');
    expect(swatches[1].style.backgroundColor).toBe('rgb(34, 34, 34)');
  });
});
