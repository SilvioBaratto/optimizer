import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { TestBed, type ComponentFixture } from '@angular/core/testing';

import {
  BacktestSparklineComponent,
  computeDrawdownRegion,
} from './backtest-sparkline';
import { BuilderStore } from '../../state/builder.store';
import type { BuilderBacktest } from '../../models/builder-result.model';

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
const LOSS_COLOR = '#ef4444';

function setCssVars(): void {
  PALETTE.forEach((value, index) => {
    document.documentElement.style.setProperty(
      `--color-chart-${index + 1}`,
      value,
    );
  });
  document.documentElement.style.setProperty('--color-loss', LOSS_COLOR);
}

function clearCssVars(): void {
  PALETTE.forEach((_, index) => {
    document.documentElement.style.removeProperty(`--color-chart-${index + 1}`);
  });
  document.documentElement.style.removeProperty('--color-loss');
}

function makeBacktest(overrides: Partial<BuilderBacktest> = {}): BuilderBacktest {
  return {
    runId: 'bt-1',
    summaryStats: {
      totalReturn: 0.18,
      maxDrawdown: -0.12,
      sharpe: 1.1,
    },
    equityCurve: {
      '2026-01-01': 1.0,
      '2026-02-01': 1.05,
      '2026-03-01': 1.1,
      '2026-04-01': 1.05,
      '2026-05-01': 1.0,
      '2026-06-01': 0.95,
    },
    createdAt: '2026-01-01T00:00:00Z',
    ...overrides,
  };
}

function setup(): {
  fixture: ComponentFixture<BacktestSparklineComponent>;
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
  const fixture = TestBed.createComponent(BacktestSparklineComponent);
  fixture.detectChanges();
  return { fixture, store, host: fixture.nativeElement as HTMLElement };
}

describe('computeDrawdownRegion', () => {
  it('when values is empty, computeDrawdownRegion returns null', () => {
    expect(computeDrawdownRegion([])).toBeNull();
  });

  it('when values is flat, computeDrawdownRegion returns null', () => {
    expect(computeDrawdownRegion([1, 1, 1, 1])).toBeNull();
  });

  it('when values is strictly rising, computeDrawdownRegion returns null', () => {
    expect(computeDrawdownRegion([1, 2, 3, 4, 5])).toBeNull();
  });

  it('when peak is at idx 2 and trough at idx 5, computeDrawdownRegion returns {startIdx:2,endIdx:5}', () => {
    expect(
      computeDrawdownRegion([1.0, 1.05, 1.1, 1.05, 1.0, 0.95]),
    ).toEqual({ startIdx: 2, endIdx: 5 });
  });

  it('when peak then dip then rally then deeper dip, computeDrawdownRegion picks the deepest peak/trough pair', () => {
    expect(
      computeDrawdownRegion([1.0, 1.1, 1.05, 1.2, 0.6]),
    ).toEqual({ startIdx: 3, endIdx: 4 });
  });
});

describe('BacktestSparklineComponent', () => {
  beforeAll(() => setCssVars());
  afterAll(() => clearCssVars());

  it('when backtest is null, the empty-state placeholder renders and no chart container is in the DOM', () => {
    const { host } = setup();
    expect(
      host.querySelector('[data-region="backtest-sparkline-empty"]'),
    ).not.toBeNull();
  });

  it('when backtest is null, hasData() is false', () => {
    const { fixture } = setup();
    expect(fixture.componentInstance.hasData()).toBe(false);
  });

  it('when backtest has an equity curve, hasData() is true and the empty placeholder is suppressed', () => {
    const { fixture, store, host } = setup();
    store.setBacktest(makeBacktest());
    fixture.detectChanges();
    expect(fixture.componentInstance.hasData()).toBe(true);
    expect(
      host.querySelector('[data-region="backtest-sparkline-empty"]'),
    ).toBeNull();
  });

  it('when backtest has an equity curve, labels() exposes the date keys in insertion order', () => {
    const { fixture, store } = setup();
    store.setBacktest(makeBacktest());
    fixture.detectChanges();
    expect(fixture.componentInstance.labels()).toEqual([
      '2026-01-01',
      '2026-02-01',
      '2026-03-01',
      '2026-04-01',
      '2026-05-01',
      '2026-06-01',
    ]);
  });

  it('when backtest has an equity curve, values() exposes each value multiplied by 100', () => {
    const { fixture, store } = setup();
    store.setBacktest(makeBacktest());
    fixture.detectChanges();
    const got = fixture.componentInstance.values();
    const want = [100, 105, 110, 105, 100, 95];
    expect(got.length).toBe(want.length);
    for (let i = 0; i < want.length; i++) {
      expect(got[i]).toBeCloseTo(want[i], 8);
    }
  });

  it('when equity curve has a null entry, values() projects null to 0', () => {
    const { fixture, store } = setup();
    store.setBacktest(
      makeBacktest({
        equityCurve: { '2026-01-01': 1.0, '2026-02-01': null, '2026-03-01': 1.05 },
      }),
    );
    fixture.detectChanges();
    expect(fixture.componentInstance.values()).toEqual([100, 0, 105]);
  });

  it('when summaryStats carry totalReturn/maxDrawdown/sharpe, stats() exposes them under cumReturn/maxDd/sharpe keys', () => {
    const { fixture, store } = setup();
    store.setBacktest(makeBacktest());
    fixture.detectChanges();
    expect(fixture.componentInstance.stats()).toEqual({
      cumReturn: 0.18,
      maxDd: -0.12,
      sharpe: 1.1,
    });
  });

  it('when backtest is null, stats() returns all-null fields', () => {
    const { fixture } = setup();
    expect(fixture.componentInstance.stats()).toEqual({
      cumReturn: null,
      maxDd: null,
      sharpe: null,
    });
  });

  it('when summaryStats values are missing or NaN, stats() collapses them to null', () => {
    const { fixture, store } = setup();
    store.setBacktest(
      makeBacktest({
        summaryStats: { totalReturn: Number.NaN, sharpe: null },
      }),
    );
    fixture.detectChanges();
    expect(fixture.componentInstance.stats()).toEqual({
      cumReturn: null,
      maxDd: null,
      sharpe: null,
    });
  });

  it('when mounted with data, the stats row exposes three [data-stat] cells', () => {
    const { fixture, store, host } = setup();
    store.setBacktest(makeBacktest());
    fixture.detectChanges();
    expect(host.querySelector('[data-region="backtest-stats"]')).not.toBeNull();
    expect(host.querySelector('[data-stat="cum-return"]')).not.toBeNull();
    expect(host.querySelector('[data-stat="max-dd"]')).not.toBeNull();
    expect(host.querySelector('[data-stat="sharpe"]')).not.toBeNull();
  });

  it('when mounted with no data, the stats row still renders so loading/empty states stay consistent', () => {
    const { host } = setup();
    expect(host.querySelector('[data-region="backtest-stats"]')).not.toBeNull();
  });

  it('when buildOptionForTest is called with a drawdown series, the option includes markArea.data shaped as [[{xAxis},{xAxis}]]', () => {
    const { fixture, store } = setup();
    store.setBacktest(makeBacktest());
    fixture.detectChanges();
    const option = fixture.componentInstance.buildOptionForTest() as {
      series: Array<{
        markArea?: { data?: Array<Array<{ xAxis: string | number }>> };
      }>;
    };
    const markArea = option.series[0].markArea;
    expect(markArea).toBeDefined();
    expect(markArea?.data?.length).toBe(1);
    const pair = markArea?.data?.[0] ?? [];
    expect(pair.length).toBe(2);
    expect(pair[0].xAxis).toBe('2026-03-01');
    expect(pair[1].xAxis).toBe('2026-06-01');
  });

  it('when buildOptionForTest is called with a strictly-rising series, the option has no markArea on the series', () => {
    const { fixture, store } = setup();
    store.setBacktest(
      makeBacktest({
        equityCurve: { d1: 1.0, d2: 1.1, d3: 1.2, d4: 1.3 },
      }),
    );
    fixture.detectChanges();
    const option = fixture.componentInstance.buildOptionForTest() as {
      series: Array<{ markArea?: unknown }>;
    };
    expect(option.series[0].markArea).toBeUndefined();
  });

  it('when buildOptionForTest is called, the xAxis category data equals labels()', () => {
    const { fixture, store } = setup();
    store.setBacktest(makeBacktest());
    fixture.detectChanges();
    const option = fixture.componentInstance.buildOptionForTest() as {
      xAxis: { data: string[] };
    };
    expect(option.xAxis.data).toEqual(fixture.componentInstance.labels());
  });
});
