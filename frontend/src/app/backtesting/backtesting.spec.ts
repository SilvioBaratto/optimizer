import { TestBed, ComponentFixture } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';

import { BacktestingComponent, toFactorLoadingSeries } from './backtesting';
import { ICON_PROVIDER } from '../icons';
import type {
  BacktestMetrics,
  BacktestResult,
  FactorLoading,
} from './backtest.model';

const EMPTY_METRICS: BacktestMetrics = {
  totalReturn: 0.12,
  annualizedReturn: 0.05,
  annualizedVol: 0.15,
  sharpe: 0.9,
  sortino: 1.1,
  maxDrawdown: -0.18,
  calmar: 0.27,
  cvar95: -0.04,
  trackingError: 0.06,
  informationRatio: 0.4,
  winRate: 0.55,
  profitFactor: 1.3,
};

const REAL_BENCHMARK_METRICS: BacktestMetrics = {
  totalReturn: 0.10,
  annualizedReturn: 0.04,
  annualizedVol: 0.13,
  sharpe: 0.6,
  sortino: 0.8,
  maxDrawdown: -0.22,
  calmar: 0.18,
  cvar95: -0.05,
  trackingError: 0.08,
  informationRatio: 0.25,
  winRate: 0.51,
  profitFactor: 1.15,
};

function makeResult(overrides: Partial<BacktestResult> = {}): BacktestResult {
  return {
    equity: [
      { date: '2024-01-01', portfolio: 100, benchmark: 100 },
      { date: '2024-12-31', portfolio: 112, benchmark: 110 },
    ],
    metrics: EMPTY_METRICS,
    drawdowns: [],
    monthlyReturns: [],
    rollingMetrics: [],
    returnDistribution: [],
    factorLoadings: [],
    ...overrides,
  };
}

describe('BacktestingComponent — benchmark literals removal (issue #434)', () => {
  let component: BacktestingComponent;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [BacktestingComponent],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
      ],
    }).compileComponents();
    component = TestBed.createComponent(BacktestingComponent).componentInstance;
  });

  describe('metricsTableRows when no backtest result is loaded', () => {
    it('shows a dash for every benchmark cell', () => {
      // Default `result()` is the EMPTY_RESULT shipped by the component
      const rows = component.metricsTableRows();
      for (const row of rows) {
        expect(row.benchmark)
          .withContext(`row "${row.metric}" must show a dash without a backtest`)
          .toBe('—');
        expect(row.benchmarkRaw)
          .withContext(`row "${row.metric}" benchmarkRaw must be null`)
          .toBeNull();
      }
    });

    it('contains every documented metric (12 rows)', () => {
      expect(component.metricsTableRows().length).toBe(12);
    });
  });

  describe('metricsTableRows with real benchmark metrics', () => {
    beforeEach(() => {
      component.result.set(
        makeResult({ benchmarkMetrics: REAL_BENCHMARK_METRICS }),
      );
    });

    it('uses real benchmark values for every numeric row', () => {
      const byMetric = new Map(
        component.metricsTableRows().map((r) => [r.metric, r]),
      );
      expect(byMetric.get('Total Return')?.benchmarkRaw).toBe(0.10);
      expect(byMetric.get('Annualized Return')?.benchmarkRaw).toBe(0.04);
      expect(byMetric.get('Sharpe Ratio')?.benchmarkRaw).toBe(0.6);
      expect(byMetric.get('Max Drawdown')?.benchmarkRaw).toBe(-0.22);
      expect(byMetric.get('Profit Factor')?.benchmarkRaw).toBe(1.15);
    });

    it('keeps tracking-error and information-ratio rows benchmark-less', () => {
      // These metrics have no meaningful "benchmark" counterpart
      const te = component.metricsTableRows().find((r) => r.metric === 'Tracking Error');
      const ir = component.metricsTableRows().find((r) => r.metric === 'Information Ratio');
      expect(te?.benchmark).toBe('—');
      expect(te?.benchmarkRaw).toBeNull();
      expect(ir?.benchmark).toBe('—');
      expect(ir?.benchmarkRaw).toBeNull();
    });
  });

  describe('KPI strip benchmark delta', () => {
    it('returns null deltas before any backtest has run', () => {
      expect(component.benchmarkDelta('totalReturn')).toBeNull();
      expect(component.benchmarkDelta('sharpe')).toBeNull();
      expect(component.benchmarkDelta('maxDrawdown')).toBeNull();
    });

    it('returns flat trends before any backtest has run', () => {
      expect(component.benchmarkTrend('totalReturn')).toBe('flat');
      expect(component.benchmarkTrend('sharpe')).toBe('flat');
    });

    it('returns blank subtitles before any backtest has run', () => {
      expect(component.benchmarkSubtitle('totalReturn')).toBe('');
    });

    it('computes deltas relative to real benchmark metrics when available', () => {
      component.result.set(
        makeResult({ benchmarkMetrics: REAL_BENCHMARK_METRICS }),
      );
      expect(component.benchmarkDelta('totalReturn'))
        .toBeCloseTo(EMPTY_METRICS.totalReturn - REAL_BENCHMARK_METRICS.totalReturn, 9);
      expect(component.benchmarkTrend('totalReturn')).toBe('up'); // 0.12 > 0.10
      expect(component.benchmarkSubtitle('totalReturn')).toContain('benchmark');
    });
  });
});


describe('toFactorLoadingSeries helper (issue #458)', () => {
  it('maps each loading to BarData { label, value }', () => {
    const out = toFactorLoadingSeries([
      { factor: 'MKT', loading: 0.8, tStat: 5, pValue: 0.001 },
    ]);
    expect(out).toEqual([{ label: 'MKT', value: 0.8 }]);
  });

  it('sorts loadings descending by loading value', () => {
    const out = toFactorLoadingSeries([
      { factor: 'B', loading: 0.1, tStat: 1, pValue: 0.1 },
      { factor: 'A', loading: 0.5, tStat: 2, pValue: 0.05 },
      { factor: 'C', loading: -0.3, tStat: -1.5, pValue: 0.07 },
    ]);
    expect(out.map((b) => b.label)).toEqual(['A', 'B', 'C']);
    expect(out.map((b) => b.value)).toEqual([0.5, 0.1, -0.3]);
  });

  it('returns [] for empty input', () => {
    expect(toFactorLoadingSeries([])).toEqual([]);
  });

  it('does not mutate the input array', () => {
    const input: FactorLoading[] = [
      { factor: 'B', loading: 0.1, tStat: 1, pValue: 0.1 },
      { factor: 'A', loading: 0.5, tStat: 2, pValue: 0.05 },
    ];
    const snapshot = input.map((f) => ({ ...f }));
    toFactorLoadingSeries(input);
    expect(input).toEqual(snapshot);
  });
});

describe('BacktestingComponent — Factor Loadings honest rendering (issue #458)', () => {
  let fixture: ComponentFixture<BacktestingComponent>;
  let component: BacktestingComponent;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [BacktestingComponent],
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        ICON_PROVIDER,
      ],
    }).compileComponents();
    fixture = TestBed.createComponent(BacktestingComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('does not call Math.sin when computing factor loading bars', () => {
    const sinSpy = spyOn(Math, 'sin').and.callThrough();
    component.result.set(
      makeResult({
        factorLoadings: [
          { factor: 'MKT', loading: 0.8, tStat: 5, pValue: 0.001 },
          { factor: 'SMB', loading: 0.1, tStat: 2, pValue: 0.05 },
        ],
      }),
    );
    component.factorLoadingBars();
    expect(sinSpy).not.toHaveBeenCalled();
  });

  it('emits one bar per factor loading sorted descending', () => {
    component.result.set(
      makeResult({
        factorLoadings: [
          { factor: 'SMB', loading: 0.2, tStat: 2, pValue: 0.05 },
          { factor: 'MKT', loading: 0.8, tStat: 5, pValue: 0.001 },
          { factor: 'HML', loading: -0.1, tStat: -1, pValue: 0.3 },
        ],
      }),
    );
    const bars = component.factorLoadingBars();
    expect(bars.length).toBe(3);
    expect(bars.map((b) => b.label)).toEqual(['MKT', 'SMB', 'HML']);
    expect(bars.map((b) => b.value)).toEqual([0.8, 0.2, -0.1]);
  });

  it('returns an empty array when factorLoadings is empty', () => {
    expect(component.factorLoadingBars()).toEqual([]);
  });

  it('renders <app-echarts-bar> on the Style tab when loadings exist', () => {
    component.result.set(
      makeResult({
        factorLoadings: [
          { factor: 'MKT', loading: 0.8, tStat: 5, pValue: 0.001 },
        ],
      }),
    );
    component.activeTab.set('style');
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelectorAll('app-echarts-bar').length).toBe(1);
    expect(el.querySelectorAll('app-echarts-stacked-area').length).toBe(0);
  });

  it('shows an empty-state and no chart when factorLoadings is empty on the Style tab', () => {
    component.result.set(makeResult({ factorLoadings: [] }));
    component.activeTab.set('style');
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('[data-testid="factor-loadings-empty"]')).not.toBeNull();
    expect(el.querySelectorAll('app-echarts-bar').length).toBe(0);
  });
});
