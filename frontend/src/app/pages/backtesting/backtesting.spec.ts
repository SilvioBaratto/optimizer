import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';

import { BacktestingComponent } from './backtesting';
import type {
  BacktestMetrics,
  BacktestResult,
} from '../../models/backtest.model';

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
