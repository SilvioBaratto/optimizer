import { TestBed, type ComponentFixture } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { BacktestResultsPanelComponent } from './backtest-results-panel';
import type { BacktestRunResponse } from '../backtest.model';
import { ICON_PROVIDER } from '../../icons';

function makeRun(overrides: Partial<BacktestRunResponse> = {}): BacktestRunResponse {
  return {
    id: 'r-1',
    portfolioId: null,
    jobId: 'j-1',
    status: 'completed',
    config: { optimizer_type: 'hrp' },
    equityCurve: {
      '2024-01-02': 1.0,
      '2024-01-03': 1.01,
      '2024-01-04': 1.02,
    },
    drawdowns: {
      '2024-01-02': 0.0,
      '2024-01-03': -0.01,
      max_drawdown: -0.015,
    },
    monthlyReturns: { '2024-01-31': 0.02 },
    yearlyReturns: { '2024-12-31': 0.12 },
    rollingMetrics: {
      rolling_sharpe: { '2024-01-02': 1.1, '2024-01-03': 1.05 },
      rolling_vol: { '2024-01-02': 0.18, '2024-01-03': 0.19 },
    },
    turnoverHistory: { '2024-01-02': 0.0 },
    cvFoldMetrics: null,
    summaryStats: { sharpe: 1.2, max_drawdown: -0.015, cagr: 0.07 },
    errorMessage: null,
    durationSeconds: 2.34,
    createdAt: '2024-02-01T00:00:00Z',
    updatedAt: '2024-02-01T00:00:01Z',
    ...overrides,
  };
}

let activeFixture: ComponentFixture<BacktestResultsPanelComponent> | null = null;

function createFixture(): ComponentFixture<BacktestResultsPanelComponent> {
  TestBed.configureTestingModule({
    providers: [provideZonelessChangeDetection(), ICON_PROVIDER],
  });
  activeFixture = TestBed.createComponent(BacktestResultsPanelComponent);
  return activeFixture;
}

describe('BacktestResultsPanelComponent', () => {
  afterEach(() => {
    // Dispose child chart components' async ECharts init so Karma can exit.
    activeFixture?.destroy();
    activeFixture = null;
  });

  it('renders the loading skeleton when `loading` is true', () => {
    const fixture = createFixture();
    fixture.componentRef.setInput('loading', true);
    fixture.detectChanges();

    const el: HTMLElement = fixture.nativeElement;
    expect(el.querySelector('[data-testid="loading"]')).toBeTruthy();
    expect(el.querySelector('[data-testid="panel"]')).toBeNull();
  });

  it('renders the error banner when `error` is set and `loading` is false', () => {
    const fixture = createFixture();
    fixture.componentRef.setInput('error', 'fetch failed');
    fixture.detectChanges();

    const err = fixture.nativeElement.querySelector('[data-testid="error"]') as HTMLElement | null;
    expect(err?.textContent).toContain('fetch failed');
  });

  it('when error is set, error element has role="alert" with non-blank text', () => {
    const fixture = createFixture();
    fixture.componentRef.setInput('error', 'Backtest run unavailable');
    fixture.detectChanges();

    const alertEl = fixture.nativeElement.querySelector('[role="alert"]') as HTMLElement | null;
    expect(alertEl).withContext('expected [role="alert"] when error is set').toBeTruthy();
    expect(alertEl?.textContent?.trim().length).toBeGreaterThan(0);
  });

  it('renders nothing when run is null and not loading/erroring', () => {
    const fixture = createFixture();
    fixture.detectChanges();

    const el: HTMLElement = fixture.nativeElement;
    expect(el.querySelector('[data-testid="panel"]')).toBeNull();
    expect(el.querySelector('[data-testid="loading"]')).toBeNull();
    expect(el.querySelector('[data-testid="error"]')).toBeNull();
  });

  it('exposes summary cells derived from run.summaryStats, skipping nulls', () => {
    const fixture = createFixture();
    fixture.componentRef.setInput(
      'run',
      makeRun({ summaryStats: { annualized_sharpe_ratio: 1.2, cagr: null, cvar: -0.03 } }),
    );
    fixture.detectChanges();

    const keys = fixture.componentInstance.summaryCells().map((c) => c.key);
    expect(keys).toEqual(['Annualized Sharpe', 'CVaR 95%']);
  });

  it('sorts equity-curve date keys ascending', () => {
    const fixture = createFixture();
    fixture.componentRef.setInput(
      'run',
      makeRun({
        equityCurve: {
          '2024-01-03': 1.01,
          '2024-01-02': 1.0,
          '2024-01-04': 1.02,
        },
      }),
    );
    fixture.detectChanges();

    expect(fixture.componentInstance.equityLabels()).toEqual([
      '2024-01-02',
      '2024-01-03',
      '2024-01-04',
    ]);
    expect(fixture.componentInstance.equityValues()).toEqual([1.0, 1.01, 1.02]);
  });

  it('filters max_drawdown summary key out of the drawdown chart input', () => {
    const fixture = createFixture();
    fixture.componentRef.setInput('run', makeRun());
    fixture.detectChanges();

    const points = fixture.componentInstance.drawdownPoints();
    expect(points.map((p) => p.date)).toEqual(['2024-01-02', '2024-01-03']);
    expect(points.every((p) => p.date !== 'max_drawdown')).toBe(true);
  });

  it('builds a rolling-metric series per key in rollingMetrics', () => {
    const fixture = createFixture();
    fixture.componentRef.setInput('run', makeRun());
    fixture.detectChanges();

    const series = fixture.componentInstance.rollingSeries();
    const names = series.map((s) => s.name);
    expect(names).toContain('Rolling Sharpe');
    expect(names).toContain('Rolling Vol');
  });

  it('renders the summary heading when a run is provided', () => {
    const fixture = createFixture();
    fixture.componentRef.setInput('run', makeRun());
    fixture.detectChanges();

    const panel = fixture.nativeElement.querySelector('[data-testid="panel"]');
    expect(panel).toBeTruthy();
    expect((panel as HTMLElement).textContent).toContain('Summary Stats');
    expect((panel as HTMLElement).textContent).toContain('Equity Curve');
    expect((panel as HTMLElement).textContent).toContain('Drawdowns');
    expect((panel as HTMLElement).textContent).toContain('Rolling Metrics');
  });
});
