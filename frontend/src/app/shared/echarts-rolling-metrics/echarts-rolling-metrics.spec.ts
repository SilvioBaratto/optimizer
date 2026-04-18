import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import {
  EchartsRollingMetricsComponent,
  RollingMetricSeries,
} from './echarts-rolling-metrics';
import { CHART_EXPORTABLE } from '../charts/chart-export.token';

function makeSharpeSeries(): RollingMetricSeries {
  return {
    name: 'Sharpe (63d)',
    formatter: 'ratio',
    values: [
      { date: '2026-01-01', value: 0.5 },
      { date: '2026-02-01', value: 0.8 },
      { date: '2026-03-01', value: 1.2 },
    ],
  };
}

function makeVolatilitySeries(): RollingMetricSeries {
  return {
    name: 'Vol (63d)',
    formatter: 'percent',
    values: [
      { date: '2026-01-01', value: 0.12 },
      { date: '2026-02-01', value: 0.15 },
      { date: '2026-03-01', value: 0.1 },
    ],
  };
}

function makeBetaSeries(): RollingMetricSeries {
  return {
    name: 'Beta (63d)',
    formatter: 'unit',
    values: [
      { date: '2026-01-01', value: 0.95 },
      { date: '2026-02-01', value: 1.05 },
      { date: '2026-03-01', value: 1.1 },
    ],
  };
}

describe('EchartsRollingMetricsComponent', () => {
  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [EchartsRollingMetricsComponent],
      providers: [provideZonelessChangeDetection()],
    });
  });

  it('creates successfully', () => {
    const fixture = TestBed.createComponent(EchartsRollingMetricsComponent);
    fixture.componentRef.setInput('series', []);
    expect(fixture.componentInstance).toBeTruthy();
  });

  it('shows the empty state when series is empty', () => {
    const fixture = TestBed.createComponent(EchartsRollingMetricsComponent);
    fixture.componentRef.setInput('series', []);
    fixture.detectChanges();

    expect(
      fixture.nativeElement.querySelector('[data-testid="empty"]'),
    ).not.toBeNull();
    expect(
      fixture.nativeElement.querySelector('[data-testid="chart"]'),
    ).toBeNull();
  });

  it('shows the empty state when every series has empty values', () => {
    const fixture = TestBed.createComponent(EchartsRollingMetricsComponent);
    const series: RollingMetricSeries[] = [
      { name: 'a', formatter: 'ratio', values: [] },
      { name: 'b', formatter: 'percent', values: [] },
    ];
    fixture.componentRef.setInput('series', series);
    fixture.detectChanges();

    expect(
      fixture.nativeElement.querySelector('[data-testid="empty"]'),
    ).not.toBeNull();
    expect(
      fixture.nativeElement.querySelector('[data-testid="chart"]'),
    ).toBeNull();
  });

  it('shows the loading placeholder when loading=true', () => {
    const fixture = TestBed.createComponent(EchartsRollingMetricsComponent);
    fixture.componentRef.setInput('series', []);
    fixture.componentRef.setInput('loading', true);
    fixture.detectChanges();

    expect(
      fixture.nativeElement.querySelector('[data-testid="loading"]'),
    ).not.toBeNull();
    expect(
      fixture.nativeElement.querySelector('[data-testid="empty"]'),
    ).toBeNull();
    expect(
      fixture.nativeElement.querySelector('[data-testid="chart"]'),
    ).toBeNull();
  });

  it('renders the chart container when at least one series has values', () => {
    const fixture = TestBed.createComponent(EchartsRollingMetricsComponent);
    fixture.componentRef.setInput('series', [makeSharpeSeries()]);
    fixture.detectChanges();

    expect(
      fixture.nativeElement.querySelector('[data-testid="chart"]'),
    ).not.toBeNull();
  });

  it('single series renders a single y-axis using the series formatter', () => {
    const fixture = TestBed.createComponent(EchartsRollingMetricsComponent);
    fixture.componentRef.setInput('series', [makeSharpeSeries()]);
    fixture.detectChanges();

    const option = fixture.componentInstance.buildOptionForTest();
    const yAxis = option['yAxis'] as Array<{ formatter?: unknown }>;
    expect(Array.isArray(yAxis)).toBe(true);
    expect(yAxis.length).toBe(1);

    const series = option['series'] as Array<{ yAxisIndex: number }>;
    expect(series.length).toBe(1);
    expect(series[0].yAxisIndex).toBe(0);
  });

  it('multi-series with identical formatters renders a single y-axis', () => {
    const fixture = TestBed.createComponent(EchartsRollingMetricsComponent);
    const a: RollingMetricSeries = {
      name: 'Sharpe A',
      formatter: 'ratio',
      values: [{ date: '2026-01-01', value: 0.5 }],
    };
    const b: RollingMetricSeries = {
      name: 'Sharpe B',
      formatter: 'ratio',
      values: [{ date: '2026-01-01', value: 0.8 }],
    };
    fixture.componentRef.setInput('series', [a, b]);
    fixture.detectChanges();

    const option = fixture.componentInstance.buildOptionForTest();
    const yAxis = option['yAxis'] as unknown[];
    expect(yAxis.length).toBe(1);

    const series = option['series'] as Array<{ yAxisIndex: number }>;
    expect(series.every((s) => s.yAxisIndex === 0)).toBe(true);
  });

  it('multi-series with two distinct formatters renders two y-axes assigned per formatter', () => {
    const fixture = TestBed.createComponent(EchartsRollingMetricsComponent);
    const sharpe = makeSharpeSeries();
    const vol = makeVolatilitySeries();
    const beta = makeBetaSeries();
    fixture.componentRef.setInput('series', [sharpe, vol, beta]);
    fixture.detectChanges();

    const option = fixture.componentInstance.buildOptionForTest();
    const yAxis = option['yAxis'] as unknown[];
    expect(yAxis.length).toBe(2);

    const series = option['series'] as Array<{ name: string; yAxisIndex: number }>;
    const idxByName = new Map(series.map((s) => [s.name, s.yAxisIndex]));
    // sharpe (ratio) was encountered first → left axis (index 0)
    expect(idxByName.get('Sharpe (63d)')).toBe(0);
    // vol (percent) was encountered second → right axis (index 1)
    expect(idxByName.get('Vol (63d)')).toBe(1);
    // beta (unit) falls back to the axis matching its formatter: no axis matches → assigned to left (index 0)
    expect(idxByName.get('Beta (63d)')).toBe(0);
  });

  it('axis formatter functions match the percent/ratio/unit rules', () => {
    const fixture = TestBed.createComponent(EchartsRollingMetricsComponent);
    fixture.componentRef.setInput('series', [makeSharpeSeries(), makeVolatilitySeries()]);
    fixture.detectChanges();

    const option = fixture.componentInstance.buildOptionForTest();
    const yAxis = option['yAxis'] as Array<{ axisLabel: { formatter: (v: number) => string } }>;
    const leftFmt = yAxis[0].axisLabel.formatter;
    const rightFmt = yAxis[1].axisLabel.formatter;
    expect(leftFmt(0.5)).toBe('0.50');
    expect(rightFmt(0.1234)).toBe('12.34%');
  });

  it('uses height input on all state containers', () => {
    const fixture = TestBed.createComponent(EchartsRollingMetricsComponent);
    fixture.componentRef.setInput('series', []);
    fixture.componentRef.setInput('height', 320);
    fixture.detectChanges();
    const emptyEl = fixture.nativeElement.querySelector(
      '[data-testid="empty"]',
    ) as HTMLDivElement;
    expect(emptyEl.style.height).toBe('320px');
  });

  it('registers itself as CHART_EXPORTABLE', () => {
    const fixture = TestBed.createComponent(EchartsRollingMetricsComponent);
    fixture.componentRef.setInput('series', [makeSharpeSeries()]);
    const exportable = fixture.debugElement.injector.get(CHART_EXPORTABLE);
    expect(exportable).toBe(fixture.componentInstance);
  });
});
