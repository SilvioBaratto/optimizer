import {
  Component,
  input,
  computed,
  ElementRef,
  viewChild,
  afterNextRender,
  effect,
  OnDestroy,
  ChangeDetectionStrategy,
} from '@angular/core';
import type { EChartsType, EChartsCoreOption } from 'echarts/core';

import { readCssVar } from '../charts/echarts-theme';
import { CHART_EXPORTABLE, type ChartExportable } from '../charts/chart-export.token';

export type RollingMetricFormatter = 'percent' | 'ratio' | 'unit';

export interface RollingMetricPoint {
  date: string;
  value: number;
}

export interface RollingMetricSeries {
  name: string;
  values: RollingMetricPoint[];
  formatter: RollingMetricFormatter;
}

@Component({
  selector: 'app-echarts-rolling-metrics',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    @if (loading()) {
      <div
        data-testid="loading"
        class="flex h-full w-full items-center justify-center text-sm text-text-secondary"
        [style.height.px]="height()"
      >
        Loading rolling metrics…
      </div>
    } @else if (!hasData()) {
      <div
        data-testid="empty"
        class="flex h-full w-full items-center justify-center text-sm text-text-secondary"
        [style.height.px]="height()"
      >
        No rolling metrics available.
      </div>
    } @else {
      <div
        #container
        data-testid="chart"
        class="w-full"
        [style.height.px]="height()"
      ></div>
    }
  `,
  providers: [{ provide: CHART_EXPORTABLE, useExisting: EchartsRollingMetricsComponent }],
})
export class EchartsRollingMetricsComponent implements OnDestroy, ChartExportable {
  readonly series = input.required<RollingMetricSeries[]>();
  readonly loading = input<boolean>(false);
  readonly height = input<number>(260);

  readonly hasData = computed(() => this.series().some((s) => s.values.length > 0));

  private readonly container = viewChild<ElementRef<HTMLElement>>('container');
  private chart?: EChartsType;
  private ro?: ResizeObserver;

  constructor() {
    afterNextRender(() => this.initChart());
    effect(() => this.renderIfReady());
  }

  getChartInstance(): EChartsType | undefined {
    return this.chart;
  }

  ngOnDestroy(): void {
    this.disposeChart();
  }

  buildOptionForTest(): EChartsCoreOption {
    return this.buildOption();
  }

  private renderIfReady(): void {
    // Track reactive dependencies so the effect re-runs on change.
    void this.series();
    void this.loading();

    if (this.loading() || !this.hasData()) {
      this.disposeChart();
      return;
    }
    if (!this.chart) {
      return;
    }
    this.chart.setOption(this.buildOption(), true);
  }

  private async initChart(): Promise<void> {
    if (this.loading() || !this.hasData()) {
      return;
    }
    const el = this.container()?.nativeElement;
    if (!el) {
      return;
    }

    const { init, use } = await import('echarts/core');
    const { LineChart } = await import('echarts/charts');
    const { GridComponent, TooltipComponent, LegendComponent } = await import(
      'echarts/components'
    );
    const { CanvasRenderer } = await import('echarts/renderers');

    use([LineChart, GridComponent, TooltipComponent, LegendComponent, CanvasRenderer]);

    this.chart = init(el, 'portfolio', { renderer: 'canvas' });
    this.chart.setOption(this.buildOption());

    this.ro = new ResizeObserver(() => this.chart?.resize());
    this.ro.observe(el);
  }

  private buildOption(): EChartsCoreOption {
    const series = this.series();
    const formatters = collectDistinctFormatters(series);
    const xLabels = collectSortedDates(series);
    const yAxis = formatters.map((f, i) => buildYAxis(f, i));
    const axisIndexByFormatter = buildAxisIndex(formatters);
    const palette = readPalette();

    return {
      tooltip: { trigger: 'axis' },
      legend: { top: 0, right: 0 },
      grid: formatters.length > 1
        ? { left: 50, right: 50, top: 32, bottom: 32 }
        : { left: 50, right: 16, top: 32, bottom: 32 },
      xAxis: { type: 'category', boundaryGap: false, data: xLabels },
      yAxis,
      series: series.map((s, i) => buildLineSeries(s, palette[i % palette.length], axisIndexByFormatter, xLabels)),
    };
  }

  private disposeChart(): void {
    this.ro?.disconnect();
    this.ro = undefined;
    this.chart?.dispose();
    this.chart = undefined;
  }
}

function collectDistinctFormatters(series: RollingMetricSeries[]): RollingMetricFormatter[] {
  const seen: RollingMetricFormatter[] = [];
  for (const s of series) {
    if (!seen.includes(s.formatter)) {
      seen.push(s.formatter);
    }
  }
  return seen.slice(0, 2);
}

function collectSortedDates(series: RollingMetricSeries[]): string[] {
  const set = new Set<string>();
  for (const s of series) {
    for (const p of s.values) {
      set.add(p.date);
    }
  }
  return Array.from(set).sort();
}

function buildAxisIndex(
  formatters: RollingMetricFormatter[],
): Map<RollingMetricFormatter, number> {
  const map = new Map<RollingMetricFormatter, number>();
  formatters.forEach((f, i) => map.set(f, i));
  return map;
}

function buildYAxis(formatter: RollingMetricFormatter, index: number): unknown {
  return {
    type: 'value',
    position: index === 0 ? 'left' : 'right',
    axisLabel: { formatter: formatValueFor(formatter) },
    splitLine: { show: index === 0 },
  };
}

function buildLineSeries(
  s: RollingMetricSeries,
  color: string,
  axisIndexByFormatter: Map<RollingMetricFormatter, number>,
  xLabels: string[],
): unknown {
  const byDate = new Map(s.values.map((p) => [p.date, p.value]));
  const data = xLabels.map((d) => (byDate.has(d) ? +byDate.get(d)!.toFixed(6) : null));
  const yAxisIndex = axisIndexByFormatter.get(s.formatter) ?? 0;
  return {
    name: s.name,
    type: 'line',
    data,
    yAxisIndex,
    symbol: 'none',
    smooth: false,
    lineStyle: { width: 1.5, color },
    itemStyle: { color },
    connectNulls: true,
  };
}

function formatValueFor(formatter: RollingMetricFormatter): (v: number) => string {
  switch (formatter) {
    case 'percent':
      return (v: number) => `${(v * 100).toFixed(2)}%`;
    case 'ratio':
    case 'unit':
      return (v: number) => v.toFixed(2);
  }
}

function readPalette(): string[] {
  const raw = [
    readCssVar('--color-chart-1'),
    readCssVar('--color-chart-2'),
    readCssVar('--color-chart-3'),
    readCssVar('--color-chart-4'),
    readCssVar('--color-chart-5'),
    readCssVar('--color-chart-6'),
  ];
  return raw.filter((c) => c.length > 0);
}
