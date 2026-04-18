import {
  Component,
  input,
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
import type { RegimeHistoryApiResponse } from '../../models/factor.model';

export interface RegimeTimelinePoint {
  date: string;
  probabilities: number[];
}

/**
 * Canonical labels for the four HMM regimes emitted by
 * `GET /market/regime/history`. Shared by every consumer of
 * {@link EchartsRegimeTimelineComponent} so labels stay consistent.
 */
export const REGIME_LABELS: string[] = ['Bull', 'Bear', 'Sideways', 'Volatile'];

/**
 * Pure mapping from the API response shape to the chart input shape.
 * Extracted so both Factor Research and Backtesting reuse the same
 * transformation (no per-page duplication).
 */
export function regimeHistoryApiToTimelinePoints(
  api: RegimeHistoryApiResponse | null,
): RegimeTimelinePoint[] {
  if (!api) return [];
  return api.points.map((p) => ({
    date: p.date,
    probabilities: [p.bullProb, p.bearProb, p.sidewaysProb, p.volatileProb],
  }));
}

@Component({
  selector: 'app-echarts-regime-timeline',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<div #container class="w-full" [style.height.px]="height()"></div>`,
  providers: [{ provide: CHART_EXPORTABLE, useExisting: EchartsRegimeTimelineComponent }],
})
export class EchartsRegimeTimelineComponent implements OnDestroy, ChartExportable {
  data = input<RegimeTimelinePoint[]>([]);
  regimeLabels = input<string[]>([]);
  height = input(260);

  private readonly container = viewChild.required<ElementRef<HTMLElement>>('container');
  private chart?: EChartsType;
  private ro?: ResizeObserver;

  constructor() {
    afterNextRender(() => this.initChart());
    effect(() => {
      const d = this.data();
      const labels = this.regimeLabels();
      if (this.chart && d.length > 0) {
        this.chart.setOption(this.buildOption(d, labels));
      }
    });
  }

  private async initChart() {
    const { init, use } = await import('echarts/core');
    const { LineChart } = await import('echarts/charts');
    const { GridComponent, TooltipComponent, LegendComponent } = await import(
      'echarts/components'
    );
    const { CanvasRenderer } = await import('echarts/renderers');

    use([LineChart, GridComponent, TooltipComponent, LegendComponent, CanvasRenderer]);

    const el = this.container().nativeElement;
    this.chart = init(el, 'portfolio', { renderer: 'canvas' });
    this.chart.setOption(this.buildOption(this.data(), this.regimeLabels()));

    this.ro = new ResizeObserver(() => this.chart?.resize());
    this.ro.observe(el);
  }

  private buildOption(
    points: RegimeTimelinePoint[],
    labels: string[],
  ): EChartsCoreOption {
    const chartColors = Array.from({ length: 8 }, (_, i) =>
      readCssVar(`--color-chart-${i + 1}`),
    );
    const xAxis = points.map((p) => p.date);
    const regimeCount = points[0]?.probabilities.length ?? 0;
    const names = this.regimeNames(labels, regimeCount);

    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'cross' },
      },
      legend: { bottom: 0, type: 'scroll' },
      grid: { left: 50, right: 16, top: 16, bottom: 40 },
      xAxis: { type: 'category', boundaryGap: false, data: xAxis },
      yAxis: {
        type: 'value',
        min: 0,
        max: 1,
        axisLabel: { formatter: (v: number) => `${Math.round(v * 100)}%` },
      },
      series: names.map((name, i) => {
        const color = chartColors[i % chartColors.length];
        return {
          name,
          type: 'line',
          stack: 'total',
          data: points.map((p) => +(p.probabilities[i] ?? 0).toFixed(4)),
          symbol: 'none',
          lineStyle: { width: 0, color },
          itemStyle: { color },
          areaStyle: { color },
        };
      }),
    };
  }

  private regimeNames(labels: string[], count: number): string[] {
    if (labels.length >= count) return labels.slice(0, count);
    return Array.from({ length: count }, (_, i) => labels[i] ?? `Regime ${i + 1}`);
  }

  getChartInstance(): EChartsType | undefined {
    return this.chart;
  }

  ngOnDestroy() {
    this.ro?.disconnect();
    this.chart?.dispose();
  }
}
