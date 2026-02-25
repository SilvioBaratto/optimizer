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

@Component({
  selector: 'app-echarts-histogram',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<div #container class="w-full" [style.height.px]="height()"></div>`,
  providers: [{ provide: CHART_EXPORTABLE, useExisting: EchartsHistogramComponent }],
})
export class EchartsHistogramComponent implements OnDestroy, ChartExportable {
  values = input<number[]>([]);
  bins = input(0);
  overlayNormal = input(false);
  height = input(220);

  private readonly container = viewChild.required<ElementRef<HTMLElement>>('container');
  private chart?: EChartsType;
  private ro?: ResizeObserver;

  constructor() {
    afterNextRender(() => this.initChart());
    effect(() => {
      const v = this.values();
      if (this.chart && v.length > 0) {
        this.chart.setOption(this.buildOption(v));
      }
    });
  }

  private async initChart() {
    const { init, use } = await import('echarts/core');
    const { BarChart, LineChart } = await import('echarts/charts');
    const { GridComponent, TooltipComponent } = await import('echarts/components');
    const { CanvasRenderer } = await import('echarts/renderers');

    use([BarChart, LineChart, GridComponent, TooltipComponent, CanvasRenderer]);

    const el = this.container().nativeElement;
    this.chart = init(el, 'portfolio', { renderer: 'canvas' });
    this.chart.setOption(this.buildOption(this.values()));

    this.ro = new ResizeObserver(() => this.chart?.resize());
    this.ro.observe(el);
  }

  private buildOption(values: number[]): EChartsCoreOption {
    const chartColor = readCssVar('--color-chart-1');
    const accentColor = readCssVar('--color-chart-3');

    const n = values.length;
    if (n === 0) return {};

    const numBins = this.bins() > 0
      ? this.bins()
      : Math.ceil(Math.log2(n) + 1);

    const sorted = [...values].sort((a, b) => a - b);
    const minVal = sorted[0];
    const maxVal = sorted[sorted.length - 1];
    const range = maxVal - minVal;

    if (range === 0) {
      return {
        grid: { left: 50, right: 10, top: 10, bottom: 30 },
        xAxis: { type: 'category', data: [minVal.toFixed(4)] },
        yAxis: { type: 'value' },
        series: [{ type: 'bar', data: [n], itemStyle: { color: chartColor } }],
      };
    }

    const binWidth = range / numBins;
    const counts = new Array<number>(numBins).fill(0);
    const binLabels: string[] = [];

    for (let i = 0; i < numBins; i++) {
      const lo = minVal + i * binWidth;
      binLabels.push(lo.toFixed(3));
    }

    for (const v of values) {
      let idx = Math.floor((v - minVal) / binWidth);
      if (idx >= numBins) idx = numBins - 1;
      counts[idx]++;
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const series: any[] = [
      {
        type: 'bar',
        data: counts,
        barWidth: '90%',
        itemStyle: {
          color: `${chartColor}cc`,
          borderRadius: [2, 2, 0, 0],
        },
      },
    ];

    if (this.overlayNormal()) {
      const mean = values.reduce((a, b) => a + b, 0) / n;
      const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
      const std = Math.sqrt(variance);

      if (std > 0) {
        const normalData: number[] = [];
        for (let i = 0; i < numBins; i++) {
          const x = minVal + (i + 0.5) * binWidth;
          const z = (x - mean) / std;
          const pdf = Math.exp(-0.5 * z * z) / (std * Math.sqrt(2 * Math.PI));
          normalData.push(+(pdf * n * binWidth).toFixed(2));
        }

        series.push({
          type: 'line',
          data: normalData,
          smooth: true,
          symbol: 'none',
          lineStyle: { color: accentColor, width: 2 },
          itemStyle: { color: accentColor },
        });
      }
    }

    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
      },
      grid: { left: 50, right: 10, top: 10, bottom: 30 },
      xAxis: {
        type: 'category',
        data: binLabels,
        axisLabel: { rotate: 45, fontSize: 9 },
      },
      yAxis: {
        type: 'value',
        name: 'Count',
        nameTextStyle: { fontSize: 10 },
      },
      series,
    };
  }

  getChartInstance(): EChartsType | undefined {
    return this.chart;
  }

  ngOnDestroy() {
    this.ro?.disconnect();
    this.chart?.dispose();
  }
}
