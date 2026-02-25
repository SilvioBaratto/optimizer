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
import { CHART_EXPORTABLE, type ChartExportable } from '../charts/chart-export.token';

@Component({
  selector: 'app-echarts-line',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<div #container class="w-full" [style.height.px]="height()"></div>`,
  providers: [{ provide: CHART_EXPORTABLE, useExisting: EchartsLineComponent }],
})
export class EchartsLineComponent implements OnDestroy, ChartExportable {
  labels = input<string[]>([]);
  values = input<number[]>([]);
  height = input(220);
  yAxisLabel = input('');
  areaFill = input(false);

  private readonly container = viewChild.required<ElementRef<HTMLElement>>('container');
  private chart?: EChartsType;
  private ro?: ResizeObserver;

  constructor() {
    afterNextRender(() => this.initChart());
    effect(() => {
      const l = this.labels();
      const v = this.values();
      if (this.chart && l.length > 0) {
        this.chart.setOption(this.buildOption(l, v));
      }
    });
  }

  private async initChart() {
    const { init, use } = await import('echarts/core');
    const { LineChart } = await import('echarts/charts');
    const { GridComponent, TooltipComponent } = await import('echarts/components');
    const { CanvasRenderer } = await import('echarts/renderers');

    use([LineChart, GridComponent, TooltipComponent, CanvasRenderer]);

    const el = this.container().nativeElement;
    this.chart = init(el, 'portfolio', { renderer: 'canvas' });
    this.chart.setOption(this.buildOption(this.labels(), this.values()));

    this.ro = new ResizeObserver(() => this.chart?.resize());
    this.ro.observe(el);
  }

  private buildOption(labels: string[], values: number[]): EChartsCoreOption {
    const yLabel = this.yAxisLabel();
    const area = this.areaFill();
    const chartColor = getComputedStyle(document.documentElement).getPropertyValue('--color-chart-1').trim();

    return {
      tooltip: {
        formatter: (params: unknown) => {
          const p = (params as Array<{ name: string; value: number }>)[0];
          return `${p.name}: ${p.value.toFixed(2)}%`;
        },
      },
      grid: { left: 50, right: 16, top: 16, bottom: 32 },
      xAxis: {
        type: 'category',
        data: labels,
      },
      yAxis: {
        type: 'value',
        name: yLabel || undefined,
        nameTextStyle: { fontSize: 10 },
        axisLabel: {
          formatter: (v: number) => `${v.toFixed(2)}%`,
        },
      },
      series: [
        {
          type: 'line',
          data: values,
          symbol: 'circle',
          symbolSize: 5,
          areaStyle: area
            ? {
                color: {
                  type: 'linear',
                  x: 0, y: 0, x2: 0, y2: 1,
                  colorStops: [
                    { offset: 0, color: `${chartColor}1f` },
                    { offset: 1, color: `${chartColor}00` },
                  ],
                },
              }
            : undefined,
        },
      ],
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
