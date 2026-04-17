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

export interface DrawdownPoint {
  date: string;
  drawdown: number;
}

@Component({
  selector: 'app-echarts-drawdown',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<div #container class="w-full" [style.height.px]="height()"></div>`,
  providers: [{ provide: CHART_EXPORTABLE, useExisting: EchartsDrawdownComponent }],
})
export class EchartsDrawdownComponent implements OnDestroy, ChartExportable {
  data = input<DrawdownPoint[]>([]);
  height = input(220);

  private readonly container = viewChild.required<ElementRef<HTMLElement>>('container');
  private chart?: EChartsType;
  private ro?: ResizeObserver;

  constructor() {
    afterNextRender(() => this.initChart());
    effect(() => {
      const d = this.data();
      if (this.chart && d.length > 0) {
        this.chart.setOption(this.buildOption(d));
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
    this.chart.setOption(this.buildOption(this.data()));

    this.ro = new ResizeObserver(() => this.chart?.resize());
    this.ro.observe(el);
  }

  private buildOption(points: DrawdownPoint[]): EChartsCoreOption {
    const lossColor = readCssVar('--color-loss');
    const labels = points.map((p) => p.date);
    const values = points.map((p) => +(p.drawdown * 100).toFixed(2));

    return {
      tooltip: {
        trigger: 'axis',
        formatter: (params: unknown) => {
          const p = (params as Array<{ name: string; value: number }>)[0];
          return `${p.name}: ${p.value.toFixed(2)}%`;
        },
      },
      grid: { left: 50, right: 16, top: 16, bottom: 32 },
      xAxis: { type: 'category', boundaryGap: false, data: labels },
      yAxis: {
        type: 'value',
        max: 0,
        axisLabel: { formatter: (v: number) => `${v.toFixed(0)}%` },
      },
      series: [
        {
          type: 'line',
          data: values,
          symbol: 'none',
          lineStyle: { width: 1.5, color: lossColor },
          itemStyle: { color: lossColor },
          areaStyle: {
            origin: 'start',
            color: {
              type: 'linear',
              x: 0, y: 0, x2: 0, y2: 1,
              colorStops: [
                { offset: 0, color: `${lossColor}00` },
                { offset: 1, color: `${lossColor}66` },
              ],
            },
          },
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
