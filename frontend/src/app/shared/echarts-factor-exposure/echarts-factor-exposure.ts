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

export interface FactorExposureItem {
  factor: string;
  exposure: number;
  group?: string;
}

@Component({
  selector: 'app-echarts-factor-exposure',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<div #container class="w-full" [style.height.px]="height()"></div>`,
  providers: [{ provide: CHART_EXPORTABLE, useExisting: EchartsFactorExposureComponent }],
})
export class EchartsFactorExposureComponent implements OnDestroy, ChartExportable {
  data = input<FactorExposureItem[]>([]);
  height = input(280);

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
    const { BarChart } = await import('echarts/charts');
    const { GridComponent, TooltipComponent, MarkLineComponent } = await import(
      'echarts/components'
    );
    const { CanvasRenderer } = await import('echarts/renderers');

    use([BarChart, GridComponent, TooltipComponent, MarkLineComponent, CanvasRenderer]);

    const el = this.container().nativeElement;
    this.chart = init(el, 'portfolio', { renderer: 'canvas' });
    this.chart.setOption(this.buildOption(this.data()));

    this.ro = new ResizeObserver(() => this.chart?.resize());
    this.ro.observe(el);
  }

  private buildOption(items: FactorExposureItem[]): EChartsCoreOption {
    const gainColor = readCssVar('--color-gain');
    const lossColor = readCssVar('--color-loss');
    const borderColor = readCssVar('--color-border');

    const categories = items.map((i) => i.factor);
    const values = items.map((i) => +i.exposure.toFixed(3));

    return {
      tooltip: {
        formatter: (params: unknown) => {
          const p = (params as Array<{ name: string; value: number }>)[0];
          return `${p.name}: ${p.value >= 0 ? '+' : ''}${p.value.toFixed(3)}`;
        },
      },
      grid: { left: 10, right: 20, top: 10, bottom: 10, containLabel: true },
      xAxis: {
        type: 'value',
        axisLabel: {
          formatter: (v: number) => `${v > 0 ? '+' : ''}${v.toFixed(2)}`,
        },
      },
      yAxis: {
        type: 'category',
        data: categories,
        inverse: true,
      },
      series: [
        {
          type: 'bar',
          data: values.map((v) => ({
            value: v,
            itemStyle: { color: v >= 0 ? gainColor : lossColor },
          })),
          barMaxWidth: 18,
          markLine: {
            silent: true,
            symbol: 'none',
            data: [{ xAxis: 0 }],
            lineStyle: { color: borderColor, width: 1 },
            label: { show: false },
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
