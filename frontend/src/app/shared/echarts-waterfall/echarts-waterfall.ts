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
  selector: 'app-echarts-waterfall',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<div #container class="w-full" [style.height.px]="height()"></div>`,
  providers: [{ provide: CHART_EXPORTABLE, useExisting: EchartsWaterfallComponent }],
})
export class EchartsWaterfallComponent implements OnDestroy, ChartExportable {
  categories = input<string[]>([]);
  values = input<number[]>([]);
  baseValue = input(0);
  height = input(220);

  private readonly container = viewChild.required<ElementRef<HTMLElement>>('container');
  private chart?: EChartsType;
  private ro?: ResizeObserver;

  constructor() {
    afterNextRender(() => this.initChart());
    effect(() => {
      const c = this.categories();
      const v = this.values();
      if (this.chart && c.length > 0) {
        this.chart.setOption(this.buildOption(c, v));
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
    this.chart.setOption(this.buildOption(this.categories(), this.values()));

    this.ro = new ResizeObserver(() => this.chart?.resize());
    this.ro.observe(el);
  }

  private buildOption(categories: string[], values: number[]): EChartsCoreOption {
    const gainColor = readCssVar('--color-gain');
    const lossColor = readCssVar('--color-loss');
    const borderColor = readCssVar('--color-border');

    const base = this.baseValue();
    const bases: number[] = [];
    const barValues: number[] = [];
    let running = base;

    for (const v of values) {
      if (v >= 0) {
        bases.push(running);
        barValues.push(v);
      } else {
        bases.push(running + v);
        barValues.push(-v);
      }
      running += v;
    }

    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: (params: unknown) => {
          const p = params as Array<{ name: string; dataIndex: number }>;
          const idx = p[0].dataIndex;
          const val = values[idx];
          const sign = val >= 0 ? '+' : '';
          return `${categories[idx]}: ${sign}${val.toFixed(2)}`;
        },
      },
      grid: { left: 50, right: 10, top: 10, bottom: 40 },
      xAxis: {
        type: 'category',
        data: categories,
        axisLabel: { rotate: categories.length > 8 ? 45 : 0 },
      },
      yAxis: {
        type: 'value',
      },
      series: [
        {
          type: 'bar',
          stack: 'waterfall',
          data: bases.map(b => ({
            value: b,
            itemStyle: { color: 'transparent' },
          })),
          emphasis: { itemStyle: { color: 'transparent' } },
          silent: true,
        },
        {
          type: 'bar',
          stack: 'waterfall',
          data: barValues.map((v, i) => ({
            value: v,
            itemStyle: {
              color: values[i] >= 0 ? gainColor : lossColor,
              borderRadius: [2, 2, 0, 0] as [number, number, number, number],
            },
          })),
          barMaxWidth: 32,
          markLine: {
            silent: true,
            symbol: 'none',
            data: [{ yAxis: base }],
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
