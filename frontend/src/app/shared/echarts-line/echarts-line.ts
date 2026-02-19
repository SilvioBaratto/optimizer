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

@Component({
  selector: 'app-echarts-line',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<div #container class="w-full" [style.height.px]="height()"></div>`,
})
export class EchartsLineComponent implements OnDestroy {
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
    this.chart = init(el, undefined, { renderer: 'canvas' });
    this.chart.setOption(this.buildOption(this.labels(), this.values()));

    this.ro = new ResizeObserver(() => this.chart?.resize());
    this.ro.observe(el);
  }

  private buildOption(labels: string[], values: number[]): EChartsCoreOption {
    const yLabel = this.yAxisLabel();
    const area = this.areaFill();

    return {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        formatter: (params: unknown) => {
          const p = (params as Array<{ name: string; value: number }>)[0];
          return `${p.name}: ${p.value.toFixed(2)}%`;
        },
        backgroundColor: '#ffffff',
        borderColor: '#e4e4e7',
        borderWidth: 1,
        textStyle: { color: '#18181b', fontSize: 12 },
      },
      grid: { left: 50, right: 16, top: 16, bottom: 32 },
      xAxis: {
        type: 'category',
        data: labels,
        axisLabel: { color: '#71717a', fontSize: 11 },
        axisLine: { lineStyle: { color: '#e4e4e7' } },
        axisTick: { show: false },
      },
      yAxis: {
        type: 'value',
        name: yLabel || undefined,
        nameTextStyle: { color: '#71717a', fontSize: 10 },
        axisLabel: {
          color: '#71717a',
          fontSize: 10,
          formatter: (v: number) => `${v.toFixed(2)}%`,
        },
        splitLine: { lineStyle: { color: '#f4f4f5' } },
        axisLine: { show: false },
      },
      series: [
        {
          type: 'line',
          data: values,
          smooth: true,
          lineStyle: { color: '#18181b', width: 2 },
          itemStyle: { color: '#18181b' },
          symbol: 'circle',
          symbolSize: 5,
          areaStyle: area
            ? {
                color: {
                  type: 'linear',
                  x: 0, y: 0, x2: 0, y2: 1,
                  colorStops: [
                    { offset: 0, color: 'rgba(24,24,27,0.12)' },
                    { offset: 1, color: 'rgba(24,24,27,0)' },
                  ],
                },
              }
            : undefined,
        },
      ],
    };
  }

  ngOnDestroy() {
    this.ro?.disconnect();
    this.chart?.dispose();
  }
}
