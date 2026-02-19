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
import { BarData } from '../bar-chart/bar-chart';

export type { BarData };

@Component({
  selector: 'app-echarts-bar',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<div #container class="w-full" [style.height.px]="height()"></div>`,
})
export class EchartsBarComponent implements OnDestroy {
  data = input<BarData[]>([]);
  height = input(180);

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
    this.chart = init(el, undefined, { renderer: 'canvas' });
    this.chart.setOption(this.buildOption(this.data()));

    this.ro = new ResizeObserver(() => this.chart?.resize());
    this.ro.observe(el);
  }

  private buildOption(bars: BarData[]): EChartsCoreOption {
    const labels = bars.map(b => {
      // Format "2024-03" → "Mar '24"
      const parts = b.label.split('-');
      if (parts.length === 2) {
        const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
        const m = parseInt(parts[1], 10) - 1;
        return `${monthNames[m]} '${parts[0].slice(2)}`;
      }
      return b.label;
    });
    const values = bars.map(b => +(b.value * 100).toFixed(2));
    const colors = values.map(v => (v >= 0 ? '#16a34a' : '#dc2626'));

    return {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        formatter: (params: unknown) => {
          const p = (params as Array<{ name: string; value: number }>)[0];
          return `${p.name}: ${p.value >= 0 ? '+' : ''}${p.value.toFixed(2)}%`;
        },
        backgroundColor: '#ffffff',
        borderColor: '#e4e4e7',
        borderWidth: 1,
        textStyle: { color: '#18181b', fontSize: 12 },
      },
      grid: { left: 40, right: 10, top: 10, bottom: 50 },
      xAxis: {
        type: 'category',
        data: labels,
        axisLabel: { color: '#71717a', fontSize: 10, rotate: 45 },
        axisLine: { lineStyle: { color: '#e4e4e7' } },
        axisTick: { show: false },
      },
      yAxis: {
        type: 'value',
        axisLabel: {
          color: '#71717a',
          fontSize: 10,
          formatter: (v: number) => `${v > 0 ? '+' : ''}${v.toFixed(1)}%`,
        },
        splitLine: { lineStyle: { color: '#f4f4f5' } },
        axisLine: { show: false },
      },
      series: [
        {
          type: 'bar',
          data: values.map((v, i) => ({
            value: v,
            itemStyle: { color: colors[i], borderRadius: [2, 2, 0, 0] as [number, number, number, number],
            },
          })),
          barMaxWidth: 24,
          markLine: {
            silent: true,
            symbol: 'none',
            data: [{ yAxis: 0 }],
            lineStyle: { color: '#e4e4e7', width: 1 },
            label: { show: false },
          },
        },
      ],
    };
  }

  ngOnDestroy() {
    this.ro?.disconnect();
    this.chart?.dispose();
  }
}
