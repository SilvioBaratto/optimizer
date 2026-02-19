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
  selector: 'app-echarts-heatmap',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<div #container class="w-full" [style.height.px]="height()"></div>`,
})
export class EchartsHeatmapComponent implements OnDestroy {
  assets = input<string[]>([]);
  matrix = input<number[][]>([]);
  height = input(340);

  private readonly container = viewChild.required<ElementRef<HTMLElement>>('container');
  private chart?: EChartsType;
  private ro?: ResizeObserver;

  constructor() {
    afterNextRender(() => this.initChart());
    effect(() => {
      const a = this.assets();
      const m = this.matrix();
      if (this.chart && a.length > 0) {
        this.chart.setOption(this.buildOption(a, m));
      }
    });
  }

  private async initChart() {
    const { init, use } = await import('echarts/core');
    const { HeatmapChart } = await import('echarts/charts');
    const { GridComponent, TooltipComponent, VisualMapComponent } = await import(
      'echarts/components'
    );
    const { CanvasRenderer } = await import('echarts/renderers');

    use([HeatmapChart, GridComponent, TooltipComponent, VisualMapComponent, CanvasRenderer]);

    const el = this.container().nativeElement;
    this.chart = init(el, undefined, { renderer: 'canvas' });
    this.chart.setOption(this.buildOption(this.assets(), this.matrix()));

    this.ro = new ResizeObserver(() => this.chart?.resize());
    this.ro.observe(el);
  }

  private buildOption(assets: string[], matrix: number[][]): EChartsCoreOption {
    const data: [number, number, number][] = [];
    for (let i = 0; i < assets.length; i++) {
      for (let j = 0; j < assets.length; j++) {
        data.push([j, i, +(matrix[i]?.[j] ?? 0).toFixed(2)]);
      }
    }

    return {
      backgroundColor: 'transparent',
      tooltip: {
        position: 'top',
        formatter: (params: unknown) => {
          const p = params as { value: [number, number, number] };
          const col = assets[p.value[0]];
          const row = assets[p.value[1]];
          return `${row} × ${col}<br/>${p.value[2].toFixed(2)}`;
        },
        backgroundColor: '#ffffff',
        borderColor: '#e4e4e7',
        borderWidth: 1,
        textStyle: { color: '#18181b', fontSize: 12 },
      },
      grid: { left: 60, right: 80, top: 10, bottom: 60 },
      xAxis: {
        type: 'category',
        data: assets,
        axisLabel: { color: '#71717a', fontSize: 10, rotate: 45 },
        axisLine: { lineStyle: { color: '#e4e4e7' } },
        axisTick: { show: false },
        splitArea: { show: true, areaStyle: { color: ['#fafafa', '#ffffff'] } },
      },
      yAxis: {
        type: 'category',
        data: assets,
        axisLabel: { color: '#71717a', fontSize: 10 },
        axisLine: { lineStyle: { color: '#e4e4e7' } },
        axisTick: { show: false },
        splitArea: { show: true, areaStyle: { color: ['#fafafa', '#ffffff'] } },
      },
      visualMap: {
        min: -1,
        max: 1,
        calculable: true,
        orient: 'vertical',
        right: 0,
        top: 'middle',
        itemHeight: 160,
        text: ['1', '-1'],
        textStyle: { color: '#71717a', fontSize: 10 },
        inRange: {
          color: ['#2563eb', '#ffffff', '#dc2626'],
        },
      },
      series: [
        {
          type: 'heatmap',
          data,
          label: {
            show: assets.length <= 10,
            fontSize: 9,
            color: '#52525b',
            formatter: (params: unknown) => {
              const p = params as { value: [number, number, number] };
              return String(p.value[2].toFixed(2));
            },
          },
          emphasis: { itemStyle: { shadowBlur: 6, shadowColor: 'rgba(0,0,0,0.2)' } },
        },
      ],
    };
  }

  ngOnDestroy() {
    this.ro?.disconnect();
    this.chart?.dispose();
  }
}
