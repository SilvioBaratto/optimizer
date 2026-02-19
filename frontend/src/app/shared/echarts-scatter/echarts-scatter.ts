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

export interface ScatterPoint {
  x: number;
  y: number;
  label?: string;
}

@Component({
  selector: 'app-echarts-scatter',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<div #container class="w-full" [style.height.px]="height()"></div>`,
})
export class EchartsScatterComponent implements OnDestroy {
  frontierPoints = input<ScatterPoint[]>([]);
  optimalPoint = input<ScatterPoint | null>(null);
  height = input(280);

  private readonly container = viewChild.required<ElementRef<HTMLElement>>('container');
  private chart?: EChartsType;
  private ro?: ResizeObserver;

  constructor() {
    afterNextRender(() => this.initChart());
    effect(() => {
      const pts = this.frontierPoints();
      if (this.chart && pts.length > 0) {
        this.chart.setOption(this.buildOption(pts, this.optimalPoint()));
      }
    });
  }

  private async initChart() {
    const { init, use } = await import('echarts/core');
    const { ScatterChart, LineChart } = await import('echarts/charts');
    const { GridComponent, TooltipComponent, LegendComponent } = await import(
      'echarts/components'
    );
    const { CanvasRenderer } = await import('echarts/renderers');

    use([ScatterChart, LineChart, GridComponent, TooltipComponent, LegendComponent, CanvasRenderer]);

    const el = this.container().nativeElement;
    this.chart = init(el, undefined, { renderer: 'canvas' });
    this.chart.setOption(this.buildOption(this.frontierPoints(), this.optimalPoint()));

    this.ro = new ResizeObserver(() => this.chart?.resize());
    this.ro.observe(el);
  }

  private buildOption(pts: ScatterPoint[], optimal: ScatterPoint | null): EChartsCoreOption {
    const frontierData = pts.map(p => [+(p.x * 100).toFixed(3), +(p.y * 100).toFixed(3)]);

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const series: any[] = [
      {
        name: 'Efficient Frontier',
        type: 'line',
        data: frontierData,
        smooth: true,
        lineStyle: { color: '#71717a', width: 1.5 },
        itemStyle: { color: '#71717a' },
        symbol: 'none',
        z: 1,
      },
      {
        name: 'Frontier Points',
        type: 'scatter',
        data: frontierData,
        symbolSize: 5,
        itemStyle: { color: '#a1a1aa', opacity: 0.7 },
        z: 2,
      },
    ];

    if (optimal) {
      series.push({
        name: 'Optimal Portfolio',
        type: 'scatter',
        data: [[+(optimal.x * 100).toFixed(3), +(optimal.y * 100).toFixed(3)]],
        symbolSize: 14,
        itemStyle: { color: '#18181b', borderColor: '#ffffff', borderWidth: 2 },
        z: 10,
        label: {
          show: true,
          formatter: optimal.label ?? 'Optimal',
          position: 'top',
          color: '#18181b',
          fontSize: 11,
          fontWeight: 'bold',
        },
      });
    }

    return {
      backgroundColor: 'transparent',
      legend: {
        bottom: 0,
        textStyle: { color: '#71717a', fontSize: 11 },
        data: ['Frontier Points', 'Optimal Portfolio'],
      },
      tooltip: {
        trigger: 'item',
        formatter: (params: unknown) => {
          const p = params as { value: [number, number]; seriesName: string };
          return `${p.seriesName}<br/>Risk: ${p.value[0].toFixed(2)}%<br/>Return: ${p.value[1].toFixed(2)}%`;
        },
        backgroundColor: '#ffffff',
        borderColor: '#e4e4e7',
        borderWidth: 1,
        textStyle: { color: '#18181b', fontSize: 12 },
      },
      grid: { left: 50, right: 16, top: 16, bottom: 40 },
      xAxis: {
        type: 'value',
        name: 'Risk (σ %)',
        nameLocation: 'middle',
        nameGap: 26,
        nameTextStyle: { color: '#71717a', fontSize: 11 },
        axisLabel: { color: '#71717a', fontSize: 10, formatter: (v: number) => `${v.toFixed(1)}%` },
        splitLine: { lineStyle: { color: '#f4f4f5' } },
        axisLine: { lineStyle: { color: '#e4e4e7' } },
      },
      yAxis: {
        type: 'value',
        name: 'Return (%)',
        nameLocation: 'middle',
        nameGap: 40,
        nameTextStyle: { color: '#71717a', fontSize: 11 },
        axisLabel: { color: '#71717a', fontSize: 10, formatter: (v: number) => `${v.toFixed(1)}%` },
        splitLine: { lineStyle: { color: '#f4f4f5' } },
        axisLine: { show: false },
      },
      series,
    };
  }

  ngOnDestroy() {
    this.ro?.disconnect();
    this.chart?.dispose();
  }
}
