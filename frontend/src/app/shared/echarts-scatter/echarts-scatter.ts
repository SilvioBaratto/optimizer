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

export interface ScatterPoint {
  x: number;
  y: number;
  label?: string;
}

@Component({
  selector: 'app-echarts-scatter',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<div #container class="w-full" [style.height.px]="height()"></div>`,
  providers: [{ provide: CHART_EXPORTABLE, useExisting: EchartsScatterComponent }],
})
export class EchartsScatterComponent implements OnDestroy, ChartExportable {
  frontierPoints = input<ScatterPoint[]>([]);
  optimalPoint = input<ScatterPoint | null>(null);
  highlightedPoints = input<ScatterPoint[]>([]);
  height = input(280);

  private readonly container = viewChild.required<ElementRef<HTMLElement>>('container');
  private chart?: EChartsType;
  private ro?: ResizeObserver;

  constructor() {
    afterNextRender(() => this.initChart());
    effect(() => {
      const pts = this.frontierPoints();
      const highlighted = this.highlightedPoints();
      const optimal = this.optimalPoint();
      if (this.chart && pts.length > 0) {
        this.chart.setOption(this.buildOption(pts, optimal, highlighted));
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
    this.chart = init(el, 'portfolio', { renderer: 'canvas' });
    this.chart.setOption(this.buildOption(this.frontierPoints(), this.optimalPoint(), this.highlightedPoints()));

    this.ro = new ResizeObserver(() => this.chart?.resize());
    this.ro.observe(el);
  }

  private buildOption(
    pts: ScatterPoint[],
    optimal: ScatterPoint | null,
    highlighted: ScatterPoint[],
  ): EChartsCoreOption {
    const style = getComputedStyle(document.documentElement);
    const chart7 = style.getPropertyValue('--color-chart-7').trim();
    const chart1 = style.getPropertyValue('--color-chart-1').trim();
    const chart3 = style.getPropertyValue('--color-chart-3').trim();
    const textColor = style.getPropertyValue('--color-text').trim();

    const frontierData = pts.map(p => [+(p.x * 100).toFixed(3), +(p.y * 100).toFixed(3)]);

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const series: any[] = [
      {
        name: 'Efficient Frontier',
        type: 'line',
        data: frontierData,
        smooth: true,
        lineStyle: { color: chart7, width: 2 },
        itemStyle: { color: chart7 },
        symbol: 'none',
        z: 1,
      },
    ];

    const colors = [chart1, chart3];
    const allPoints = [
      ...(optimal ? [optimal] : []),
      ...highlighted,
    ];

    for (let i = 0; i < allPoints.length; i++) {
      const pt = allPoints[i];
      const color = colors[i % colors.length];
      series.push({
        name: pt.label ?? `Portfolio ${i + 1}`,
        type: 'scatter',
        data: [[+(pt.x * 100).toFixed(3), +(pt.y * 100).toFixed(3)]],
        symbolSize: 12,
        itemStyle: { color, borderColor: '#ffffff', borderWidth: 2 },
        z: 10,
        label: {
          show: true,
          formatter: pt.label ?? '',
          position: i === 0 ? 'top' : 'right',
          color: textColor,
          fontSize: 11,
          fontWeight: 'bold',
        },
      });
    }

    return {
      legend: { show: false },
      tooltip: {
        trigger: 'item',
        formatter: (params: unknown) => {
          const p = params as { value: [number, number]; seriesName: string };
          return `${p.seriesName}<br/>Risk: ${p.value[0].toFixed(2)}%<br/>Return: ${p.value[1].toFixed(2)}%`;
        },
      },
      grid: { left: 50, right: 16, top: 24, bottom: 24 },
      xAxis: {
        type: 'value',
        name: 'Risk (\u03c3 %)',
        nameLocation: 'middle',
        nameGap: 26,
        axisLabel: { formatter: (v: number) => `${v.toFixed(1)}%` },
      },
      yAxis: {
        type: 'value',
        name: 'Return (%)',
        nameLocation: 'middle',
        nameGap: 40,
        axisLabel: { formatter: (v: number) => `${v.toFixed(1)}%` },
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
