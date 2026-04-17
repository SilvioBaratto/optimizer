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

export interface YieldCurveData {
  label: string;
  maturities: string[];
  yields: number[];
}

@Component({
  selector: 'app-echarts-yield-curve',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<div #container class="w-full" [style.height.px]="height()"></div>`,
  providers: [{ provide: CHART_EXPORTABLE, useExisting: EchartsYieldCurveComponent }],
})
export class EchartsYieldCurveComponent implements OnDestroy, ChartExportable {
  data = input<YieldCurveData[]>([]);
  height = input(260);

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
    const { GridComponent, TooltipComponent, LegendComponent } = await import(
      'echarts/components'
    );
    const { CanvasRenderer } = await import('echarts/renderers');

    use([LineChart, GridComponent, TooltipComponent, LegendComponent, CanvasRenderer]);

    const el = this.container().nativeElement;
    this.chart = init(el, 'portfolio', { renderer: 'canvas' });
    this.chart.setOption(this.buildOption(this.data()));

    this.ro = new ResizeObserver(() => this.chart?.resize());
    this.ro.observe(el);
  }

  private buildOption(curves: YieldCurveData[]): EChartsCoreOption {
    const chartColors = Array.from({ length: 8 }, (_, i) =>
      readCssVar(`--color-chart-${i + 1}`),
    );
    const xAxis = curves[0]?.maturities ?? [];

    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'cross' },
        formatter: (params: unknown) => {
          const ps = params as Array<{ seriesName: string; value: number; axisValueLabel: string; color: string }>;
          if (!Array.isArray(ps) || !ps.length) return '';
          let html = `<div style="font-size:12px"><b>${ps[0].axisValueLabel}</b>`;
          for (const p of ps) {
            html += `<br/><span style="color:${p.color}">&#9679;</span> ${p.seriesName}: ${p.value.toFixed(2)}%`;
          }
          return html + '</div>';
        },
      },
      legend: { bottom: 0, type: 'scroll' },
      grid: { left: 50, right: 16, top: 16, bottom: 40 },
      xAxis: { type: 'category', data: xAxis, boundaryGap: false },
      yAxis: {
        type: 'value',
        axisLabel: { formatter: (v: number) => `${v.toFixed(1)}%` },
      },
      series: curves.map((curve, i) => {
        const color = chartColors[i % chartColors.length];
        return {
          name: curve.label,
          type: 'line',
          data: curve.yields,
          symbol: 'circle',
          symbolSize: 5,
          smooth: true,
          lineStyle: { width: 2, color },
          itemStyle: { color },
        };
      }),
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
