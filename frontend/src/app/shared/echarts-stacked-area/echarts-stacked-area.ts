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

export interface AreaSeries {
  name: string;
  values: number[];
  color?: string;
}

@Component({
  selector: 'app-echarts-stacked-area',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<div #container class="w-full" [style.height.px]="height()"></div>`,
  providers: [{ provide: CHART_EXPORTABLE, useExisting: EchartsStackedAreaComponent }],
})
export class EchartsStackedAreaComponent implements OnDestroy, ChartExportable {
  labels = input<string[]>([]);
  series = input<AreaSeries[]>([]);
  height = input(280);

  private readonly container = viewChild.required<ElementRef<HTMLElement>>('container');
  private chart?: EChartsType;
  private ro?: ResizeObserver;

  constructor() {
    afterNextRender(() => this.initChart());
    effect(() => {
      const l = this.labels();
      const s = this.series();
      if (this.chart && l.length > 0) {
        this.chart.setOption(this.buildOption(l, s));
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
    this.chart.setOption(this.buildOption(this.labels(), this.series()));

    this.ro = new ResizeObserver(() => this.chart?.resize());
    this.ro.observe(el);
  }

  private buildOption(labels: string[], series: AreaSeries[]): EChartsCoreOption {
    const chartColors = Array.from({ length: 8 }, (_, i) =>
      readCssVar(`--color-chart-${i + 1}`)
    );

    return {
      tooltip: {
        trigger: 'axis',
      },
      legend: {
        bottom: 0,
      },
      grid: { left: 50, right: 16, top: 16, bottom: 40 },
      xAxis: {
        type: 'category',
        boundaryGap: false,
        data: labels,
      },
      yAxis: {
        type: 'value',
      },
      series: series.map((s, i) => {
        const color = s.color || chartColors[i % chartColors.length];
        return {
          name: s.name,
          type: 'line',
          stack: 'total',
          data: s.values,
          symbol: 'none',
          lineStyle: { width: 1.5, color },
          itemStyle: { color },
          areaStyle: { color: `${color}33` },
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
