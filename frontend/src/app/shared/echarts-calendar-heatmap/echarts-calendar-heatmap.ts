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
  selector: 'app-echarts-calendar-heatmap',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<div #container class="w-full" [style.height.px]="height()"></div>`,
  providers: [{ provide: CHART_EXPORTABLE, useExisting: EchartsCalendarHeatmapComponent }],
})
export class EchartsCalendarHeatmapComponent implements OnDestroy, ChartExportable {
  years = input<string[]>([]);
  months = input<string[]>([]);
  data = input<number[][]>([]);
  height = input(220);

  private readonly container = viewChild.required<ElementRef<HTMLElement>>('container');
  private chart?: EChartsType;
  private ro?: ResizeObserver;

  constructor() {
    afterNextRender(() => this.initChart());
    effect(() => {
      const y = this.years();
      const m = this.months();
      const d = this.data();
      if (this.chart && y.length > 0 && m.length > 0) {
        this.chart.setOption(this.buildOption(y, m, d));
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
    this.chart = init(el, 'portfolio', { renderer: 'canvas' });
    this.chart.setOption(this.buildOption(this.years(), this.months(), this.data()));

    this.ro = new ResizeObserver(() => this.chart?.resize());
    this.ro.observe(el);
  }

  private buildOption(years: string[], months: string[], data: number[][]): EChartsCoreOption {
    const heatmapColors = Array.from({ length: 7 }, (_, i) =>
      readCssVar(`--color-heatmap-${i + 1}`)
    );

    const tuples: [number, number, number][] = [];
    let absMax = 0;

    for (let row = 0; row < years.length; row++) {
      for (let col = 0; col < months.length; col++) {
        const val = data[row]?.[col] ?? 0;
        tuples.push([col, row, val]);
        const abs = Math.abs(val);
        if (abs > absMax) absMax = abs;
      }
    }

    if (absMax === 0) absMax = 0.01;

    return {
      tooltip: {
        position: 'top',
        formatter: (params: unknown) => {
          const p = params as { value: [number, number, number] };
          const year = years[p.value[1]];
          const month = months[p.value[0]];
          const val = p.value[2];
          const pct = (val * 100).toFixed(1);
          const sign = val >= 0 ? '+' : '';
          return `${month} ${year}<br/>${sign}${pct}%`;
        },
      },
      grid: { left: 60, right: 80, top: 10, bottom: 30 },
      xAxis: {
        type: 'category',
        data: months,
        position: 'bottom',
        splitArea: { show: false },
      },
      yAxis: {
        type: 'category',
        data: years,
        splitArea: { show: false },
      },
      visualMap: {
        min: -absMax,
        max: absMax,
        calculable: true,
        orient: 'vertical',
        right: 0,
        top: 'middle',
        itemHeight: 120,
        inRange: { color: heatmapColors },
        text: [
          `+${(absMax * 100).toFixed(0)}%`,
          `-${(absMax * 100).toFixed(0)}%`,
        ],
      },
      series: [
        {
          type: 'heatmap',
          data: tuples,
          label: {
            show: true,
            fontSize: 9,
            formatter: (params: unknown) => {
              const p = params as { value: [number, number, number] };
              const val = p.value[2];
              const pct = (val * 100).toFixed(1);
              const sign = val >= 0 ? '+' : '';
              return `${sign}${pct}%`;
            },
          },
          emphasis: {
            itemStyle: { shadowBlur: 6, shadowColor: 'rgba(0,0,0,0.2)' },
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
