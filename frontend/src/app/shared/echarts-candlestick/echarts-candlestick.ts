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

export interface CandlestickData {
  date: string;
  open: number;
  close: number;
  high: number;
  low: number;
  volume?: number;
}

@Component({
  selector: 'app-echarts-candlestick',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<div #container class="w-full" [style.height.px]="height()"></div>`,
  providers: [{ provide: CHART_EXPORTABLE, useExisting: EchartsCandlestickComponent }],
})
export class EchartsCandlestickComponent implements OnDestroy, ChartExportable {
  data = input<CandlestickData[]>([]);
  height = input(360);

  private readonly container = viewChild.required<ElementRef<HTMLElement>>('container');
  private chart?: EChartsType;
  private ro?: ResizeObserver;

  constructor() {
    afterNextRender(() => this.initChart());
    effect(() => {
      const d = this.data();
      if (this.chart && d.length > 0) {
        this.chart.setOption(this.buildOption(d), true);
      }
    });
  }

  private async initChart() {
    const { init, use } = await import('echarts/core');
    const { CandlestickChart, BarChart } = await import('echarts/charts');
    const {
      GridComponent,
      TooltipComponent,
      DataZoomComponent,
      LegendComponent,
    } = await import('echarts/components');
    const { CanvasRenderer } = await import('echarts/renderers');

    use([
      CandlestickChart,
      BarChart,
      GridComponent,
      TooltipComponent,
      DataZoomComponent,
      LegendComponent,
      CanvasRenderer,
    ]);

    const el = this.container().nativeElement;
    this.chart = init(el, 'portfolio', { renderer: 'canvas' });
    this.chart.setOption(this.buildOption(this.data()));

    this.ro = new ResizeObserver(() => this.chart?.resize());
    this.ro.observe(el);
  }

  private buildOption(rows: CandlestickData[]): EChartsCoreOption {
    const gainColor = readCssVar('--color-gain');
    const lossColor = readCssVar('--color-loss');

    const dates = rows.map((r) => r.date);
    const ohlc = rows.map((r) => [r.open, r.close, r.low, r.high]);
    const hasVolume = rows.some((r) => r.volume !== undefined);
    const volumes = rows.map((r) => r.volume ?? 0);

    return {
      tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
      legend: { bottom: 0 },
      grid: hasVolume
        ? [
            { left: 50, right: 16, top: 16, height: '60%' },
            { left: 50, right: 16, top: '76%', height: '16%' },
          ]
        : [{ left: 50, right: 16, top: 16, bottom: 56 }],
      xAxis: hasVolume
        ? [
            { type: 'category', data: dates, boundaryGap: true, gridIndex: 0 },
            {
              type: 'category',
              data: dates,
              boundaryGap: true,
              gridIndex: 1,
              axisLabel: { show: false },
              axisTick: { show: false },
            },
          ]
        : [{ type: 'category', data: dates, boundaryGap: true }],
      yAxis: hasVolume
        ? [
            { scale: true, gridIndex: 0 },
            { scale: true, gridIndex: 1, splitNumber: 2 },
          ]
        : [{ scale: true }],
      dataZoom: [
        {
          type: 'inside',
          xAxisIndex: hasVolume ? [0, 1] : 0,
          start: 0,
          end: 100,
        },
        {
          type: 'slider',
          xAxisIndex: hasVolume ? [0, 1] : 0,
          bottom: hasVolume ? 0 : 24,
          height: 18,
        },
      ],
      series: [
        {
          name: 'Price',
          type: 'candlestick',
          data: ohlc,
          itemStyle: {
            color: gainColor,
            color0: lossColor,
            borderColor: gainColor,
            borderColor0: lossColor,
          },
          ...(hasVolume ? { xAxisIndex: 0, yAxisIndex: 0 } : {}),
        },
        ...(hasVolume
          ? [
              {
                name: 'Volume',
                type: 'bar' as const,
                data: volumes.map((v, i) => ({
                  value: v,
                  itemStyle: {
                    color: rows[i].close >= rows[i].open ? gainColor : lossColor,
                  },
                })),
                xAxisIndex: 1,
                yAxisIndex: 1,
                barMaxWidth: 8,
              },
            ]
          : []),
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
