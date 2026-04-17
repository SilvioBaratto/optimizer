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

export interface RebalancingDiffData {
  asset: string;
  oldWeight: number;
  newWeight: number;
}

@Component({
  selector: 'app-echarts-rebalancing-diff',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<div #container class="w-full" [style.height.px]="height()"></div>`,
  providers: [{ provide: CHART_EXPORTABLE, useExisting: EchartsRebalancingDiffComponent }],
})
export class EchartsRebalancingDiffComponent implements OnDestroy, ChartExportable {
  data = input<RebalancingDiffData[]>([]);
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
    const { BarChart } = await import('echarts/charts');
    const { GridComponent, TooltipComponent, LegendComponent } = await import(
      'echarts/components'
    );
    const { CanvasRenderer } = await import('echarts/renderers');

    use([BarChart, GridComponent, TooltipComponent, LegendComponent, CanvasRenderer]);

    const el = this.container().nativeElement;
    this.chart = init(el, 'portfolio', { renderer: 'canvas' });
    this.chart.setOption(this.buildOption(this.data()));

    this.ro = new ResizeObserver(() => this.chart?.resize());
    this.ro.observe(el);
  }

  private buildOption(rows: RebalancingDiffData[]): EChartsCoreOption {
    const gainColor = readCssVar('--color-gain');
    const lossColor = readCssVar('--color-loss');
    const mutedColor = readCssVar('--color-text-tertiary');

    const assets = rows.map((r) => r.asset);
    const oldValues = rows.map((r) => +(r.oldWeight * 100).toFixed(2));
    const newValues = rows.map((r) => +(r.newWeight * 100).toFixed(2));
    const deltas = rows.map((r) => +((r.newWeight - r.oldWeight) * 100).toFixed(2));

    return {
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      legend: { data: ['Old', 'New'], bottom: 0 },
      grid: { left: 10, right: 10, top: 10, bottom: 40, containLabel: true },
      xAxis: { type: 'category', data: assets, axisLabel: { rotate: 30 } },
      yAxis: {
        type: 'value',
        axisLabel: { formatter: (v: number) => `${v.toFixed(0)}%` },
      },
      series: [
        {
          name: 'Old',
          type: 'bar',
          data: oldValues,
          itemStyle: { color: mutedColor },
          barMaxWidth: 22,
        },
        {
          name: 'New',
          type: 'bar',
          data: newValues.map((v, i) => ({
            value: v,
            itemStyle: { color: deltas[i] >= 0 ? gainColor : lossColor },
          })),
          barMaxWidth: 22,
          label: {
            show: true,
            position: 'top',
            formatter: (p: { dataIndex: number }) => {
              const d = deltas[p.dataIndex];
              return `${d >= 0 ? '+' : ''}${d.toFixed(1)}%`;
            },
            color: mutedColor,
            fontSize: 10,
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
