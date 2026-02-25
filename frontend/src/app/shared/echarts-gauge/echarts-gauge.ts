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

export interface GaugeThreshold {
  value: number;
  color: string;
}

@Component({
  selector: 'app-echarts-gauge',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<div #container class="w-full" [style.height.px]="height()"></div>`,
  providers: [{ provide: CHART_EXPORTABLE, useExisting: EchartsGaugeComponent }],
})
export class EchartsGaugeComponent implements OnDestroy, ChartExportable {
  value = input(0);
  min = input(0);
  max = input(100);
  thresholds = input<GaugeThreshold[]>([]);
  label = input('');
  height = input(100);

  private readonly container = viewChild.required<ElementRef<HTMLElement>>('container');
  private chart?: EChartsType;
  private ro?: ResizeObserver;

  constructor() {
    afterNextRender(() => this.initChart());
    effect(() => {
      const v = this.value();
      const t = this.thresholds();
      if (this.chart) {
        this.chart.setOption(this.buildOption(v, t));
      }
    });
  }

  private async initChart() {
    const { init, use } = await import('echarts/core');
    const { GaugeChart } = await import('echarts/charts');
    const { TooltipComponent } = await import('echarts/components');
    const { CanvasRenderer } = await import('echarts/renderers');

    use([GaugeChart, TooltipComponent, CanvasRenderer]);

    const el = this.container().nativeElement;
    this.chart = init(el, 'portfolio', { renderer: 'canvas' });
    this.chart.setOption(this.buildOption(this.value(), this.thresholds()));

    this.ro = new ResizeObserver(() => this.chart?.resize());
    this.ro.observe(el);
  }

  private buildOption(value: number, thresholds: GaugeThreshold[]): EChartsCoreOption {
    const minVal = this.min();
    const maxVal = this.max();
    const range = maxVal - minVal;
    const textColor = readCssVar('--color-text');
    const textSecondary = readCssVar('--color-text-secondary');

    const axisLineColors: [number, string][] = thresholds.length > 0
      ? thresholds
          .slice()
          .sort((a, b) => a.value - b.value)
          .map(t => [(t.value - minVal) / range, t.color] as [number, string])
      : [[1, readCssVar('--color-chart-1')]];

    // Ensure last segment reaches 1
    if (axisLineColors.length > 0 && axisLineColors[axisLineColors.length - 1][0] < 1) {
      axisLineColors.push([1, axisLineColors[axisLineColors.length - 1][1]]);
    }

    return {
      series: [
        {
          type: 'gauge',
          min: minVal,
          max: maxVal,
          startAngle: 200,
          endAngle: -20,
          radius: '90%',
          center: ['50%', '60%'],
          pointer: {
            length: '60%',
            width: 4,
            itemStyle: { color: textColor },
          },
          axisLine: {
            lineStyle: {
              width: 12,
              color: axisLineColors,
            },
          },
          axisTick: { show: false },
          splitLine: { show: false },
          axisLabel: {
            distance: 18,
            fontSize: 9,
            color: textSecondary,
          },
          detail: {
            valueAnimation: true,
            formatter: `{value}`,
            fontSize: 16,
            fontWeight: 'bold',
            color: textColor,
            offsetCenter: [0, '30%'],
          },
          title: {
            show: !!this.label(),
            offsetCenter: [0, '55%'],
            fontSize: 10,
            color: textSecondary,
          },
          data: [{ value, name: this.label() }],
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
