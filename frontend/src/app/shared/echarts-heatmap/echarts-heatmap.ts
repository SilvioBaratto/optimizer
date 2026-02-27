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

@Component({
  selector: 'app-echarts-heatmap',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<div #container class="w-full" [style.height.px]="height()"></div>`,
  providers: [{ provide: CHART_EXPORTABLE, useExisting: EchartsHeatmapComponent }],
})
export class EchartsHeatmapComponent implements OnDestroy, ChartExportable {
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
    const { GridComponent, TooltipComponent, VisualMapComponent, DataZoomComponent } = await import(
      'echarts/components'
    );
    const { CanvasRenderer } = await import('echarts/renderers');

    use([HeatmapChart, GridComponent, TooltipComponent, VisualMapComponent, DataZoomComponent, CanvasRenderer]);

    const el = this.container().nativeElement;
    this.chart = init(el, 'portfolio', { renderer: 'canvas' });
    this.chart.setOption(this.buildOption(this.assets(), this.matrix()));

    this.ro = new ResizeObserver(() => {
      this.chart?.resize();
      this.chart?.setOption(this.buildOption(this.assets(), this.matrix()));
    });
    this.ro.observe(el);
  }

  private buildOption(assets: string[], matrix: number[][]): EChartsCoreOption {
    const data: [number, number, number][] = [];
    for (let i = 0; i < assets.length; i++) {
      for (let j = 0; j < assets.length; j++) {
        data.push([j, i, +(matrix[i]?.[j] ?? 0).toFixed(2)]);
      }
    }

    const containerWidth = this.container().nativeElement.clientWidth;
    const isNarrow = containerWidth < 500;
    const useDataZoom = isNarrow && assets.length > 10;

    const longestLabel = assets.reduce((max, a) => a.length > max.length ? a : max, '');
    const labelPx = longestLabel.length * (isNarrow ? 6 : 8) + 12;
    const gridLeft = Math.max(isNarrow ? 40 : 60, labelPx);
    // Show ~10 columns at a time on narrow screens via dataZoom
    const zoomEnd = useDataZoom ? Math.min(100, (10 / assets.length) * 100) : 100;

    return {
      tooltip: {
        trigger: 'item',
        axisPointer: { type: 'none' },
        confine: true,
        formatter: (params: unknown) => {
          const raw = Array.isArray(params) ? params[0] : params;
          const p = raw as { value: [number, number, number]; data: [number, number, number] };
          const tuple = p.value ?? p.data;
          if (!tuple) return '';
          const col = assets[tuple[0]];
          const row = assets[tuple[1]];
          return `<b>${row} \u00d7 ${col}</b><br/>${tuple[2].toFixed(2)}`;
        },
      },
      grid: {
        left: gridLeft,
        right: isNarrow ? 16 : 80,
        top: 10,
        bottom: useDataZoom ? 80 : (isNarrow ? 40 : 60),
      },
      xAxis: {
        type: 'category',
        data: assets,
        axisLabel: { rotate: 45, fontSize: isNarrow ? 9 : 12 },
        splitArea: { show: true, areaStyle: { color: ['#fafafa', '#ffffff'] } },
      },
      yAxis: {
        type: 'category',
        data: assets,
        axisLabel: { fontSize: isNarrow ? 9 : 12 },
        splitArea: { show: true, areaStyle: { color: ['#fafafa', '#ffffff'] } },
      },
      ...(useDataZoom ? {
        dataZoom: [
          {
            type: 'slider',
            xAxisIndex: 0,
            start: 0,
            end: zoomEnd,
            bottom: 40,
            height: 20,
            borderColor: 'transparent',
          },
          {
            type: 'inside',
            xAxisIndex: 0,
            start: 0,
            end: zoomEnd,
          },
        ],
      } : {}),
      visualMap: {
        min: -1,
        max: 1,
        calculable: true,
        orient: isNarrow ? 'horizontal' as const : 'vertical' as const,
        ...(isNarrow
          ? { left: 'center', bottom: useDataZoom ? 10 : 0, itemWidth: 12, itemHeight: 100 }
          : { right: 0, top: 'middle', itemHeight: 160 }),
        text: ['1', '-1'],
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

  getChartInstance(): EChartsType | undefined {
    return this.chart;
  }

  ngOnDestroy() {
    this.ro?.disconnect();
    this.chart?.dispose();
  }
}
