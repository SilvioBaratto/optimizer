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

export interface PieSegment {
  label: string;
  value: number;
  color: string;
}

const NARROW_BREAKPOINT_PX = 500;
const DEFAULT_LABEL_THRESHOLD = 8;

@Component({
  selector: 'app-echarts-donut',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<div #container class="w-full" [style.height.px]="height()"></div>`,
  providers: [{ provide: CHART_EXPORTABLE, useExisting: EchartsDonutComponent }],
})
export class EchartsDonutComponent implements OnDestroy, ChartExportable {
  segments = input<PieSegment[]>([]);
  height = input(280);
  labelThreshold = input<number>(DEFAULT_LABEL_THRESHOLD);

  private readonly container = viewChild.required<ElementRef<HTMLElement>>('container');
  private chart?: EChartsType;
  private ro?: ResizeObserver;

  constructor() {
    afterNextRender(() => this.initChart());
    effect(() => {
      const segs = this.segments();
      if (this.chart && segs.length > 0) {
        this.chart.setOption(this.buildOptionForContainer(segs));
      }
    });
  }

  getChartInstance(): EChartsType | undefined {
    return this.chart;
  }

  ngOnDestroy() {
    this.ro?.disconnect();
    this.chart?.dispose();
  }

  /**
   * Exposed for unit tests: build the ECharts option for a given container
   * width without depending on DOM layout.
   */
  buildOptionForTest(segs: PieSegment[], containerWidth: number): EChartsCoreOption {
    return this.buildOption(segs, containerWidth);
  }

  private async initChart() {
    const { init, use } = await import('echarts/core');
    const { PieChart } = await import('echarts/charts');
    const { TooltipComponent, LegendComponent } = await import('echarts/components');
    const { CanvasRenderer } = await import('echarts/renderers');

    use([PieChart, TooltipComponent, LegendComponent, CanvasRenderer]);

    const el = this.container().nativeElement;
    this.chart = init(el, 'portfolio', { renderer: 'canvas' });
    this.chart.setOption(this.buildOptionForContainer(this.segments()));

    this.ro = new ResizeObserver(() => {
      this.chart?.resize();
      this.chart?.setOption(this.buildOptionForContainer(this.segments()));
    });
    this.ro.observe(el);
  }

  private buildOptionForContainer(segs: PieSegment[]): EChartsCoreOption {
    const width = this.container().nativeElement.clientWidth;
    return this.buildOption(segs, width);
  }

  private buildOption(segs: PieSegment[], containerWidth: number): EChartsCoreOption {
    const isNarrow = containerWidth < NARROW_BREAKPOINT_PX;
    const overflowing = segs.length > this.labelThreshold();
    const showInlineLabels = isNarrow && !overflowing;

    return {
      tooltip: { trigger: 'item', formatter: '{b}: {d}%' },
      legend: this.buildLegend(segs, isNarrow, overflowing),
      series: [this.buildSeries(segs, isNarrow, showInlineLabels)],
    };
  }

  private buildLegend(
    segs: PieSegment[],
    isNarrow: boolean,
    overflowing: boolean,
  ): Record<string, unknown> {
    const placement = isNarrow
      ? { left: 'center', bottom: 0 }
      : { right: 0, top: 'middle' };
    const legend: Record<string, unknown> = {
      orient: isNarrow ? 'horizontal' : 'vertical',
      ...placement,
      textStyle: { fontSize: isNarrow ? 11 : 12 },
      formatter: (name: string) => this.formatLegendEntry(segs, name),
    };
    if (overflowing) {
      legend['type'] = 'scroll';
    }
    return legend;
  }

  private formatLegendEntry(segs: PieSegment[], name: string): string {
    const seg = segs.find((s) => s.label === name);
    if (!seg) return name;
    const total = segs.reduce((acc, s) => acc + s.value, 0);
    const pct = total > 0 ? ((seg.value / total) * 100).toFixed(1) : '0';
    return `${name}  ${pct}%`;
  }

  private buildSeries(
    segs: PieSegment[],
    isNarrow: boolean,
    showInlineLabels: boolean,
  ): Record<string, unknown> {
    return {
      type: 'pie',
      radius: isNarrow ? ['30%', '55%'] : ['40%', '70%'],
      center: isNarrow ? ['50%', '40%'] : ['35%', '50%'],
      data: segs.map((s) => ({
        name: s.label,
        value: s.value,
        itemStyle: { color: s.color },
      })),
      label: {
        show: showInlineLabels,
        position: 'outside',
        formatter: '{d}%',
        fontSize: 10,
      },
    };
  }
}
