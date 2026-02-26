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
import { PieSegment } from '../pie-chart/pie-chart';
import { CHART_EXPORTABLE, type ChartExportable } from '../charts/chart-export.token';

export type { PieSegment };

@Component({
  selector: 'app-echarts-donut',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<div #container class="w-full" [style.height.px]="height()"></div>`,
  providers: [{ provide: CHART_EXPORTABLE, useExisting: EchartsDonutComponent }],
})
export class EchartsDonutComponent implements OnDestroy, ChartExportable {
  segments = input<PieSegment[]>([]);
  height = input(280);

  private readonly container = viewChild.required<ElementRef<HTMLElement>>('container');
  private chart?: EChartsType;
  private ro?: ResizeObserver;

  constructor() {
    afterNextRender(() => this.initChart());
    effect(() => {
      const segs = this.segments();
      if (this.chart && segs.length > 0) {
        this.chart.setOption(this.buildOption(segs));
      }
    });
  }

  private async initChart() {
    const { init, use } = await import('echarts/core');
    const { PieChart } = await import('echarts/charts');
    const { TooltipComponent, LegendComponent } = await import('echarts/components');
    const { CanvasRenderer } = await import('echarts/renderers');

    use([PieChart, TooltipComponent, LegendComponent, CanvasRenderer]);

    const el = this.container().nativeElement;
    this.chart = init(el, 'portfolio', { renderer: 'canvas' });
    this.chart.setOption(this.buildOption(this.segments()));

    this.ro = new ResizeObserver(() => {
      this.chart?.resize();
      this.chart?.setOption(this.buildOption(this.segments()));
    });
    this.ro.observe(el);
  }

  private buildOption(segs: PieSegment[]): EChartsCoreOption {
    const containerWidth = this.container().nativeElement.clientWidth;
    const isNarrow = containerWidth < 500;

    return {
      tooltip: {
        trigger: 'item',
        formatter: '{b}: {d}%',
      },
      legend: {
        orient: isNarrow ? 'horizontal' as const : 'vertical' as const,
        ...(isNarrow
          ? { left: 'center', bottom: 0 }
          : { right: 0, top: 'middle' }),
        textStyle: { fontSize: isNarrow ? 11 : 12 },
        formatter: (name: string) => {
          const seg = segs.find(s => s.label === name);
          if (!seg) return name;
          const total = segs.reduce((acc, s) => acc + s.value, 0);
          const pct = total > 0 ? ((seg.value / total) * 100).toFixed(1) : '0';
          return `${name}  ${pct}%`;
        },
      },
      series: [
        {
          type: 'pie',
          radius: isNarrow ? ['30%', '55%'] : ['40%', '70%'],
          center: isNarrow ? ['50%', '40%'] : ['35%', '50%'],
          data: segs.map(s => ({ name: s.label, value: s.value, itemStyle: { color: s.color } })),
          label: { show: isNarrow, position: 'outside', formatter: '{d}%', fontSize: 10 },
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
