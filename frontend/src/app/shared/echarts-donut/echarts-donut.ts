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

export type { PieSegment };

@Component({
  selector: 'app-echarts-donut',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<div #container class="w-full" [style.height.px]="height()"></div>`,
})
export class EchartsDonutComponent implements OnDestroy {
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
    this.chart = init(el, undefined, { renderer: 'canvas' });
    this.chart.setOption(this.buildOption(this.segments()));

    this.ro = new ResizeObserver(() => this.chart?.resize());
    this.ro.observe(el);
  }

  private buildOption(segs: PieSegment[]): EChartsCoreOption {
    return {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'item',
        formatter: '{b}: {d}%',
        backgroundColor: '#ffffff',
        borderColor: '#e4e4e7',
        borderWidth: 1,
        textStyle: { color: '#18181b', fontSize: 12 },
      },
      legend: {
        orient: 'vertical',
        right: 0,
        top: 'middle',
        textStyle: { color: '#71717a', fontSize: 11 },
        itemWidth: 10,
        itemHeight: 10,
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
          radius: ['40%', '70%'],
          center: ['35%', '50%'],
          data: segs.map(s => ({ name: s.label, value: s.value, itemStyle: { color: s.color } })),
          label: { show: false },
          emphasis: {
            itemStyle: { shadowBlur: 6, shadowOffsetX: 0, shadowColor: 'rgba(0,0,0,0.15)' },
          },
        },
      ],
    };
  }

  ngOnDestroy() {
    this.ro?.disconnect();
    this.chart?.dispose();
  }
}
