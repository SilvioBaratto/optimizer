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

export interface SunburstNode {
  name: string;
  value?: number;
  children?: SunburstNode[];
}

@Component({
  selector: 'app-echarts-sunburst',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<div #container class="w-full" [style.height.px]="height()"></div>`,
  providers: [{ provide: CHART_EXPORTABLE, useExisting: EchartsSunburstComponent }],
})
export class EchartsSunburstComponent implements OnDestroy, ChartExportable {
  data = input<SunburstNode[]>([]);
  height = input(340);

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
    const { SunburstChart } = await import('echarts/charts');
    const { TooltipComponent } = await import('echarts/components');
    const { CanvasRenderer } = await import('echarts/renderers');

    use([SunburstChart, TooltipComponent, CanvasRenderer]);

    const el = this.container().nativeElement;
    this.chart = init(el, 'portfolio', { renderer: 'canvas' });
    this.chart.setOption(this.buildOption(this.data()));

    this.ro = new ResizeObserver(() => this.chart?.resize());
    this.ro.observe(el);
  }

  private buildOption(data: SunburstNode[]): EChartsCoreOption {
    const chartColors = Array.from({ length: 8 }, (_, i) =>
      readCssVar(`--color-chart-${i + 1}`)
    );

    const colorize = (nodes: SunburstNode[], depth: number, parentColor?: string): unknown[] =>
      nodes.map((node, i) => {
        const color = depth === 0
          ? chartColors[i % chartColors.length]
          : `${parentColor}99`;
        return {
          name: node.name,
          value: node.value,
          itemStyle: { color },
          children: node.children
            ? colorize(node.children, depth + 1, depth === 0 ? chartColors[i % chartColors.length] : parentColor)
            : undefined,
        };
      });

    return {
      tooltip: {
        trigger: 'item',
        formatter: (params: unknown) => {
          const p = params as { name: string; value: number; treePathInfo: Array<{ name: string; value: number }> };
          const root = p.treePathInfo?.[0];
          const pct = root && root.value > 0 ? ((p.value / root.value) * 100).toFixed(1) : p.value.toFixed(1);
          return `<b>${p.name}</b><br/>${pct}%`;
        },
      },
      series: [
        {
          type: 'sunburst',
          data: colorize(data, 0),
          radius: ['20%', '85%'],
          sort: undefined,
          label: {
            rotate: 0,
            fontSize: 10,
            overflow: 'truncate',
            ellipsis: '..',
          },
          emphasis: {
            focus: 'ancestor',
          },
          levels: [
            {},
            {
              r0: '20%', r: '50%',
              label: { fontSize: 11, fontWeight: 600, padding: 2 },
            },
            {
              r0: '50%', r: '75%',
              label: { fontSize: 9, padding: 1 },
            },
            {
              r0: '75%', r: '85%',
              label: { show: false },
            },
          ],
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
