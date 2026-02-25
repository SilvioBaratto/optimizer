import {
  Component,
  input,
  computed,
  ElementRef,
  viewChild,
  effect,
  OnDestroy,
  ChangeDetectionStrategy,
} from '@angular/core';
import { EchartsWaterfallComponent } from '../../shared/echarts-waterfall/echarts-waterfall';
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { ChartToolbarComponent } from '../../shared/chart-toolbar/chart-toolbar';
import { readCssVar } from '../../shared/charts/echarts-theme';
import type { MultiLevelAttribution } from '../../models/attribution.model';
import type { EChartsType, EChartsCoreOption } from 'echarts/core';

@Component({
  selector: 'app-saa-taa-panel',
  imports: [EchartsWaterfallComponent, DataTableComponent, ChartToolbarComponent],
  templateUrl: './saa-taa-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class SaaTaaPanelComponent implements OnDestroy {
  multiLevel = input.required<MultiLevelAttribution[]>();

  private readonly groupedBarContainer = viewChild<ElementRef<HTMLElement>>('groupedBarChart');
  private chart?: EChartsType;
  private ro?: ResizeObserver;

  waterfallCategories = computed(() => this.multiLevel().map(m => m.name));

  waterfallValues = computed(() => this.multiLevel().map(m => m.contribution));

  summaryTableColumns: TableColumn[] = [
    { key: 'level', label: 'Level', sortable: true },
    { key: 'name', label: 'Name', sortable: true },
    { key: 'weight', label: 'Weight', type: 'percentage', sortable: true },
    { key: 'returnPct', label: 'Return', type: 'percentage', sortable: true },
    { key: 'contribution', label: 'Contribution', type: 'percentage', sortable: true, colorBySign: true },
  ];

  summaryTableRows = computed(() =>
    this.multiLevel().map(m => ({
      level: m.level,
      name: m.name,
      weight: m.weight,
      returnPct: m.returnPct,
      contribution: m.contribution,
    })),
  );

  constructor() {
    effect(() => {
      const container = this.groupedBarContainer();
      const data = this.multiLevel();
      if (container && !this.chart && data.length > 0) {
        void this.initGroupedBarChart();
      } else if (this.chart && data.length > 0) {
        this.chart.setOption(this.buildGroupedBarOption(data));
      }
    });
  }

  private async initGroupedBarChart() {
    const container = this.groupedBarContainer();
    if (!container) return;

    const { init, use } = await import('echarts/core');
    const { BarChart } = await import('echarts/charts');
    const { GridComponent, TooltipComponent, LegendComponent } = await import('echarts/components');
    const { CanvasRenderer } = await import('echarts/renderers');

    use([BarChart, GridComponent, TooltipComponent, LegendComponent, CanvasRenderer]);

    const el = container.nativeElement;
    this.chart = init(el, 'portfolio', { renderer: 'canvas' });
    this.chart.setOption(this.buildGroupedBarOption(this.multiLevel()));

    this.ro = new ResizeObserver(() => this.chart?.resize());
    this.ro.observe(el);
  }

  private buildGroupedBarOption(data: MultiLevelAttribution[]): EChartsCoreOption {
    const color1 = readCssVar('--color-chart-1');
    const color2 = readCssVar('--color-chart-2');

    const names = data.map(m => m.name);
    const weights = data.map(m => +(m.weight * 100).toFixed(2));
    const returns = data.map(m => +(m.returnPct * 100).toFixed(2));

    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: (params: unknown) => {
          const p = params as Array<{ seriesName: string; name: string; value: number }>;
          return p.map(s => `${s.seriesName}: ${s.value.toFixed(2)}%`).join('<br/>');
        },
      },
      legend: { bottom: 0 },
      grid: { left: 50, right: 16, top: 16, bottom: 40 },
      xAxis: {
        type: 'category',
        data: names,
        axisLabel: { rotate: names.length > 4 ? 30 : 0 },
      },
      yAxis: {
        type: 'value',
        axisLabel: { formatter: (v: number) => `${v.toFixed(1)}%` },
      },
      series: [
        {
          name: 'Weight',
          type: 'bar',
          data: weights,
          itemStyle: {
            color: color1,
            borderRadius: [2, 2, 0, 0] as [number, number, number, number],
          },
          barMaxWidth: 28,
        },
        {
          name: 'Return',
          type: 'bar',
          data: returns,
          itemStyle: {
            color: color2,
            borderRadius: [2, 2, 0, 0] as [number, number, number, number],
          },
          barMaxWidth: 28,
        },
      ],
    };
  }

  ngOnDestroy() {
    this.ro?.disconnect();
    this.chart?.dispose();
  }
}
