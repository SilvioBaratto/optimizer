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
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { ChartToolbarComponent } from '../../shared/chart-toolbar/chart-toolbar';
import { readCssVar } from '../../shared/charts/echarts-theme';
import type { DriftEntry } from '../../models/rebalancing.model';
import type { EChartsType, EChartsCoreOption } from 'echarts/core';

@Component({
  selector: 'app-status-panel',
  imports: [DataTableComponent, ChartToolbarComponent],
  templateUrl: './status-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class StatusPanelComponent implements OnDestroy {
  driftEntries = input<DriftEntry[]>([]);

  private readonly chartContainer = viewChild<ElementRef<HTMLElement>>('driftChart');
  private chart?: EChartsType;
  private ro?: ResizeObserver;

  readonly driftColumns: TableColumn[] = [
    { key: 'ticker', label: 'Ticker', sortable: true },
    { key: 'name', label: 'Name', sortable: true },
    { key: 'sector', label: 'Sector', sortable: true },
    { key: 'targetWeight', label: 'Target', type: 'percentage', sortable: true },
    { key: 'currentWeight', label: 'Current', type: 'percentage', sortable: true },
    { key: 'drift', label: 'Drift', type: 'percentage', sortable: true, colorBySign: true },
    { key: 'driftAbsolute', label: 'Abs Drift', type: 'percentage', sortable: true },
    {
      key: 'status',
      label: 'Status',
      type: 'badge',
      sortable: true,
      badgeMap: {
        true: { value: 'BREACHED', colorClass: 'bg-red-500/15 text-red-400' },
        false: { value: 'OK', colorClass: 'bg-emerald-500/15 text-emerald-400' },
      },
    },
  ];

  driftRows = computed(() =>
    this.driftEntries().map(entry => ({
      ticker: entry.ticker,
      name: entry.name,
      sector: entry.sector,
      targetWeight: entry.targetWeight,
      currentWeight: entry.currentWeight,
      drift: entry.drift,
      driftAbsolute: entry.driftAbsolute,
      status: String(entry.breached),
    })),
  );

  breachedCount = computed(() => this.driftEntries().filter(e => e.breached).length);

  hasBreaches = computed(() => this.breachedCount() > 0);

  constructor() {
    effect(() => {
      const container = this.chartContainer();
      const entries = this.driftEntries();
      if (container && !this.chart && entries.length > 0) {
        void this.initChart();
      } else if (this.chart && entries.length > 0) {
        this.chart.setOption(this.buildGroupedBarOption(entries));
      }
    });
  }

  private async initChart() {
    const container = this.chartContainer();
    if (!container) return;

    const { init, use } = await import('echarts/core');
    const { BarChart } = await import('echarts/charts');
    const { GridComponent, TooltipComponent, LegendComponent } = await import('echarts/components');
    const { CanvasRenderer } = await import('echarts/renderers');

    use([BarChart, GridComponent, TooltipComponent, LegendComponent, CanvasRenderer]);

    const el = container.nativeElement;
    this.chart = init(el, 'portfolio', { renderer: 'canvas' });
    this.chart.setOption(this.buildGroupedBarOption(this.driftEntries()));

    this.ro = new ResizeObserver(() => this.chart?.resize());
    this.ro.observe(el);
  }

  private buildGroupedBarOption(entries: DriftEntry[]): EChartsCoreOption {
    const color1 = readCssVar('--color-chart-1');
    const color2 = readCssVar('--color-chart-2');

    const categories = entries.map(e => e.ticker);
    const currentWeights = entries.map(e => +(e.currentWeight * 100).toFixed(4));
    const targetWeights = entries.map(e => +(e.targetWeight * 100).toFixed(4));

    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: (params: unknown) => {
          const p = params as Array<{ name: string; seriesName: string; value: number }>;
          const ticker = p[0].name;
          return p.map(s => `${s.seriesName}: ${s.value.toFixed(2)}%`).join('<br/>') + `<br/><b>${ticker}</b>`;
        },
      },
      legend: { data: ['Current Weight', 'Target Weight'], top: 0 },
      grid: { left: 50, right: 16, top: 36, bottom: 40 },
      xAxis: {
        type: 'category',
        data: categories,
        axisLabel: { rotate: categories.length > 8 ? 45 : 0 },
      },
      yAxis: {
        type: 'value',
        axisLabel: { formatter: (v: number) => `${v.toFixed(1)}%` },
      },
      series: [
        {
          name: 'Current Weight',
          type: 'bar',
          data: currentWeights,
          itemStyle: { color: color1, borderRadius: [2, 2, 0, 0] },
          barMaxWidth: 20,
        },
        {
          name: 'Target Weight',
          type: 'bar',
          data: targetWeights,
          itemStyle: { color: color2, borderRadius: [2, 2, 0, 0] },
          barMaxWidth: 20,
        },
      ],
    };
  }

  ngOnDestroy() {
    this.ro?.disconnect();
    this.chart?.dispose();
  }
}
