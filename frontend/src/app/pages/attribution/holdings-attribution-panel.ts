import {
  Component,
  input,
  signal,
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
import type { HoldingsAttribution } from '../../models/attribution.model';
import type { EChartsType, EChartsCoreOption } from 'echarts/core';

type ViewMode = 'top10' | 'bottom10' | 'all';

interface ViewModeOption {
  value: ViewMode;
  label: string;
}

@Component({
  selector: 'app-holdings-attribution-panel',
  imports: [DataTableComponent, ChartToolbarComponent],
  templateUrl: './holdings-attribution-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class HoldingsAttributionPanelComponent implements OnDestroy {
  holdings = input.required<HoldingsAttribution[]>();

  viewMode = signal<ViewMode>('top10');
  groupBySector = signal(false);

  private readonly scatterContainer = viewChild<ElementRef<HTMLElement>>('scatterChart');
  private chart?: EChartsType;
  private ro?: ResizeObserver;

  readonly viewModes: ViewModeOption[] = [
    { value: 'top10', label: 'Top 10' },
    { value: 'bottom10', label: 'Bottom 10' },
    { value: 'all', label: 'All' },
  ];

  readonly holdingsTableColumns: TableColumn[] = [
    { key: 'ticker', label: 'Ticker', sortable: true },
    { key: 'name', label: 'Name', sortable: true },
    { key: 'sector', label: 'Sector', sortable: true },
    { key: 'weight', label: 'Weight', type: 'percentage', sortable: true },
    { key: 'returnPct', label: 'Return', type: 'percentage', sortable: true, colorBySign: true },
    { key: 'contribution', label: 'Contribution', type: 'percentage', sortable: true, colorBySign: true },
  ];

  filteredRows = computed(() => {
    const all = this.holdings();
    const mode = this.viewMode();
    const bySector = this.groupBySector();

    let rows: HoldingsAttribution[];
    if (mode === 'top10') {
      rows = all.slice(0, 10);
    } else if (mode === 'bottom10') {
      rows = all.slice(-10);
    } else {
      rows = [...all];
    }

    if (bySector) {
      rows = [...rows].sort((a, b) => {
        const sectorCmp = a.sector.localeCompare(b.sector);
        return sectorCmp !== 0 ? sectorCmp : b.contribution - a.contribution;
      });
    }

    return rows.map(h => ({
      ticker: h.ticker,
      name: h.name,
      sector: h.sector,
      weight: h.weight,
      returnPct: h.returnPct,
      contribution: h.contribution,
    }));
  });

  constructor() {
    effect(() => {
      const container = this.scatterContainer();
      const data = this.holdings();
      if (container && !this.chart && data.length > 0) {
        void this.initScatterChart();
      } else if (this.chart && data.length > 0) {
        this.chart.setOption(this.buildScatterOption(data));
      }
    });
  }

  private async initScatterChart() {
    const container = this.scatterContainer();
    if (!container) return;

    const { init, use } = await import('echarts/core');
    const { ScatterChart } = await import('echarts/charts');
    const { GridComponent, TooltipComponent, LegendComponent } = await import('echarts/components');
    const { CanvasRenderer } = await import('echarts/renderers');

    use([ScatterChart, GridComponent, TooltipComponent, LegendComponent, CanvasRenderer]);

    const el = container.nativeElement;
    this.chart = init(el, 'portfolio', { renderer: 'canvas' });
    this.chart.setOption(this.buildScatterOption(this.holdings()));

    this.ro = new ResizeObserver(() => this.chart?.resize());
    this.ro.observe(el);
  }

  private buildScatterOption(data: HoldingsAttribution[]): EChartsCoreOption {
    const colorGain = readCssVar('--color-gain');
    const colorLoss = readCssVar('--color-loss');

    const positivePoints = data
      .filter(h => h.contribution >= 0)
      .map(h => this.toScatterPoint(h));

    const negativePoints = data
      .filter(h => h.contribution < 0)
      .map(h => this.toScatterPoint(h));

    return {
      tooltip: {
        trigger: 'item',
        formatter: (params: unknown) => {
          const p = params as { data: [number, number, number, string] };
          const [weight, ret, contrib, ticker] = p.data;
          return [
            `<b>${ticker}</b>`,
            `Weight: ${weight.toFixed(2)}%`,
            `Return: ${ret.toFixed(2)}%`,
            `Contribution: ${(contrib * 100).toFixed(3)}%`,
          ].join('<br/>');
        },
      },
      grid: { left: 56, right: 24, top: 16, bottom: 48 },
      xAxis: {
        type: 'value',
        name: 'Weight (%)',
        nameLocation: 'middle',
        nameGap: 32,
        axisLabel: { formatter: (v: number) => `${v.toFixed(1)}%` },
      },
      yAxis: {
        type: 'value',
        name: 'Return (%)',
        nameLocation: 'middle',
        nameGap: 44,
        axisLabel: { formatter: (v: number) => `${v.toFixed(1)}%` },
      },
      legend: {
        data: ['Positive', 'Negative'],
        bottom: 0,
      },
      series: [
        {
          name: 'Positive',
          type: 'scatter',
          data: positivePoints,
          symbolSize: (d: [number, number, number, string]) =>
            Math.min(30, Math.max(4, Math.abs(d[2]) * 10000 * 3)),
          itemStyle: { color: colorGain, opacity: 0.75 },
          label: {
            show: true,
            formatter: (params: unknown) => {
              const p = params as { data: [number, number, number, string] };
              return Math.abs(p.data[2]) > 0.001 ? p.data[3] : '';
            },
            position: 'top',
            fontSize: 10,
          },
        },
        {
          name: 'Negative',
          type: 'scatter',
          data: negativePoints,
          symbolSize: (d: [number, number, number, string]) =>
            Math.min(30, Math.max(4, Math.abs(d[2]) * 10000 * 3)),
          itemStyle: { color: colorLoss, opacity: 0.75 },
          label: {
            show: true,
            formatter: (params: unknown) => {
              const p = params as { data: [number, number, number, string] };
              return Math.abs(p.data[2]) > 0.001 ? p.data[3] : '';
            },
            position: 'top',
            fontSize: 10,
          },
        },
      ],
    };
  }

  private toScatterPoint(h: HoldingsAttribution): [number, number, number, string] {
    return [
      Math.round(h.weight * 10000) / 100,
      Math.round(h.returnPct * 10000) / 100,
      h.contribution,
      h.ticker,
    ];
  }

  ngOnDestroy() {
    this.ro?.disconnect();
    this.chart?.dispose();
  }
}
