import {
  Component,
  input,
  computed,
  ElementRef,
  viewChild,
  afterNextRender,
  effect,
  OnDestroy,
  ChangeDetectionStrategy,
} from '@angular/core';
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { readCssVar } from '../../shared/charts/echarts-theme';
import type { TAASignal, FactorReturnSeries, MacroRegime } from '../../models/factor.model';
import type { EChartsType, EChartsCoreOption } from 'echarts/core';

const REGIME_BADGE_MAP: Record<MacroRegime, { value: string; colorClass: string }> = {
  expansion: { value: 'Expansion', colorClass: 'bg-gain/15 text-gain' },
  slowdown: { value: 'Slowdown', colorClass: 'bg-chart-4/15 text-chart-4' },
  recession: { value: 'Recession', colorClass: 'bg-loss/15 text-loss' },
  recovery: { value: 'Recovery', colorClass: 'bg-chart-1/15 text-chart-1' },
};

@Component({
  selector: 'app-taa-panel',
  imports: [DataTableComponent],
  templateUrl: './taa-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class TaaPanelComponent implements OnDestroy {
  signals = input<TAASignal[]>([]);
  factorReturns = input<FactorReturnSeries[]>([]);

  private readonly barChartContainer = viewChild<ElementRef<HTMLElement>>('barChart');
  private readonly lineChartContainer = viewChild<ElementRef<HTMLElement>>('lineChart');
  private barChart?: EChartsType;
  private lineChart?: EChartsType;
  private barRo?: ResizeObserver;
  private lineRo?: ResizeObserver;

  readonly signalColumns: TableColumn[] = [
    { key: 'factor', label: 'Factor', type: 'text', sortable: true },
    { key: 'currentWeight', label: 'Current Weight', type: 'percentage', sortable: true },
    { key: 'tiltedWeight', label: 'Tilted Weight', type: 'percentage', sortable: true },
    { key: 'deltaBps', label: 'Delta (bps)', type: 'bps', colorBySign: true, sortable: true },
    { key: 'tiltReason', label: 'Reason', type: 'text' },
    {
      key: 'regime',
      label: 'Regime',
      type: 'badge',
      badgeMap: REGIME_BADGE_MAP as Record<string, { value: string; colorClass: string }>,
    },
  ];

  signalRows = computed(() =>
    this.signals().map(s => ({
      factor: s.factor.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase()),
      currentWeight: s.currentWeight,
      tiltedWeight: s.tiltedWeight,
      deltaBps: (s.tiltedWeight - s.currentWeight) * 10000,
      tiltReason: s.tiltReason,
      regime: s.regime,
    }) as Record<string, unknown>),
  );

  constructor() {
    afterNextRender(() => {
      void this.initBarChart();
      void this.initLineChart();
    });

    effect(() => {
      const s = this.signals();
      if (this.barChart && s.length > 0) {
        this.barChart.setOption(this.buildBarOption(s));
      }
    });

    effect(() => {
      const fr = this.factorReturns();
      if (this.lineChart && fr.length > 0) {
        this.lineChart.setOption(this.buildLineOption(fr));
      }
    });
  }

  private async initBarChart() {
    const container = this.barChartContainer();
    if (!container) return;

    const { init, use } = await import('echarts/core');
    const { BarChart } = await import('echarts/charts');
    const { GridComponent, TooltipComponent, LegendComponent } = await import('echarts/components');
    const { CanvasRenderer } = await import('echarts/renderers');

    use([BarChart, GridComponent, TooltipComponent, LegendComponent, CanvasRenderer]);

    const el = container.nativeElement;
    this.barChart = init(el, 'portfolio', { renderer: 'canvas' });
    this.barChart.setOption(this.buildBarOption(this.signals()));

    this.barRo = new ResizeObserver(() => this.barChart?.resize());
    this.barRo.observe(el);
  }

  private async initLineChart() {
    const container = this.lineChartContainer();
    if (!container) return;

    const { init, use } = await import('echarts/core');
    const { LineChart } = await import('echarts/charts');
    const { GridComponent, TooltipComponent, LegendComponent } = await import('echarts/components');
    const { CanvasRenderer } = await import('echarts/renderers');

    use([LineChart, GridComponent, TooltipComponent, LegendComponent, CanvasRenderer]);

    const el = container.nativeElement;
    this.lineChart = init(el, 'portfolio', { renderer: 'canvas' });
    this.lineChart.setOption(this.buildLineOption(this.factorReturns()));

    this.lineRo = new ResizeObserver(() => this.lineChart?.resize());
    this.lineRo.observe(el);
  }

  private buildBarOption(signals: TAASignal[]): EChartsCoreOption {
    const color1 = readCssVar('--color-chart-1');
    const color2 = readCssVar('--color-chart-3');

    const labels = signals.map(s =>
      s.factor.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase()),
    );

    return {
      tooltip: { trigger: 'axis' },
      legend: { bottom: 0, data: ['Current', 'Tilted'] },
      grid: { left: 60, right: 16, top: 16, bottom: 48 },
      xAxis: {
        type: 'category',
        data: labels,
        axisLabel: { rotate: 20, fontSize: 10 },
      },
      yAxis: {
        type: 'value',
        axisLabel: { formatter: (v: number) => `${(v * 100).toFixed(0)}%` },
      },
      series: [
        {
          name: 'Current',
          type: 'bar',
          data: signals.map(s => +(s.currentWeight * 100).toFixed(2)),
          itemStyle: { color: color1 },
          barMaxWidth: 28,
        },
        {
          name: 'Tilted',
          type: 'bar',
          data: signals.map(s => +(s.tiltedWeight * 100).toFixed(2)),
          itemStyle: { color: color2 },
          barMaxWidth: 28,
        },
      ],
    };
  }

  private buildLineOption(factorReturns: FactorReturnSeries[]): EChartsCoreOption {
    const colors = Array.from({ length: 8 }, (_, i) => readCssVar(`--color-chart-${i + 1}`));

    // Sample every 5th point for performance
    const seriesData = factorReturns.map((fr, i) => {
      const sampled = fr.points.filter((_, idx) => idx % 5 === 0);
      return {
        name: fr.factor.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
        type: 'line' as const,
        data: sampled.map(p => +(p.cumReturn * 100).toFixed(3)),
        symbol: 'none',
        lineStyle: { width: 1.5, color: colors[i % colors.length] },
        itemStyle: { color: colors[i % colors.length] },
      };
    });

    const labels =
      factorReturns[0]?.points
        .filter((_, idx) => idx % 5 === 0)
        .map(p => p.date) ?? [];

    return {
      tooltip: {
        trigger: 'axis',
        formatter: (params: unknown) => {
          const items = params as Array<{ seriesName: string; value: number; color: string }>;
          const date = (items[0] as { axisValue?: string })?.axisValue ?? '';
          const lines = items
            .map(p => `<span style="color:${p.color}">&#9679;</span> ${p.seriesName}: ${p.value.toFixed(2)}%`)
            .join('<br/>');
          return `${date}<br/>${lines}`;
        },
      },
      legend: { bottom: 0, type: 'scroll' },
      grid: { left: 50, right: 16, top: 16, bottom: 56 },
      xAxis: {
        type: 'category',
        data: labels,
        axisLabel: { rotate: 0, fontSize: 10 },
      },
      yAxis: {
        type: 'value',
        axisLabel: { formatter: (v: number) => `${v.toFixed(0)}%` },
      },
      series: seriesData,
    };
  }

  ngOnDestroy() {
    this.barRo?.disconnect();
    this.lineRo?.disconnect();
    this.barChart?.dispose();
    this.lineChart?.dispose();
  }
}
