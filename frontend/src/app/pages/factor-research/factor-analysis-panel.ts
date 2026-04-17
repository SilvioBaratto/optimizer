import {
  Component,
  input,
  output,
  signal,
  computed,
  ElementRef,
  viewChild,
  effect,
  OnDestroy,
  ChangeDetectionStrategy,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { EchartsHeatmapComponent } from '../../shared/echarts-heatmap/echarts-heatmap';
import { ChartToolbarComponent } from '../../shared/chart-toolbar/chart-toolbar';
import { readCssVar } from '../../shared/charts/echarts-theme';
import type {
  FactorICReport,
  FactorQuintileSpreadApiResponse,
  FactorReturnSeries,
  FactorValidateResponse,
} from '../../models/factor.model';
import type { EChartsType, EChartsCoreOption } from 'echarts/core';

const IC_BADGE_MAP: Record<string, { value: string; colorClass: string }> = {
  true: { value: 'YES', colorClass: 'bg-emerald-500/15 text-emerald-400' },
  false: { value: 'NO', colorClass: 'bg-zinc-500/15 text-zinc-400' },
};

function computePairwiseCorrelation(series: FactorReturnSeries[]): number[][] {
  const n = series.length;
  const matrix: number[][] = Array.from({ length: n }, () => new Array(n).fill(0) as number[]);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) {
        matrix[i][j] = 1;
        continue;
      }

      const a = series[i].points.map(p => p.cumReturn);
      const b = series[j].points.map(p => p.cumReturn);
      const len = Math.min(a.length, b.length);

      if (len < 2) {
        matrix[i][j] = 0;
        continue;
      }

      const meanA = a.slice(0, len).reduce((s, v) => s + v, 0) / len;
      const meanB = b.slice(0, len).reduce((s, v) => s + v, 0) / len;

      let cov = 0;
      let varA = 0;
      let varB = 0;

      for (let k = 0; k < len; k++) {
        const da = a[k] - meanA;
        const db = b[k] - meanB;
        cov += da * db;
        varA += da * da;
        varB += db * db;
      }

      const denom = Math.sqrt(varA * varB);
      matrix[i][j] = denom > 0 ? Math.round((cov / denom) * 100) / 100 : 0;
    }
  }

  return matrix;
}

@Component({
  selector: 'app-factor-analysis-panel',
  imports: [FormsModule, DataTableComponent, EchartsHeatmapComponent, ChartToolbarComponent],
  templateUrl: './factor-analysis-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class FactorAnalysisPanelComponent implements OnDestroy {
  factorReturns = input<FactorReturnSeries[]>([]);
  icReports = input<FactorICReport[]>([]);
  validateReport = input<FactorValidateResponse | null>(null);
  quintileSpread = input<FactorQuintileSpreadApiResponse | null>(null);
  factorName = input<string>('momentum_12_1');

  validate = output<string>();
  runQuintileSpread = output<string>();

  readonly formFactor = signal<string>(this.factorName());

  readonly validateRows = computed(() => {
    const report = this.validateReport();
    if (!report) return [];
    return [{
      factor: report.factor_type ?? '—',
      validationType: report.validation_type,
      icMean: report.ic_mean ?? 0,
      icStd: report.ic_std ?? 0,
      icir: report.icir ?? 0,
      tStat: report.t_stat ?? 0,
      pValue: report.p_value ?? 0,
      vif: report.vif ?? 0,
      significant: String((report.p_value ?? 1) < 0.05),
    } as Record<string, unknown>];
  });

  readonly validateColumns: TableColumn[] = [
    { key: 'factor', label: 'Factor' },
    { key: 'validationType', label: 'Validation' },
    { key: 'icMean', label: 'IC mean', type: 'ratio', align: 'right', colorBySign: true },
    { key: 'icStd', label: 'IC std', type: 'ratio', align: 'right' },
    { key: 'icir', label: 'ICIR', type: 'ratio', align: 'right', colorBySign: true },
    { key: 'tStat', label: 't-stat', type: 'ratio', align: 'right', colorBySign: true },
    { key: 'pValue', label: 'p-value', type: 'ratio', align: 'right' },
    { key: 'vif', label: 'VIF', type: 'ratio', align: 'right' },
    { key: 'significant', label: 'Significant', type: 'badge', badgeMap: IC_BADGE_MAP },
  ];

  readonly quintileLabels = computed(() => {
    const data = this.quintileSpread();
    return data ? Object.keys(data.quintile_cumulative_returns) : [];
  });

  readonly spreadSeries = computed(() => {
    const data = this.quintileSpread();
    return data?.spread_cumulative_return ?? [];
  });

  readonly annualizedSpread = computed(() => {
    const data = this.quintileSpread();
    return data ? data.annualized_spread : null;
  });

  submitValidate(): void {
    this.validate.emit(this.formFactor());
  }

  submitQuintile(): void {
    this.runQuintileSpread.emit(this.formFactor());
  }

  private readonly lineChartContainer = viewChild<ElementRef<HTMLElement>>('lineChart');
  private lineChart?: EChartsType;
  private lineRo?: ResizeObserver;

  correlationAssets = computed(() =>
    this.factorReturns().map(fr =>
      fr.factor.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
    ),
  );

  correlationMatrix = computed(() => computePairwiseCorrelation(this.factorReturns()));

  readonly icColumns: TableColumn[] = [
    { key: 'factor', label: 'Factor', type: 'text', sortable: true },
    { key: 'group', label: 'Group', type: 'text', sortable: true },
    { key: 'ic', label: 'IC', type: 'ratio', sortable: true, colorBySign: true },
    { key: 'icir', label: 'ICIR', type: 'ratio', sortable: true, colorBySign: true },
    { key: 'tStat', label: 't-Stat', type: 'ratio', sortable: true, colorBySign: true },
    { key: 'pValue', label: 'p-Value', type: 'ratio', sortable: true },
    { key: 'vif', label: 'VIF', type: 'ratio', sortable: true },
    {
      key: 'significant',
      label: 'Significant',
      type: 'badge',
      badgeMap: IC_BADGE_MAP,
    },
  ];

  icRows = computed(() =>
    this.icReports().map(r => ({
      factor: r.factor.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
      group: r.group.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
      ic: r.ic,
      icir: r.icir,
      tStat: r.tStat,
      pValue: r.pValue,
      vif: r.vif,
      significant: String(r.significant),
    }) as Record<string, unknown>),
  );

  constructor() {
    effect((onCleanup) => {
      const container = this.lineChartContainer();
      const fr = this.factorReturns();
      if (container && !this.lineChart && fr.length > 0) {
        void this.initLineChart();
      } else if (this.lineChart && fr.length > 0) {
        this.lineChart.setOption(this.buildLineOption(fr));
      }
      onCleanup(() => {
        this.lineRo?.disconnect();
        this.lineChart?.dispose();
        this.lineChart = undefined;
        this.lineRo = undefined;
      });
    });
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

  private buildLineOption(factorReturns: FactorReturnSeries[]): EChartsCoreOption {
    const colors = Array.from({ length: 8 }, (_, i) => readCssVar(`--color-chart-${i + 1}`));

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
          const items = params as Array<{ seriesName: string; value: number; color: string; axisValue?: string }>;
          const date = items[0]?.axisValue ?? '';
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
        axisLabel: { fontSize: 10 },
      },
      yAxis: {
        type: 'value',
        axisLabel: { formatter: (v: number) => `${v.toFixed(0)}%` },
      },
      series: seriesData,
    };
  }

  ngOnDestroy() {
    this.lineRo?.disconnect();
    this.lineChart?.dispose();
  }
}
