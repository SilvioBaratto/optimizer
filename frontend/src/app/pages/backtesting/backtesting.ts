import {
  Component,
  signal,
  computed,
  inject,
  ElementRef,
  viewChild,
  effect,
  OnDestroy,
  ChangeDetectionStrategy,
} from '@angular/core';
import { LucideAngularModule } from 'lucide-angular';
import type { EChartsType, EChartsCoreOption } from 'echarts/core';
import { PageHeaderComponent } from '../../shared/components/page-header/page-header';
import { TabGroupComponent, Tab } from '../../shared/components/tab-group/tab-group';
import { StatCardComponent } from '../../shared/stat-card/stat-card';
import { EchartsCalendarHeatmapComponent } from '../../shared/echarts-calendar-heatmap/echarts-calendar-heatmap';
import { EchartsHistogramComponent } from '../../shared/echarts-histogram/echarts-histogram';
import { EchartsBarComponent, BarData } from '../../shared/echarts-bar/echarts-bar';
import { EchartsStackedAreaComponent, AreaSeries } from '../../shared/echarts-stacked-area/echarts-stacked-area';
import { ChartToolbarComponent } from '../../shared/chart-toolbar/chart-toolbar';
import { FormatService } from '../../services/format.service';
import { readCssVar } from '../../shared/charts/echarts-theme';
import { CHART_EXPORTABLE, type ChartExportable } from '../../shared/charts/chart-export.token';
import { MOCK_BACKTEST_CONFIG, MOCK_BACKTEST_RESULT } from '../../mocks/backtest-mocks';
import { ModalService } from '../../shared/modal/modal.service';
import { ExportReportModalComponent } from '../../shared/modal/export-report-modal';
import { MockFetchService } from '../../services/mock-fetch.service';

interface MetricsRow {
  metric: string;
  portfolio: string;
  benchmark: string;
  portfolioRaw: number;
  benchmarkRaw: number | null;
}

@Component({
  selector: 'app-backtesting',
  imports: [
    LucideAngularModule,
    PageHeaderComponent,
    TabGroupComponent,
    StatCardComponent,
    EchartsCalendarHeatmapComponent,
    EchartsHistogramComponent,
    EchartsBarComponent,
    EchartsStackedAreaComponent,
    ChartToolbarComponent,
  ],
  templateUrl: './backtesting.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
  providers: [
    { provide: CHART_EXPORTABLE, useExisting: BacktestingComponent },
  ],
})
export class BacktestingComponent implements OnDestroy, ChartExportable {
  private readonly fmt = inject(FormatService);
  private readonly modalService = inject(ModalService);
  private readonly mockFetch = inject(MockFetchService);

  // ── Loading / error state ──────────────────────────────────────────────────
  readonly isLoading = signal(true);
  readonly hasError = signal(false);
  readonly errorMessage = signal('');

  // ── State ──────────────────────────────────────────────────────────────────
  readonly activeTab = signal('overview');
  readonly rollingWindow = signal<'1Y' | '3Y'>('1Y');
  readonly styleWindow = signal<'1Y' | '3Y'>('1Y');
  readonly logScale = signal(false);

  // ── Backtest configuration (user-selectable) ─────────────────────────────
  readonly selectedBenchmark = signal('SPY');
  readonly selectedStartDate = signal('2021-03-01');
  readonly selectedEndDate = signal('2026-02-25');

  readonly benchmarks = [
    { label: 'SPY', value: 'SPY' },
    { label: 'MSCI World (URTH)', value: 'URTH' },
    { label: '60/40 Balanced (VBINX)', value: 'VBINX' },
    { label: 'QQQ', value: 'QQQ' },
    { label: 'IWM', value: 'IWM' },
  ];

  // ── Static data ────────────────────────────────────────────────────────────
  readonly config = MOCK_BACKTEST_CONFIG;
  readonly result = MOCK_BACKTEST_RESULT;

  // ── Tab definitions (computed for dynamic badge) ─────────────────────────
  readonly drawdownCount = computed(() =>
    [...this.result.drawdowns].sort((a, b) => a.depth - b.depth).slice(0, 10).length
  );

  readonly tabs = computed<Tab[]>(() => [
    { id: 'overview', label: 'Overview' },
    { id: 'metrics', label: 'Metrics' },
    { id: 'monthly', label: 'Monthly Returns' },
    { id: 'drawdowns', label: 'Drawdowns', badge: this.drawdownCount() },
    { id: 'rolling', label: 'Rolling Metrics' },
    { id: 'distribution', label: 'Distribution' },
    { id: 'style', label: 'Style Analysis' },
  ]);

  // ── Equity curve data ──────────────────────────────────────────────────────
  readonly equityLabels = computed(() =>
    this.result.equity.map(p => p.date)
  );

  readonly portfolioValues = computed(() =>
    this.result.equity.map(p => p.portfolio)
  );

  readonly benchmarkValues = computed(() =>
    this.result.equity.map(p => p.benchmark)
  );

  readonly underwaterValues = computed(() => {
    const portfolio = this.portfolioValues();
    let peak = portfolio[0] ?? 0;
    return portfolio.map(v => {
      if (v > peak) peak = v;
      return peak > 0 ? ((v - peak) / peak) * 100 : 0;
    });
  });

  // ── Monthly heatmap data ───────────────────────────────────────────────────
  readonly monthlyHeatmapYears = computed(() => {
    const years = [...new Set(this.result.monthlyReturns.map(c => String(c.year)))];
    return years.sort();
  });

  readonly monthlyHeatmapMonths = computed(() =>
    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
  );

  readonly monthlyHeatmapData = computed(() => {
    const years = this.monthlyHeatmapYears();
    const cells = this.result.monthlyReturns;
    const byKey = new Map<string, number>();
    for (const cell of cells) {
      byKey.set(`${cell.year}-${cell.month}`, cell.value);
    }
    return years.map(yr =>
      Array.from({ length: 12 }, (_, mo) => byKey.get(`${yr}-${mo + 1}`) ?? 0)
    );
  });

  // ── Metrics table ──────────────────────────────────────────────────────────
  readonly metricsTableRows = computed<MetricsRow[]>(() => {
    const m = this.result.metrics;
    const pct = (v: number) => this.fmt.formatPercent(v);
    const ratio = (v: number) => this.fmt.formatRatio(v);

    return [
      { metric: 'Total Return', portfolio: pct(m.totalReturn), benchmark: pct(0.482), portfolioRaw: m.totalReturn, benchmarkRaw: 0.482 },
      { metric: 'Annualized Return', portfolio: pct(m.annualizedReturn), benchmark: pct(0.081), portfolioRaw: m.annualizedReturn, benchmarkRaw: 0.081 },
      { metric: 'Annualized Volatility', portfolio: pct(m.annualizedVol), benchmark: pct(0.172), portfolioRaw: m.annualizedVol, benchmarkRaw: 0.172 },
      { metric: 'Sharpe Ratio', portfolio: ratio(m.sharpe), benchmark: ratio(0.471), portfolioRaw: m.sharpe, benchmarkRaw: 0.471 },
      { metric: 'Sortino Ratio', portfolio: ratio(m.sortino), benchmark: ratio(0.648), portfolioRaw: m.sortino, benchmarkRaw: 0.648 },
      { metric: 'Max Drawdown', portfolio: pct(m.maxDrawdown), benchmark: pct(-0.247), portfolioRaw: m.maxDrawdown, benchmarkRaw: -0.247 },
      { metric: 'Calmar Ratio', portfolio: ratio(m.calmar), benchmark: ratio(0.328), portfolioRaw: m.calmar, benchmarkRaw: 0.328 },
      { metric: 'CVaR 95%', portfolio: pct(m.cvar95), benchmark: pct(-0.031), portfolioRaw: m.cvar95, benchmarkRaw: -0.031 },
      { metric: 'Tracking Error', portfolio: pct(m.trackingError), benchmark: '—', portfolioRaw: m.trackingError, benchmarkRaw: null },
      { metric: 'Information Ratio', portfolio: ratio(m.informationRatio), benchmark: '—', portfolioRaw: m.informationRatio, benchmarkRaw: null },
      { metric: 'Win Rate', portfolio: pct(m.winRate), benchmark: pct(0.522), portfolioRaw: m.winRate, benchmarkRaw: 0.522 },
      { metric: 'Profit Factor', portfolio: ratio(m.profitFactor), benchmark: ratio(1.18), portfolioRaw: m.profitFactor, benchmarkRaw: 1.18 },
    ];
  });

  // ── Drawdown table ─────────────────────────────────────────────────────────
  readonly drawdownTableRows = computed(() =>
    [...this.result.drawdowns]
      .sort((a, b) => a.depth - b.depth)
      .slice(0, 10)
      .map((d, i) => ({
        rank: i + 1,
        start: d.start,
        trough: d.trough,
        end: d.end ?? 'Ongoing',
        depth: d.depth,
        duration: d.duration,
        recovery: d.recovery ?? '—',
      }))
  );

  // ── Drawdown depth histogram values ───────────────────────────────────────
  readonly drawdownDepthValues = computed(() =>
    this.result.drawdowns.map(d => d.depth)
  );

  // ── Rolling metrics (uses rollingWindow) ──────────────────────────────────
  private filterByWindow(window: '1Y' | '3Y') {
    const metrics = this.result.rollingMetrics;
    const cutoffMonths = window === '1Y' ? 12 : 36;
    const last = metrics[metrics.length - 1];
    if (!last) return [];
    const cutoff = new Date(last.date);
    cutoff.setMonth(cutoff.getMonth() - cutoffMonths);
    return metrics.filter(m => new Date(m.date) >= cutoff);
  }

  readonly rollingLabels = computed(() =>
    this.filterByWindow(this.rollingWindow()).map(m => m.date)
  );

  readonly rollingSharpeValues = computed(() =>
    this.filterByWindow(this.rollingWindow()).map(m => m.sharpe)
  );

  readonly rollingVolValues = computed(() =>
    this.filterByWindow(this.rollingWindow()).map(m => m.volatility)
  );

  readonly rollingBetaValues = computed(() =>
    this.filterByWindow(this.rollingWindow()).map(m => m.beta)
  );

  // ── Style tab rolling exposure (uses independent styleWindow) ─────────────
  readonly styleLabels = computed(() =>
    this.filterByWindow(this.styleWindow()).map(m => m.date)
  );

  // ── Distribution data ──────────────────────────────────────────────────────
  readonly distributionValues = computed(() =>
    this.result.returnDistribution.map(b => (b.binStart + b.binEnd) / 2)
  );

  readonly distributionFullValues = computed(() => {
    const result: number[] = [];
    for (const bin of this.result.returnDistribution) {
      const mid = (bin.binStart + bin.binEnd) / 2;
      for (let i = 0; i < bin.count; i++) {
        result.push(mid);
      }
    }
    return result;
  });

  readonly distributionStats = computed(() => {
    const vals = this.distributionFullValues();
    if (vals.length === 0) return { mean: 0, median: 0, std: 0, skewness: 0, kurtosis: 0, jbStat: 0 };
    const n = vals.length;
    const mean = vals.reduce((a, b) => a + b, 0) / n;
    const sorted = [...vals].sort((a, b) => a - b);
    const median = n % 2 === 0
      ? ((sorted[n / 2 - 1] ?? 0) + (sorted[n / 2] ?? 0)) / 2
      : (sorted[Math.floor(n / 2)] ?? 0);
    const variance = vals.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
    const std = Math.sqrt(variance);
    const skewness = std > 0 ? vals.reduce((a, b) => a + ((b - mean) / std) ** 3, 0) / n : 0;
    const kurtosis = std > 0 ? vals.reduce((a, b) => a + ((b - mean) / std) ** 4, 0) / n - 3 : 0;
    const jbStat = (n / 6) * (skewness ** 2 + (kurtosis ** 2) / 4);
    return { mean, median, std, skewness, kurtosis, jbStat };
  });

  // ── QQ Plot data ───────────────────────────────────────────────────────────
  readonly qqPlotData = computed(() => {
    const vals = this.distributionFullValues();
    if (vals.length === 0) return { points: [] as [number, number][], refLine: [] as [number, number][] };
    const n = vals.length;
    const sorted = [...vals].sort((a, b) => a - b);
    const mean = sorted.reduce((a, b) => a + b, 0) / n;
    const std = Math.sqrt(sorted.reduce((a, b) => a + (b - mean) ** 2, 0) / n) || 1;

    const points: [number, number][] = sorted.map((v, i) => {
      const p = (i + 0.5) / n;
      const theoretical = mean + std * normalQuantile(p);
      return [+(theoretical * 100).toFixed(4), +(v * 100).toFixed(4)];
    });

    const xMin = points[0]?.[0] ?? -3;
    const xMax = points[points.length - 1]?.[0] ?? 3;
    const refLine: [number, number][] = [[xMin, xMin], [xMax, xMax]];
    return { points, refLine };
  });

  // ── Factor loadings bar & table ────────────────────────────────────────────
  readonly factorBarData = computed<BarData[]>(() =>
    this.result.factorLoadings.map(f => ({
      label: f.factor,
      value: f.loading,
    }))
  );

  readonly factorTableRows = computed(() =>
    this.result.factorLoadings.map(f => ({
      factor: f.factor,
      loading: f.loading,
      tStat: f.tStat,
      pValue: f.pValue,
      significance: f.pValue < 0.001 ? '***' : f.pValue < 0.01 ? '**' : f.pValue < 0.05 ? '*' : 'ns',
    }))
  );

  // ── Rolling factor exposure stacked area (uses styleWindow) ──────────────
  readonly rollingExposureSeries = computed<AreaSeries[]>(() => {
    const chartColors = Array.from({ length: 6 }, (_, i) =>
      readCssVar(`--color-chart-${i + 1}`)
    );
    return this.result.factorLoadings.map((f, i) => ({
      name: f.factor,
      values: this.styleLabels().map((_, j) =>
        parseFloat((f.loading + Math.sin(j * 0.05 + i) * 0.03).toFixed(3))
      ),
      color: chartColors[i % chartColors.length],
    }));
  });

  // ── Inline ECharts: Overview tab (always rendered first) ──────────────────
  private readonly equityContainer = viewChild<ElementRef<HTMLElement>>('equityChart');
  private equityChart?: EChartsType;
  private equityRo?: ResizeObserver;

  private readonly underwaterContainer = viewChild<ElementRef<HTMLElement>>('underwaterChart');
  private underwaterChart?: EChartsType;
  private underwaterRo?: ResizeObserver;

  // ── Inline ECharts: Rolling tab (lazy init) ─────────────────────────────
  private readonly rollingSharpeContainer = viewChild<ElementRef<HTMLElement>>('rollingSharpeChart');
  private rollingSharpeChart?: EChartsType;
  private rollingSharpeRo?: ResizeObserver;

  private readonly rollingVolContainer = viewChild<ElementRef<HTMLElement>>('rollingVolChart');
  private rollingVolChart?: EChartsType;
  private rollingVolRo?: ResizeObserver;

  private readonly rollingBetaContainer = viewChild<ElementRef<HTMLElement>>('rollingBetaChart');
  private rollingBetaChart?: EChartsType;
  private rollingBetaRo?: ResizeObserver;

  // ── Inline ECharts: Distribution tab (lazy init) ────────────────────────
  private readonly qqContainer = viewChild<ElementRef<HTMLElement>>('qqChart');
  private qqChart?: EChartsType;
  private qqRo?: ResizeObserver;

  private echartsLoaded = false;

  constructor() {
    this.loadData();

    // Init/dispose overview charts when containers appear/disappear (tab or loading change)
    effect((onCleanup) => {
      const eqEl = this.equityContainer();
      const uwEl = this.underwaterContainer();
      if (eqEl && uwEl && !this.equityChart) {
        void this.initOverviewCharts();
      }
      onCleanup(() => {
        this.equityRo?.disconnect();
        this.equityChart?.dispose();
        this.equityChart = undefined;
        this.equityRo = undefined;
        this.underwaterRo?.disconnect();
        this.underwaterChart?.dispose();
        this.underwaterChart = undefined;
        this.underwaterRo = undefined;
      });
    });

    // Re-render equity chart on log scale toggle
    effect(() => {
      const _logScale = this.logScale();
      if (this.equityChart) {
        this.equityChart.setOption(this.buildEquityOption());
      }
    });

    // Init/dispose rolling charts when their tab becomes active/inactive
    effect((onCleanup) => {
      const sharpeEl = this.rollingSharpeContainer();
      const volEl = this.rollingVolContainer();
      const betaEl = this.rollingBetaContainer();
      if (sharpeEl && volEl && betaEl && !this.rollingSharpeChart) {
        void this.initRollingCharts();
      }
      onCleanup(() => {
        this.rollingSharpeRo?.disconnect();
        this.rollingSharpeChart?.dispose();
        this.rollingSharpeChart = undefined;
        this.rollingSharpeRo = undefined;
        this.rollingVolRo?.disconnect();
        this.rollingVolChart?.dispose();
        this.rollingVolChart = undefined;
        this.rollingVolRo = undefined;
        this.rollingBetaRo?.disconnect();
        this.rollingBetaChart?.dispose();
        this.rollingBetaChart = undefined;
        this.rollingBetaRo = undefined;
      });
    });

    // Init/dispose QQ chart when distribution tab becomes active/inactive
    effect((onCleanup) => {
      const qqEl = this.qqContainer();
      if (qqEl && !this.qqChart) {
        void this.initQQChart();
      }
      onCleanup(() => {
        this.qqRo?.disconnect();
        this.qqChart?.dispose();
        this.qqChart = undefined;
        this.qqRo = undefined;
      });
    });

    // Update rolling charts when window changes
    effect(() => {
      const _window = this.rollingWindow();
      const labels = this.rollingLabels();
      const sharpe = this.rollingSharpeValues();
      const vol = this.rollingVolValues();
      const beta = this.rollingBetaValues();
      if (this.rollingSharpeChart && labels.length > 0) {
        this.rollingSharpeChart.setOption(this.buildRollingOption('Sharpe Ratio', labels, sharpe, '--color-chart-1'));
      }
      if (this.rollingVolChart && labels.length > 0) {
        this.rollingVolChart.setOption(this.buildRollingOption('Volatility', labels, vol, '--color-chart-3', true));
      }
      if (this.rollingBetaChart && labels.length > 0) {
        this.rollingBetaChart.setOption(this.buildRollingOption('Beta', labels, beta, '--color-chart-5'));
      }
    });
  }

  loadData(): void {
    this.isLoading.set(true);
    this.hasError.set(false);

    this.mockFetch.fetch({
      config: MOCK_BACKTEST_CONFIG,
      result: MOCK_BACKTEST_RESULT,
    }).then(() => {
      this.isLoading.set(false);
    }).catch((err: Error) => {
      this.hasError.set(true);
      this.errorMessage.set(err.message);
      this.isLoading.set(false);
    });
  }

  retry(): void {
    this.loadData();
  }

  private async initOverviewCharts() {
    await this.loadEcharts();
    this.initChartInstance(this.equityContainer, c => { this.equityChart = c; this.equityChart.setOption(this.buildEquityOption()); }, r => this.equityRo = r);
    this.initChartInstance(this.underwaterContainer, c => { this.underwaterChart = c; this.underwaterChart.setOption(this.buildUnderwaterOption()); }, r => this.underwaterRo = r);
  }

  private async loadEcharts() {
    if (this.echartsLoaded) return;
    const { use } = await import('echarts/core');
    const { LineChart, ScatterChart } = await import('echarts/charts');
    const { GridComponent, TooltipComponent, LegendComponent, DataZoomComponent, MarkLineComponent } = await import('echarts/components');
    const { CanvasRenderer } = await import('echarts/renderers');
    use([LineChart, ScatterChart, GridComponent, TooltipComponent, LegendComponent, DataZoomComponent, MarkLineComponent, CanvasRenderer]);
    this.echartsLoaded = true;
  }

  private async initChartInstance(
    containerSignal: ReturnType<typeof viewChild<ElementRef<HTMLElement>>>,
    setup: (chart: EChartsType) => void,
    setRo: (ro: ResizeObserver) => void,
  ) {
    const ref = containerSignal();
    if (!ref) return;
    const { init } = await import('echarts/core');
    const el = ref.nativeElement;
    const chart = init(el, 'portfolio', { renderer: 'canvas' });
    setup(chart);
    const ro = new ResizeObserver(() => chart.resize());
    ro.observe(el);
    setRo(ro);
  }

  // ── Chart initializers (lazy, safe for @if blocks) ────────────────────────
  private async initRollingCharts() {
    await this.loadEcharts();
    const labels = this.rollingLabels();
    await this.initChartInstance(this.rollingSharpeContainer, c => { this.rollingSharpeChart = c; c.setOption(this.buildRollingOption('Sharpe Ratio', labels, this.rollingSharpeValues(), '--color-chart-1')); }, r => this.rollingSharpeRo = r);
    await this.initChartInstance(this.rollingVolContainer, c => { this.rollingVolChart = c; c.setOption(this.buildRollingOption('Volatility', labels, this.rollingVolValues(), '--color-chart-3', true)); }, r => this.rollingVolRo = r);
    await this.initChartInstance(this.rollingBetaContainer, c => { this.rollingBetaChart = c; c.setOption(this.buildRollingOption('Beta', labels, this.rollingBetaValues(), '--color-chart-5')); }, r => this.rollingBetaRo = r);

    // Sync hover across all three rolling charts
    if (this.rollingSharpeChart && this.rollingVolChart && this.rollingBetaChart) {
      const { connect } = await import('echarts/core');
      connect([this.rollingSharpeChart, this.rollingVolChart, this.rollingBetaChart]);
    }
  }

  private async initQQChart() {
    await this.loadEcharts();
    this.initChartInstance(this.qqContainer, c => { this.qqChart = c; c.setOption(this.buildQQOption()); }, r => this.qqRo = r);
  }

  // ── Chart option builders ──────────────────────────────────────────────────
  private buildEquityOption(): EChartsCoreOption {
    const labels = this.equityLabels();
    const portfolio = this.portfolioValues();
    const benchmark = this.benchmarkValues();
    const chart1 = readCssVar('--color-chart-1');
    const chart7 = readCssVar('--color-chart-7');
    const isLog = this.logScale();

    return {
      tooltip: {
        trigger: 'axis',
        formatter: (params: unknown) => {
          const ps = params as Array<{ seriesName: string; value: number; axisValueLabel: string; color: string }>;
          if (!ps.length) return '';
          let html = `<div style="font-size:12px"><b>${ps[0].axisValueLabel}</b>`;
          for (const p of ps) {
            html += `<br/><span style="color:${p.color}">&#9679;</span> ${p.seriesName}: ${p.value.toFixed(2)}`;
          }
          return html + '</div>';
        },
      },
      legend: {
        data: ['Portfolio', 'Benchmark (SPY)'],
        top: 0,
        right: 0,
      },
      grid: {
        left: window.innerWidth < 640 ? 40 : 55,
        right: 16,
        top: 28,
        bottom: window.innerWidth < 640 ? 50 : 36,
      },
      xAxis: {
        type: 'category',
        data: labels,
        axisLabel: {
          formatter: (v: string) => v.slice(0, 7),
          interval: Math.floor(labels.length / (window.innerWidth < 640 ? 3 : 6)),
          rotate: window.innerWidth < 640 ? 30 : 0,
          fontSize: window.innerWidth < 640 ? 10 : 12,
        },
      },
      yAxis: {
        type: isLog ? 'log' : 'value',
        axisLabel: { formatter: (v: number) => v.toFixed(0) },
      },
      dataZoom: [
        { type: 'inside', start: 0, end: 100 },
        { type: 'slider', start: 0, end: 100, height: 18, bottom: 2 },
      ],
      series: [
        {
          name: 'Portfolio',
          type: 'line',
          data: portfolio,
          symbol: 'none',
          lineStyle: { width: 2, color: chart1 },
          itemStyle: { color: chart1 },
          areaStyle: { color: `${chart1}15` },
        },
        {
          name: 'Benchmark (SPY)',
          type: 'line',
          data: benchmark,
          symbol: 'none',
          lineStyle: { width: 1.5, color: chart7, type: 'dashed' },
          itemStyle: { color: chart7 },
        },
      ],
    };
  }

  private buildUnderwaterOption(): EChartsCoreOption {
    const labels = this.equityLabels();
    const underwater = this.underwaterValues();
    const lossColor = readCssVar('--color-loss');

    return {
      tooltip: {
        trigger: 'axis',
        formatter: (params: unknown) => {
          const ps = params as Array<{ value: number; axisValueLabel: string }>;
          if (!ps.length) return '';
          return `${ps[0].axisValueLabel}<br/>${(ps[0].value ?? 0).toFixed(2)}%`;
        },
      },
      grid: {
        left: window.innerWidth < 640 ? 40 : 55,
        right: 16,
        top: 10,
        bottom: window.innerWidth < 640 ? 44 : 30,
      },
      xAxis: {
        type: 'category',
        data: labels,
        axisLabel: {
          formatter: (v: string) => v.slice(0, 7),
          interval: Math.floor(labels.length / (window.innerWidth < 640 ? 3 : 6)),
          rotate: window.innerWidth < 640 ? 30 : 0,
          fontSize: window.innerWidth < 640 ? 10 : 12,
        },
      },
      yAxis: {
        type: 'value',
        axisLabel: { formatter: (v: number) => `${v.toFixed(0)}%` },
        max: 0,
      },
      series: [
        {
          name: 'Drawdown',
          type: 'line',
          data: underwater,
          symbol: 'none',
          lineStyle: { width: 1, color: lossColor },
          itemStyle: { color: lossColor },
          areaStyle: { color: `${lossColor}4d` },
        },
      ],
    };
  }

  private buildRollingOption(
    name: string,
    labels: string[],
    values: number[],
    colorVar: string,
    isPercent = false,
  ): EChartsCoreOption {
    const color = readCssVar(colorVar);
    const borderColor = readCssVar('--color-border');

    return {
      tooltip: {
        trigger: 'axis',
        formatter: (params: unknown) => {
          const ps = params as Array<{ value: number; axisValueLabel: string }>;
          if (!ps.length) return '';
          const v = ps[0].value ?? 0;
          const fmt = isPercent ? `${(v * 100).toFixed(1)}%` : v.toFixed(3);
          return `${ps[0].axisValueLabel}<br/>${name}: ${fmt}`;
        },
      },
      grid: {
        left: window.innerWidth < 640 ? 40 : 50,
        right: 16,
        top: 10,
        bottom: window.innerWidth < 640 ? 44 : 30,
      },
      xAxis: {
        type: 'category',
        data: labels,
        axisLabel: {
          formatter: (v: string) => v.slice(0, 7),
          interval: Math.floor(labels.length / (window.innerWidth < 640 ? 3 : 6)),
          rotate: window.innerWidth < 640 ? 30 : 0,
          fontSize: window.innerWidth < 640 ? 10 : 12,
        },
      },
      yAxis: {
        type: 'value',
        axisLabel: {
          formatter: isPercent
            ? (v: number) => `${(v * 100).toFixed(0)}%`
            : (v: number) => v.toFixed(2),
        },
      },
      series: [
        {
          name,
          type: 'line',
          data: values,
          symbol: 'none',
          lineStyle: { width: 1.5, color },
          itemStyle: { color },
          areaStyle: { color: `${color}20` },
          markLine: {
            silent: true,
            symbol: 'none',
            data: [{ yAxis: 0 }],
            lineStyle: { color: borderColor, width: 1, type: 'dashed' },
            label: { show: false },
          },
        },
      ],
    };
  }

  private buildQQOption(): EChartsCoreOption {
    const { points, refLine } = this.qqPlotData();
    const chart1 = readCssVar('--color-chart-1');
    const chart7 = readCssVar('--color-chart-7');
    const textSecondary = readCssVar('--color-text-secondary');

    return {
      tooltip: {
        trigger: 'item',
        formatter: (params: unknown) => {
          const p = params as { value: [number, number]; seriesName: string };
          if (!Array.isArray(p.value)) return '';
          return `Theoretical: ${p.value[0].toFixed(3)}%<br/>Sample: ${p.value[1].toFixed(3)}%`;
        },
      },
      legend: { show: false },
      grid: { left: 16, right: 16, top: 20, bottom: 36, containLabel: true },
      xAxis: {
        type: 'value',
        name: 'Theoretical Quantile (%)',
        nameLocation: 'middle',
        nameGap: 28,
        nameTextStyle: { fontSize: 11, color: textSecondary },
        axisLabel: { fontSize: 11, formatter: (v: number) => `${v.toFixed(1)}%` },
      },
      yAxis: {
        type: 'value',
        name: 'Sample Quantile (%)',
        nameLocation: 'middle',
        nameGap: 45,
        nameTextStyle: { fontSize: 11, color: textSecondary },
        axisLabel: { fontSize: 11, formatter: (v: number) => `${v.toFixed(1)}%` },
      },
      series: [
        {
          name: 'Returns',
          type: 'scatter',
          data: points,
          symbolSize: 5,
          itemStyle: { color: chart1, opacity: 0.7 },
        },
        {
          name: '45° Reference',
          type: 'line',
          data: refLine,
          symbol: 'none',
          lineStyle: { color: chart7, width: 1.5, type: 'dashed' },
          itemStyle: { color: chart7 },
        },
      ],
    };
  }

  // ── ChartExportable (delegates to equity chart) ────────────────────────────
  getChartInstance(): EChartsType | undefined {
    return this.equityChart;
  }

  onBenchmarkChange(event: Event): void {
    this.selectedBenchmark.set((event.target as HTMLSelectElement).value);
  }

  onStartDateChange(event: Event): void {
    this.selectedStartDate.set((event.target as HTMLInputElement).value);
  }

  onEndDateChange(event: Event): void {
    this.selectedEndDate.set((event.target as HTMLInputElement).value);
  }

  openReportModal(): void {
    this.modalService.open({ component: ExportReportModalComponent, title: 'Export Report', size: 'lg' });
  }

  // ── Tab helpers ────────────────────────────────────────────────────────────
  onTabChange(id: string): void {
    this.activeTab.set(id);
  }

  setRollingWindow(w: '1Y' | '3Y'): void {
    this.rollingWindow.set(w);
  }

  setStyleWindow(w: '1Y' | '3Y'): void {
    this.styleWindow.set(w);
  }

  toggleLogScale(): void {
    this.logScale.update(v => !v);
  }

  // ── Formatting helpers ─────────────────────────────────────────────────────
  formatPct(v: number): string {
    return this.fmt.formatPercent(v);
  }

  formatRatio(v: number, decimals = 2): string {
    return this.fmt.formatRatio(v, decimals);
  }

  formatCurrency(v: number): string {
    return this.fmt.formatCurrency(v);
  }

  signClass(v: number): string {
    if (v > 0) return 'text-gain';
    if (v < 0) return 'text-loss';
    return 'text-flat';
  }

  sigBadgeClass(sig: string): string {
    if (sig === '***') return 'bg-accent/10 text-text font-bold';
    if (sig === '**') return 'bg-accent/10 text-text-secondary';
    if (sig === '*') return 'bg-surface-inset text-text-tertiary';
    return 'bg-surface-inset text-text-tertiary opacity-50';
  }

  // ── Lifecycle ──────────────────────────────────────────────────────────────
  ngOnDestroy() {
    this.equityRo?.disconnect();
    this.equityChart?.dispose();
    this.underwaterRo?.disconnect();
    this.underwaterChart?.dispose();
    this.rollingSharpeRo?.disconnect();
    this.rollingSharpeChart?.dispose();
    this.rollingVolRo?.disconnect();
    this.rollingVolChart?.dispose();
    this.rollingBetaRo?.disconnect();
    this.rollingBetaChart?.dispose();
    this.qqRo?.disconnect();
    this.qqChart?.dispose();
  }
}

// ── Utility: inverse normal CDF (Beasley-Springer-Moro approximation) ──────
function normalQuantile(p: number): number {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  const a = [2.515517, 0.802853, 0.010328];
  const b = [1.432788, 0.189269, 0.001308];
  const t = p < 0.5 ? Math.sqrt(-2 * Math.log(p)) : Math.sqrt(-2 * Math.log(1 - p));
  const num = a[0] + t * (a[1] + t * a[2]);
  const den = 1 + t * (b[0] + t * (b[1] + t * b[2]));
  const result = t - num / den;
  return p < 0.5 ? -result : result;
}
