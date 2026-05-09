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
import { EchartsDrawdownComponent } from '../../shared/echarts-drawdown/echarts-drawdown';
import { ChartToolbarComponent } from '../../shared/chart-toolbar/chart-toolbar';
import { JobProgressTrackerComponent } from '../../shared/job-progress-tracker/job-progress-tracker';
import { FormatService } from '../../services/format.service';
import { BacktestService } from '../../services/backtest.service';
import { PortfolioContextService } from '../../services/portfolio-context.service';
import { readCssVar } from '../../shared/charts/echarts-theme';
import { CHART_EXPORTABLE, type ChartExportable } from '../../shared/charts/chart-export.token';
import { ModalService } from '../../shared/modal/modal.service';
import { ExportReportModalComponent } from '../../shared/modal/export-report-modal';
import { DestroyRef } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { WalkForwardPanelComponent } from './walk-forward-panel';
import { BacktestResultsPanelComponent } from './backtest-results-panel';
import type {
  BacktestConfig,
  BacktestMetrics,
  BacktestResult,
  BacktestRunResponse,
  FactorLoading,
} from '../../models/backtest.model';

// Empty defaults used until a backtest run completes — signals swap them
// for real data fetched via BacktestService.
const DEFAULT_CONFIG: BacktestConfig = {
  startDate: '2021-03-01',
  endDate: '2026-02-25',
  initialCapital: 10_000_000,
  rebalanceFrequency: 'quarterly',
  transactionCostBps: 10,
  benchmark: 'SPY',
};

const EMPTY_RESULT: BacktestResult = {
  equity: [],
  metrics: {
    totalReturn: 0, annualizedReturn: 0, annualizedVol: 0, sharpe: 0,
    sortino: 0, maxDrawdown: 0, calmar: 0, cvar95: 0,
    trackingError: 0, informationRatio: 0, winRate: 0, profitFactor: 0,
  },
  drawdowns: [],
  monthlyReturns: [],
  rollingMetrics: [],
  returnDistribution: [],
  factorLoadings: [],
};

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
    EchartsDrawdownComponent,
    ChartToolbarComponent,
    JobProgressTrackerComponent,
    WalkForwardPanelComponent,
    BacktestResultsPanelComponent,
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
  private readonly backtest = inject(BacktestService);
  private readonly portfolioContext = inject(PortfolioContextService);
  private readonly destroyRef = inject(DestroyRef);

  // ── Loading / error state ──────────────────────────────────────────────────
  readonly isLoading = signal(false);
  readonly hasError = signal(false);
  readonly errorMessage = signal('');

  // ── Run state ─────────────────────────────────────────────────────────────
  readonly runJobId = signal<string | null>(null);
  readonly runRunId = signal<string | null>(null);
  readonly runError = signal<string | null>(null);
  readonly isRunning = computed(() => this.runJobId() !== null);

  // Results panel state (issue #465): populated by `getBacktestRun` after
  // the job completes. `runResponseLoading` / `runResponseError` drive the
  // panel's skeleton and error banner.
  readonly runResponse = signal<BacktestRunResponse | null>(null);
  readonly runResponseLoading = signal(false);
  readonly runResponseError = signal<string | null>(null);

  // ── State ──────────────────────────────────────────────────────────────────
  readonly activeTab = signal('overview');
  readonly rollingWindow = signal<'1Y' | '3Y'>('1Y');
  readonly logScale = signal(false);

  // ── Backtest configuration (user-selectable) ─────────────────────────────
  readonly selectedBenchmark = signal('SPY');
  readonly selectedStartDate = signal('2021-03-01');
  readonly selectedEndDate = signal('2026-02-25');
  readonly tickersRaw = signal('AAPL, MSFT, GOOGL, AMZN, NVDA');

  readonly tickers = computed<string[]>(() =>
    this.tickersRaw()
      .split(',')
      .map((s) => s.trim().toUpperCase())
      .filter((s) => s.length > 0),
  );

  onTickersChange(event: Event): void {
    this.tickersRaw.set((event.target as HTMLInputElement).value);
  }

  readonly benchmarks = [
    { label: 'SPY', value: 'SPY' },
    { label: 'MSCI World (URTH)', value: 'URTH' },
    { label: '60/40 Balanced (VBINX)', value: 'VBINX' },
    { label: 'QQQ', value: 'QQQ' },
    { label: 'IWM', value: 'IWM' },
  ];

  // ── Reactive data (signals replaced when a backtest run completes) ───────
  readonly config = signal<BacktestConfig>(DEFAULT_CONFIG);
  readonly result = signal<BacktestResult>(EMPTY_RESULT);

  readonly hasResult = computed(() => this.result().equity.length > 0);

  // ── Tab definitions (computed for dynamic badge) ─────────────────────────
  readonly drawdownCount = computed(() =>
    [...this.result().drawdowns].sort((a, b) => a.depth - b.depth).slice(0, 10).length
  );

  readonly tabs = computed<Tab[]>(() => {
    const base: Tab[] = [
      { id: 'overview', label: 'Overview' },
      { id: 'metrics', label: 'Metrics' },
      { id: 'monthly', label: 'Monthly Returns' },
      { id: 'drawdowns', label: 'Drawdowns', badge: this.drawdownCount() },
      { id: 'rolling', label: 'Rolling Metrics' },
      { id: 'distribution', label: 'Distribution' },
    ];
    // Style Analysis and Regimes are placeholder tabs (factor regression and
    // regime decomposition are not computed by the backtest backend yet).
    // Surface them only when actual data exists, otherwise hide to keep the
    // tab bar honest.
    if (this.result().factorLoadings.length > 0) {
      base.push({ id: 'style', label: 'Style Analysis' });
    }
    return base;
  });

  // ── Equity curve data ──────────────────────────────────────────────────────
  readonly equityLabels = computed(() =>
    this.result().equity.map(p => p.date)
  );

  readonly portfolioValues = computed(() =>
    this.result().equity.map(p => p.portfolio)
  );

  readonly benchmarkValues = computed(() =>
    this.result().equity.map(p => p.benchmark)
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
    const years = [...new Set(this.result().monthlyReturns.map(c => String(c.year)))];
    return years.sort();
  });

  readonly monthlyHeatmapMonths = computed(() =>
    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
  );

  readonly monthlyHeatmapData = computed(() => {
    const years = this.monthlyHeatmapYears();
    const cells = this.result().monthlyReturns;
    const byKey = new Map<string, number>();
    for (const cell of cells) {
      byKey.set(`${cell.year}-${cell.month}`, cell.value);
    }
    return years.map(yr =>
      Array.from({ length: 12 }, (_, mo) => byKey.get(`${yr}-${mo + 1}`) ?? 0)
    );
  });

  // ── Metrics table ──────────────────────────────────────────────────────────
  // Per-metric benchmark availability: tracking error and information ratio
  // are portfolio-vs-benchmark constructs and have no benchmark counterpart.
  private static readonly BENCHMARK_LESS_METRICS: ReadonlySet<keyof BacktestMetrics> =
    new Set(['trackingError', 'informationRatio']);

  readonly metricsTableRows = computed<MetricsRow[]>(() => {
    const m = this.result().metrics;
    const pct = (v: number) => this.fmt.formatPercent(v);
    const ratio = (v: number) => this.fmt.formatRatio(v);

    return [
      this.metricsRow('Total Return', 'totalReturn', pct, m),
      this.metricsRow('Annualized Return', 'annualizedReturn', pct, m),
      this.metricsRow('Annualized Volatility', 'annualizedVol', pct, m),
      this.metricsRow('Sharpe Ratio', 'sharpe', ratio, m),
      this.metricsRow('Sortino Ratio', 'sortino', ratio, m),
      this.metricsRow('Max Drawdown', 'maxDrawdown', pct, m),
      this.metricsRow('Calmar Ratio', 'calmar', ratio, m),
      this.metricsRow('CVaR 95%', 'cvar95', pct, m),
      this.metricsRow('Tracking Error', 'trackingError', pct, m),
      this.metricsRow('Information Ratio', 'informationRatio', ratio, m),
      this.metricsRow('Win Rate', 'winRate', pct, m),
      this.metricsRow('Profit Factor', 'profitFactor', ratio, m),
    ];
  });

  private metricsRow(
    label: string,
    key: keyof BacktestMetrics,
    formatter: (v: number) => string,
    metrics: BacktestMetrics,
  ): MetricsRow {
    const portfolioRaw = metrics[key];
    const benchmarkRaw = this.benchmarkValue(key);
    return {
      metric: label,
      portfolio: formatter(portfolioRaw),
      benchmark: benchmarkRaw == null ? '—' : formatter(benchmarkRaw),
      portfolioRaw,
      benchmarkRaw,
    };
  }

  private benchmarkValue(key: keyof BacktestMetrics): number | null {
    if (BacktestingComponent.BENCHMARK_LESS_METRICS.has(key)) return null;
    return this.result().benchmarkMetrics?.[key] ?? null;
  }

  // ── KPI strip benchmark helpers (issue #434) ─────────────────────────────
  benchmarkDelta(key: keyof BacktestMetrics): number | null {
    const benchmark = this.result().benchmarkMetrics;
    if (!benchmark) return null;
    return this.result().metrics[key] - benchmark[key];
  }

  benchmarkTrend(key: keyof BacktestMetrics): 'up' | 'down' | 'flat' {
    const delta = this.benchmarkDelta(key);
    if (delta === null || delta === 0) return 'flat';
    return delta > 0 ? 'up' : 'down';
  }

  benchmarkSubtitle(key: keyof BacktestMetrics): string {
    const benchmark = this.result().benchmarkMetrics;
    if (!benchmark) return '';
    return `vs ${this.fmt.formatPercent(benchmark[key])} benchmark`;
  }

  // ── Drawdown table ─────────────────────────────────────────────────────────
  readonly drawdownTableRows = computed(() =>
    [...this.result().drawdowns]
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
  // Drawdown time series wired to EchartsDrawdownComponent (underwater curve)
  readonly drawdownSeries = computed(() => {
    const portfolio = this.portfolioValues();
    const labels = this.equityLabels();
    let peak = portfolio[0] ?? 0;
    return labels.map((date, i) => {
      const v = portfolio[i] ?? peak;
      if (v > peak) peak = v;
      return { date, drawdown: peak > 0 ? (v - peak) / peak : 0 };
    });
  });

  readonly drawdownDepthValues = computed(() =>
    this.result().drawdowns.map(d => d.depth)
  );

  // ── Rolling metrics (uses rollingWindow) ──────────────────────────────────
  private filterByWindow(window: '1Y' | '3Y') {
    const metrics = this.result().rollingMetrics;
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

  // ── Distribution data ──────────────────────────────────────────────────────
  readonly distributionValues = computed(() =>
    this.result().returnDistribution.map(b => (b.binStart + b.binEnd) / 2)
  );

  readonly distributionFullValues = computed(() => {
    const result: number[] = [];
    for (const bin of this.result().returnDistribution) {
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
  readonly factorLoadingBars = computed<BarData[]>(() =>
    toFactorLoadingSeries(this.result().factorLoadings),
  );

  readonly factorTableRows = computed(() =>
    this.result().factorLoadings.map(f => ({
      factor: f.factor,
      loading: f.loading,
      tStat: f.tStat,
      pValue: f.pValue,
      significance: f.pValue < 0.001 ? '***' : f.pValue < 0.01 ? '**' : f.pValue < 0.05 ? '*' : 'ns',
    }))
  );

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

    // Sync local backtest date inputs with the global PortfolioContextService
    // date range — header presets (1Y, 3Y, etc.) propagate to the run window.
    effect(() => {
      const range = this.portfolioContext.dateRange();
      this.selectedStartDate.set(range.start.toISOString().slice(0, 10));
      this.selectedEndDate.set(range.end.toISOString().slice(0, 10));
    });

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

    // Re-render equity chart on log scale toggle OR when result data changes.
    effect(() => {
      const _logScale = this.logScale();
      const _result = this.result();
      if (this.equityChart) {
        this.equityChart.setOption(this.buildEquityOption());
      }
      if (this.underwaterChart) {
        this.underwaterChart.setOption(this.buildUnderwaterOption());
      }
    });

    // Re-render QQ chart on result change.
    effect(() => {
      const _result = this.result();
      if (this.qqChart) {
        this.qqChart.setOption(this.buildQQOption());
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
    this.hasError.set(false);
    this.isLoading.set(false);
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

  // ── Run / progress integration ────────────────────────────────────────────
  onRunBacktest(tickers?: string[]): void {
    if (this.isRunning()) return;
    const list = tickers && tickers.length > 0 ? tickers : this.tickers();
    if (list.length === 0) {
      this.runError.set('Provide at least one ticker.');
      return;
    }
    this.runError.set(null);
    this.backtest
      .runBacktest({
        tickers: list,
        start_date: this.selectedStartDate(),
        end_date: this.selectedEndDate(),
        pipeline_config: { benchmark: this.selectedBenchmark() },
      })
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => {
          this.runJobId.set(res.jobId);
          this.runRunId.set(res.runId);
        },
        error: (err: Error) => this.runError.set(err.message ?? 'Backtest failed'),
      });
  }

  onJobCompleted(): void {
    const runId = this.runRunId();
    this.runJobId.set(null);
    if (!runId) return;
    this.runResponse.set(null);
    this.runResponseError.set(null);
    this.runResponseLoading.set(true);
    this.result.set(EMPTY_RESULT);
    this.backtest
      .getBacktestRun(runId)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (run) => {
          this.runResponse.set(run);
          this.result.set(mapRunResponseToResult(run));
          this.runResponseLoading.set(false);
        },
        error: (err: Error) => {
          this.runResponseLoading.set(false);
          this.runResponseError.set(err?.message ?? 'Failed to load completed run');
        },
      });
  }

  onJobFailed(message: string): void {
    this.runError.set(message || 'Backtest job failed');
    this.runJobId.set(null);
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

// ── Pure transform: factor loadings → sorted horizontal bar data ──────────
export function toFactorLoadingSeries(loadings: readonly FactorLoading[]): BarData[] {
  return [...loadings]
    .sort((a, b) => b.loading - a.loading)
    .map((f) => ({ label: f.factor, value: f.loading }));
}

/**
 * Map a backend `BacktestRunResponse` to the legacy `BacktestResult` shape
 * consumed by the KPI strip and the 8 tabs of the page (issue: BUG-034).
 *
 * Backend conventions:
 *  - `equityCurve`: Record<dateISO, cumulative_return>. Cumulative simple
 *    returns relative to the run start. We expose `(1 + r)` as the
 *    portfolio "value" series (initial capital = 1.0). Benchmark is not
 *    returned by the backend yet; we leave it equal to the portfolio so
 *    the chart renders with a single visible series.
 *  - `drawdowns`: Record<dateISO, drawdown_value>. Time series of
 *    instantaneous drawdown (negative or zero). Detect drawdown events
 *    (start → trough → end) by walking the series.
 *  - `monthlyReturns`: Record<dateISO, value> with date == last calendar
 *    day of the month.
 *  - `rollingMetrics`: Record<metric_name, Record<dateISO, value>>. We
 *    extract the union of dates and pull `sharpe`, `volatility`, `beta`
 *    when present; missing metrics fall back to NaN-safe 0.
 *  - `summaryStats`: free-form snake_case mapping. Annualised keys are
 *    preferred when both raw and annualised exist.
 */
export function mapRunResponseToResult(
  run: { equityCurve: Record<string, number | null>;
         drawdowns: Record<string, number | null>;
         monthlyReturns: Record<string, number | null>;
         rollingMetrics: Record<string, Record<string, number | null>>;
         summaryStats: Record<string, number | null>; } | null,
): BacktestResult {
  if (!run) return EMPTY_RESULT;

  const equity = mapEquityCurve(run.equityCurve);
  const drawdowns = detectDrawdownEvents(run.drawdowns);
  const monthlyReturns = mapMonthlyReturns(run.monthlyReturns);
  const rollingMetrics = mapRollingMetrics(run.rollingMetrics);
  const metrics = mapSummaryToMetrics(
    run.summaryStats as Record<string, unknown>,
    equity,
  );
  const returnDistribution = buildReturnDistribution(run.equityCurve);

  return {
    equity,
    metrics,
    drawdowns,
    monthlyReturns,
    rollingMetrics,
    returnDistribution,
    factorLoadings: [],
  };
}

function mapEquityCurve(
  curve: Record<string, number | null>,
): { date: string; portfolio: number; benchmark: number }[] {
  const isIsoDate = (s: string) => /^\d{4}-\d{2}-\d{2}/.test(s);
  const entries = Object.entries(curve)
    .filter(([k, v]) => isIsoDate(k) && typeof v === 'number' && Number.isFinite(v))
    .sort(([a], [b]) => a.localeCompare(b));
  return entries.map(([date, value]) => {
    const portfolio = 1 + (value as number);
    return { date, portfolio, benchmark: portfolio };
  });
}

function detectDrawdownEvents(
  drawdowns: Record<string, number | null>,
): { start: string; trough: string; end: string | null; depth: number; duration: number; recovery: number | null }[] {
  const isIsoDate = (s: string) => /^\d{4}-\d{2}-\d{2}/.test(s);
  const entries = Object.entries(drawdowns)
    .filter(([k, v]) => isIsoDate(k) && typeof v === 'number' && Number.isFinite(v))
    .sort(([a], [b]) => a.localeCompare(b));
  if (entries.length === 0) return [];

  const events: { start: string; trough: string; end: string | null; depth: number; duration: number; recovery: number | null }[] = [];
  let inDrawdown = false;
  let start = '';
  let trough = '';
  let troughDepth = 0;

  for (const [date, raw] of entries) {
    const v = raw as number;
    if (v < 0 && !inDrawdown) {
      inDrawdown = true;
      start = date;
      trough = date;
      troughDepth = v;
    } else if (v < 0 && inDrawdown) {
      if (v < troughDepth) {
        trough = date;
        troughDepth = v;
      }
    } else if (v >= 0 && inDrawdown) {
      events.push({
        start,
        trough,
        end: date,
        depth: troughDepth,
        duration: daysBetween(start, date),
        recovery: daysBetween(trough, date),
      });
      inDrawdown = false;
    }
  }
  if (inDrawdown) {
    const lastDate = entries[entries.length - 1]?.[0] ?? trough;
    events.push({
      start,
      trough,
      end: null,
      depth: troughDepth,
      duration: daysBetween(start, lastDate),
      recovery: null,
    });
  }
  return events.sort((a, b) => a.depth - b.depth);
}

function daysBetween(a: string, b: string): number {
  const da = new Date(a).getTime();
  const db = new Date(b).getTime();
  return Math.max(0, Math.round((db - da) / 86_400_000));
}

function mapMonthlyReturns(
  monthly: Record<string, number | null>,
): { year: number; month: number; value: number }[] {
  const isIsoDate = (s: string) => /^\d{4}-\d{2}-\d{2}/.test(s);
  return Object.entries(monthly)
    .filter(([k, v]) => isIsoDate(k) && typeof v === 'number' && Number.isFinite(v))
    .map(([date, value]) => {
      const d = new Date(date);
      return { year: d.getUTCFullYear(), month: d.getUTCMonth() + 1, value: value as number };
    });
}

function mapRollingMetrics(
  rolling: Record<string, Record<string, number | null>>,
): { date: string; sharpe: number; volatility: number; beta: number }[] {
  const sharpeMap = pickRollingSeries(rolling, ['sharpe', 'sharpe_ratio', 'rolling_sharpe']);
  const volMap = pickRollingSeries(rolling, ['volatility', 'vol', 'rolling_volatility']);
  const betaMap = pickRollingSeries(rolling, ['beta', 'rolling_beta']);
  const dates = new Set<string>([
    ...Object.keys(sharpeMap),
    ...Object.keys(volMap),
    ...Object.keys(betaMap),
  ]);
  return Array.from(dates)
    .sort()
    .map((date) => ({
      date,
      sharpe: sharpeMap[date] ?? 0,
      volatility: volMap[date] ?? 0,
      beta: betaMap[date] ?? 0,
    }));
}

function pickRollingSeries(
  rolling: Record<string, Record<string, number | null>>,
  candidates: string[],
): Record<string, number> {
  const keys = Object.keys(rolling);
  for (const c of candidates) {
    const found = keys.find((k) => k.toLowerCase().includes(c));
    if (found) {
      const series = rolling[found] ?? {};
      const out: Record<string, number> = {};
      for (const [date, value] of Object.entries(series)) {
        if (typeof value === 'number' && Number.isFinite(value)) out[date] = value;
      }
      return out;
    }
  }
  return {};
}

function mapSummaryToMetrics(
  stats: Record<string, unknown>,
  equity: { portfolio: number }[],
): BacktestMetrics {
  const inSample = (stats['in_sample'] && typeof stats['in_sample'] === 'object'
    ? (stats['in_sample'] as Record<string, unknown>)
    : {}) as Record<string, unknown>;

  const get = (...keys: string[]): number => {
    for (const k of keys) {
      const v = stats[k];
      if (typeof v === 'number' && Number.isFinite(v)) return v;
      const nested = inSample[k];
      if (typeof nested === 'number' && Number.isFinite(nested)) return nested;
    }
    return 0;
  };
  const totalReturnFromEquity =
    equity.length > 0 ? (equity[equity.length - 1].portfolio ?? 1) - 1 : 0;

  return {
    totalReturn: get('total_return', 'Total Return') || totalReturnFromEquity,
    annualizedReturn: get('Annualized Mean', 'annualized_mean', 'annualized_return'),
    annualizedVol: get(
      'Annualized Standard Deviation',
      'annualized_standard_deviation',
      'annualized_volatility',
    ),
    sharpe: get(
      'Annualized Sharpe Ratio',
      'annualized_sharpe_ratio',
      'Sharpe Ratio',
      'sharpe_ratio',
    ),
    sortino: get(
      'Annualized Sortino Ratio',
      'annualized_sortino_ratio',
      'Sortino Ratio',
      'sortino_ratio',
    ),
    maxDrawdown: -Math.abs(get('MAX Drawdown', 'max_drawdown')),
    calmar: get('Calmar Ratio', 'calmar_ratio'),
    cvar95: -Math.abs(get('CVaR at 95%', 'cvar_95', 'cvar')),
    trackingError: get('Tracking Error', 'tracking_error'),
    informationRatio: get('Information Ratio', 'information_ratio'),
    winRate: get('Win Rate', 'win_rate'),
    profitFactor: get('Profit Factor', 'profit_factor'),
  };
}

function buildReturnDistribution(
  curve: Record<string, number | null>,
): { binStart: number; binEnd: number; count: number; frequency: number }[] {
  const sorted = Object.entries(curve)
    .filter(([, v]) => typeof v === 'number' && Number.isFinite(v))
    .sort(([a], [b]) => a.localeCompare(b));
  if (sorted.length < 2) return [];

  const dailyReturns: number[] = [];
  for (let i = 1; i < sorted.length; i++) {
    const prev = (sorted[i - 1][1] as number) + 1;
    const curr = (sorted[i][1] as number) + 1;
    if (prev > 0) dailyReturns.push(curr / prev - 1);
  }
  if (dailyReturns.length === 0) return [];

  const min = Math.min(...dailyReturns);
  const max = Math.max(...dailyReturns);
  const binCount = 30;
  const width = (max - min) / binCount || 1;
  const bins = Array.from({ length: binCount }, (_, i) => ({
    binStart: min + i * width,
    binEnd: min + (i + 1) * width,
    count: 0,
    frequency: 0,
  }));
  for (const r of dailyReturns) {
    const idx = Math.min(binCount - 1, Math.max(0, Math.floor((r - min) / width)));
    bins[idx].count += 1;
  }
  for (const b of bins) b.frequency = b.count / dailyReturns.length;
  return bins;
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
