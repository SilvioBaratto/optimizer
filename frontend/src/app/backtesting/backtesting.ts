import {
  Component,
  signal,
  computed,
  inject,
  ChangeDetectionStrategy,
} from '@angular/core';
import { LucideAngularModule } from 'lucide-angular';
import { PageHeaderComponent } from '../shared/components/page-header/page-header';
import { TabGroupComponent, Tab } from '../shared/components/tab-group/tab-group';
import { StatCardComponent } from '../shared/stat-card/stat-card';
import { EchartsCalendarHeatmapComponent } from '../shared/echarts-calendar-heatmap/echarts-calendar-heatmap';
import { EchartsHistogramComponent } from '../shared/echarts-histogram/echarts-histogram';
import { EchartsBarComponent, BarData } from '../shared/echarts-bar/echarts-bar';
import { EchartsDrawdownComponent } from '../shared/echarts-drawdown/echarts-drawdown';
import { ChartToolbarComponent } from '../shared/chart-toolbar/chart-toolbar';
import { JobProgressTrackerComponent } from '../shared/job-progress-tracker/job-progress-tracker';
import { FormatService } from '../core/services/format.service';
import { BacktestService } from './backtest.service';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { TickerSeedingService } from '../core/services/ticker-seeding.service';
import { BacktestWeightResolverService } from './backtest-weight-resolver';
import { CHART_EXPORTABLE, type ChartExportable } from '../shared/charts/chart-export.token';
import { ModalService } from '../shared/modal/modal.service';
import { ExportReportModalComponent } from '../shared/modal/export-report-modal';
import { DestroyRef } from '@angular/core';
import { takeUntilDestroyed, toObservable } from '@angular/core/rxjs-interop';
import { EMPTY, catchError, switchMap, take } from 'rxjs';
import { WalkForwardPanelComponent } from './walk-forward-panel/walk-forward-panel';
import { BacktestResultsPanelComponent } from './backtest-results-panel/backtest-results-panel';
import { BacktestingSetupFormComponent, type BacktestRunConfig } from './backtesting-setup-form/backtesting-setup-form';
import {
  clearBacktestRun,
  loadBacktestRun,
  saveBacktestRun,
} from './backtesting-run-storage';
import type {
  BacktestConfig,
  BacktestKpiResult,
  BacktestMetrics,
  BacktestResult,
  BacktestResultEnvelope,
  BacktestRunResponse,
  FactorLoading,
} from './backtest.model';

export const DEFAULT_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'] as const;

function periodToIsoRange(period: string): { start: string; end: string } {
  const end = new Date();
  const start = new Date();
  switch (period) {
    case '1M': start.setMonth(start.getMonth() - 1); break;
    case '3M': start.setMonth(start.getMonth() - 3); break;
    case '6M': start.setMonth(start.getMonth() - 6); break;
    case 'YTD': start.setMonth(0, 1); break;
    case '3Y': start.setFullYear(start.getFullYear() - 3); break;
    case '5Y': start.setFullYear(start.getFullYear() - 5); break;
    case 'Max': start.setFullYear(2000, 0, 1); break;
    default: start.setFullYear(start.getFullYear() - 1);
  }
  return { start: start.toISOString().slice(0, 10), end: end.toISOString().slice(0, 10) };
}

/** Order-insensitive equality check for two string arrays. */
function arraysEqualUnordered(a: string[], b: string[]): boolean {
  if (a.length !== b.length) return false;
  const aSorted = [...a].sort();
  const bSorted = [...b].sort();
  return aSorted.every((v, i) => v === bSorted[i]);
}

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
  benchmarkTestId: string;
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
    BacktestingSetupFormComponent,
  ],
  templateUrl: './backtesting.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
  providers: [
    { provide: CHART_EXPORTABLE, useExisting: BacktestingComponent },
  ],
})
export class BacktestingComponent implements ChartExportable {
  private readonly fmt = inject(FormatService);
  private readonly modalService = inject(ModalService);
  private readonly backtest = inject(BacktestService);
  private readonly portfolioContext = inject(PortfolioContextService);
  private readonly tickerSeeding = inject(TickerSeedingService);
  private readonly weightResolver = inject(BacktestWeightResolverService);
  private readonly destroyRef = inject(DestroyRef);

  /** Tracks the last array written by the seeding pipeline for the re-seed guard. */
  private readonly lastSeed = signal<string[] | null>(null);

  // ── Loading / error state ──────────────────────────────────────────────────
  readonly isLoading = signal(false);
  readonly hasError = signal(false);
  readonly errorMessage = signal('');

  // ── Run state ─────────────────────────────────────────────────────────────
  readonly runJobId = signal<string | null>(null);
  readonly runRunId = signal<string | null>(null);
  readonly runError = signal<string | null>(null);
  readonly isRunning = computed(() => this.runJobId() !== null);
  readonly isPolling = computed(() => this.runJobId() !== null);
  readonly walkForwardError = signal<string | null>(null);

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

  /** Public aliases consumed by the ticker-seeding spec (criterion d). */
  readonly startDate = this.selectedStartDate;
  readonly endDate = this.selectedEndDate;

  readonly tickersRaw = signal(DEFAULT_TICKERS.join(', '));

  readonly tickers = computed<string[]>(() =>
    this.tickersRaw()
      .split(',')
      .map((s) => s.trim().toUpperCase())
      .filter((s) => s.length > 0),
  );

  /** Ticker universe resolved from the portfolio's latest optimization snapshot. */
  readonly resolvedTickers = signal<string[]>([...DEFAULT_TICKERS]);

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

  /**
   * True once a real backtest result is loaded (issue #996, 13a/13b). Drives the
   * KPI em-dash pre-run state and the restore/empty-state branch. `hasResult` is
   * kept as an alias so existing specs continue to compile against one predicate.
   */
  readonly hasLoadedResult = computed(() => this.result().equity.length > 0);
  readonly hasResult = this.hasLoadedResult;

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
    const testId = 'benchmark-' + label.toLowerCase().replace(/\s+/g, '-');
    return {
      metric: label,
      portfolio: formatter(portfolioRaw),
      benchmark: benchmarkRaw == null ? '—' : formatter(benchmarkRaw),
      portfolioRaw,
      benchmarkRaw,
      benchmarkTestId: testId,
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

  constructor() {
    this.loadData();
    this.initTickerSeeding();
    this.initWeightResolver();
    this.restorePersistedRun();
    this.initResultStream();
  }

  loadData(): void {
    this.hasError.set(false);
    this.isLoading.set(false);
  }

  retry(): void {
    this.loadData();
  }

  private initTickerSeeding(): void {
    toObservable(this.portfolioContext.currentPortfolioName)
      .pipe(
        switchMap((name) =>
          this.tickerSeeding.seedFromPortfolio(name, [...DEFAULT_TICKERS]),
        ),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe((seeded) => this.applySeed(seeded));
  }

  private initWeightResolver(): void {
    toObservable(this.portfolioContext.currentPortfolioId)
      .pipe(
        switchMap((id) => this.weightResolver.resolve(id)),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe((res) => {
        const tickers = res.source === 'stored' && Object.keys(res.weights).length > 0
          ? Object.keys(res.weights)
          : [...DEFAULT_TICKERS];
        this.resolvedTickers.set(tickers);
      });
  }

  private applySeed(seeded: string[]): void {
    const prior = this.lastSeed();
    const current = this.tickers();
    const isUnseeded = prior === null;
    const matchesPriorSeed = prior !== null && arraysEqualUnordered(current, prior);
    if (isUnseeded || matchesPriorSeed) {
      this.lastSeed.set(seeded);
      this.tickersRaw.set(seeded.join(', '));
    }
  }

  // ── ChartExportable (stub — charts live in backtest-results-panel) ──────────
  getChartInstance(): undefined {
    return undefined;
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

  /** KPI percent value, or em-dash when no run is loaded (issue #996, 13b). */
  displayPct(v: number): string {
    return this.hasLoadedResult() ? this.formatPct(v) : '—';
  }

  /** KPI ratio value, or em-dash when no run is loaded (issue #996, 13b). */
  displayRatio(v: number, decimals = 2): string {
    return this.hasLoadedResult() ? this.formatRatio(v, decimals) : '—';
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
    const list = tickers && tickers.length > 0 ? tickers : this.resolvedTickers();
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

  onRun(): void {
    this.onRunBacktest(this.tickers());
  }

  onSetupFormRun(config: BacktestRunConfig): void {
    const { start, end } = periodToIsoRange(config.period);
    this.selectedBenchmark.set(config.benchmark);
    this.selectedStartDate.set(start);
    this.selectedEndDate.set(end);
    this.weightResolver
      .resolve(config.portfolio)
      .pipe(take(1), takeUntilDestroyed(this.destroyRef))
      .subscribe((res) => {
        const tickers =
          res.source === 'stored' && Object.keys(res.weights).length > 0
            ? Object.keys(res.weights)
            : this.resolvedTickers();
        this.onRunBacktest(tickers);
      });
  }

  onJobCompleted(runId?: string): void {
    const id = runId ?? this.runRunId();
    this.runJobId.set(null);
    if (!id) return;
    this.hydrateRun(id);
  }

  /**
   * Re-hydrate a persisted run on construct (issue #996, 13a). Only restores when
   * the stored run belongs to the currently selected portfolio; otherwise the
   * stale entry is discarded and the page shows the empty state.
   */
  private restorePersistedRun(): void {
    const stored = loadBacktestRun();
    if (!stored) return;
    if (stored.portfolioId !== this.portfolioContext.currentPortfolioId()) {
      clearBacktestRun();
      return;
    }
    this.runRunId.set(stored.runId);
    this.hydrateRun(stored.runId);
  }

  /**
   * On initial load, fetch the latest run result once for an already-selected
   * portfolio so the KPI strip populates the moment a backtest completes
   * (issue #1030, criterion 12). Reads the first `selectedPortfolio` value
   * only (`take(1)`): when a portfolio is already known synchronously the
   * result is fetched; when none is selected yet the stream stays inert and
   * the KPIs remain em-dash. A failed lookup is swallowed so a missing run
   * never breaks the stream.
   */
  private initResultStream(): void {
    toObservable(this.portfolioContext.selectedPortfolio)
      .pipe(
        take(1),
        switchMap((portfolio) =>
          portfolio
            ? this.backtest.getResult(portfolio.id).pipe(catchError(() => EMPTY))
            : EMPTY,
        ),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe((env) => this.onResultEnvelope(env));
  }

  /** Swap a completed result envelope into KPI/result state; ignore the rest. */
  private onResultEnvelope(env: BacktestResultEnvelope | null): void {
    if (!env || env.status !== 'completed' || !env.result) return;
    this.result.set(mapKpiResultToResult(env.result));
  }

  /** Fetch a completed run and swap it into the results/KPI/tab state. */
  private hydrateRun(runId: string): void {
    this.runResponse.set(null);
    this.runResponseError.set(null);
    this.runResponseLoading.set(true);
    this.result.set(EMPTY_RESULT);
    this.backtest
      .getBacktestRun(runId)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (run) => this.onRunHydrated(run),
        error: (err: Error) => this.onRunHydrateError(err),
      });
  }

  private onRunHydrated(run: BacktestRunResponse): void {
    this.runResponse.set(run);
    this.result.set(mapRunResponseToResult(run));
    this.runResponseLoading.set(false);
    this.persistRun(run.id);
  }

  private onRunHydrateError(err: Error): void {
    this.runResponseLoading.set(false);
    this.runResponseError.set(err?.message ?? 'Failed to load completed run');
  }

  /** Persist `{ runId, portfolioId }` so the run survives navigation / F5. */
  private persistRun(runId: string): void {
    saveBacktestRun({
      runId,
      portfolioId: this.portfolioContext.currentPortfolioId(),
    });
  }

  onJobFailed(message: string): void {
    this.runError.set(message || 'Backtest job failed');
    this.runJobId.set(null);
  }

  onRunWalkForward(): void {
    this.walkForwardError.set(null);
    this.backtest
      .runWalkForward({
        tickers: this.tickers(),
        start_date: this.selectedStartDate(),
        end_date: this.selectedEndDate(),
        cv_type: 'walk_forward',
        cv_config: {},
        optimizer_type: 'hrp',
        optimizer_config: {},
      })
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: () => { /* job polling handled by WalkForwardPanelComponent */ },
        error: (err: Error) => this.walkForwardError.set(err.message ?? 'Walk-forward failed'),
      });
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
/**
 * Map the lightweight `getResult` KPI payload (issue #1030, criterion 12) onto
 * the legacy `BacktestResult` shape so the KPI strip renders formatted values.
 * Only the four headline metrics plus the equity curve are populated; the tab
 * panels stay on their empty defaults until a full run is hydrated.
 */
export function mapKpiResultToResult(payload: BacktestKpiResult): BacktestResult {
  const dates = payload.equityCurve?.dates ?? [];
  const values = payload.equityCurve?.values ?? [];
  const equity = dates.map((date, i) => {
    const portfolio = values[i] ?? 0;
    return { date, portfolio, benchmark: portfolio };
  });
  return {
    ...EMPTY_RESULT,
    equity,
    metrics: {
      ...EMPTY_RESULT.metrics,
      totalReturn: payload.totalReturn,
      sharpe: payload.sharpeRatio,
      maxDrawdown: payload.maxDrawdown,
      informationRatio: payload.informationRatio,
    },
  };
}

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
