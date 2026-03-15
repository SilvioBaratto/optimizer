import {
  Component,
  signal,
  computed,
  inject,
  ChangeDetectionStrategy,
  DestroyRef,
  ElementRef,
  viewChild,
  effect,
  OnDestroy,
} from '@angular/core';
import { LucideAngularModule } from 'lucide-angular';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { finalize } from 'rxjs';
import { PageHeaderComponent } from '../../shared/components/page-header/page-header';
import { TabGroupComponent, Tab } from '../../shared/components/tab-group/tab-group';
import { StatCardComponent } from '../../shared/stat-card/stat-card';
import { MacroIntelligenceService } from '../../services/macro-intelligence.service';
import { SECTOR_ROTATION_TABLE, COUNTRY_NAME_MAP } from '../../constants/macro-intelligence.constants';
import { PHASE_DEFAULTS, COMPOSITE_SCORE_THRESHOLDS, COMPOSITE_CHART_AXIS } from '../../constants/macro-intelligence.constants';
import type {
  MacroCalibrationResponse,
  FredObservationPoint,
  CountryMacroData,
  BondYieldSnapshot,
  MacroNewsItem,
  MacroNewsTheme,
  MacroNewsSummaryResponse,
  BusinessCyclePhase,
  MacroRegimeLabel,
  SectorRotationStance,
  CompositeScorePoint,
} from '../../models/macro-intelligence.model';
import type { EChartsType, EChartsCoreOption } from 'echarts/core';

type ChartTab = 'yield-curve' | 'credit-spreads' | 'composite-history';
type IntelTab = 'macro-brief' | 'themed-news' | 'country-summary';

const PHASE_LABELS: Record<BusinessCyclePhase, string> = {
  EARLY_EXPANSION: 'Early Expansion',
  MID_EXPANSION: 'Mid Expansion',
  LATE_EXPANSION: 'Late Expansion',
  CONTRACTION: 'Contraction',
};

@Component({
  selector: 'app-macro-intelligence',
  imports: [
    LucideAngularModule,
    PageHeaderComponent,
    TabGroupComponent,
    StatCardComponent,
  ],
  templateUrl: './macro-intelligence.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class MacroIntelligenceComponent implements OnDestroy {
  private readonly macroService = inject(MacroIntelligenceService);
  private readonly destroyRef = inject(DestroyRef);

  // ── Loading ──
  readonly isLoading = signal(true);
  readonly isRefreshing = signal(false);
  readonly hasError = signal(false);
  readonly errorMessage = signal('');

  // ── Data signals ──
  readonly macroCalibration = signal<MacroCalibrationResponse | null>(null);
  readonly fredPmi = signal<FredObservationPoint[]>([]);
  readonly fredYieldSpread = signal<FredObservationPoint[]>([]);
  readonly fredHyOas = signal<FredObservationPoint[]>([]);
  readonly fredVix = signal<FredObservationPoint[]>([]);
  readonly fredIgOas = signal<FredObservationPoint[]>([]);
  readonly countryData = signal<CountryMacroData[]>([]);
  readonly bondYieldsToday = signal<BondYieldSnapshot[]>([]);
  readonly bondYields1YAgo = signal<BondYieldSnapshot[]>([]);
  readonly newsThemes = signal<MacroNewsTheme[]>([]);
  readonly newsItems = signal<MacroNewsItem[]>([]);
  readonly compositeHistory = signal<CompositeScorePoint[]>([]);
  readonly countrySummaries = signal<MacroNewsSummaryResponse[]>([]);

  // ── UI state ──
  readonly selectedCountry = signal('US');
  readonly activeChartTab = signal<ChartTab>('yield-curve');
  readonly activeIntelTab = signal<IntelTab>('macro-brief');
  readonly selectedNewsTheme = signal<string | null>(null);

  // ── Chart refs ──
  private readonly chartContainer = viewChild<ElementRef<HTMLElement>>('chartContainer');
  private chart?: EChartsType;
  private chartRo?: ResizeObserver;
  private chartInitialized = false;

  // ── Tabs ──
  readonly chartTabs = computed<Tab[]>(() => [
    { id: 'yield-curve', label: 'Yield Curve' },
    { id: 'credit-spreads', label: 'Credit Spreads' },
    { id: 'composite-history', label: 'Score History' },
  ]);

  readonly intelTabs = computed<Tab[]>(() => [
    { id: 'macro-brief', label: 'Macro Brief' },
    { id: 'themed-news', label: 'Themed News', badge: this.newsItems().length },
    { id: 'country-summary', label: 'Country Summary', badge: this.countrySummaries().length },
  ]);

  // ── Computed: S_t composite score ──
  readonly pmiScore = computed(() => {
    const data = this.fredPmi();
    if (!data.length) return 0;
    const latest = data[data.length - 1].value;
    if (latest > COMPOSITE_SCORE_THRESHOLDS.PMI_BULL) return 1;
    if (latest < COMPOSITE_SCORE_THRESHOLDS.PMI_BEAR) return -1;
    return 0;
  });

  readonly spreadScore = computed(() => {
    const data = this.fredYieldSpread();
    if (!data.length) return 0;
    const latest = data[data.length - 1].value;
    if (latest > COMPOSITE_SCORE_THRESHOLDS.YIELD_SPREAD_BULL) return 1;
    if (latest < COMPOSITE_SCORE_THRESHOLDS.YIELD_SPREAD_BEAR) return -1;
    return 0;
  });

  readonly hyScore = computed(() => {
    const data = this.fredHyOas();
    if (!data.length) return 0;
    const latest = data[data.length - 1].value;
    if (latest < COMPOSITE_SCORE_THRESHOLDS.HY_OAS_BULL) return 1;
    if (latest > COMPOSITE_SCORE_THRESHOLDS.HY_OAS_BEAR) return -1;
    return 0;
  });

  readonly compositeScore = computed(() =>
    this.pmiScore() + this.spreadScore() + this.hyScore()
  );

  readonly regimeLabel = computed<MacroRegimeLabel>(() => {
    const s = this.compositeScore();
    if (s >= COMPOSITE_CHART_AXIS.BULL_THRESHOLD) return 'Expansionary';
    if (s <= COMPOSITE_CHART_AXIS.BEAR_THRESHOLD) return 'Contractionary';
    return 'Transitional';
  });

  readonly regimeDotClass = computed(() => {
    const label = this.regimeLabel();
    if (label === 'Expansionary') return 'bg-gain';
    if (label === 'Contractionary') return 'bg-loss';
    return 'bg-warning';
  });

  // ── Computed: Indicator helpers ──
  readonly latestPmi = computed(() => {
    const d = this.fredPmi();
    return d.length ? d[d.length - 1].value : 0;
  });

  readonly latestSpread = computed(() => {
    const d = this.fredYieldSpread();
    return d.length ? d[d.length - 1].value : 0;
  });

  readonly latestHyOas = computed(() => {
    const d = this.fredHyOas();
    return d.length ? d[d.length - 1].value : 0;
  });

  readonly latestVix = computed(() => {
    const d = this.fredVix();
    return d.length ? d[d.length - 1].value : 0;
  });

  readonly pmiSparkline = computed(() => this.fredPmi().slice(-24).map(p => p.value));
  readonly spreadSparkline = computed(() => this.fredYieldSpread().slice(-90).map(p => p.value));
  readonly hyOasSparkline = computed(() => this.fredHyOas().slice(-90).map(p => p.value));
  readonly vixSparkline = computed(() => this.fredVix().slice(-90).map(p => p.value));

  readonly pmiDelta = computed(() => this.computeDelta(this.fredPmi(), 30));
  readonly spreadDelta = computed(() => this.computeDelta(this.fredYieldSpread(), 30));
  readonly hyOasDelta = computed(() => this.computeDelta(this.fredHyOas(), 30));
  readonly vixDelta = computed(() => this.computeDelta(this.fredVix(), 30));

  // ── Computed: Effective calibration (fallback when LLM unavailable) ──
  readonly effectiveCalibration = computed<MacroCalibrationResponse>(() => {
    const cal = this.macroCalibration();
    if (cal !== null) return cal;

    const phase = this.phaseFromCompositeScore(this.compositeScore());
    const defaults = PHASE_DEFAULTS[phase];
    return {
      phase,
      delta: defaults.delta,
      tau: defaults.tau,
      confidence: 0,
      rationale: 'LLM unavailable — using phase defaults',
      macro_summary: '',
      timestamp: new Date().toISOString(),
      bl_config: {
        views: [],
        tau: defaults.tau,
        prior_config: {
          mu_estimator: 'equilibrium',
          risk_aversion: defaults.delta,
          cov_estimator: 'ledoit_wolf',
        },
      },
    };
  });

  // ── Computed: Sector recommendations ──
  readonly currentPhase = computed<BusinessCyclePhase>(() =>
    this.effectiveCalibration().phase
  );

  readonly currentPhaseLabel = computed(() => PHASE_LABELS[this.currentPhase()]);

  readonly sectorTable = SECTOR_ROTATION_TABLE;

  readonly overweightSectors = computed(() =>
    this.sectorTable
      .filter(r => r.phases[this.currentPhase()] === 'OW')
      .map(r => r.sector)
  );

  readonly underweightSectors = computed(() =>
    this.sectorTable
      .filter(r => r.phases[this.currentPhase()] === 'UW')
      .map(r => r.sector)
  );

  // ── Computed: Filtered news ──
  readonly filteredNews = computed(() => {
    const theme = this.selectedNewsTheme();
    const items = this.newsItems();
    return theme ? items.filter(n => n.theme === theme) : items;
  });

  // ── Computed: Selected country data ──
  readonly selectedCountryData = computed(() =>
    this.countryData().find(c => c.country_code === this.selectedCountry())
  );

  // ── Phase columns for sector table ──
  readonly phaseColumns: BusinessCyclePhase[] = ['EARLY_EXPANSION', 'MID_EXPANSION', 'LATE_EXPANSION', 'CONTRACTION'];

  constructor() {
    this.loadData();
    effect(() => {
      const tab = this.activeChartTab();
      const country = this.selectedCountry();
      const bondsToday = this.bondYieldsToday();
      const bonds1YAgo = this.bondYields1YAgo();
      const hyOas = this.fredHyOas();
      const igOas = this.fredIgOas();
      const history = this.compositeHistory();
      const loading = this.isLoading();
      const container = this.chartContainer();

      // Chart container only exists after loading completes
      if (loading || !container) return;

      if (!this.chartInitialized) {
        this.initChart();
      } else {
        this.updateChart();
      }
    });
  }

  // ── Data loading ──
  loadData(showSkeleton = true, forceRefreshCalibration = false): void {
    if (showSkeleton) {
      this.isLoading.set(true);
      // Chart container will be destroyed by @if — clean up stale instance
      if (this.chartInitialized) {
        this.chartRo?.disconnect();
        this.chart?.dispose();
        this.chart = undefined;
        this.chartRo = undefined;
        this.chartInitialized = false;
      }
    }
    this.hasError.set(false);

    const subs = [
      this.macroService.getMacroCalibration(forceRefreshCalibration).subscribe({
        next: d => this.macroCalibration.set(d),
        error: () => this.macroCalibration.set(null),
      }),
      this.macroService.getFredPmi().subscribe(d => this.fredPmi.set(d)),
      this.macroService.getFredYieldSpread().subscribe(d => this.fredYieldSpread.set(d)),
      this.macroService.getFredHyOas().subscribe(d => this.fredHyOas.set(d)),
      this.macroService.getFredVix().subscribe(d => this.fredVix.set(d)),
      this.macroService.getFredIgOas().subscribe(d => this.fredIgOas.set(d)),
      this.macroService.getCountryData().subscribe(d => this.countryData.set(d)),
      this.macroService.getBondYieldsToday().subscribe(d => this.bondYieldsToday.set(d)),
      this.macroService.getBondYields1YAgo().subscribe(d => this.bondYields1YAgo.set(d)),
      this.macroService.getNewsThemes().subscribe(d => this.newsThemes.set(d)),
      this.macroService.getNews().subscribe(d => { this.newsItems.set(d); this.isLoading.set(false); }),
      this.macroService.getCompositeHistory().subscribe(d => this.compositeHistory.set(d)),
      this.macroService.getCountrySummaries().subscribe(d => this.countrySummaries.set(d)),
    ];
  }

  retry(): void {
    this.loadData();
  }

  refreshData(): void {
    if (this.isRefreshing()) return;
    this.isRefreshing.set(true);

    this.macroService.triggerRefresh().pipe(
      takeUntilDestroyed(this.destroyRef),
      finalize(() => this.isRefreshing.set(false)),
    ).subscribe({
      next: () => this.loadData(false, true),
      error: (err) => {
        console.error('Refresh failed', err);
        this.loadData(false, true);
      },
    });
  }

  // ── Chart ──
  private async initChart(): Promise<void> {
    if (this.chartInitialized) return;
    this.chartInitialized = true;

    const el = this.chartContainer()?.nativeElement;
    if (!el) { this.chartInitialized = false; return; }

    const { init, use } = await import('echarts/core');
    const { LineChart, BarChart } = await import('echarts/charts');
    const { GridComponent, TooltipComponent, LegendComponent, MarkLineComponent } = await import('echarts/components');
    const { CanvasRenderer } = await import('echarts/renderers');

    use([LineChart, BarChart, GridComponent, TooltipComponent, LegendComponent, MarkLineComponent, CanvasRenderer]);

    this.chart = init(el, 'portfolio', { renderer: 'canvas' });
    this.chartRo = new ResizeObserver(() => this.chart?.resize());
    this.chartRo.observe(el);
    this.updateChart();
  }

  private updateChart(): void {
    if (!this.chart) return;
    const tab = this.activeChartTab();
    if (tab === 'yield-curve') this.chart.setOption(this.buildYieldCurveOption(), true);
    else if (tab === 'credit-spreads') this.chart.setOption(this.buildCreditSpreadOption(), true);
    else this.chart.setOption(this.buildCompositeHistoryOption(), true);
  }

  private buildYieldCurveOption(): EChartsCoreOption {
    const country = this.selectedCountry();
    const todaySnap = this.bondYieldsToday().find(s => s.country === country);
    const agoSnap = this.bondYields1YAgo().find(s => s.country === country);
    const maturities = todaySnap?.curve.map(p => p.maturity) ?? [];

    return {
      tooltip: { trigger: 'axis' },
      legend: { bottom: 0 },
      grid: { left: 50, right: 20, top: 20, bottom: 40 },
      xAxis: { type: 'category', data: maturities },
      yAxis: { type: 'value', axisLabel: { formatter: (v: number) => `${v.toFixed(1)}%` } },
      series: [
        {
          name: 'Today',
          type: 'line',
          data: todaySnap?.curve.map(p => p.yield_pct) ?? [],
          smooth: true,
          symbol: 'circle',
          symbolSize: 6,
        },
        {
          name: '1Y Ago',
          type: 'line',
          data: agoSnap?.curve.map(p => p.yield_pct) ?? [],
          smooth: true,
          symbol: 'diamond',
          symbolSize: 6,
          lineStyle: { type: 'dashed' },
        },
      ],
    };
  }

  private buildCreditSpreadOption(): EChartsCoreOption {
    const hyData = this.fredHyOas();
    const igData = this.fredIgOas();
    const labels = hyData.map(p => p.date);

    const style = typeof document !== 'undefined'
      ? getComputedStyle(document.documentElement)
      : null;
    const borderColor = style?.getPropertyValue('--color-border').trim() ?? '#e5e7eb';

    return {
      tooltip: { trigger: 'axis' },
      legend: { bottom: 0 },
      grid: { left: 55, right: 55, top: 20, bottom: 40 },
      xAxis: { type: 'category', data: labels },
      yAxis: [
        { type: 'value', name: 'HY OAS (bp)', position: 'left', axisLabel: { formatter: (v: number) => `${v}` } },
        { type: 'value', name: 'IG OAS (bp)', position: 'right', axisLabel: { formatter: (v: number) => `${v}` } },
      ],
      series: [
        {
          name: 'HY OAS',
          type: 'line',
          data: hyData.map(p => p.value),
          smooth: true,
          markLine: {
            silent: true,
            symbol: 'none',
            data: [
              { yAxis: COMPOSITE_SCORE_THRESHOLDS.HY_OAS_BULL, lineStyle: { color: '#22c55e', type: 'dashed' }, label: { formatter: `${COMPOSITE_SCORE_THRESHOLDS.HY_OAS_BULL} (Bull)`, position: 'insideEndTop' } },
              { yAxis: COMPOSITE_SCORE_THRESHOLDS.HY_OAS_BEAR, lineStyle: { color: '#ef4444', type: 'dashed' }, label: { formatter: `${COMPOSITE_SCORE_THRESHOLDS.HY_OAS_BEAR} (Bear)`, position: 'insideEndTop' } },
            ],
          },
        },
        {
          name: 'IG OAS',
          type: 'line',
          yAxisIndex: 1,
          data: igData.map(p => p.value),
          smooth: true,
        },
      ],
    };
  }

  private buildCompositeHistoryOption(): EChartsCoreOption {
    const history = this.compositeHistory();

    return {
      tooltip: {
        trigger: 'axis',
        formatter: (params: unknown) => {
          const p = (params as Array<{ name: string; value: number }>)[0];
          return `${p.name}: S<sub>t</sub> = ${p.value > 0 ? '+' : ''}${p.value}`;
        },
      },
      grid: { left: 40, right: 20, top: 20, bottom: 40 },
      xAxis: {
        type: 'category',
        data: history.map(h => h.month),
        axisLabel: { rotate: 45 },
      },
      yAxis: { type: 'value', min: COMPOSITE_CHART_AXIS.MIN, max: COMPOSITE_CHART_AXIS.MAX },
      series: [
        {
          type: 'bar',
          data: history.map(h => ({
            value: h.score,
            itemStyle: {
              color: h.score >= COMPOSITE_CHART_AXIS.BULL_THRESHOLD ? '#22c55e' : h.score <= COMPOSITE_CHART_AXIS.BEAR_THRESHOLD ? '#ef4444' : '#f59e0b',
              borderRadius: [2, 2, 0, 0] as [number, number, number, number],
            },
          })),
          barMaxWidth: 20,
        },
      ],
    };
  }

  // ── Helpers ──
  onChartTabChange(tab: string): void {
    this.activeChartTab.set(tab as ChartTab);
  }

  onIntelTabChange(tab: string): void {
    this.activeIntelTab.set(tab as IntelTab);
  }

  selectCountry(code: string): void {
    this.selectedCountry.set(code);
  }

  selectNewsTheme(themeId: string | null): void {
    this.selectedNewsTheme.set(themeId);
  }

  getPhaseLabel(phase: BusinessCyclePhase): string {
    return PHASE_LABELS[phase];
  }

  getStanceClass(stance: SectorRotationStance): string {
    if (stance === 'OW') return 'text-gain font-semibold';
    if (stance === 'UW') return 'text-loss font-semibold';
    return 'text-text-tertiary';
  }

  getScoreSign(score: number): string {
    return score > 0 ? `+${score}` : `${score}`;
  }

  getScoreClass(score: number): string {
    if (score > 0) return 'text-gain';
    if (score < 0) return 'text-loss';
    return 'text-text-tertiary';
  }

  getTrend(delta: number): 'up' | 'down' | 'flat' {
    if (delta > 0.001) return 'up';
    if (delta < -0.001) return 'down';
    return 'flat';
  }

  relativeTime(isoString: string): string {
    const now = Date.now();
    const then = new Date(isoString).getTime();
    const diffMin = Math.floor((now - then) / 60000);
    if (diffMin < 1) return 'just now';
    if (diffMin < 60) return `${diffMin}m ago`;
    const diffHr = Math.floor(diffMin / 60);
    if (diffHr < 24) return `${diffHr}h ago`;
    const diffDay = Math.floor(diffHr / 24);
    return `${diffDay}d ago`;
  }

  parseMacroSummary(summary: string): { label: string; value: string }[] {
    if (!summary.trim()) return [];
    return summary.split('|').map(segment => {
      const trimmed = segment.trim();
      const colonIdx = trimmed.indexOf(':');
      if (colonIdx === -1) return { label: trimmed, value: '' };
      return {
        label: trimmed.slice(0, colonIdx).trim(),
        value: trimmed.slice(colonIdx + 1).trim(),
      };
    });
  }

  buildMiniSparkline(data: number[]): string {
    if (data.length < 2) return '';
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;
    return data
      .map((v, i) => {
        const x = (i / (data.length - 1)) * 100;
        const y = 24 - ((v - min) / range) * 20 - 2;
        return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)} ${y.toFixed(1)}`;
      })
      .join(' ');
  }

  getSentimentClass(sentiment: string | null): string {
    switch (sentiment) {
      case 'BULLISH': return 'text-gain bg-gain/10';
      case 'BEARISH': return 'text-loss bg-loss/10';
      case 'MIXED': return 'text-warning bg-warning/10';
      default: return 'text-text-tertiary bg-surface-inset';
    }
  }

  getSentimentLabel(sentiment: string | null): string {
    return sentiment ?? 'No data';
  }

  getCountrySummaryFor(countryCode: string): MacroNewsSummaryResponse | undefined {
    const dbName = COUNTRY_NAME_MAP[countryCode];
    return this.countrySummaries().find(s => s.country === dbName);
  }

  private phaseFromCompositeScore(score: number): BusinessCyclePhase {
    if (score >= COMPOSITE_CHART_AXIS.BULL_THRESHOLD) return 'EARLY_EXPANSION';
    if (score >= 0) return 'MID_EXPANSION';
    if (score >= COMPOSITE_CHART_AXIS.BEAR_THRESHOLD + 1) return 'LATE_EXPANSION';
    return 'CONTRACTION';
  }

  private computeDelta(data: FredObservationPoint[], lookback: number): number {
    if (data.length < lookback + 1) return 0;
    const latest = data[data.length - 1].value;
    const prior = data[data.length - 1 - lookback].value;
    return prior !== 0 ? (latest - prior) / Math.abs(prior) : 0;
  }

  ngOnDestroy(): void {
    this.chartRo?.disconnect();
    this.chart?.dispose();
  }
}
