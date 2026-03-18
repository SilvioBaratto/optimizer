import {
  Component,
  signal,
  inject,
  computed,
  effect,
  ElementRef,
  viewChild,
  OnDestroy,
  DestroyRef,
  ChangeDetectionStrategy,
} from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { RouterLink } from '@angular/router';
import { LucideAngularModule } from 'lucide-angular';
import { interval, switchMap, startWith } from 'rxjs';
import type { EChartsType, EChartsCoreOption } from 'echarts/core';
import { StatCardComponent } from '../../shared/stat-card/stat-card';
import { PageHeaderComponent } from '../../shared/components/page-header/page-header';
import { EchartsSunburstComponent, SunburstNode } from '../../shared/echarts-sunburst/echarts-sunburst';
import { FormatService } from '../../services/format.service';
import { DashboardService } from '../../services/dashboard.service';
import { PortfolioContextService } from '../../services/portfolio-context.service';
import { readCssVar } from '../../shared/charts/echarts-theme';
import type { DashboardKPI, ActivityType, MarketRegime } from '../../models/dashboard.model';
import type { ActivityFeedItem, MarketContext, RegimeInfo, DriftEntry, EquityCurvePoint, AssetClassReturn } from '../../models/dashboard.model';
import { ModalService } from '../../shared/modal/modal.service';
import { ExportReportModalComponent } from '../../shared/modal/export-report-modal';

const MARKET_REFRESH_MS = 5 * 60 * 1000; // 5 minutes

@Component({
  selector: 'app-dashboard',
  imports: [RouterLink, LucideAngularModule, StatCardComponent, PageHeaderComponent, EchartsSunburstComponent],
  templateUrl: './dashboard.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class DashboardComponent implements OnDestroy {
  readonly fmt = inject(FormatService);
  private readonly modalService = inject(ModalService);
  private readonly dashboardSvc = inject(DashboardService);
  private readonly destroyRef = inject(DestroyRef);
  private readonly portfolioCtx = inject(PortfolioContextService);

  // Portfolio name from context with fallback
  readonly portfolioName = computed(
    () => this.portfolioCtx.currentPortfolioId() ?? 'Global Multi-Factor',
  );

  // Per-domain loading states
  readonly isLoadingPortfolio = signal(true);
  readonly isLoadingMarket = signal(true);

  // Aggregate loading / error
  readonly isLoading = computed(() => this.isLoadingPortfolio() && this.isLoadingMarket());
  readonly hasError = signal(false);
  readonly errorMessage = signal('');
  readonly revealIndex = signal(0);

  // #153 — KPI Strip
  readonly kpis = signal<DashboardKPI[]>([]);
  readonly nav = signal(0);
  readonly dailyChange = signal(0);

  // #154 — Equity Curve + Allocation
  readonly equityCurve = signal<EquityCurvePoint[]>([]);
  readonly allocationData = signal<SunburstNode[]>([]);
  readonly driftTable = signal<DriftEntry[]>([]);

  // Allocation legend
  readonly allocationSectors = computed(() => {
    const data = this.allocationData();
    const total = data.reduce((sum, s) => sum + (s.value ?? 0), 0);
    return data
      .map((sector, i) => ({
        name: sector.name,
        weight: (sector.value ?? 0) / total,
        holdings: sector.children?.length ?? 0,
        colorIndex: i,
      }))
      .sort((a, b) => b.weight - a.weight);
  });
  readonly allocationTotalHoldings = computed(() =>
    this.allocationData().reduce((sum, s) => sum + (s.children?.length ?? 0), 0),
  );

  // #155 — Activity Feed + Market Context
  readonly activityFeed = signal<ActivityFeedItem[]>([]);
  readonly marketContext = signal<MarketContext>({
    vix: 0, vixChange: 0, sp500Return: 0,
    tenYearYield: 0, yieldChange: 0, usdIndex: 0, usdChange: 0,
  });
  readonly regimeInfo = signal<RegimeInfo>({
    current: 'sideways', probability: 0, since: '', hmmStates: [],
  });
  readonly assetClassReturns = signal<AssetClassReturn[]>([]);

  // Macro stat-card helpers
  readonly macroVixDelta = computed(() => {
    const ctx = this.marketContext();
    return ctx.vix ? ctx.vixChange / ctx.vix : 0;
  });
  readonly macroVixTrend = computed((): 'up' | 'down' | 'flat' =>
    this.marketContext().vixChange < 0 ? 'up' : this.marketContext().vixChange > 0 ? 'down' : 'flat');
  readonly macroYieldDelta = computed(() => this.marketContext().yieldChange / 100);
  readonly macroYieldTrend = computed((): 'up' | 'down' | 'flat' =>
    this.marketContext().yieldChange < 0 ? 'down' : this.marketContext().yieldChange > 0 ? 'up' : 'flat');
  readonly macroUsdDelta = computed(() => {
    const ctx = this.marketContext();
    return ctx.usdIndex ? ctx.usdChange / ctx.usdIndex : 0;
  });
  readonly macroUsdTrend = computed((): 'up' | 'down' | 'flat' =>
    this.marketContext().usdChange > 0 ? 'up' : this.marketContext().usdChange < 0 ? 'down' : 'flat');

  readonly subtitle = computed(() => {
    const navVal = this.nav();
    const change = this.dailyChange();
    const navStr = this.fmt.formatCurrencyCompact(navVal);
    const changeStr = this.fmt.formatPercent(change);
    return `NAV ${navStr}  |  ${change >= 0 ? '+' : ''}${changeStr} today`;
  });

  // Equity curve chart
  private readonly equityCurveContainer = viewChild<ElementRef<HTMLElement>>('equityCurve');
  private equityChart?: EChartsType;
  private equityRo?: ResizeObserver;

  constructor() {
    this.loadPortfolioData();
    this.startMarketRefresh();

    effect((onCleanup) => {
      const el = this.equityCurveContainer();
      if (el && !this.equityChart) {
        void this.initEquityCurveChart();
      }
      onCleanup(() => {
        this.equityRo?.disconnect();
        this.equityChart?.dispose();
        this.equityChart = undefined;
        this.equityRo = undefined;
      });
    });
  }

  private loadPortfolioData(): void {
    const name = this.portfolioName();

    this.isLoadingPortfolio.set(true);
    this.hasError.set(false);
    this.revealIndex.set(0);

    let completedCount = 0;
    const totalCalls = 6;

    const onComplete = () => {
      completedCount++;
      if (completedCount >= totalCalls) {
        this.isLoadingPortfolio.set(false);
        this.startRevealAnimation();
      }
    };

    const onError = (err: Error) => {
      this.hasError.set(true);
      this.errorMessage.set(err.message);
      this.isLoadingPortfolio.set(false);
    };

    this.dashboardSvc.getPerformanceMetrics(name)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: res => {
          this.kpis.set(res.kpis as DashboardKPI[]);
          this.nav.set(res.nav);
          this.dailyChange.set(res.navChangePct);
          onComplete();
        },
        error: onError,
      });

    this.dashboardSvc.getEquityCurve(name, 'SPY', '3Y')
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: res => {
          this.equityCurve.set(res.points as EquityCurvePoint[]);
          onComplete();
          if (this.equityChart) {
            this.equityChart.setOption(this.buildEquityCurveOption());
          }
        },
        error: onError,
      });

    this.dashboardSvc.getAllocation(name)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: res => {
          this.allocationData.set(res.nodes as SunburstNode[]);
          onComplete();
        },
        error: onError,
      });

    this.dashboardSvc.getDrift(name)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: res => {
          this.driftTable.set(res.entries as DriftEntry[]);
          onComplete();
        },
        error: onError,
      });

    this.dashboardSvc.getActivity(name)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: res => {
          this.activityFeed.set(res.items as ActivityFeedItem[]);
          onComplete();
        },
        error: onError,
      });

    this.dashboardSvc.getAssetClassReturns(name)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: res => {
          this.assetClassReturns.set(res.returns as AssetClassReturn[]);
          onComplete();
        },
        error: onError,
      });
  }

  private startMarketRefresh(): void {
    interval(MARKET_REFRESH_MS)
      .pipe(
        startWith(0),
        switchMap(() => this.dashboardSvc.getMarketSnapshot()),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe({
        next: res => {
          this.marketContext.set({
            vix: res.vix,
            vixChange: res.vixChange,
            sp500Return: res.sp500Return,
            tenYearYield: res.tenYearYield,
            yieldChange: res.yieldChange,
            usdIndex: res.usdIndex,
            usdChange: res.usdChange,
          });
          this.isLoadingMarket.set(false);
        },
        error: () => this.isLoadingMarket.set(false),
      });

    interval(MARKET_REFRESH_MS)
      .pipe(
        startWith(0),
        switchMap(() => this.dashboardSvc.getRegimeState()),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe({
        next: res => {
          this.regimeInfo.set({
            current: res.current as MarketRegime,
            probability: res.probability,
            since: res.since,
            hmmStates: res.hmmStates as { regime: MarketRegime; probability: number }[],
          });
        },
        error: () => { /* regime failure is non-critical */ },
      });
  }

  private startRevealAnimation(): void {
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReducedMotion) {
      this.revealIndex.set(10);
    } else {
      let idx = 0;
      const timer = setInterval(() => {
        idx++;
        this.revealIndex.set(idx);
        if (idx >= 10) clearInterval(timer);
      }, 50);
    }
  }

  retry(): void {
    this.loadPortfolioData();
  }

  formatKpiValue(kpi: DashboardKPI): string {
    switch (kpi.format) {
      case 'percent': return this.fmt.formatPercent(kpi.value);
      case 'currency': return this.fmt.formatCurrencyCompact(kpi.value);
      case 'ratio': return this.fmt.formatRatio(kpi.value);
      default: return kpi.value.toLocaleString();
    }
  }

  kpiTrend(kpi: DashboardKPI): 'up' | 'down' | 'flat' {
    if (kpi.change > 0) return 'up';
    if (kpi.change < 0) return 'down';
    return 'flat';
  }

  kpiDelta(kpi: DashboardKPI): number {
    if (kpi.format === 'currency') {
      const navVal = this.nav();
      return navVal ? kpi.change / navVal : 0;
    }
    return kpi.change;
  }

  openReportModal(): void {
    this.modalService.open({ component: ExportReportModalComponent, title: 'Export Report', size: 'lg' });
  }

  // Activity feed helpers
  activityDotClass(type: string): string {
    const map: Record<string, string> = {
      optimization: 'bg-info',
      rebalance: 'bg-success',
      regime_change: 'bg-warning',
      alert: 'bg-danger',
      ai_decision: 'bg-agent-risk',
      trade: 'bg-chart-5',
    };
    return map[type] ?? 'bg-text-tertiary';
  }

  relativeTime(timestamp: string): string {
    const now = Date.now();
    const diff = now - new Date(timestamp).getTime();
    const mins = Math.floor(diff / 60000);
    const hrs = Math.floor(mins / 60);
    const days = Math.floor(hrs / 24);
    if (mins < 1) return 'just now';
    if (mins < 60) return `${mins}m ago`;
    if (hrs < 24) return `${hrs}h ago`;
    return `${days}d ago`;
  }

  // Regime helpers
  regimeColorClass(regime: MarketRegime): string {
    const map: Record<MarketRegime, string> = {
      bull: 'bg-regime-bull-bg text-regime-bull',
      bear: 'bg-regime-bear-bg text-regime-bear',
      sideways: 'bg-regime-neutral-bg text-regime-neutral',
      volatile: 'bg-regime-crisis-bg text-regime-crisis',
    };
    return map[regime];
  }

  regimeBarColor(regime: MarketRegime): string {
    const map: Record<MarketRegime, string> = {
      bull: 'bg-regime-bull',
      bear: 'bg-regime-bear',
      sideways: 'bg-regime-neutral',
      volatile: 'bg-regime-crisis',
    };
    return map[regime];
  }

  // Asset class returns heatmap color
  returnCellBg(value: number): string {
    if (value > 0.05) return 'background-color: var(--color-heatmap-7); color: white';
    if (value > 0.02) return 'background-color: var(--color-heatmap-6); color: white';
    if (value > 0) return 'background-color: var(--color-heatmap-5)';
    if (value === 0) return 'background-color: var(--color-heatmap-4)';
    if (value > -0.02) return 'background-color: var(--color-heatmap-3)';
    if (value > -0.05) return 'background-color: var(--color-heatmap-2); color: white';
    return 'background-color: var(--color-heatmap-1); color: white';
  }

  // Equity curve chart initialization
  private async initEquityCurveChart() {
    const el = this.equityCurveContainer()?.nativeElement;
    if (!el) return;

    const { init, use } = await import('echarts/core');
    const { LineChart } = await import('echarts/charts');
    const { GridComponent, TooltipComponent, LegendComponent, DataZoomComponent, MarkAreaComponent } = await import('echarts/components');
    const { CanvasRenderer } = await import('echarts/renderers');

    use([LineChart, GridComponent, TooltipComponent, LegendComponent, DataZoomComponent, MarkAreaComponent, CanvasRenderer]);

    this.equityChart = init(el, 'portfolio', { renderer: 'canvas' });
    this.equityChart.setOption(this.buildEquityCurveOption());

    this.equityRo = new ResizeObserver(() => this.equityChart?.resize());
    this.equityRo.observe(el);
  }

  private buildEquityCurveOption(): EChartsCoreOption {
    const data = this.equityCurve();
    const dates = data.map(d => d.date);
    const portfolio = data.map(d => d.portfolio);
    const benchmark = data.map(d => d.benchmark);

    const chart1 = readCssVar('--color-chart-1');
    const chart7 = readCssVar('--color-chart-7');
    const lossMuted = readCssVar('--color-loss-muted');

    // Compute drawdown regions (>5% from peak)
    const markAreas: Array<[{ xAxis: string }, { xAxis: string }]> = [];
    let peak = portfolio[0];
    let inDrawdown = false;
    let ddStart = '';

    for (let i = 0; i < portfolio.length; i++) {
      if (portfolio[i] > peak) peak = portfolio[i];
      const dd = (portfolio[i] - peak) / peak;
      if (dd < -0.05 && !inDrawdown) {
        inDrawdown = true;
        ddStart = dates[i];
      } else if (dd >= -0.05 && inDrawdown) {
        inDrawdown = false;
        markAreas.push([{ xAxis: ddStart }, { xAxis: dates[i] }]);
      }
    }
    if (inDrawdown) {
      markAreas.push([{ xAxis: ddStart }, { xAxis: dates[dates.length - 1] }]);
    }

    return {
      tooltip: {
        trigger: 'axis',
        formatter: (params: unknown) => {
          const ps = params as Array<{ seriesName: string; value: number; axisValueLabel: string; color: string }>;
          if (!ps.length) return '';
          let html = `<div style="font-size:12px"><b>${ps[0].axisValueLabel}</b>`;
          for (const p of ps) {
            html += `<br/><span style="color:${p.color}">\u25CF</span> ${p.seriesName}: ${p.value.toFixed(2)}`;
          }
          return html + '</div>';
        },
      },
      legend: {
        data: ['Portfolio', 'Benchmark (SPY)'],
        top: 0,
        right: 0,
      },
      grid: { left: 50, right: 16, top: 36, bottom: 56 },
      xAxis: {
        type: 'category',
        data: dates,
        axisLabel: {
          formatter: (v: string) => v.slice(0, 7),
          interval: Math.floor(dates.length / 6),
          hideOverlap: true,
        },
      },
      yAxis: {
        type: 'value',
        axisLabel: {
          formatter: (v: number) => v.toFixed(0),
        },
      },
      dataZoom: [
        { type: 'inside', start: 0, end: 100 },
        { type: 'slider', start: 0, end: 100, height: 20, bottom: 4 },
      ],
      series: [
        {
          name: 'Portfolio',
          type: 'line',
          data: portfolio,
          symbol: 'none',
          lineStyle: { width: 2, color: chart1 },
          itemStyle: { color: chart1 },
          markArea: markAreas.length > 0
            ? {
                silent: true,
                itemStyle: { color: `${lossMuted}18` },
                data: markAreas,
              }
            : undefined,
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

  ngOnDestroy() {
    this.equityRo?.disconnect();
    this.equityChart?.dispose();
  }
}
