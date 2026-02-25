import {
  Component,
  signal,
  computed,
  inject,
  ElementRef,
  viewChild,
  afterNextRender,
  effect,
  OnDestroy,
  ChangeDetectionStrategy,
} from '@angular/core';
import { DatePipe } from '@angular/common';
import type { EChartsType, EChartsCoreOption } from 'echarts/core';
import { PageHeaderComponent } from '../../shared/components/page-header/page-header';
import { TabGroupComponent, Tab } from '../../shared/components/tab-group/tab-group';
import { StatCardComponent } from '../../shared/stat-card/stat-card';
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { EchartsDonutComponent } from '../../shared/echarts-donut/echarts-donut';
import { FormatService } from '../../services/format.service';
import { MockFetchService } from '../../services/mock-fetch.service';
import { readCssVar } from '../../shared/charts/echarts-theme';
import type { AgentRole, DecisionFeedItem, AgentStatus } from '../../models/ai-control.model';
import {
  MOCK_AGENT_STATUSES,
  MOCK_DECISION_FEED,
  MOCK_WORKFLOW_STATES,
  MOCK_AI_FUND_SUMMARY,
  MOCK_AGENT_CONFIGS_EXTENDED,
  MOCK_ORCHESTRATION_CONFIG,
  MOCK_VETO_LOG,
  MOCK_COMPARISON_DATA,
} from '../../mocks/ai-control-mocks';

type ControlTab = 'overview' | 'portfolio' | 'history' | 'agents' | 'compare';

const AGENT_COLOR_MAP: Record<AgentRole, { var: string; bgVar: string; label: string }> = {
  portfolio_manager: { var: '--color-agent-pm', bgVar: '--color-agent-pm-bg', label: 'PM' },
  risk_analyst: { var: '--color-agent-risk', bgVar: '--color-agent-risk-bg', label: 'Risk' },
  factor_researcher: { var: '--color-agent-analyst', bgVar: '--color-agent-analyst-bg', label: 'Analyst' },
  execution_agent: { var: '--color-agent-cio', bgVar: '--color-agent-cio-bg', label: 'Exec' },
};

const STATUS_CLASS: Record<string, string> = {
  active: 'bg-gain',
  idle: 'bg-text-tertiary',
  paused: 'bg-warning',
  error: 'bg-loss',
};

const OUTCOME_BADGE: Record<string, { value: string; colorClass: string }> = {
  executed: { value: 'Executed', colorClass: 'bg-gain/15 text-gain' },
  approved: { value: 'Approved', colorClass: 'bg-accent/15 text-accent' },
  pending: { value: 'Pending', colorClass: 'bg-warning/15 text-warning' },
  rejected: { value: 'Rejected', colorClass: 'bg-loss/15 text-loss' },
};

const TYPE_BADGE: Record<string, { value: string; colorClass: string }> = {
  rebalance: { value: 'Rebalance', colorClass: 'bg-accent/15 text-accent' },
  risk_alert: { value: 'Risk Alert', colorClass: 'bg-loss/15 text-loss' },
  factor_tilt: { value: 'Factor Tilt', colorClass: 'bg-chart-3/15 text-[var(--color-chart-3)]' },
  trade: { value: 'Trade', colorClass: 'bg-gain/15 text-gain' },
  veto: { value: 'Veto', colorClass: 'bg-loss/15 text-loss' },
  regime_change: { value: 'Regime', colorClass: 'bg-chart-5/15 text-[var(--color-chart-5)]' },
};

@Component({
  selector: 'app-ai-control-room',
  imports: [
    DatePipe,
    PageHeaderComponent,
    TabGroupComponent,
    StatCardComponent,
    DataTableComponent,
    EchartsDonutComponent,
  ],
  templateUrl: './ai-control-room.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class AiControlRoomComponent implements OnDestroy {
  private readonly fmt = inject(FormatService);
  private readonly mockFetch = inject(MockFetchService);

  // ── Loading state ──
  isLoading = signal(true);
  hasError = signal(false);
  errorMessage = signal('');

  // ── State ──
  readonly activeTab = signal<ControlTab>('overview');
  readonly feedFilter = signal<AgentRole | 'all'>('all');
  readonly expandedDecisions = signal<Set<string>>(new Set());

  // ── Static data ──
  readonly agents = MOCK_AGENT_STATUSES;
  readonly feed = MOCK_DECISION_FEED;
  readonly workflow = MOCK_WORKFLOW_STATES;
  readonly fund = MOCK_AI_FUND_SUMMARY;
  readonly agentConfigs = MOCK_AGENT_CONFIGS_EXTENDED;
  readonly orchestration = MOCK_ORCHESTRATION_CONFIG;
  readonly vetoLog = MOCK_VETO_LOG;
  readonly comparison = MOCK_COMPARISON_DATA;

  // ── Tabs ──
  readonly tabs = computed<Tab[]>(() => [
    { id: 'overview', label: 'Overview' },
    { id: 'portfolio', label: 'AI Fund' },
    { id: 'history', label: 'History', badge: this.feed.length },
    { id: 'agents', label: 'Agents' },
    { id: 'compare', label: 'Compare' },
  ]);

  // ── Top-level stats ──
  readonly totalDecisionsToday = computed(() =>
    this.agents.reduce((sum, a) => sum + a.decisionsToday, 0)
  );
  readonly activeAgentCount = computed(() =>
    this.agents.filter(a => a.status === 'active').length
  );
  readonly avgConfidence = computed(() => {
    const c = this.agents.reduce((sum, a) => sum + a.confidence, 0) / this.agents.length;
    return (c * 100).toFixed(1) + '%';
  });
  readonly fundNav = computed(() =>
    this.fmt.formatCurrency(this.fund.aum)
  );

  // ── Overview: filtered feed ──
  readonly filteredFeed = computed(() => {
    const filter = this.feedFilter();
    const items = filter === 'all' ? this.feed : this.feed.filter(d => d.agent === filter);
    return items.slice(0, 20);
  });

  readonly feedFilterOptions: { value: AgentRole | 'all'; label: string }[] = [
    { value: 'all', label: 'All' },
    { value: 'portfolio_manager', label: 'PM' },
    { value: 'risk_analyst', label: 'Risk' },
    { value: 'factor_researcher', label: 'Analyst' },
    { value: 'execution_agent', label: 'Exec' },
  ];

  // ── Portfolio: donut segments ──
  readonly holdingsDonutSegments = computed(() => {
    const chartColors = [
      readCssVar('--color-chart-1'),
      readCssVar('--color-chart-2'),
      readCssVar('--color-chart-3'),
      readCssVar('--color-chart-4'),
      readCssVar('--color-chart-5'),
      readCssVar('--color-chart-6'),
      readCssVar('--color-chart-7'),
      readCssVar('--color-chart-8'),
    ];
    return this.fund.holdings
      .filter(h => h.ticker !== 'CASH')
      .slice(0, 8)
      .map((h, i) => ({
        label: h.ticker,
        value: h.weight * 100,
        color: chartColors[i % chartColors.length],
      }));
  });

  // ── History: table columns ──
  readonly historyColumns: TableColumn[] = [
    { key: 'timestamp', label: 'Time', sortable: true, type: 'date', dateFormat: 'medium' },
    { key: 'agent', label: 'Agent', sortable: true, type: 'badge', badgeMap: Object.fromEntries(Object.entries(AGENT_COLOR_MAP).map(([k, v]) => [k, { value: v.label, colorClass: `bg-[${v.var}]/15 text-[var(${v.var})]` }])) },
    { key: 'type', label: 'Type', sortable: true, type: 'badge', badgeMap: TYPE_BADGE },
    { key: 'title', label: 'Decision', sortable: true },
    { key: 'outcome', label: 'Outcome', sortable: true, type: 'badge', badgeMap: OUTCOME_BADGE },
    { key: 'confidence', label: 'Confidence', sortable: true, type: 'percentage', align: 'right' },
  ];

  readonly historyRows = computed(() =>
    this.feed.map(d => ({
      timestamp: d.timestamp,
      agent: d.agent,
      type: d.type,
      title: d.title,
      outcome: d.outcome,
      confidence: d.confidence,
    }))
  );

  // ── Agents: veto log columns ──
  readonly vetoColumns: TableColumn[] = [
    { key: 'timestamp', label: 'Time', sortable: true, type: 'date', dateFormat: 'medium' },
    { key: 'vetoAgent', label: 'Veto By', sortable: true, type: 'badge', badgeMap: Object.fromEntries(Object.entries(AGENT_COLOR_MAP).map(([k, v]) => [k, { value: v.label, colorClass: `bg-[${v.var}]/15 text-[var(${v.var})]` }])) },
    { key: 'targetAgent', label: 'Target', sortable: true, type: 'badge', badgeMap: Object.fromEntries(Object.entries(AGENT_COLOR_MAP).map(([k, v]) => [k, { value: v.label, colorClass: `bg-[${v.var}]/15 text-[var(${v.var})]` }])) },
    { key: 'action', label: 'Action', sortable: true },
    { key: 'reason', label: 'Reason', sortable: false },
    { key: 'overridden', label: 'Status', sortable: true, type: 'badge', badgeMap: { true: { value: 'Overridden', colorClass: 'bg-warning/15 text-warning' }, false: { value: 'Enforced', colorClass: 'bg-gain/15 text-gain' } } },
  ];

  readonly vetoRows = computed(() =>
    this.vetoLog.map(v => ({
      timestamp: v.timestamp,
      vetoAgent: v.vetoAgent,
      targetAgent: v.targetAgent,
      action: v.action,
      reason: v.reason,
      overridden: String(v.overridden),
    }))
  );

  // ── Compare: table columns ──
  readonly comparisonColumns: TableColumn[] = [
    { key: 'metric', label: 'Metric', sortable: true },
    { key: 'user', label: 'You', sortable: true, align: 'right', format: (v) => this.formatComparisonValue(v as number, '') },
    { key: 'ai', label: 'AI Fund', sortable: true, align: 'right', format: (v) => this.formatComparisonValue(v as number, '') },
    { key: 'benchmark', label: 'Benchmark', sortable: true, align: 'right', format: (v) => this.formatComparisonValue(v as number, '') },
  ];

  readonly comparisonRows = computed(() =>
    this.comparison.metrics.map(m => ({
      metric: m.metric,
      user: m.user,
      ai: m.ai,
      benchmark: m.benchmark,
      unit: m.unit,
    }))
  );

  readonly divergenceColumns: TableColumn[] = [
    { key: 'date', label: 'Date', sortable: true, type: 'date', dateFormat: 'short' },
    { key: 'asset', label: 'Asset', sortable: true },
    { key: 'userAction', label: 'Your Action', sortable: false },
    { key: 'aiAction', label: 'AI Action', sortable: false },
    { key: 'delta', label: 'Delta', sortable: true, type: 'percentage', align: 'right', colorBySign: true },
  ];

  readonly divergenceRows = computed(() =>
    this.comparison.divergences.map(d => ({
      date: d.date,
      asset: d.asset,
      userAction: d.userAction,
      aiAction: d.aiAction,
      userOutcome: d.userOutcome,
      aiOutcome: d.aiOutcome,
      delta: d.delta,
    }))
  );

  // ── Compare: inline equity chart ──
  private readonly equityContainer = viewChild<ElementRef<HTMLElement>>('equityChart');
  private equityChartInstance?: EChartsType;
  private ro?: ResizeObserver;

  constructor() {
    this.loadData();
    afterNextRender(() => {
      if (this.equityContainer()) {
        this.initEquityChart();
      }
    });
    effect(() => {
      const tab = this.activeTab();
      if (tab === 'compare' && !this.equityChartInstance) {
        setTimeout(() => this.initEquityChart(), 50);
      }
    });
  }

  loadData(): void {
    this.isLoading.set(true);
    this.hasError.set(false);
    this.mockFetch
      .fetch({
        agents: MOCK_AGENT_STATUSES,
        feed: MOCK_DECISION_FEED,
        workflow: MOCK_WORKFLOW_STATES,
        fund: MOCK_AI_FUND_SUMMARY,
        agentConfigs: MOCK_AGENT_CONFIGS_EXTENDED,
        orchestration: MOCK_ORCHESTRATION_CONFIG,
        vetoLog: MOCK_VETO_LOG,
        comparison: MOCK_COMPARISON_DATA,
      })
      .then(() => {
        this.isLoading.set(false);
      })
      .catch((err: Error) => {
        this.hasError.set(true);
        this.errorMessage.set(err.message);
        this.isLoading.set(false);
      });
  }

  retry(): void {
    this.loadData();
  }

  private async initEquityChart() {
    const el = this.equityContainer()?.nativeElement;
    if (!el || this.equityChartInstance) return;

    const { init, use } = await import('echarts/core');
    const { LineChart } = await import('echarts/charts');
    const { GridComponent, TooltipComponent, LegendComponent } = await import('echarts/components');
    const { CanvasRenderer } = await import('echarts/renderers');

    use([LineChart, GridComponent, TooltipComponent, LegendComponent, CanvasRenderer]);

    this.equityChartInstance = init(el, 'portfolio', { renderer: 'canvas' });
    this.equityChartInstance.setOption(this.buildEquityOption());

    this.ro = new ResizeObserver(() => this.equityChartInstance?.resize());
    this.ro.observe(el);
  }

  private buildEquityOption(): EChartsCoreOption {
    const equity = this.comparison.equity;
    const textSecondary = readCssVar('--color-text-secondary');
    const border = readCssVar('--color-border-muted');

    return {
      tooltip: { trigger: 'axis' },
      legend: {
        data: ['You', 'AI Fund', 'Benchmark'],
        bottom: 0,
      },
      grid: { top: 10, right: 16, bottom: 36, left: 48 },
      xAxis: {
        type: 'category',
        data: equity.map(p => p.date),
        axisLabel: {
          fontSize: 10,
          color: textSecondary,
          formatter: (v: string) => v.slice(5),
        },
        axisLine: { lineStyle: { color: border } },
        axisTick: { show: false },
      },
      yAxis: {
        type: 'value',
        axisLabel: { fontSize: 10, color: textSecondary },
        splitLine: { lineStyle: { color: border, type: 'dashed' } },
        axisLine: { show: false },
        axisTick: { show: false },
      },
      series: [
        {
          name: 'You',
          type: 'line',
          data: equity.map(p => p.user),
          smooth: false,
          lineStyle: { width: 2 },
          symbol: 'none',
          itemStyle: { color: readCssVar('--color-chart-1') },
        },
        {
          name: 'AI Fund',
          type: 'line',
          data: equity.map(p => p.ai),
          smooth: false,
          lineStyle: { width: 2 },
          symbol: 'none',
          itemStyle: { color: readCssVar('--color-gain') },
        },
        {
          name: 'Benchmark',
          type: 'line',
          data: equity.map(p => p.benchmark),
          smooth: false,
          lineStyle: { width: 1.5, type: 'dashed' },
          symbol: 'none',
          itemStyle: { color: readCssVar('--color-text-tertiary') },
        },
      ],
    };
  }

  // ── Helpers ──
  getAgentColorVar(role: AgentRole): string {
    return `var(${AGENT_COLOR_MAP[role].var})`;
  }

  getAgentBgVar(role: AgentRole): string {
    return `var(${AGENT_COLOR_MAP[role].bgVar})`;
  }

  getAgentLabel(role: AgentRole): string {
    return AGENT_COLOR_MAP[role].label;
  }

  getStatusDotClass(status: string): string {
    return STATUS_CLASS[status] ?? 'bg-text-tertiary';
  }

  getOutcomeBadge(outcome: string): { value: string; colorClass: string } {
    return OUTCOME_BADGE[outcome] ?? { value: outcome, colorClass: '' };
  }

  getTypeBadge(type: string): { value: string; colorClass: string } {
    return TYPE_BADGE[type] ?? { value: type, colorClass: '' };
  }

  getWorkflowStepStatusClass(status: string): string {
    switch (status) {
      case 'completed': return 'bg-gain text-white';
      case 'running': return 'bg-accent text-white animate-pulse';
      case 'pending': return 'bg-surface-inset text-text-tertiary';
      case 'skipped': return 'bg-warning/20 text-warning';
      default: return 'bg-surface-inset text-text-tertiary';
    }
  }

  getWorkflowConnectorClass(status: string): string {
    return status === 'completed' ? 'bg-gain' : 'bg-border';
  }

  getRiskBarWidth(utilization: number): string {
    return `${Math.min(utilization * 100, 100)}%`;
  }

  getRiskBarClass(status: string): string {
    switch (status) {
      case 'ok': return 'bg-gain';
      case 'warning': return 'bg-warning';
      case 'breach': return 'bg-loss';
      default: return 'bg-gain';
    }
  }

  getSignalBarWidth(value: number): string {
    return `${Math.abs(value) * 50}%`;
  }

  getSignalColor(label: string): string {
    switch (label) {
      case 'bullish': return 'bg-gain';
      case 'bearish': return 'bg-loss';
      default: return 'bg-text-tertiary';
    }
  }

  getSignalPosition(value: number): string {
    return value >= 0 ? 'left: 50%' : `left: ${50 + value * 50}%`;
  }

  getCostPercentage(used: number, budget: number): number {
    return budget > 0 ? (used / budget) * 100 : 0;
  }

  formatPercent(v: number): string {
    return this.fmt.formatPercent(v);
  }

  formatCurrency(v: number): string {
    return this.fmt.formatCurrency(v);
  }

  onTabChange(tab: string): void {
    this.activeTab.set(tab as ControlTab);
  }

  toggleDecision(id: string): void {
    this.expandedDecisions.update(set => {
      const next = new Set(set);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }

  isExpanded(id: string): boolean {
    return this.expandedDecisions().has(id);
  }

  private formatComparisonValue(v: number, _unit: string): string {
    if (v === 0) return '—';
    if (Math.abs(v) < 1 && Math.abs(v) > 0) return this.fmt.formatPercent(v);
    if (Math.abs(v) >= 100) return this.fmt.formatCurrency(v);
    return v.toFixed(2);
  }

  ngOnDestroy(): void {
    this.ro?.disconnect();
    this.equityChartInstance?.dispose();
  }
}
