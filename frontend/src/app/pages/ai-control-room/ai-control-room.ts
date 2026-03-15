import {
  Component,
  signal,
  computed,
  inject,
  ChangeDetectionStrategy,
} from '@angular/core';
import { DatePipe } from '@angular/common';
import { LucideAngularModule } from 'lucide-angular';
import { PageHeaderComponent } from '../../shared/components/page-header/page-header';
import { TabGroupComponent, Tab } from '../../shared/components/tab-group/tab-group';
import { StatCardComponent } from '../../shared/stat-card/stat-card';
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { MockFetchService } from '../../services/mock-fetch.service';
import type { AgentRole } from '../../models/ai-control.model';
import {
  MOCK_AGENT_STATUSES,
  MOCK_DECISION_FEED,
  MOCK_VETO_LOG,
} from '../../mocks/ai-control-mocks';

type ControlTab = 'overview' | 'history' | 'agents';

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
    LucideAngularModule,
    DatePipe,
    PageHeaderComponent,
    TabGroupComponent,
    StatCardComponent,
    DataTableComponent,
  ],
  templateUrl: './ai-control-room.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class AiControlRoomComponent {
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
  readonly vetoLog = MOCK_VETO_LOG;

  // ── Tabs ──
  readonly tabs = computed<Tab[]>(() => [
    { id: 'overview', label: 'Overview' },
    { id: 'history', label: 'History', badge: this.feed.length },
    { id: 'agents', label: 'Agents' },
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

  // ── History: table columns ──
  readonly historyColumns: TableColumn[] = [
    { key: 'timestamp', label: 'Time', sortable: true, type: 'date', dateFormat: 'medium' },
    { key: 'agent', label: 'Agent', sortable: true, type: 'badge', badgeMap: Object.fromEntries(Object.entries(AGENT_COLOR_MAP).map(([k, v]) => [k, { value: v.label, colorClass: `bg-[${v.var}]/15 text-[var(${v.var})]` }])) },
    { key: 'type', label: 'Type', sortable: true, type: 'badge', badgeMap: TYPE_BADGE, hiddenOnMobile: true },
    { key: 'title', label: 'Decision', sortable: true, hiddenOnMobile: true },
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
    { key: 'action', label: 'Action', sortable: true, hiddenOnMobile: true },
    { key: 'reason', label: 'Reason', sortable: false, hiddenOnMobile: true },
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

  constructor() {
    this.loadData();
  }

  loadData(): void {
    this.isLoading.set(true);
    this.hasError.set(false);
    this.mockFetch
      .fetch({
        agents: MOCK_AGENT_STATUSES,
        feed: MOCK_DECISION_FEED,
        vetoLog: MOCK_VETO_LOG,
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
}
