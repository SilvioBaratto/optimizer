import {
  Component,
  signal,
  computed,
  inject,
  ChangeDetectionStrategy,
} from '@angular/core';
import { DatePipe } from '@angular/common';
import { LucideAngularModule } from 'lucide-angular';
import { PageHeaderComponent } from '../shared/components/page-header/page-header';
import { TabGroupComponent, Tab } from '../shared/components/tab-group/tab-group';
import { StatCardComponent } from '../shared/stat-card/stat-card';
import { DataTableComponent, TableColumn } from '../shared/data-table/data-table';
import { PipelineStatusComponent } from './pipeline-status/pipeline-status';
import type {
  AgentRole,
  AgentStatus,
  DecisionFeedItem,
  VetoLogEntry,
} from './ai-control.model';
import { FEATURE_AI_DECISION_LOG_TOKEN } from '../core/config/feature-flags';

type ControlTab = 'overview' | 'history' | 'agents' | 'pipeline';

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
    PipelineStatusComponent,
  ],
  templateUrl: './ai-control-room.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class AiControlRoomComponent {
  // Compile-time feature flag gating the Overview/History/Agents tabs and the
  // decision-log KPI row. Overridable in tests via
  // `TestBed.configureTestingModule` provider swap.
  readonly decisionLogEnabled = inject(FEATURE_AI_DECISION_LOG_TOKEN);

  // ── Loading state ──
  isLoading = signal(true);
  hasError = signal(false);
  errorMessage = signal('');

  // ── State ──
  readonly activeTab = signal<ControlTab>(
    this.decisionLogEnabled ? 'overview' : 'pipeline',
  );
  readonly feedFilter = signal<AgentRole | 'all'>('all');
  readonly expandedDecisions = signal<Set<string>>(new Set());

  // ── Decision-log data sources ──
  // These three arrays back the Overview / History / Agents tabs, which are
  // only rendered when `decisionLogEnabled` (FEATURE_AI_DECISION_LOG) is true.
  // No backend endpoint exists yet for agent statuses, the decision feed, or
  // the veto log; the feature flag ships OFF so production never shows stale
  // data. When the endpoints land, wire a service here — see the follow-up
  // tracked under milestone "Cycle 5 — Observability" in the audit backlog.
  readonly agents: AgentStatus[] = [];
  readonly feed: DecisionFeedItem[] = [];
  readonly vetoLog: VetoLogEntry[] = [];

  // ── Tabs ──
  readonly tabs = computed<Tab[]>(() => {
    if (!this.decisionLogEnabled) {
      return [{ id: 'pipeline', label: 'Pipeline' }];
    }
    return [
      { id: 'overview', label: 'Overview' },
      { id: 'history', label: 'History', badge: this.feed.length },
      { id: 'agents', label: 'Agents' },
      { id: 'pipeline', label: 'Pipeline' },
    ];
  });

  // ── Top-level stats ──
  readonly totalDecisionsToday = computed(() =>
    this.agents.reduce((sum, a) => sum + a.decisionsToday, 0)
  );
  readonly activeAgentCount = computed(() =>
    this.agents.filter(a => a.status === 'active').length
  );
  readonly avgConfidence = computed(() => {
    if (this.agents.length === 0) return '0.0%';
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
    this.hasError.set(false);
    this.isLoading.set(false);
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
