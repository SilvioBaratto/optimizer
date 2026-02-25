import { Component, signal, computed, inject, ChangeDetectionStrategy } from '@angular/core';
import { PageHeaderComponent } from '../../shared/components/page-header/page-header';
import { TabGroupComponent, Tab } from '../../shared/components/tab-group/tab-group';
import { StatCardComponent } from '../../shared/stat-card/stat-card';
import { FormatService } from '../../services/format.service';
import { MockFetchService } from '../../services/mock-fetch.service';
import { StatusPanelComponent } from './status-panel';
import { PolicyPanelComponent } from './policy-panel';
import { TradePreviewPanelComponent } from './trade-preview-panel';
import { HistoryPanelComponent } from './history-panel';
import { WhatifPanelComponent } from './whatif-panel';
import {
  MOCK_DRIFT_TABLE,
  MOCK_REBALANCING_POLICIES,
  MOCK_TRADE_PREVIEW,
  MOCK_TRADE_SUMMARY,
  MOCK_REBALANCING_HISTORY,
} from '../../mocks/rebalancing-mocks';
import type { RebalancingPolicy } from '../../models/rebalancing.model';

@Component({
  selector: 'app-rebalancing',
  imports: [
    PageHeaderComponent,
    TabGroupComponent,
    StatCardComponent,
    StatusPanelComponent,
    PolicyPanelComponent,
    TradePreviewPanelComponent,
    HistoryPanelComponent,
    WhatifPanelComponent,
  ],
  templateUrl: './rebalancing.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class RebalancingComponent {
  private fmt = inject(FormatService);
  private readonly mockFetch = inject(MockFetchService);

  isLoading = signal(true);
  hasError = signal(false);
  errorMessage = signal('');

  activeTab = signal('status');
  policies = signal<RebalancingPolicy[]>(MOCK_REBALANCING_POLICIES);

  readonly driftEntries = MOCK_DRIFT_TABLE;
  readonly trades = MOCK_TRADE_PREVIEW;
  readonly summary = MOCK_TRADE_SUMMARY;
  readonly history = MOCK_REBALANCING_HISTORY;

  constructor() {
    this.loadData();
  }

  loadData(): void {
    this.isLoading.set(true);
    this.hasError.set(false);
    this.mockFetch
      .fetch({
        driftEntries: MOCK_DRIFT_TABLE,
        policies: MOCK_REBALANCING_POLICIES,
        trades: MOCK_TRADE_PREVIEW,
        summary: MOCK_TRADE_SUMMARY,
        history: MOCK_REBALANCING_HISTORY,
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

  tabs: Tab[] = [
    { id: 'status',        label: 'Drift Status' },
    { id: 'policy',        label: 'Policy' },
    { id: 'trade-preview', label: 'Trade Preview' },
    { id: 'history',       label: 'History' },
    { id: 'what-if',       label: 'What-If' },
  ];

  kpiMaxDrift = computed(() => {
    const max = Math.max(...this.driftEntries.map(e => e.driftAbsolute), 0);
    return this.fmt.formatBps(max);
  });

  kpiBreachedAssets = computed(() =>
    String(this.driftEntries.filter(e => e.breached).length),
  );

  kpiActivePolicy = computed(() => {
    const active = this.policies().find(p => p.active);
    return active?.name ?? '--';
  });

  kpiEstRebalCost = computed(() =>
    this.fmt.formatCurrency(this.summary.totalCost),
  );

  onActivatePolicy(policyId: string) {
    this.policies.update(list =>
      list.map(p => ({ ...p, active: p.id === policyId })),
    );
  }
}
