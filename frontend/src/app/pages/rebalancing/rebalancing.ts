import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  computed,
  effect,
  inject,
  signal,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { LucideAngularModule } from 'lucide-angular';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';

import { PageHeaderComponent } from '../../shared/components/page-header/page-header';
import { TabGroupComponent, Tab } from '../../shared/components/tab-group/tab-group';
import { StatCardComponent } from '../../shared/stat-card/stat-card';
import { FormatService } from '../../services/format.service';
import { PortfolioApiService } from '../../services/portfolio-api.service';
import { RebalancingService } from '../../services/rebalancing.service';

import { StatusPanelComponent } from './status-panel';
import { PolicyPanelComponent } from './policy-panel';
import { TradePreviewPanelComponent } from './trade-preview-panel';
import { HistoryPanelComponent } from './history-panel';
import { WhatifPanelComponent } from './whatif-panel';

import type {
  DriftApiResponse,
  DriftEntryDto,
  RebalanceDecideApiResponse,
  RebalanceDecideRequest,
  RebalancePreviewApiResponse,
  RebalancingPolicyCreatePayload,
  RebalancingPolicyDto,
} from '../../models/rebalancing.model';
import type { PortfolioDto, SnapshotDto } from '../../models/portfolio-api.model';

@Component({
  selector: 'app-rebalancing',
  imports: [
    FormsModule,
    LucideAngularModule,
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
  private readonly fmt = inject(FormatService);
  private readonly portfolioApi = inject(PortfolioApiService);
  private readonly rebalancing = inject(RebalancingService);
  private readonly destroyRef = inject(DestroyRef);

  readonly isLoading = signal<boolean>(false);
  readonly hasError = signal<boolean>(false);
  readonly errorMessage = signal<string>('');

  readonly activeTab = signal<string>('status');
  readonly driftThreshold = signal<number>(0.05);

  // Portfolio state
  readonly portfolios = signal<PortfolioDto[]>([]);
  readonly selectedPortfolio = signal<string>('');

  // Data signals
  readonly driftResponse = signal<DriftApiResponse | null>(null);
  readonly policies = signal<RebalancingPolicyDto[]>([]);
  readonly previewResponse = signal<RebalancePreviewApiResponse | null>(null);
  readonly snapshots = signal<SnapshotDto[]>([]);
  readonly decideResponse = signal<RebalanceDecideApiResponse | null>(null);

  readonly panelErrors = signal<Record<string, string | null>>({});
  readonly pendingActivateId = signal<string | null>(null);

  readonly tabs: Tab[] = [
    { id: 'status',        label: 'Drift Status' },
    { id: 'policy',        label: 'Policy' },
    { id: 'trade-preview', label: 'Trade Preview' },
    { id: 'history',       label: 'History' },
    { id: 'what-if',       label: 'What-If' },
  ];

  readonly activePolicy = computed<RebalancingPolicyDto | null>(
    () => this.policies().find((p) => p.isActive) ?? null,
  );

  readonly kpiMaxDrift = computed(() => {
    const drift = this.driftResponse();
    if (!drift || drift.entries.length === 0) return '—';
    const max = Math.max(...drift.entries.map((e) => Math.abs(e.drift)));
    return this.fmt.formatBps(max);
  });

  readonly kpiBreachedAssets = computed(() =>
    String(this.driftResponse()?.breachedCount ?? 0),
  );

  readonly kpiActivePolicy = computed(() => this.activePolicy()?.name ?? '—');

  readonly kpiEstRebalCost = computed(() => {
    const dec = this.decideResponse();
    if (!dec) return '—';
    return this.fmt.formatPercent(dec.estimatedCost);
  });

  constructor() {
    this.loadPortfolios();
    effect(() => this.onPortfolioChange());
  }

  retry(): void {
    this.hasError.set(false);
    this.loadPortfolios();
  }

  onPortfolioSelect(name: string): void {
    this.selectedPortfolio.set(name);
  }

  onThresholdChange(value: number): void {
    this.driftThreshold.set(value);
    this.fetchDrift();
  }

  requestActivate(policyId: string): void {
    this.pendingActivateId.set(policyId);
  }

  cancelActivate(): void {
    this.pendingActivateId.set(null);
  }

  confirmActivate(): void {
    const id = this.pendingActivateId();
    const name = this.selectedPortfolio();
    if (!id || !name) return;
    this.rebalancing
      .activatePolicy(name, id)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: () => this.onActivateSuccess(id),
        error: (err: Error) => this.setPanelError('policy', err.message ?? 'Activate failed'),
      });
  }

  onCreatePolicy(payload: RebalancingPolicyCreatePayload): void {
    const name = this.selectedPortfolio();
    if (!name) return;
    this.rebalancing
      .createPolicy(name, payload)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: () => this.reloadPolicies(),
        error: (err: Error) => this.setPanelError('policy', err.message ?? 'Create failed'),
      });
  }

  onRunDecide(body: RebalanceDecideRequest): void {
    this.rebalancing
      .decide(body)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => this.decideResponse.set(res),
        error: (err: Error) => this.setPanelError('whatif', err.message ?? 'Decide failed'),
      });
  }

  private loadPortfolios(): void {
    this.isLoading.set(true);
    this.portfolioApi
      .list()
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (list) => this.applyPortfolios(list.items),
        error: (err: Error) => this.failInitialLoad(err.message ?? 'Failed to load portfolios'),
      });
  }

  private applyPortfolios(items: PortfolioDto[]): void {
    this.portfolios.set(items);
    if (items.length > 0 && !this.selectedPortfolio()) {
      this.selectedPortfolio.set(items[0].name);
    }
    this.isLoading.set(false);
  }

  private failInitialLoad(message: string): void {
    this.errorMessage.set(message);
    this.hasError.set(true);
    this.isLoading.set(false);
  }

  private onPortfolioChange(): void {
    const name = this.selectedPortfolio();
    if (!name) return;
    this.fetchDrift();
    this.reloadPolicies();
    this.fetchPreview();
    this.reloadSnapshots();
  }

  private fetchDrift(): void {
    const name = this.selectedPortfolio();
    if (!name) return;
    this.rebalancing
      .getDrift(name, this.driftThreshold())
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => this.driftResponse.set(res),
        error: (err: Error) => this.setPanelError('drift', err.message ?? 'Drift failed'),
      });
  }

  private reloadPolicies(): void {
    const name = this.selectedPortfolio();
    if (!name) return;
    this.rebalancing
      .listPolicies(name)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (list) => this.policies.set(list.items),
        error: (err: Error) => this.setPanelError('policy', err.message ?? 'Policy load failed'),
      });
  }

  private fetchPreview(): void {
    const name = this.selectedPortfolio();
    if (!name) return;
    this.rebalancing
      .getPreview(name)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => this.previewResponse.set(res),
        error: (err: Error) => this.setPanelError('preview', err.message ?? 'Preview failed'),
      });
  }

  private reloadSnapshots(): void {
    const name = this.selectedPortfolio();
    if (!name) return;
    this.rebalancing
      .getSnapshots(name)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => this.snapshots.set(res.items),
        error: (err: Error) => this.setPanelError('history', err.message ?? 'History failed'),
      });
  }

  private onActivateSuccess(id: string): void {
    this.pendingActivateId.set(null);
    this.policies.update((list) =>
      list.map((p) => ({ ...p, isActive: p.id === id })),
    );
    // Refresh preview since active policy changed.
    this.fetchPreview();
  }

  private setPanelError(key: string, message: string): void {
    this.panelErrors.update((prev) => ({ ...prev, [key]: message }));
  }

  // Convenience accessor for template derivations
  driftEntries(): DriftEntryDto[] {
    return this.driftResponse()?.entries ?? [];
  }
}
