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
import { EMPTY, catchError } from 'rxjs';

import { PageHeaderComponent } from '../shared/components/page-header/page-header';
import { TabGroupComponent, Tab } from '../shared/components/tab-group/tab-group';
import { StatCardComponent } from '../shared/stat-card/stat-card';
import { PageErrorBannerComponent } from '../shared/components/page-error-banner/page-error-banner';
import { FormatService } from '../core/services/format.service';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { RebalancingService } from './rebalancing.service';

import { StatusPanelComponent } from './status-panel/status-panel';
import { PolicyPanelComponent } from './policy-panel/policy-panel';
import { TradePreviewPanelComponent } from './trade-preview-panel/trade-preview-panel';
import { HistoryPanelComponent } from './history-panel/history-panel';
import { WhatifPanelComponent } from './whatif-panel/whatif-panel';

import type {
  DriftApiResponse,
  DriftEntryDto,
  RebalanceDecideApiResponse,
  RebalanceDecideRequest,
  RebalancePreviewApiResponse,
  RebalancingPolicyCreatePayload,
  RebalancingPolicyDto,
} from './rebalancing.model';
import type { SnapshotDto } from '../core/models/portfolio-api.model';
import type { ApiError } from '../core/models/api-error.model';

function friendlyPreviewError(err: ApiError, portfolioName: string): string {
  if (err.status === 404) {
    return `No preview available for '${portfolioName}' yet — run a rebalance first.`;
  }
  if (err.status >= 500) {
    return `Preview is temporarily unavailable: ${err.message}`;
  }
  return err.message || 'Preview failed';
}

@Component({
  selector: 'app-rebalancing',
  imports: [
    FormsModule,
    LucideAngularModule,
    PageHeaderComponent,
    TabGroupComponent,
    StatCardComponent,
    PageErrorBannerComponent,
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
  private readonly portfolioCtx = inject(PortfolioContextService);
  private readonly rebalancing = inject(RebalancingService);
  private readonly destroyRef = inject(DestroyRef);

  readonly isLoading = signal<boolean>(false);
  readonly hasError = signal<boolean>(false);
  readonly errorMessage = signal<string>('');

  readonly activeTab = signal<string>('status');
  readonly driftThreshold = signal<number>(0.05);

  // Derived from PortfolioContextService — name takes priority over raw id
  readonly selectedPortfolio = computed<string>(
    () => this.portfolioCtx.currentPortfolioName() ?? this.portfolioCtx.currentPortfolioId() ?? '',
  );

  // Data signals
  readonly driftResponse = signal<DriftApiResponse | null>(null);
  readonly policies = signal<RebalancingPolicyDto[]>([]);
  readonly previewResponse = signal<RebalancePreviewApiResponse | null>(null);
  readonly snapshots = signal<SnapshotDto[]>([]);
  readonly decideResponse = signal<RebalanceDecideApiResponse | null>(null);

  readonly panelErrors = signal<Record<string, string | null>>({});
  readonly pendingActivateId = signal<string | null>(null);

  readonly hasAnyPanelError = computed(() =>
    Object.values(this.panelErrors()).some((v) => v !== null),
  );

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
    effect(() => {
      const name = this.portfolioCtx.currentPortfolioName();
      const id = this.portfolioCtx.currentPortfolioId();
      const portfolio = name ?? id;
      if (!portfolio) return;
      this.loadAllPanels(portfolio);
    });
  }

  retry(): void {
    this.hasError.set(false);
    this.errorMessage.set('');
    const portfolio = this.selectedPortfolio();
    if (portfolio) this.loadAllPanels(portfolio);
  }

  onThresholdChange(value: number): void {
    this.driftThreshold.set(value);
    const portfolio = this.selectedPortfolio();
    if (portfolio) this.fetchDrift(portfolio);
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
        next: () => this.reloadPolicies(name),
        error: (err: Error) => this.setPanelError('policy', err.message ?? 'Create failed'),
      });
  }

  onRunDecide(body: RebalanceDecideRequest): void {
    this.rebalancing
      .decide(body)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => {
          this.decideResponse.set(res);
          this.setPanelError('whatif', null);
        },
        error: (err: Error) => this.setPanelError('whatif', err.message ?? 'Decide failed'),
      });
  }

  private loadAllPanels(portfolio: string): void {
    this.fetchDrift(portfolio);
    this.reloadPolicies(portfolio);
    this.fetchPreview(portfolio);
    this.reloadSnapshots(portfolio);
  }

  private fetchDrift(name: string): void {
    this.rebalancing
      .getDrift(name, this.driftThreshold())
      .pipe(
        catchError((err: Error) => {
          this.setPanelError('drift', err.message ?? 'Drift failed');
          return EMPTY;
        }),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe((res) => {
        this.driftResponse.set(res);
        this.setPanelError('drift', null);
      });
  }

  private reloadPolicies(name: string): void {
    this.rebalancing
      .listPolicies(name)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (list) => {
          this.policies.set(list.items);
          this.setPanelError('policy', null);
        },
        error: (err: Error) => this.setPanelError('policy', err.message ?? 'Policy load failed'),
      });
  }

  private fetchPreview(name: string): void {
    this.rebalancing
      .getPreview(name)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => {
          this.previewResponse.set(res);
          this.setPanelError('preview', null);
        },
        error: (err: ApiError) =>
          this.setPanelError('preview', friendlyPreviewError(err, name)),
      });
  }

  private reloadSnapshots(name: string): void {
    this.rebalancing
      .getSnapshots(name)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => {
          this.snapshots.set(res.items);
          this.setPanelError('history', null);
        },
        error: (err: Error) => this.setPanelError('history', err.message ?? 'History failed'),
      });
  }

  private onActivateSuccess(id: string): void {
    this.pendingActivateId.set(null);
    this.setPanelError('policy', null);
    this.policies.update((list) =>
      list.map((p) => ({ ...p, isActive: p.id === id })),
    );
    const portfolio = this.selectedPortfolio();
    if (portfolio) this.fetchPreview(portfolio);
  }

  private setPanelError(key: string, message: string | null): void {
    this.panelErrors.update((prev) => ({ ...prev, [key]: message }));
  }

  panelError(key: string): string | null {
    return this.panelErrors()[key] ?? null;
  }

  // Convenience accessor for template derivations
  driftEntries(): DriftEntryDto[] {
    return this.driftResponse()?.entries ?? [];
  }
}
