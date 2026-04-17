import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  computed,
  inject,
  signal,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { LucideAngularModule } from 'lucide-angular';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';

import { PageHeaderComponent } from '../../shared/components/page-header/page-header';
import { TabGroupComponent, Tab } from '../../shared/components/tab-group/tab-group';
import { UniversePanelComponent } from './universe-panel';
import { WeightPanelComponent } from './weight-panel';
import { IpsPanelComponent } from './ips-panel';
import { InstrumentDetailFlyoutComponent } from '../../shared/instrument-detail-flyout/instrument-detail-flyout';

import { PortfolioApiService } from '../../services/portfolio-api.service';
import type { PortfolioDto } from '../../models/portfolio-api.model';
import type { Instrument } from '../../models/universe.model';
import type { Constraint, IPS } from '../../models/portfolio-builder.model';

const DEFAULT_IPS: IPS = {
  name: '',
  riskProfile: 'moderate',
  targetReturn: 0,
  maxVolatility: 0,
  maxDrawdown: 0,
  rebalanceFrequency: 'quarterly',
  constraints: [],
};

const CURRENCY_CODES = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD'];
const DEFAULT_BENCHMARK = 'SPY';

type PortfolioStatus = 'idle' | 'creating' | 'success' | 'error';
type SnapshotStatus = 'idle' | 'saving' | 'success' | 'error';

@Component({
  selector: 'app-portfolio-builder',
  imports: [
    FormsModule,
    LucideAngularModule,
    PageHeaderComponent,
    TabGroupComponent,
    UniversePanelComponent,
    WeightPanelComponent,
    IpsPanelComponent,
    InstrumentDetailFlyoutComponent,
  ],
  templateUrl: './portfolio-builder.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class PortfolioBuilderComponent {
  private readonly portfolioApi = inject(PortfolioApiService);
  private readonly destroyRef = inject(DestroyRef);

  readonly isLoading = signal(false);
  readonly hasError = signal(false);
  readonly errorMessage = signal('');

  readonly activeTab = signal<string>('universe');
  readonly flyoutInstrumentId = signal<string | null>(null);

  // Portfolios
  readonly portfolios = signal<PortfolioDto[]>([]);
  readonly activePortfolio = signal<PortfolioDto | null>(null);

  // Create-portfolio form
  readonly showCreateModal = signal<boolean>(false);
  readonly formName = signal<string>('');
  readonly formDescription = signal<string>('');
  readonly formCurrency = signal<string>('USD');
  readonly formBenchmark = signal<string>(DEFAULT_BENCHMARK);
  readonly formStatus = signal<PortfolioStatus>('idle');
  readonly formError = signal<string | null>(null);

  // Snapshot state
  readonly snapshotStatus = signal<SnapshotStatus>('idle');
  readonly snapshotError = signal<string | null>(null);

  readonly currencyCodes = CURRENCY_CODES;

  readonly tabs: Tab[] = [
    { id: 'universe', label: 'Universe' },
    { id: 'weights', label: 'Weights & Constraints' },
    { id: 'ips', label: 'IPS & Summary' },
  ];

  readonly formIsValid = computed(() => {
    const name = this.formName().trim();
    return name.length >= 2 && name.length <= 64 && this.formCurrency().length > 0;
  });

  constructor() {
    this.loadPortfolios();
  }

  loadPortfolios(): void {
    this.isLoading.set(true);
    this.hasError.set(false);
    this.portfolioApi
      .list()
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (list) => {
          this.portfolios.set(list.items);
          if (list.items.length > 0) this.activePortfolio.set(list.items[0]);
          this.isLoading.set(false);
        },
        error: (err: Error) => {
          this.errorMessage.set(err.message ?? 'Failed to load portfolios');
          this.hasError.set(true);
          this.isLoading.set(false);
        },
      });
  }

  retry(): void {
    this.loadPortfolios();
  }

  openCreateModal(): void {
    this.resetForm();
    this.showCreateModal.set(true);
  }

  closeCreateModal(): void {
    this.showCreateModal.set(false);
  }

  submitCreate(): void {
    if (!this.formIsValid() || this.formStatus() === 'creating') return;
    this.formStatus.set('creating');
    this.formError.set(null);
    this.portfolioApi
      .create({
        name: this.formName().trim(),
        description: this.formDescription().trim() || null,
        currency: this.formCurrency(),
        benchmark_ticker: this.formBenchmark(),
      })
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (created) => this.handleCreated(created),
        error: (err: Error) => this.handleCreateError(err.message ?? 'Create failed'),
      });
  }

  saveSnapshot(): void {
    const portfolio = this.activePortfolio();
    if (!portfolio || this.snapshotStatus() === 'saving') return;
    this.snapshotStatus.set('saving');
    this.snapshotError.set(null);
    this.portfolioApi
      .createSnapshot(portfolio.name, {
        snapshot_date: this.todayIso(),
        snapshot_type: 'manual',
        weights: {},
      })
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: () => this.snapshotStatus.set('success'),
        error: (err: Error) => {
          this.snapshotError.set(err.message ?? 'Snapshot failed');
          this.snapshotStatus.set('error');
        },
      });
  }

  onTickerSelected(instrument: Instrument): void {
    this.flyoutInstrumentId.set(instrument.id);
  }

  closeFlyout(): void {
    this.flyoutInstrumentId.set(null);
  }

  setActivePortfolio(portfolio: PortfolioDto): void {
    this.activePortfolio.set(portfolio);
  }

  onActiveSelect(name: string): void {
    const match = this.portfolios().find((p) => p.name === name);
    if (match) this.activePortfolio.set(match);
  }

  readonly defaultIps: IPS = DEFAULT_IPS;

  noopConstraints(_constraints: Constraint[]): void {
    // Weights panel is deferred to downstream issues.
  }

  noopIps(_ips: IPS): void {
    // IPS panel is deferred to downstream issues.
  }

  private handleCreated(created: PortfolioDto): void {
    this.portfolios.update((list) => [...list, created]);
    this.activePortfolio.set(created);
    this.formStatus.set('success');
    this.showCreateModal.set(false);
  }

  private handleCreateError(message: string): void {
    this.formError.set(message);
    this.formStatus.set('error');
  }

  private resetForm(): void {
    this.formName.set('');
    this.formDescription.set('');
    this.formCurrency.set('USD');
    this.formBenchmark.set(DEFAULT_BENCHMARK);
    this.formStatus.set('idle');
    this.formError.set(null);
  }

  private todayIso(): string {
    return new Date().toISOString().slice(0, 10);
  }
}
