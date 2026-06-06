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

import { ModalService } from '../../shared/modal/modal.service';
import { ExportReportModalComponent } from '../../shared/modal/export-report-modal';
import { PageHeaderComponent } from '../../shared/components/page-header/page-header';
import { TabGroupComponent, Tab } from '../../shared/components/tab-group/tab-group';
import { StatCardComponent } from '../../shared/stat-card/stat-card';
import { FormatService } from '../../core/services/format.service';
import { AttributionService } from '../../services/attribution.service';
import { PortfolioApiService } from '../../core/services/portfolio-api.service';

import { BrinsonPanelComponent } from './brinson-panel';
import { SaaTaaPanelComponent } from './saa-taa-panel';
import { FactorAttributionPanelComponent } from './factor-attribution-panel';
import { HoldingsAttributionPanelComponent } from './holdings-attribution-panel';

import type {
  BrinsonApiResponse,
  FactorAttributionApiResponse,
} from '../../models/attribution.model';
import type { PortfolioDto } from '../../core/models/portfolio-api.model';

@Component({
  selector: 'app-attribution',
  imports: [
    FormsModule,
    LucideAngularModule,
    PageHeaderComponent,
    TabGroupComponent,
    StatCardComponent,
    BrinsonPanelComponent,
    SaaTaaPanelComponent,
    FactorAttributionPanelComponent,
    HoldingsAttributionPanelComponent,
  ],
  templateUrl: './attribution.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class AttributionComponent {
  private readonly fmt = inject(FormatService);
  private readonly modalService = inject(ModalService);
  private readonly attribution = inject(AttributionService);
  private readonly portfolioApi = inject(PortfolioApiService);
  private readonly destroyRef = inject(DestroyRef);

  readonly isLoading = signal<boolean>(false);
  readonly hasError = signal<boolean>(false);
  readonly errorMessage = signal<string>('');

  readonly activeTab = signal<string>('brinson');

  readonly portfolios = signal<PortfolioDto[]>([]);
  readonly selectedPortfolio = signal<string>('');
  readonly benchmarkWeightsRaw = signal<string>('SPY:1.0');
  readonly startDate = signal<string>(this.defaultStart());
  readonly endDate = signal<string>(this.todayIso());

  readonly portfolioWeights = signal<Record<string, number>>({});

  readonly brinsonResponse = signal<BrinsonApiResponse | null>(null);
  readonly factorResponse = signal<FactorAttributionApiResponse | null>(null);

  readonly brinsonLoading = signal<boolean>(false);
  readonly factorLoading = signal<boolean>(false);
  readonly brinsonError = signal<string | null>(null);
  readonly factorError = signal<string | null>(null);

  readonly tabs: Tab[] = [
    { id: 'brinson', label: 'Brinson-Fachler' },
    { id: 'multi-level', label: 'Multi-Level' },
    { id: 'factor', label: 'Factor Attribution' },
    { id: 'holdings', label: 'Holdings' },
  ];

  readonly kpiActiveReturn = computed(() =>
    this.brinsonResponse()
      ? this.fmt.formatPercent(this.brinsonResponse()!.totalActiveReturn)
      : '—',
  );
  readonly kpiAllocation = computed(() =>
    this.brinsonResponse()
      ? this.fmt.formatPercent(this.brinsonResponse()!.totalAllocation)
      : '—',
  );
  readonly kpiSelection = computed(() =>
    this.brinsonResponse()
      ? this.fmt.formatPercent(this.brinsonResponse()!.totalSelection)
      : '—',
  );
  readonly kpiResidual = computed(() =>
    this.factorResponse() ? this.fmt.formatPercent(this.factorResponse()!.residual) : '—',
  );

  readonly benchmarkWeights = computed<Record<string, number>>(() =>
    this.parseWeights(this.benchmarkWeightsRaw()),
  );

  readonly isFormValid = computed(() => {
    const pw = this.portfolioWeights();
    const bw = this.benchmarkWeights();
    if (!this.sumsToOne(pw) || !this.sumsToOne(bw)) return false;
    return this.startDate() < this.endDate();
  });

  readonly isFactorFormValid = computed(() => {
    if (!this.sumsToOne(this.portfolioWeights())) return false;
    return this.startDate() < this.endDate();
  });

  /**
   * BUG-045 guard: when the entire benchmark resolves to a single
   * "Unclassified" sector (typical with single-ETF benchmarks like
   * `SPY:1.0`), the per-sector Brinson decomposition assigns the entire
   * benchmark return to that pseudo-sector and the resulting Allocation /
   * Selection effects are mathematically valid but semantically misleading.
   * Detect that and surface a warning so the user can choose a benchmark
   * with proper sector composition.
   */
  readonly benchAllUnclassified = computed<boolean>(() => {
    const r = this.brinsonResponse();
    if (!r) return false;
    const total = r.sectors.reduce((acc, s) => acc + s.benchmarkWeight, 0);
    if (total <= 0) return false;
    const unclassified = r.sectors
      .filter((s) => s.sector === 'Unclassified')
      .reduce((acc, s) => acc + s.benchmarkWeight, 0);
    return unclassified / total > 0.95;
  });

  constructor() {
    this.loadPortfolios();
    effect(() => this.onPortfolioChange());
  }

  retry(): void {
    this.hasError.set(false);
    this.loadPortfolios();
  }

  openReportModal(): void {
    this.modalService.open({
      component: ExportReportModalComponent,
      title: 'Export Report',
      size: 'lg',
    });
  }

  onPortfolioSelect(name: string): void {
    this.selectedPortfolio.set(name);
  }

  runBrinson(): void {
    if (!this.isFormValid()) return;
    this.brinsonLoading.set(true);
    this.brinsonError.set(null);
    this.attribution
      .brinson({
        portfolio_weights: this.portfolioWeights(),
        benchmark_weights: this.benchmarkWeights(),
        start_date: this.startDate(),
        end_date: this.endDate(),
      })
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => this.onBrinsonSuccess(res),
        error: (err: Error) => this.onBrinsonError(err.message ?? 'Brinson failed'),
      });
  }

  runFactor(): void {
    if (!this.sumsToOne(this.portfolioWeights())) return;
    if (!(this.startDate() < this.endDate())) return;
    this.factorLoading.set(true);
    this.factorError.set(null);
    this.attribution
      .factor({
        portfolio_weights: this.portfolioWeights(),
        start_date: this.startDate(),
        end_date: this.endDate(),
      })
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => this.onFactorSuccess(res),
        error: (err: Error) => this.onFactorError(err.message ?? 'Factor failed'),
      });
  }

  private onBrinsonSuccess(res: BrinsonApiResponse): void {
    this.brinsonResponse.set(res);
    this.brinsonLoading.set(false);
  }

  private onBrinsonError(message: string): void {
    this.brinsonError.set(message);
    this.brinsonLoading.set(false);
  }

  private onFactorSuccess(res: FactorAttributionApiResponse): void {
    this.factorResponse.set(res);
    this.factorLoading.set(false);
  }

  private onFactorError(message: string): void {
    this.factorError.set(message);
    this.factorLoading.set(false);
  }

  private loadPortfolios(): void {
    this.isLoading.set(true);
    this.portfolioApi
      .list()
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (list) => this.applyPortfolios(list.items),
        error: (err: Error) => this.failLoad(err.message ?? 'Failed to load portfolios'),
      });
  }

  private applyPortfolios(items: PortfolioDto[]): void {
    this.portfolios.set(items);
    if (items.length > 0 && !this.selectedPortfolio()) {
      this.selectedPortfolio.set(items[0].name);
    }
    this.isLoading.set(false);
  }

  private failLoad(message: string): void {
    this.errorMessage.set(message);
    this.hasError.set(true);
    this.isLoading.set(false);
  }

  private onPortfolioChange(): void {
    const name = this.selectedPortfolio();
    if (!name) return;
    this.portfolioApi
      .getLatestSnapshot(name)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (snap) => this.portfolioWeights.set(snap.weights ?? {}),
        error: () => this.portfolioWeights.set({}),
      });
  }

  private parseWeights(raw: string): Record<string, number> {
    const result: Record<string, number> = {};
    for (const pair of raw.split(',')) {
      const [key, value] = pair.split(':').map((s) => s.trim());
      const num = Number(value);
      if (key && Number.isFinite(num)) result[key.toUpperCase()] = num;
    }
    return result;
  }

  private sumsToOne(weights: Record<string, number>): boolean {
    if (Object.keys(weights).length === 0) return false;
    const sum = Object.values(weights).reduce((a, b) => a + b, 0);
    return Math.abs(sum - 1) < 0.01;
  }

  private defaultStart(): string {
    const d = new Date();
    d.setFullYear(d.getFullYear() - 1);
    return d.toISOString().slice(0, 10);
  }

  private todayIso(): string {
    return new Date().toISOString().slice(0, 10);
  }
}
