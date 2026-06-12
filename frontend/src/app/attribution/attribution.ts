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

import { ModalService } from '../shared/modal/modal.service';
import { ExportReportModalComponent } from '../shared/modal/export-report-modal';
import { PageHeaderComponent } from '../shared/components/page-header/page-header';
import { TabGroupComponent, Tab } from '../shared/components/tab-group/tab-group';
import { StatCardComponent } from '../shared/stat-card/stat-card';
import { PageErrorBannerComponent } from '../shared/components/page-error-banner/page-error-banner';
import { FormatService } from '../core/services/format.service';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { AttributionService } from './attribution.service';

import { BrinsonPanelComponent } from './brinson-panel/brinson-panel';
import { SaaTaaPanelComponent } from './saa-taa-panel/saa-taa-panel';
import { FactorAttributionPanelComponent } from './factor-attribution-panel/factor-attribution-panel';
import { HoldingsAttributionPanelComponent } from './holdings-attribution-panel/holdings-attribution-panel';

import type {
  BrinsonApiResponse,
  FactorAttributionApiResponse,
} from './attribution.model';
import type { PortfolioDto } from '../core/models/portfolio-api.model';

@Component({
  selector: 'app-attribution',
  imports: [
    FormsModule,
    LucideAngularModule,
    PageHeaderComponent,
    TabGroupComponent,
    StatCardComponent,
    PageErrorBannerComponent,
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
  private readonly portfolioCtx = inject(PortfolioContextService);
  private readonly destroyRef = inject(DestroyRef);

  // ── Portfolio context ─────────────────────────────────────────────────────
  readonly portfolioName = computed(
    () => this.portfolioCtx.currentPortfolioName() ?? this.portfolioCtx.currentPortfolioId(),
  );

  // ── Global loading / error state ──────────────────────────────────────────
  readonly isLoading = signal<boolean>(false);
  readonly hasError = signal<boolean>(false);
  readonly errorMessage = signal<string>('');

  // ── Tab navigation ────────────────────────────────────────────────────────
  readonly activeTab = signal<string>('brinson');

  // ── Legacy manual-entry signals (kept for render-coverage spec) ───────────
  /** Kept for backward compat with render-coverage spec and manual form flow. */
  readonly portfolios = signal<PortfolioDto[]>([]);
  readonly selectedPortfolio = signal<string>('');
  readonly benchmarkWeightsRaw = signal<string>('SPY:1.0');
  readonly startDate = signal<string>(this.defaultStart());
  readonly endDate = signal<string>(this.todayIso());
  readonly portfolioWeights = signal<Record<string, number>>({});

  // ── Panel data ────────────────────────────────────────────────────────────
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

  // ── KPI computed views ────────────────────────────────────────────────────
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
    if (!r || !r.sectors?.length) return false;
    const total = r.sectors.reduce((acc, s) => acc + s.benchmarkWeight, 0);
    if (total <= 0) return false;
    const unclassified = r.sectors
      .filter((s) => s.sector === 'Unclassified')
      .reduce((acc, s) => acc + s.benchmarkWeight, 0);
    return unclassified / total > 0.95;
  });

  constructor() {
    // React to portfolio context changes and refetch attribution data.
    effect(() => this.onPortfolioContextChange());
  }

  // ── Public actions ────────────────────────────────────────────────────────

  retry(): void {
    this.hasError.set(false);
    this.errorMessage.set('');
    const name = this.portfolioName();
    if (name) this.loadAttributionData(name);
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

  // ── Portfolio context wiring ──────────────────────────────────────────────

  private onPortfolioContextChange(): void {
    const name = this.portfolioName();
    if (!name) return;
    this.loadAttributionData(name);
  }

  private loadAttributionData(portfolioName: string): void {
    this.hasError.set(false);
    this.errorMessage.set('');

    this.attribution
      .getBrinsonAttribution(portfolioName, this.startDate(), this.endDate())
      .pipe(
        catchError((err: Error) => {
          this.hasError.set(true);
          this.errorMessage.set(err?.message ?? 'Failed to load attribution data');
          return EMPTY;
        }),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe((res) => this.brinsonResponse.set(res));

    this.attribution
      .getFactorAttribution(portfolioName, this.startDate(), this.endDate())
      .pipe(
        catchError(() => EMPTY),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe((res) => this.factorResponse.set(res));
  }

  // ── Private panel handlers ────────────────────────────────────────────────

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

  // ── Utilities ─────────────────────────────────────────────────────────────

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
