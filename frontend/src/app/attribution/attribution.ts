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
import { HttpErrorResponse } from '@angular/common/http';
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
import { PortfolioApiService } from '../core/services/portfolio-api.service';
import { AttributionService } from './attribution.service';

import { BrinsonPanelComponent } from './brinson-panel/brinson-panel';
import { SaaTaaPanelComponent } from './saa-taa-panel/saa-taa-panel';
import { FactorAttributionPanelComponent } from './factor-attribution-panel/factor-attribution-panel';
import { HoldingsAttributionPanelComponent } from './holdings-attribution-panel/holdings-attribution-panel';

import type {
  BrinsonApiResponse,
  FactorAttributionApiResponse,
} from './attribution.model';

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
  private readonly portfolioApi = inject(PortfolioApiService);
  private readonly destroyRef = inject(DestroyRef);

  // ── Portfolio context ─────────────────────────────────────────────────────
  // Wait for the resolved name — never fall through to the raw UUID.
  // During cold-boot (id restored, name still resolving), portfolioName() is
  // null and the effect returns early, preventing a UUID-keyed snapshot call.
  readonly portfolioName = computed(() => this.portfolioCtx.currentPortfolioName());
  readonly selectedPortfolio = computed(() => this.portfolioCtx.selectedPortfolio());

  // ── Global loading / error state ──────────────────────────────────────────
  readonly isLoading = signal<boolean>(false);
  readonly hasError = signal<boolean>(false);
  readonly errorMessage = signal<string>('');

  // ── Tab navigation ────────────────────────────────────────────────────────
  readonly activeTab = signal<string>('brinson');

  // ── Manual-entry signals (benchmark + date range for the run-attribution form)
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
  /** Set when the factor endpoint returns 404 — drives the panel's empty state. */
  readonly factorNotAvailable = signal<boolean>(false);

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
    this.resetFactorErrorState();
    this.attribution
      .factor({
        portfolio_weights: this.portfolioWeights(),
        start_date: this.startDate(),
        end_date: this.endDate(),
      })
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => this.onFactorSuccess(res),
        error: (err: unknown) => this.onFactorError(err),
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

    const portfolio = this.selectedPortfolio();
    const benchmarkTicker = portfolio?.benchmark_ticker ?? 'SPY';
    const benchmarkWeights: Record<string, number> = { [benchmarkTicker]: 1.0 };
    const startDate = this.startDate();
    const endDate = this.endDate();

    this.portfolioApi
      .getLatestSnapshot(portfolioName)
      .pipe(
        takeUntilDestroyed(this.destroyRef),
        catchError((err: Error) => {
          this.hasError.set(true);
          this.errorMessage.set(err?.message ?? 'Failed to load portfolio snapshot');
          return EMPTY;
        }),
      )
      .subscribe((snapshot) => {
        const weights = snapshot.weights;
        this.portfolioWeights.set(weights);

        this.attribution
          .brinson({ portfolio_weights: weights, benchmark_weights: benchmarkWeights, start_date: startDate, end_date: endDate })
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
          .factor({ portfolio_weights: weights, start_date: startDate, end_date: endDate })
          .pipe(takeUntilDestroyed(this.destroyRef))
          .subscribe({
            next: (res) => this.onFactorSuccess(res),
            error: (err: unknown) => this.onFactorError(err),
          });
      });
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
    this.resetFactorErrorState();
  }

  /**
   * Classify a factor() failure: a 404 means "no factor scores in the DB" and
   * routes to the panel's inline empty state; any other status stays a visible
   * error (issue #997, 13c).
   */
  private onFactorError(err: unknown): void {
    this.factorLoading.set(false);
    if (err instanceof HttpErrorResponse && err.status === 404) {
      this.factorNotAvailable.set(true);
      this.factorError.set(null);
      return;
    }
    this.factorNotAvailable.set(false);
    this.factorError.set(this.errorMessageOf(err));
  }

  private resetFactorErrorState(): void {
    this.factorError.set(null);
    this.factorNotAvailable.set(false);
  }

  private errorMessageOf(err: unknown): string {
    if (err instanceof HttpErrorResponse) return err.message;
    if (err instanceof Error) return err.message;
    return 'Factor failed';
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
