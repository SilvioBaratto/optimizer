import {
  Component,
  signal,
  inject,
  computed,
  ChangeDetectionStrategy,
  DestroyRef,
} from '@angular/core';
import { LucideAngularModule } from 'lucide-angular';
import { takeUntilDestroyed, toObservable } from '@angular/core/rxjs-interop';
import { switchMap } from 'rxjs';
import { ModalService } from '../shared/modal/modal.service';
import { ExportReportModalComponent } from '../shared/modal/export-report-modal';
import { PageHeaderComponent } from '../shared/components/page-header/page-header';
import { OptimizerPanelComponent } from './optimizer-panel/optimizer-panel';
import { OptimizerRunRequest } from './optimizer-panel/optimizer-panel';
import { ResultsPanelComponent } from './results-panel/results-panel';
import { JobProgressTrackerComponent } from '../shared/job-progress-tracker/job-progress-tracker';
import {
  OptimizationService,
  isAcceptedOptimizerType,
  type OptimizeResult,
} from './optimization.service';
import { FormatService } from '../core/services/format.service';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { TickerSeedingService } from '../core/services/ticker-seeding.service';
import type { OptimizationRunResponse } from '../core/models/optimization.model';

type ApplyStatus = 'idle' | 'saving' | 'success' | 'error';

function arraysEqual(a: string[], b: string[]): boolean {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

const DEFAULT_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'JPM', 'V'];

@Component({
  selector: 'app-optimization-studio',
  imports: [
    LucideAngularModule,
    PageHeaderComponent,
    OptimizerPanelComponent,
    ResultsPanelComponent,
    JobProgressTrackerComponent,
  ],
  templateUrl: './optimization-studio.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class OptimizationStudioComponent {
  private readonly modalService = inject(ModalService);
  private readonly optimization = inject(OptimizationService);
  private readonly fmt = inject(FormatService);
  private readonly portfolioContext = inject(PortfolioContextService);
  private readonly tickerSeeding = inject(TickerSeedingService);
  private readonly destroyRef = inject(DestroyRef);

  readonly isLoading = signal(false);
  readonly hasError = signal(false);
  readonly errorMessage = signal('');

  readonly isRunning = signal(false);
  readonly runJobId = signal<string | null>(null);
  readonly runRunId = signal<string | null>(null);
  readonly runResult = signal<OptimizationRunResponse | null>(null);
  readonly runError = signal<string | null>(null);

  readonly tickers = signal<string[]>(DEFAULT_TICKERS);
  private readonly lastSeed = signal<string[]>([]);
  readonly startDate = signal<string>(this.fmt.defaultStartIso());
  readonly endDate = signal<string>(this.fmt.todayIso());

  readonly applyStatus = signal<ApplyStatus>('idle');
  readonly applyError = signal<string | null>(null);
  readonly appliedPortfolioName = signal<string | null>(null);

  readonly hasResult = computed(() => this.runResult() !== null);
  readonly canSave = computed(() => this.portfolioContext.hasPortfolio() && this.hasResult());
  readonly isPolling = computed(() => this.runJobId() !== null && !this.hasResult());

  constructor() {
    toObservable(this.portfolioContext.currentPortfolioName)
      .pipe(
        switchMap((name) =>
          this.tickerSeeding.seedFromPortfolio(name, DEFAULT_TICKERS),
        ),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe((seeded) => {
        const current = this.tickers();
        const last = this.lastSeed();
        if (last.length === 0 || arraysEqual(current, last)) {
          this.tickers.set(seeded);
          this.lastSeed.set(seeded);
        }
      });
  }

  loadData(): void {
    this.hasError.set(false);
    this.isLoading.set(false);
  }

  retry(): void {
    this.runError.set(null);
    this.loadData();
  }

  onRunPipeline(request: OptimizerRunRequest): void {
    if (this.isRunning()) return;
    if (!isAcceptedOptimizerType(request.optimizerType)) {
      this.handleRunError(`Unsupported optimizer type: ${request.optimizerType}`);
      return;
    }
    this.beginRun();
    this.submitRun(request);
  }

  onJobCompleted(runId?: string): void {
    const id = runId ?? this.runRunId();
    if (!id) return this.handleRunError('Optimization run id unavailable');
    this.fetchRun(id);
  }

  onJobFailed(message: string): void {
    this.handleRunError(message || 'Job failed');
  }

  onApplyWeights(weights: Record<string, number>): void {
    this.applyError.set(null);
    this.appliedPortfolioName.set(null);
    const portfolioRef = this.portfolioContext.currentPortfolioId();
    if (!portfolioRef) {
      this.applyStatus.set('error');
      this.applyError.set('No active portfolio selected. Pick a portfolio in the sidebar.');
      return;
    }
    this.applyStatus.set('saving');
    this.optimization
      .applyWeightsToPortfolio(portfolioRef, weights, this.fmt.todayIso())
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (name) => {
          this.appliedPortfolioName.set(name);
          this.applyStatus.set('success');
        },
        error: (err: Error) => {
          this.applyStatus.set('error');
          this.applyError.set(err.message ?? 'Snapshot failed');
        },
      });
  }

  openReportModal(): void {
    this.modalService.open({
      component: ExportReportModalComponent,
      title: 'Export Report',
      size: 'lg',
    });
  }

  private submitRun(request: OptimizerRunRequest): void {
    const body = this.optimization.buildOptimizeBody(
      request, this.tickers(), this.startDate(), this.endDate(),
    );
    this.optimization
      .optimize(body)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => this.handleOptimizeResponse(res),
        error: (err: Error) => this.handleRunError(err.message ?? 'Optimization failed'),
      });
  }

  private fetchRun(id: string): void {
    this.optimization
      .getOptimizationRun(id)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (run) => this.completeWithRun(run),
        error: (err: Error) => this.handleRunError(err.message ?? 'Fetch failed'),
      });
  }

  private beginRun(): void {
    this.runError.set(null);
    this.runResult.set(null);
    this.runJobId.set(null);
    this.runRunId.set(null);
    this.isRunning.set(true);
  }

  private handleOptimizeResponse(res: OptimizeResult): void {
    if (OptimizationService.isAsyncResponse(res)) {
      this.runJobId.set(res.job_id);
      this.runRunId.set(res.run_id);
      return;
    }
    this.completeWithRun(res);
  }

  private completeWithRun(run: OptimizationRunResponse): void {
    this.runResult.set(run);
    this.runJobId.set(null);
    this.isRunning.set(false);
  }

  private handleRunError(message: string): void {
    this.runError.set(message);
    this.isRunning.set(false);
    this.runJobId.set(null);
  }
}
