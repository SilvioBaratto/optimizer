import {
  Component,
  signal,
  inject,
  computed,
  effect,
  ChangeDetectionStrategy,
  DestroyRef,
} from '@angular/core';
import { LucideAngularModule } from 'lucide-angular';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { ModalService } from '../../shared/modal/modal.service';
import { ExportReportModalComponent } from '../../shared/modal/export-report-modal';
import { PageHeaderComponent } from '../../shared/components/page-header/page-header';
import { PipelineBuilderComponent } from './pipeline-builder';
import { PreprocessingPanelComponent } from './preprocessing-panel';
import { MomentPanelComponent } from './moment-panel';
import { ViewPanelComponent } from './view-panel';
import { OptimizerPanelComponent } from './optimizer-panel';
import { OptimizerRunRequest } from './optimizer-panel';
import { ResultsPanelComponent } from './results-panel';
import { JobProgressTrackerComponent } from '../../shared/job-progress-tracker/job-progress-tracker';
import {
  OptimizationService,
  type OptimizeResult,
} from '../../services/optimization.service';
import { PortfolioApiService } from '../../services/portfolio-api.service';
import { PortfolioContextService } from '../../services/portfolio-context.service';
import type {
  OptimizationRunResponse,
  OptimizeRequest,
  PipelineNode,
} from '../../models/optimization.model';

const PIPELINE_STORAGE_KEY = 'optimizer.savedPipeline';

interface SavedPipeline {
  tickers: string[];
  startDate: string;
  endDate: string;
  savedAt: string;
}

type ApplyStatus = 'idle' | 'saving' | 'success' | 'error';

const DEFAULT_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'JPM', 'V'];

const CONFIG_NODE_IDS = new Set(['p2', 'p3', 'p4']);
const RUN_NODE_IDS = new Set(['p5', 'p6']);

const INITIAL_PIPELINE_NODES: PipelineNode[] = [
  { id: 'p2', label: 'Pre-Selection', status: 'pending', detail: 'Configure before running' },
  { id: 'p3', label: 'Moment Estimation', status: 'pending', detail: 'mu / covariance estimators' },
  { id: 'p4', label: 'View Formation', status: 'pending', detail: 'Black-Litterman / pooling' },
  { id: 'p5', label: 'Optimization', status: 'pending', detail: 'Objective & risk measure' },
  { id: 'p6', label: 'Results', status: 'pending', detail: 'Frontier + weights' },
];

@Component({
  selector: 'app-optimization-studio',
  imports: [
    LucideAngularModule,
    PageHeaderComponent,
    PipelineBuilderComponent,
    PreprocessingPanelComponent,
    MomentPanelComponent,
    ViewPanelComponent,
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
  private readonly portfolioApi = inject(PortfolioApiService);
  private readonly portfolioContext = inject(PortfolioContextService);
  private readonly destroyRef = inject(DestroyRef);

  readonly isLoading = signal(false);
  readonly hasError = signal(false);
  readonly errorMessage = signal('');

  readonly activeNode = signal<string | null>(null);
  readonly pipelineNodes = signal<PipelineNode[]>(INITIAL_PIPELINE_NODES);
  readonly isRunning = signal(false);
  readonly runJobId = signal<string | null>(null);
  readonly runResult = signal<OptimizationRunResponse | null>(null);
  readonly runError = signal<string | null>(null);

  readonly tickers = signal<string[]>(DEFAULT_TICKERS);
  readonly startDate = signal<string>(this.defaultStart());
  readonly endDate = signal<string>(this.todayIso());

  readonly applyStatus = signal<ApplyStatus>('idle');
  readonly applyError = signal<string | null>(null);
  readonly appliedPortfolioName = signal<string | null>(null);
  readonly pipelineStatus = signal<string | null>(null);

  readonly hasResult = computed(() => this.runResult() !== null);
  readonly isPolling = computed(() => this.runJobId() !== null && !this.hasResult());

  constructor() {
    // Sync optimization date range with the global PortfolioContextService.
    // Header presets (1Y, 3Y, etc.) update dateRange(); here we propagate to
    // the local startDate/endDate signals so /optimize uses the chosen window.
    effect(() => {
      const range = this.portfolioContext.dateRange();
      this.startDate.set(this.toIso(range.start));
      this.endDate.set(this.toIso(range.end));
    });
  }

  private toIso(d: Date): string {
    return d.toISOString().slice(0, 10);
  }

  loadData(): void {
    this.hasError.set(false);
    this.isLoading.set(false);
  }

  retry(): void {
    this.runError.set(null);
    this.loadData();
  }

  onNodeSelect(nodeId: string): void {
    this.activeNode.update((current) => (current === nodeId ? null : nodeId));
  }

  onRunPipeline(request: OptimizerRunRequest): void {
    if (this.isRunning()) return;
    this.beginRun();
    const body = this.buildOptimizeBody(request);
    this.optimization
      .optimize(body)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => this.handleOptimizeResponse(res),
        error: (err: Error) => this.handleRunError(err.message ?? 'Optimization failed'),
      });
  }

  onJobCompleted(runId: string): void {
    this.optimization
      .getOptimizationRun(runId)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (run) => this.completeWithRun(run),
        error: (err: Error) => this.handleRunError(err.message ?? 'Fetch failed'),
      });
  }

  onJobFailed(message: string): void {
    this.handleRunError(message || 'Job failed');
  }

  openReportModal(): void {
    this.modalService.open({
      component: ExportReportModalComponent,
      title: 'Export Report',
      size: 'lg',
    });
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
    this.portfolioApi
      .list()
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (list) => {
          const target = list.items.find(
            (p) => p.id === portfolioRef || p.name === portfolioRef,
          );
          if (!target) {
            this.applyStatus.set('error');
            this.applyError.set(`Portfolio ${portfolioRef} not found.`);
            return;
          }
          this.persistSnapshot(target.name, weights);
        },
        error: (err: Error) => {
          this.applyStatus.set('error');
          this.applyError.set(err.message ?? 'Failed to load portfolios');
        },
      });
  }

  loadPipeline(): void {
    this.pipelineStatus.set(null);
    if (typeof localStorage === 'undefined') return;
    let raw: string | null = null;
    try {
      raw = localStorage.getItem(PIPELINE_STORAGE_KEY);
    } catch {
      this.pipelineStatus.set('Local storage unavailable.');
      return;
    }
    if (!raw) {
      this.pipelineStatus.set('No saved pipeline found.');
      return;
    }
    try {
      const parsed = JSON.parse(raw) as SavedPipeline;
      if (Array.isArray(parsed.tickers)) this.tickers.set([...parsed.tickers]);
      if (typeof parsed.startDate === 'string') this.startDate.set(parsed.startDate);
      if (typeof parsed.endDate === 'string') this.endDate.set(parsed.endDate);
      this.pipelineStatus.set(`Pipeline loaded (saved ${parsed.savedAt ?? 'unknown'}).`);
    } catch {
      this.pipelineStatus.set('Saved pipeline is corrupt.');
    }
  }

  savePipeline(): void {
    this.pipelineStatus.set(null);
    if (typeof localStorage === 'undefined') {
      this.pipelineStatus.set('Local storage unavailable.');
      return;
    }
    const payload: SavedPipeline = {
      tickers: this.tickers(),
      startDate: this.startDate(),
      endDate: this.endDate(),
      savedAt: new Date().toISOString(),
    };
    try {
      localStorage.setItem(PIPELINE_STORAGE_KEY, JSON.stringify(payload));
      this.pipelineStatus.set('Pipeline saved.');
    } catch {
      this.pipelineStatus.set('Failed to save pipeline.');
    }
  }

  private persistSnapshot(portfolioName: string, weights: Record<string, number>): void {
    this.portfolioApi
      .createSnapshot(portfolioName, {
        snapshot_date: this.todayIso(),
        snapshot_type: 'optimization',
        weights,
      })
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: () => {
          this.appliedPortfolioName.set(portfolioName);
          this.applyStatus.set('success');
        },
        error: (err: Error) => {
          this.applyStatus.set('error');
          this.applyError.set(err.message ?? 'Snapshot failed');
        },
      });
  }

  private beginRun(): void {
    this.runError.set(null);
    this.runResult.set(null);
    this.runJobId.set(null);
    this.isRunning.set(true);
    this.updateRunNodeStatus('running');
  }

  private buildOptimizeBody(request: OptimizerRunRequest): OptimizeRequest {
    return {
      tickers: this.tickers(),
      start_date: this.startDate(),
      end_date: this.endDate(),
      optimizer_type: request.optimizerType,
      config: request.config,
    };
  }

  private handleOptimizeResponse(res: OptimizeResult): void {
    if (OptimizationService.isAsyncResponse(res)) {
      this.runJobId.set(res.job_id);
      return;
    }
    this.completeWithRun(res);
  }

  private completeWithRun(run: OptimizationRunResponse): void {
    this.runResult.set(run);
    this.runJobId.set(null);
    this.isRunning.set(false);
    this.updateRunNodeStatus('completed');
  }

  private handleRunError(message: string): void {
    this.runError.set(message);
    this.isRunning.set(false);
    this.runJobId.set(null);
    this.updateRunNodeStatus('error');
  }

  private updateRunNodeStatus(status: PipelineNode['status']): void {
    // Only Optimization (p5) and Results (p6) reflect actual execution state.
    // Config nodes (Pre-Selection, Moment Estimation, View Formation) stay
    // 'pending' because they are user-configured, not automatically run.
    this.pipelineNodes.update((nodes) =>
      nodes.map((n) => {
        if (RUN_NODE_IDS.has(n.id)) return { ...n, status };
        if (CONFIG_NODE_IDS.has(n.id)) {
          return { ...n, status: 'pending' as const };
        }
        return n;
      }),
    );
  }

  private defaultStart(): string {
    const d = new Date();
    d.setFullYear(d.getFullYear() - 3);
    return d.toISOString().slice(0, 10);
  }

  private todayIso(): string {
    return new Date().toISOString().slice(0, 10);
  }
}
