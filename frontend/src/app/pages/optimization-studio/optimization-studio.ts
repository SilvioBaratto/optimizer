import {
  Component,
  signal,
  inject,
  computed,
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
import type {
  OptimizationRunResponse,
  OptimizeRequest,
  PipelineNode,
} from '../../models/optimization.model';

const DEFAULT_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'JPM', 'V'];

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

  readonly hasResult = computed(() => this.runResult() !== null);
  readonly isPolling = computed(() => this.runJobId() !== null && !this.hasResult());

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

  private beginRun(): void {
    this.runError.set(null);
    this.runResult.set(null);
    this.runJobId.set(null);
    this.isRunning.set(true);
    this.markNodesStatus('running');
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
    this.markNodesStatus('completed');
  }

  private handleRunError(message: string): void {
    this.runError.set(message);
    this.isRunning.set(false);
    this.runJobId.set(null);
    this.markNodesStatus('error');
  }

  private markNodesStatus(status: PipelineNode['status']): void {
    this.pipelineNodes.update((nodes) => nodes.map((n) => ({ ...n, status })));
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
