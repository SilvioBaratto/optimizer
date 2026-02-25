import { Component, signal, inject, ChangeDetectionStrategy } from '@angular/core';
import { ModalService } from '../../shared/modal/modal.service';
import { ExportReportModalComponent } from '../../shared/modal/export-report-modal';
import { PageHeaderComponent } from '../../shared/components/page-header/page-header';
import { ProgressBarComponent } from '../../shared/progress-bar/progress-bar';
import { PipelineBuilderComponent } from './pipeline-builder';
import { PreprocessingPanelComponent } from './preprocessing-panel';
import { MomentPanelComponent } from './moment-panel';
import { ViewPanelComponent } from './view-panel';
import { OptimizerPanelComponent } from './optimizer-panel';
import { ResultsPanelComponent } from './results-panel';
import { MockFetchService } from '../../services/mock-fetch.service';
import { PipelineNode } from '../../models/optimization.model';
import { MOCK_PIPELINE_NODES } from '../../mocks/optimization-mocks';

@Component({
  selector: 'app-optimization-studio',
  imports: [
    PageHeaderComponent,
    ProgressBarComponent,
    PipelineBuilderComponent,
    PreprocessingPanelComponent,
    MomentPanelComponent,
    ViewPanelComponent,
    OptimizerPanelComponent,
    ResultsPanelComponent,
  ],
  templateUrl: './optimization-studio.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class OptimizationStudioComponent {
  private readonly modalService = inject(ModalService);
  private readonly mockFetch = inject(MockFetchService);

  readonly isLoading = signal(true);
  readonly hasError = signal(false);
  readonly errorMessage = signal('');

  activeNode = signal<string | null>(null);
  pipelineNodes = signal<PipelineNode[]>(MOCK_PIPELINE_NODES);
  isRunning = signal(false);
  runProgress = signal(0);

  constructor() {
    this.loadData();
  }

  loadData(): void {
    this.isLoading.set(true);
    this.hasError.set(false);

    this.mockFetch.fetch({ nodes: MOCK_PIPELINE_NODES }).then(data => {
      this.pipelineNodes.set(data.nodes);
      this.isLoading.set(false);
    }).catch((err: Error) => {
      this.hasError.set(true);
      this.errorMessage.set(err.message);
      this.isLoading.set(false);
    });
  }

  retry(): void {
    this.loadData();
  }

  onNodeSelect(nodeId: string): void {
    this.activeNode.update(current => (current === nodeId ? null : nodeId));
  }

  onRunPipeline(): void {
    this.isRunning.set(true);
    this.runProgress.set(0);

    const steps = [10, 25, 45, 65, 80, 100];
    let stepIndex = 0;

    const advance = () => {
      if (stepIndex < steps.length) {
        this.runProgress.set(steps[stepIndex]);
        stepIndex++;
        setTimeout(advance, 400);
      } else {
        this.isRunning.set(false);
      }
    };

    setTimeout(advance, 200);
  }

  openReportModal(): void {
    this.modalService.open({ component: ExportReportModalComponent, title: 'Export Report', size: 'lg' });
  }
}
