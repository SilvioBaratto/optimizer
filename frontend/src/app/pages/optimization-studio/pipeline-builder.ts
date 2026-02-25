import { Component, input, output, computed, ChangeDetectionStrategy } from '@angular/core';
import { PipelineNode, PipelineNodeStatus } from '../../models/optimization.model';

@Component({
  selector: 'app-pipeline-builder',
  templateUrl: './pipeline-builder.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class PipelineBuilderComponent {
  nodes = input<PipelineNode[]>([]);
  activeNode = input<string | null>(null);

  nodeSelect = output<string>();

  completedCount = computed(() => this.nodes().filter(n => n.status === 'completed').length);
  totalCount = computed(() => this.nodes().length);

  statusColor(status: PipelineNodeStatus): string {
    const map: Record<PipelineNodeStatus, string> = {
      pending: 'bg-text-tertiary',
      running: 'bg-accent animate-pulse',
      completed: 'bg-gain',
      error: 'bg-loss',
    };
    return map[status];
  }

  statusLabel(status: PipelineNodeStatus): string {
    const map: Record<PipelineNodeStatus, string> = {
      pending: 'Pending',
      running: 'Running',
      completed: 'Done',
      error: 'Error',
    };
    return map[status];
  }

  cardClass(nodeId: string): string {
    const base = 'flex-shrink-0 w-36 p-3 rounded-lg border transition-all cursor-pointer text-left';
    return nodeId === this.activeNode()
      ? `${base} border-accent bg-accent/5`
      : `${base} border-border bg-surface-raised hover:border-border-muted`;
  }

  formatDuration(ms: number | undefined): string {
    if (ms === undefined) return '';
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  }
}
