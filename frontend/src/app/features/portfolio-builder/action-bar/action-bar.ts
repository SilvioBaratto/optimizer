import {
  ChangeDetectionStrategy,
  Component,
  computed,
  inject,
} from '@angular/core';

import { StepStatus } from '../../../models/pipeline-builder.model';
import { BuilderStore } from '../builder.store';

@Component({
  selector: 'app-action-bar',
  template: `
    <div
      class="flex items-center justify-between gap-3 border-t border-border bg-surface px-4 py-3"
    >
      <span
        data-region="action-status"
        aria-live="polite"
        class="text-data-xs text-text-tertiary"
      >
        {{ statusLabel() }}
      </span>
      <div class="flex items-center gap-2">
        <button
          type="button"
          data-action="optimize"
          [disabled]="runDisabled()"
          [attr.aria-disabled]="runDisabled()"
          (click)="onOptimize()"
          class="rounded-md border border-border bg-surface px-3 py-1.5 text-data-sm text-text-primary hover:bg-surface-hover disabled:cursor-not-allowed disabled:opacity-50"
        >
          Optimize
        </button>
        <button
          type="button"
          data-action="rebalance"
          [disabled]="runDisabled()"
          [attr.aria-disabled]="runDisabled()"
          (click)="onRebalance()"
          class="rounded-md border border-border bg-surface px-3 py-1.5 text-data-sm text-text-primary hover:bg-surface-hover disabled:cursor-not-allowed disabled:opacity-50"
        >
          Rebalance
        </button>
        <button
          type="button"
          data-action="export"
          [disabled]="exportDisabled()"
          [attr.aria-disabled]="exportDisabled()"
          (click)="onExport()"
          class="rounded-md border border-border bg-surface px-3 py-1.5 text-data-sm text-text-primary hover:bg-surface-hover disabled:cursor-not-allowed disabled:opacity-50"
        >
          Export
        </button>
      </div>
    </div>
  `,
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ActionBarComponent {
  private readonly store = inject(BuilderStore);

  readonly runDisabled = computed(
    () =>
      this.store.sessionId() === null ||
      this.store.status() === StepStatus.Running,
  );

  readonly exportDisabled = computed(() => this.store.sessionId() === null);

  readonly statusLabel = computed(() => labelFor(this.store.status()));

  onOptimize(): void {
    this.store.optimize();
  }

  onRebalance(): void {
    this.store.rebalance();
  }

  onExport(): void {
    this.store.export();
  }
}

function labelFor(status: StepStatus): string {
  if (status === StepStatus.Running) return 'Running…';
  if (status === StepStatus.Error) return 'Error';
  if (status === StepStatus.Completed) return 'Completed';
  return '';
}
