import { ChangeDetectionStrategy, Component, input } from '@angular/core';

import { LoadingSkeletonComponent } from '../../../shared/loading-skeleton/loading-skeleton';
import type { BuilderResultStatus } from '../builder-result.model';

type RetryFn = () => void;

@Component({
  selector: 'app-canvas-state-overlay',
  imports: [LoadingSkeletonComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    @switch (status()) {
      @case ('running') {
        <div
          data-region="overlay-loading"
          class="flex h-full flex-col items-stretch gap-3 p-4"
        >
          <div
            data-element="spinner"
            class="h-6 w-6 animate-spin self-center rounded-full border-2 border-accent border-t-transparent"
          ></div>
          <app-loading-skeleton height="0.75rem" />
          <app-loading-skeleton height="0.75rem" width="80%" />
          <app-loading-skeleton height="0.75rem" width="60%" />
        </div>
      }
      @case ('error') {
        <div
          data-region="overlay-error"
          class="flex h-full flex-col items-center justify-center gap-3 p-4"
        >
          <span class="text-data-sm text-loss">Something went wrong.</span>
          <button
            type="button"
            data-action="retry"
            (click)="onRetry()"
            class="rounded border border-border px-3 py-1 text-data-xs text-text-primary hover:bg-surface-raised"
          >
            Retry
          </button>
        </div>
      }
      @case ('idle') {
        <div
          data-region="overlay-idle"
          class="flex h-full flex-col items-center justify-center gap-2 p-4 text-text-tertiary"
        >
          <span class="text-data-sm">Run pipeline to see results.</span>
        </div>
      }
      @default {
        <div [class.relative]="status() === 'stale'" class="h-full">
          <div
            data-element="stale-content"
            [class.opacity-60]="status() === 'stale'"
            [class.pointer-events-none]="status() === 'stale'"
            class="h-full"
          >
            <ng-content />
          </div>
          @if (status() === 'stale') {
            <span
              data-region="overlay-stale-badge"
              class="absolute right-2 top-2 rounded bg-surface-raised px-2 py-0.5 text-data-xs text-text-tertiary"
            >
              Stale
            </span>
          }
        </div>
      }
    }
  `,
})
export class CanvasStateOverlayComponent {
  readonly status = input.required<BuilderResultStatus>();
  readonly retryFn = input<RetryFn | undefined>(undefined);

  protected onRetry(): void {
    this.retryFn()?.();
  }
}
