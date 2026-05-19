import { ChangeDetectionStrategy, Component, inject, input } from '@angular/core';

import type { WeightItem } from '../../../models/pipeline-builder.model';
import { BuilderStore, type CanvasView } from '../builder.store';
import { CanvasStateOverlayComponent } from '../canvas-state-overlay/canvas-state-overlay';
import { AllocationDonutComponent } from './allocation-donut/allocation-donut';
import { BacktestSparklineComponent } from './backtest-sparkline/backtest-sparkline';
import { EfficientFrontierComponent } from './efficient-frontier/efficient-frontier';
import { WeightBarsComponent } from './weight-bars/weight-bars';

interface ViewOption {
  readonly id: CanvasView;
  readonly label: string;
}

@Component({
  selector: 'app-canvas-pane',
  imports: [
    AllocationDonutComponent,
    BacktestSparklineComponent,
    CanvasStateOverlayComponent,
    EfficientFrontierComponent,
    WeightBarsComponent,
  ],
  template: `
    <section
      data-region="canvas-pane"
      class="flex h-full w-full flex-col gap-3 p-4"
    >
      <div
        data-region="canvas-view-toggle"
        class="flex gap-1 rounded-md border border-border p-1"
      >
        @for (view of views; track view.id) {
          <button
            type="button"
            data-view-btn
            [attr.data-view-id]="view.id"
            class="flex-1 rounded px-3 py-1.5 text-data-xs font-medium text-text-secondary"
            [class.ring-2]="store.currentView() === view.id"
            [class.ring-accent]="store.currentView() === view.id"
            (click)="store.setCurrentView(view.id)"
          >
            {{ view.label }}
          </button>
        }
      </div>
      <div class="flex-1">
        <app-canvas-state-overlay
          [status]="store.resultStatus()"
          [retryFn]="onRetry"
        >
          <div data-region="canvas-view-content" class="h-full">
            @switch (store.currentView()) {
              @case ('donut') {
                <app-allocation-donut />
              }
              @case ('bars') {
                <app-weight-bars />
              }
              @case ('frontier') {
                <app-efficient-frontier />
              }
              @case ('backtest') {
                <app-backtest-sparkline />
              }
            }
          </div>
        </app-canvas-state-overlay>
      </div>
    </section>
  `,
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class CanvasPaneComponent {
  protected readonly store = inject(BuilderStore);

  readonly weights = input<WeightItem[] | undefined>(undefined);
  readonly frontier = input<Record<string, number>[] | undefined>(undefined);
  readonly sessionId = input<string | null | undefined>(undefined);

  readonly views: readonly ViewOption[] = [
    { id: 'donut', label: 'Donut' },
    { id: 'bars', label: 'Bars' },
    { id: 'frontier', label: 'Frontier' },
    { id: 'backtest', label: 'Backtest' },
  ] as const;

  protected readonly onRetry = (): void => this.store.triggerResultRun();
}
