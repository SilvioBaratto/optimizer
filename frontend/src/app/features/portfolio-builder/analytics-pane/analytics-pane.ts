import {
  ChangeDetectionStrategy,
  Component,
  Signal,
  computed,
  inject,
} from '@angular/core';

import {
  MetricColorPipe,
  type MetricKind,
  safeNum,
} from '../../../shared/pipes/metric-color.pipe';
import { BuilderStore } from '../builder.store';
import { CanvasStateOverlayComponent } from '../canvas-state-overlay/canvas-state-overlay';

export interface MetricCard {
  readonly id: string;
  readonly label: string;
  readonly value: number | null;
  readonly kind: MetricKind;
  readonly colored: boolean;
  readonly unit: string;
}

@Component({
  selector: 'app-analytics-pane',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [CanvasStateOverlayComponent, MetricColorPipe],
  template: `
    <section data-region="analytics-pane" class="h-full">
      <app-canvas-state-overlay
        [status]="store.resultStatus()"
        [retryFn]="onRetry"
      >
        <div class="grid grid-cols-1 gap-3 p-4 sm:grid-cols-2">
          @for (card of cards(); track card.id) {
            @let metric = card.value | metricColor: card.kind;
            <div
              data-card="metric"
              [attr.data-card-id]="card.id"
              class="flex flex-col gap-1 rounded-lg border border-border bg-surface-raised p-3"
            >
              <span class="text-data-xs text-text-tertiary">{{ card.label }}</span>
              <span
                data-cell="value"
                class="tabular-nums text-right text-data-md"
                [class]="card.colored ? metric.colorClass : 'text-text-primary'"
              >
                {{ metric.text }}
              </span>
              <span class="text-right text-data-xs text-text-tertiary">
                {{ card.unit }}
              </span>
            </div>
          }
        </div>
      </app-canvas-state-overlay>
    </section>
  `,
})
export class AnalyticsPaneComponent {
  protected readonly store = inject(BuilderStore);

  readonly cards: Signal<MetricCard[]> = computed(() => {
    const metrics = this.store.result()?.metrics;
    const stats = this.store.backtest()?.summaryStats;
    return [
      this.card('return', 'Annualised Return', metrics?.['annualized_return'], 'percent', true, 'per year'),
      this.card('vol', 'Annualised Volatility', metrics?.['annualized_volatility'], 'percent', false, 'σ p.a.'),
      this.card('sharpe', 'Sharpe', metrics?.['annualized_sharpe_ratio'], 'ratio', false, 'ratio'),
      this.card('maxdd', 'Max Drawdown', metrics?.['max_drawdown'], 'percent', true, 'peak-to-trough'),
      this.card('turnover', 'Turnover', stats?.['turnover'] ?? stats?.['mean_turnover'], 'percent', false, 'rebal avg'),
      this.card('cost', 'Rebalancing Cost', stats?.['cost_bps_actual'] ?? stats?.['cost_bps'], 'ratio', false, 'bps'),
    ];
  });

  protected readonly onRetry = (): void => this.store.triggerResultRun();

  private card(
    id: string,
    label: string,
    raw: unknown,
    kind: MetricKind,
    colored: boolean,
    unit: string,
  ): MetricCard {
    return { id, label, value: safeNum(raw), kind, colored, unit };
  }
}
