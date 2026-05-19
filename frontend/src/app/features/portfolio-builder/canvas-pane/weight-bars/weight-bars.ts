import {
  ChangeDetectionStrategy,
  Component,
  Signal,
  computed,
  inject,
} from '@angular/core';

import { tickerColorMap } from '../../../../shared/charts/ticker-color-map';
import { MetricColorPipe } from '../../../../shared/pipes/metric-color.pipe';
import { BuilderStore } from '../../builder.store';

const MAX_ROWS = 20;

export interface WeightRow {
  readonly ticker: string;
  readonly weight: number;
  readonly color: string;
}

@Component({
  selector: 'app-weight-bars',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [MetricColorPipe],
  template: `
    @if (rows().length > 0) {
      <ul
        data-region="weight-bars"
        class="flex flex-col gap-1 rounded-lg border border-border p-2"
      >
        @for (row of rows(); track row.ticker) {
          @let metric = row.weight | metricColor: 'percent';
          <li
            data-row="weight-bar"
            class="flex items-center gap-3 px-2 py-1.5"
            [attr.aria-label]="row.ticker + ' ' + metric.text"
          >
            <span
              data-cell="swatch"
              class="h-3 w-3 shrink-0 rounded-sm"
              [style.background-color]="row.color"
            ></span>
            <span class="flex-1 text-data-sm text-text-secondary">
              {{ row.ticker }}
            </span>
            <span
              data-cell="weight"
              class="tabular-nums text-right text-data-sm"
              [class]="metric.colorClass"
            >
              {{ metric.text }}
            </span>
          </li>
        }
      </ul>
    } @else {
      <div
        data-region="weight-bars-empty"
        class="flex h-[240px] w-full items-center justify-center rounded-lg border border-dashed border-border text-data-xs text-text-tertiary"
      >
        No allocation yet
      </div>
    }
  `,
})
export class WeightBarsComponent {
  private readonly store = inject(BuilderStore);

  readonly rows: Signal<WeightRow[]> = computed(() =>
    this.toRows(this.store.result()?.weights),
  );

  private toRows(weights: Record<string, number> | undefined): WeightRow[] {
    if (!weights) return [];
    const sorted = Object.keys(weights)
      .sort((a, b) => Math.abs(weights[b]) - Math.abs(weights[a]))
      .slice(0, MAX_ROWS);
    const palette = tickerColorMap(sorted);
    return sorted.map((ticker) => ({
      ticker,
      weight: weights[ticker],
      color: palette[ticker],
    }));
  }
}
