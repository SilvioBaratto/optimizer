import {
  ChangeDetectionStrategy,
  Component,
  Signal,
  computed,
  inject,
  signal,
} from '@angular/core';

import type { BaseToggle, DriftRow } from '../../../../models/drift.model';
import type { BarData } from '../../../../shared/bar-chart/bar-chart';
import { tickerColorMap } from '../../../../shared/charts/ticker-color-map';
import { EchartsBarComponent } from '../../../../shared/echarts-bar/echarts-bar';
import {
  EchartsDonutComponent,
  type PieSegment,
} from '../../../../shared/echarts-donut/echarts-donut';
import { BuilderStore } from '../../builder.store';

export type ViewMode = 'bars' | 'donut';

export interface DriftBarRow {
  readonly ticker: string;
  readonly target: number;
  readonly current: number;
  readonly delta: number;
  readonly color: string;
}

const BASES: readonly BaseToggle[] = ['invested', 'total'] as const;
const VIEW_MODES: readonly ViewMode[] = ['bars', 'donut'] as const;

function sortByAbsDeltaDesc(rows: readonly DriftRow[]): DriftRow[] {
  return [...rows].sort(
    (a, b) => Math.abs(b.delta_weight) - Math.abs(a.delta_weight),
  );
}

function buildPalette(rows: readonly DriftRow[]): Record<string, string> {
  const byCurrent = [...rows].sort(
    (a, b) => Math.abs(b.current_weight) - Math.abs(a.current_weight),
  );
  return tickerColorMap(byCurrent.map((r) => r.ticker));
}

@Component({
  selector: 'app-drift-overlay',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [EchartsBarComponent, EchartsDonutComponent],
  template: `
    <section
      data-region="drift-overlay"
      class="flex h-full w-full flex-col gap-3"
    >
      <div
        data-region="drift-header"
        class="flex items-center justify-between gap-2"
      >
        <div class="flex items-center gap-2 text-data-xs text-text-secondary">
          <span class="font-medium text-text">Drift</span>
          <span class="rounded bg-surface-raised px-1.5 py-0.5 capitalize">
            {{ store.deployableBase() }}
          </span>
        </div>
        <div
          data-region="base-toggle"
          class="flex gap-1 rounded-md border border-border p-0.5"
        >
          @for (base of bases; track base) {
            <button
              type="button"
              data-base-btn
              [attr.data-base-id]="base"
              class="rounded px-2.5 py-1 text-data-xs font-medium capitalize text-text-secondary"
              [class.ring-2]="store.deployableBase() === base"
              [class.ring-accent]="store.deployableBase() === base"
              (click)="onBaseChange(base)"
            >
              {{ base }}
            </button>
          }
        </div>
        <div
          data-region="view-mode-toggle"
          class="flex gap-1 rounded-md border border-border p-0.5"
        >
          @for (mode of viewModes; track mode) {
            <button
              type="button"
              data-mode-btn
              [attr.data-mode-id]="mode"
              class="rounded px-2.5 py-1 text-data-xs font-medium capitalize text-text-secondary"
              [class.ring-2]="viewMode() === mode"
              [class.ring-accent]="viewMode() === mode"
              (click)="setViewMode(mode)"
            >
              {{ mode }}
            </button>
          }
          <button
            type="button"
            data-region="drift-refresh"
            class="rounded px-2.5 py-1 text-data-xs font-medium text-text-secondary"
            (click)="onRefresh()"
            aria-label="Refresh drift"
          >
            ⟳
          </button>
        </div>
      </div>
      <div data-region="drift-chart" class="min-h-0 flex-1">
        @if (rows().length > 0) {
          @switch (viewMode()) {
            @case ('bars') {
              <app-echarts-bar [data]="barData()" [height]="240" />
            }
            @case ('donut') {
              <app-echarts-donut
                [segments]="donutSegments()"
                [height]="240"
                [labelThreshold]="5"
              />
            }
          }
        } @else {
          <div
            data-region="drift-empty"
            class="flex h-[240px] w-full items-center justify-center rounded-lg border border-dashed border-border text-data-xs text-text-tertiary"
          >
            No drift data
          </div>
        }
      </div>
    </section>
  `,
})
export class DriftOverlayComponent {
  protected readonly store = inject(BuilderStore);
  protected readonly bases = BASES;
  protected readonly viewModes = VIEW_MODES;
  private readonly _viewMode = signal<ViewMode>('bars');

  readonly viewMode = this._viewMode.asReadonly();

  readonly barSeries: Signal<DriftBarRow[]> = computed(() => {
    const drift = this.store.drift();
    if (!drift) return [];
    const palette = buildPalette(drift.drift);
    return sortByAbsDeltaDesc(drift.drift).map((r) => ({
      ticker: r.ticker,
      target: r.target_weight,
      current: r.current_weight,
      delta: r.delta_weight,
      color: palette[r.ticker],
    }));
  });

  readonly rows: Signal<DriftBarRow[]> = this.barSeries;

  readonly barData: Signal<BarData[]> = computed(() =>
    this.barSeries().map((r) => ({
      label: r.ticker,
      value: r.delta,
      color: r.color,
    })),
  );

  readonly donutSegments: Signal<PieSegment[]> = computed(() =>
    this.barSeries().map((r) => ({
      label: r.ticker,
      value: r.current,
      color: r.color,
    })),
  );

  setViewMode(mode: ViewMode): void {
    this._viewMode.set(mode);
  }

  onBaseChange(base: BaseToggle): void {
    this.store.setDeployableBase(base);
  }

  onRefresh(): void {
    this.store.triggerDriftRun();
  }
}
