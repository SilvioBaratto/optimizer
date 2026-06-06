import {
  ChangeDetectionStrategy,
  Component,
  Signal,
  computed,
  inject,
} from '@angular/core';

import {
  EchartsDonutComponent,
  type PieSegment,
} from '../../../shared/echarts-donut/echarts-donut';
import { tickerColorMap } from '../../../shared/charts/ticker-color-map';
import { BuilderStore } from '../../state/builder.store';

@Component({
  selector: 'app-allocation-donut',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [EchartsDonutComponent],
  template: `
    @if (segments().length > 0) {
      <app-echarts-donut [segments]="segments()" [height]="240" />
    } @else {
      <div
        data-region="allocation-donut-empty"
        class="flex h-[240px] w-full items-center justify-center rounded-lg border border-dashed border-border text-data-xs text-text-tertiary"
      >
        No allocation yet
      </div>
    }
  `,
})
export class AllocationDonutComponent {
  private readonly store = inject(BuilderStore);

  readonly segments: Signal<PieSegment[]> = computed(() =>
    this.toSegments(this.store.result()?.weights),
  );

  private toSegments(weights: Record<string, number> | undefined): PieSegment[] {
    if (!weights) return [];
    const sorted = Object.keys(weights).sort(
      (a, b) => Math.abs(weights[b]) - Math.abs(weights[a]),
    );
    const palette = tickerColorMap(sorted);
    return sorted.map((ticker) => ({
      label: ticker,
      value: weights[ticker],
      color: palette[ticker],
    }));
  }
}
