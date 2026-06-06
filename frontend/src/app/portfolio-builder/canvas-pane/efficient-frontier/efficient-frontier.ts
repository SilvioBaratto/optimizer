import {
  ChangeDetectionStrategy,
  Component,
  Signal,
  computed,
  inject,
} from '@angular/core';

import type { EfficientFrontierPoint } from '../../../core/models/optimization.model';
import { EchartsEfficientFrontierComponent } from '../../../shared/echarts-efficient-frontier/echarts-efficient-frontier';
import { BuilderStore } from '../../state/builder.store';

function safeNum(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

@Component({
  selector: 'app-efficient-frontier',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [EchartsEfficientFrontierComponent],
  template: `
    @if (frontierPoints().length > 0) {
      <app-echarts-efficient-frontier
        [frontier]="frontierPoints()"
        [optimal]="currentPortfolioMarker()"
        [height]="280"
      />
    } @else {
      <div
        data-region="efficient-frontier-empty"
        class="flex h-[280px] w-full items-center justify-center rounded-lg border border-dashed border-border text-data-xs text-text-tertiary"
      >
        No efficient frontier yet
      </div>
    }
  `,
})
export class EfficientFrontierComponent {
  private readonly store = inject(BuilderStore);

  readonly frontierPoints: Signal<EfficientFrontierPoint[]> = computed(
    () => this.store.frontier()?.points ?? [],
  );

  readonly currentPortfolioMarker: Signal<EfficientFrontierPoint | null> =
    computed(() => this.composeMarker(this.store.result()?.metrics));

  private composeMarker(
    metrics: Record<string, unknown> | undefined,
  ): EfficientFrontierPoint | null {
    if (!metrics) return null;
    const risk = safeNum(metrics['annualized_volatility']);
    const ret = safeNum(metrics['annualized_return']);
    const sharpe = safeNum(metrics['annualized_sharpe_ratio']);
    if (risk === null || ret === null || sharpe === null) return null;
    return { risk, return: ret, sharpe, label: 'You' };
  }
}
