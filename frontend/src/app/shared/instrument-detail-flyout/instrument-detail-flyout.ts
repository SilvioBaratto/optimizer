import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  computed,
  effect,
  inject,
  input,
  output,
  signal,
} from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { LucideAngularModule } from 'lucide-angular';

import { YfinanceService } from '../../services/yfinance.service';
import type {
  AnalystRecommendation,
  PriceHistory,
  TickerProfile,
} from '../../models/yfinance.model';

const DEFAULT_PRICE_LIMIT = 90;

@Component({
  selector: 'app-instrument-detail-flyout',
  imports: [LucideAngularModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: './instrument-detail-flyout.html',
})
export class InstrumentDetailFlyoutComponent {
  private readonly yfinance = inject(YfinanceService);
  private readonly destroyRef = inject(DestroyRef);

  readonly instrumentId = input<string | null>(null);
  readonly closed = output<void>();

  readonly profile = signal<TickerProfile | null>(null);
  readonly prices = signal<PriceHistory[]>([]);
  readonly recommendations = signal<AnalystRecommendation[]>([]);

  readonly latestRecommendation = computed<AnalystRecommendation | null>(() => {
    const list = this.recommendations();
    return list.length > 0 ? list[0] : null;
  });

  readonly spark = computed(() => this.buildSpark(this.prices()));

  constructor() {
    effect(() => {
      const id = this.instrumentId();
      this.resetState();
      if (id) this.loadAll(id);
    });
  }

  onClose(): void {
    this.closed.emit();
  }

  private loadAll(id: string): void {
    this.yfinance
      .getProfile(id)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((p) => this.profile.set(p));
    this.yfinance
      .getPrices(id, { limit: DEFAULT_PRICE_LIMIT })
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((ps) => this.prices.set(ps));
    this.yfinance
      .getRecommendations(id)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((rs) => this.recommendations.set(rs));
  }

  private resetState(): void {
    this.profile.set(null);
    this.prices.set([]);
    this.recommendations.set([]);
  }

  private buildSpark(prices: PriceHistory[]): { path: string; min: number; max: number } {
    const series = prices
      .map((p) => p.close)
      .filter((v): v is number => typeof v === 'number' && Number.isFinite(v));
    if (series.length === 0) return { path: '', min: 0, max: 0 };
    const min = Math.min(...series);
    const max = Math.max(...series);
    const range = max - min || 1;
    const dx = series.length > 1 ? 100 / (series.length - 1) : 0;
    const path = series
      .map((v, i) => {
        const x = i * dx;
        const y = 30 - ((v - min) / range) * 30;
        return `${i === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join(' ');
    return { path, min, max };
  }
}
