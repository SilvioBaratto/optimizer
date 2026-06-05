import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  computed,
  inject,
  input,
  output,
  signal,
} from '@angular/core';
import { takeUntilDestroyed, toObservable } from '@angular/core/rxjs-interop';
import { Observable, of } from 'rxjs';
import { catchError, startWith, switchMap } from 'rxjs/operators';
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
  host: {
    '(document:keydown.escape)': 'onEscape()',
  },
})
export class InstrumentDetailFlyoutComponent {
  private readonly yfinance = inject(YfinanceService);
  private readonly destroyRef = inject(DestroyRef);

  readonly instrumentId = input<string | null>(null);
  readonly closed = output<void>();

  onEscape(): void {
    if (this.instrumentId()) this.closed.emit();
  }

  readonly profile = signal<TickerProfile | null>(null);
  readonly prices = signal<PriceHistory[]>([]);
  readonly recommendations = signal<AnalystRecommendation[]>([]);

  readonly latestRecommendation = computed<AnalystRecommendation | null>(() => {
    const list = this.recommendations();
    return list.length > 0 ? list[0] : null;
  });

  readonly spark = computed(() => this.buildSpark(this.prices()));

  constructor() {
    // switchMap on the id stream cancels an in-flight request when the id
    // changes, so a slow stale response can never overwrite a newer one.
    // `startWith(empty)` resets each section the instant the id changes;
    // `catchError` surfaces a failure as the empty state (gated by @if).
    this.bindSection((id) => this.yfinance.getProfile(id), this.profile.set.bind(this.profile), null);
    this.bindSection(
      (id) => this.yfinance.getPrices(id, { limit: DEFAULT_PRICE_LIMIT }),
      this.prices.set.bind(this.prices),
      [],
    );
    this.bindSection(
      (id) => this.yfinance.getRecommendations(id),
      this.recommendations.set.bind(this.recommendations),
      [],
    );
  }

  onClose(): void {
    this.closed.emit();
  }

  private bindSection<T>(
    fetch: (id: string) => Observable<T>,
    setter: (value: T) => void,
    empty: T,
  ): void {
    toObservable(this.instrumentId)
      .pipe(
        switchMap((id) =>
          id
            ? fetch(id).pipe(startWith(empty), catchError(() => of(empty)))
            : of(empty),
        ),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe(setter);
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
