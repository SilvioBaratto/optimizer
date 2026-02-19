import { Component, input, signal, inject, DestroyRef, ChangeDetectionStrategy, effect } from '@angular/core';
import { DatePipe } from '@angular/common';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { forkJoin } from 'rxjs';
import { YfinanceService } from '../../services/yfinance.service';
import { StatCardComponent } from '../../shared/stat-card/stat-card';
import { LineChartComponent, LineChartData } from '../../shared/line-chart/line-chart';
import { BarChartComponent, BarData } from '../../shared/bar-chart/bar-chart';
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { LoadingSkeletonComponent } from '../../shared/loading-skeleton/loading-skeleton';
import { EmptyStateComponent } from '../../shared/empty-state/empty-state';
import { TickerProfile, PriceHistory, AnalystRecommendation, InsiderTransaction, TickerNews } from '../../models/yfinance.model';

@Component({
  selector: 'app-instrument-detail',
  imports: [DatePipe, StatCardComponent, LineChartComponent, BarChartComponent, DataTableComponent, LoadingSkeletonComponent, EmptyStateComponent],
  templateUrl: './instrument-detail.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class InstrumentDetailComponent {
  instrumentId = input<string | null>(null);

  private readonly yfinanceService = inject(YfinanceService);
  private readonly destroyRef = inject(DestroyRef);

  profile = signal<TickerProfile | null>(null);
  prices = signal<PriceHistory[]>([]);
  recommendations = signal<AnalystRecommendation[]>([]);
  insiders = signal<InsiderTransaction[]>([]);
  news = signal<TickerNews[]>([]);
  loading = signal(false);

  priceChartData = signal<LineChartData[]>([]);
  recChartData = signal<BarData[]>([]);

  insiderColumns: TableColumn[] = [
    { key: 'date', label: 'Date', sortable: true },
    { key: 'insider', label: 'Insider' },
    { key: 'transaction_type', label: 'Type' },
    { key: 'shares', label: 'Shares', align: 'right', format: (v) => Number(v).toLocaleString() },
    { key: 'value', label: 'Value', align: 'right', format: (v) => '$' + Number(v).toLocaleString() },
  ];

  constructor() {
    effect(() => {
      const id = this.instrumentId();
      if (id) this.loadData(id);
    });
  }

  private loadData(id: string) {
    this.loading.set(true);
    forkJoin({
      profile: this.yfinanceService.getProfile(id),
      prices: this.yfinanceService.getPrices(id),
      recommendations: this.yfinanceService.getRecommendations(id),
      insiders: this.yfinanceService.getInsiderTransactions(id),
      news: this.yfinanceService.getNews(id),
    }).pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe(result => {
        this.profile.set(result.profile);
        this.prices.set(result.prices);
        this.recommendations.set(result.recommendations);
        this.insiders.set(result.insiders);
        this.news.set(result.news);

        this.priceChartData.set(
          result.prices.map(p => ({ time: p.date, value: p.close }))
        );

        this.recChartData.set(
          result.recommendations.map(r => ({
            label: r.period,
            value: r.strong_buy + r.buy - r.sell - r.strong_sell,
          }))
        );

        this.loading.set(false);
      });
  }

  formatMarketCap(cap: number): string {
    if (cap >= 1e12) return '$' + (cap / 1e12).toFixed(1) + 'T';
    if (cap >= 1e9) return '$' + (cap / 1e9).toFixed(1) + 'B';
    if (cap >= 1e6) return '$' + (cap / 1e6).toFixed(1) + 'M';
    return '$' + cap.toLocaleString();
  }
}
