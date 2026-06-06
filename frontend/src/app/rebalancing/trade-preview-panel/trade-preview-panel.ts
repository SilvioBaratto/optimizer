import {
  ChangeDetectionStrategy,
  Component,
  computed,
  input,
} from '@angular/core';
import { StatCardComponent } from '../../shared/stat-card/stat-card';
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { ChartToolbarComponent } from '../../shared/chart-toolbar/chart-toolbar';
import { EchartsWaterfallComponent } from '../../shared/echarts-waterfall/echarts-waterfall';
import type {
  RebalancePreviewApiResponse,
  TradeItemDto,
} from '../rebalancing.model';

@Component({
  selector: 'app-trade-preview-panel',
  imports: [StatCardComponent, DataTableComponent, ChartToolbarComponent, EchartsWaterfallComponent],
  templateUrl: './trade-preview-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class TradePreviewPanelComponent {
  readonly preview = input<RebalancePreviewApiResponse | null>(null);

  readonly tradeColumns: TableColumn[] = [
    { key: 'ticker', label: 'Ticker', sortable: true },
    {
      key: 'side',
      label: 'Side',
      type: 'badge',
      sortable: true,
      badgeMap: {
        buy: { value: 'BUY', colorClass: 'bg-gain/15 text-gain' },
        sell: { value: 'SELL', colorClass: 'bg-loss/15 text-loss' },
      },
    },
    { key: 'currentWeight', label: 'Current', type: 'percentage', sortable: true, align: 'right' },
    { key: 'targetWeight', label: 'Target', type: 'percentage', sortable: true, align: 'right' },
    { key: 'weightDelta', label: 'Δ weight', type: 'percentage', sortable: true, align: 'right', colorBySign: true },
    { key: 'shares', label: 'Shares', align: 'right' },
  ];

  readonly trades = computed<TradeItemDto[]>(() => this.preview()?.trades ?? []);

  readonly totalTurnover = computed(() => {
    const prev = this.preview();
    if (!prev) return 0;
    return (
      prev.trades.reduce((sum, t) => sum + Math.abs(t.weightDelta), 0) / 2
    );
  });

  readonly tradeRows = computed(() => {
    const prev = this.preview();
    if (!prev) return [];
    return prev.trades.map((t) => ({
      ticker: t.ticker,
      side: t.side,
      currentWeight: prev.currentWeights[t.ticker] ?? 0,
      targetWeight: prev.targetWeights[t.ticker] ?? 0,
      weightDelta: t.weightDelta,
      shares: t.shares !== null ? t.shares.toFixed(2) : '—',
    }) as Record<string, unknown>);
  });

  readonly waterfallCategories = computed(() => this.trades().map((t) => t.ticker));
  readonly waterfallValues = computed(() =>
    this.trades().map((t) => Math.round(t.weightDelta * 10000)),
  );

  readonly kpiTurnover = computed(() => `${(this.totalTurnover() * 100).toFixed(2)}%`);
  readonly kpiTradeCount = computed(() => String(this.trades().length));
  readonly kpiPortfolioValue = computed(() => {
    const v = this.preview()?.portfolioValue;
    return v !== null && v !== undefined ? v.toLocaleString() : '—';
  });
}
