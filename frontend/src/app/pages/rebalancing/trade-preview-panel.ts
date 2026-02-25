import {
  Component,
  input,
  computed,
  inject,
  ChangeDetectionStrategy,
} from '@angular/core';
import { StatCardComponent } from '../../shared/stat-card/stat-card';
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { ChartToolbarComponent } from '../../shared/chart-toolbar/chart-toolbar';
import { EchartsWaterfallComponent } from '../../shared/echarts-waterfall/echarts-waterfall';
import { FormatService } from '../../services/format.service';
import type { TradePreview, TradeSummary } from '../../models/rebalancing.model';

@Component({
  selector: 'app-trade-preview-panel',
  imports: [StatCardComponent, DataTableComponent, ChartToolbarComponent, EchartsWaterfallComponent],
  templateUrl: './trade-preview-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class TradePreviewPanelComponent {
  trades = input<TradePreview[]>([]);
  summary = input<TradeSummary | null>(null);

  private fmt = inject(FormatService);

  readonly tradeColumns: TableColumn[] = [
    { key: 'ticker', label: 'Ticker', sortable: true },
    { key: 'name', label: 'Name', sortable: true },
    {
      key: 'action',
      label: 'Action',
      type: 'badge',
      sortable: true,
      badgeMap: {
        buy: { value: 'BUY', colorClass: 'bg-emerald-500/15 text-emerald-400' },
        sell: { value: 'SELL', colorClass: 'bg-red-500/15 text-red-400' },
      },
    },
    { key: 'shares', label: 'Shares', type: 'number', sortable: true },
    { key: 'notional', label: 'Notional', type: 'currency', sortable: true },
    { key: 'fromWeight', label: 'From Weight', type: 'percentage', sortable: true },
    { key: 'toWeight', label: 'To Weight', type: 'percentage', sortable: true },
    { key: 'estimatedCost', label: 'Est. Cost', type: 'currency', sortable: true },
  ];

  tradeRows = computed(() =>
    this.trades().map(t => ({
      ticker: t.ticker,
      name: t.name,
      action: t.action,
      shares: t.shares,
      notional: t.notional,
      fromWeight: t.fromWeight,
      toWeight: t.toWeight,
      estimatedCost: t.estimatedCost,
    })),
  );

  waterfallCategories = computed(() => this.trades().map(t => t.ticker));

  waterfallValues = computed(() =>
    this.trades().map(t => Math.round((t.toWeight - t.fromWeight) * 10000)),
  );

  kpiTurnover = computed(() => {
    const s = this.summary();
    return s ? this.fmt.formatPercent(s.totalTurnover) : '--';
  });

  kpiCost = computed(() => {
    const s = this.summary();
    return s ? this.fmt.formatCurrency(s.totalCost) : '--';
  });

  kpiBuys = computed(() => {
    const s = this.summary();
    return s ? String(s.buys) : '--';
  });

  kpiSells = computed(() => {
    const s = this.summary();
    return s ? String(s.sells) : '--';
  });

  kpiNetCashFlow = computed(() => {
    const s = this.summary();
    return s ? this.fmt.formatCurrency(s.netCashFlow) : '--';
  });
}
