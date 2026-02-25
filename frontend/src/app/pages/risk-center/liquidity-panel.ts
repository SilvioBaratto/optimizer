import { Component, input, computed, inject, ChangeDetectionStrategy } from '@angular/core';
import { StatCardComponent } from '../../shared/stat-card/stat-card';
import { EchartsBarComponent, BarData } from '../../shared/echarts-bar/echarts-bar';
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { ChartToolbarComponent } from '../../shared/chart-toolbar/chart-toolbar';
import { FormatService } from '../../services/format.service';
import type { LiquidityMetric } from '../../models/risk.model';

interface LiquidityBucket {
  label: string;
  count: number;
}

@Component({
  selector: 'app-liquidity-panel',
  imports: [StatCardComponent, EchartsBarComponent, DataTableComponent, ChartToolbarComponent],
  templateUrl: './liquidity-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class LiquidityPanelComponent {
  metrics = input<LiquidityMetric[]>([]);

  private fmt = inject(FormatService);

  totalCost = computed(() => {
    const sum = this.metrics().reduce((s, m) => s + m.liquidityCost, 0);
    return this.fmt.formatBps(sum);
  });

  maxDays = computed(() => {
    const max = Math.max(...this.metrics().map(m => m.daysToLiquidate), 0);
    return `${max.toFixed(1)}d`;
  });

  weightedAvgDays = computed(() => {
    const data = this.metrics();
    const totalWeight = data.reduce((s, m) => s + m.weight, 0);
    if (totalWeight === 0) return '--';
    const wavg = data.reduce((s, m) => s + m.daysToLiquidate * m.weight, 0) / totalWeight;
    return `${wavg.toFixed(2)}d`;
  });

  bucketDistribution = computed<BarData[]>(() => {
    const buckets: LiquidityBucket[] = [
      { label: '< 0.5d', count: 0 },
      { label: '0.5-1d', count: 0 },
      { label: '1-2d', count: 0 },
      { label: '2-5d', count: 0 },
      { label: '5d+', count: 0 },
    ];

    for (const m of this.metrics()) {
      const d = m.daysToLiquidate;
      if (d < 0.5) buckets[0].count++;
      else if (d < 1) buckets[1].count++;
      else if (d < 2) buckets[2].count++;
      else if (d < 5) buckets[3].count++;
      else buckets[4].count++;
    }

    return buckets.map(b => ({ label: b.label, value: b.count / Math.max(this.metrics().length, 1) }));
  });

  tableColumns: TableColumn[] = [
    { key: 'ticker', label: 'Ticker', sortable: true },
    { key: 'name', label: 'Name', sortable: true },
    { key: 'avgDailyVolume', label: 'Avg Daily Volume', type: 'number', sortable: true },
    { key: 'daysToLiquidate', label: 'Days to Liquidate', type: 'ratio', sortable: true },
    { key: 'liquidityCost', label: 'Cost (bps)', type: 'bps', sortable: true },
    { key: 'weight', label: 'Weight', type: 'percentage', sortable: true },
  ];

  tableRows = computed(() =>
    this.metrics().map(m => ({
      ticker: m.ticker,
      name: m.name,
      avgDailyVolume: m.avgDailyVolume,
      daysToLiquidate: m.daysToLiquidate,
      liquidityCost: m.liquidityCost,
      weight: m.weight,
    })),
  );
}
