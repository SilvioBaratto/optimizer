import { Component, input, computed, ChangeDetectionStrategy } from '@angular/core';
import { StatCardComponent } from '../../shared/stat-card/stat-card';
import { EchartsBarComponent, BarData } from '../../shared/echarts-bar/echarts-bar';
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { ChartToolbarComponent } from '../../shared/chart-toolbar/chart-toolbar';
import type { ConcentrationMetric } from '../../models/risk.model';

@Component({
  selector: 'app-concentration-panel',
  imports: [StatCardComponent, EchartsBarComponent, DataTableComponent, ChartToolbarComponent],
  templateUrl: './concentration-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ConcentrationPanelComponent {
  metrics = input<ConcentrationMetric[]>([]);

  hhi = computed(() => {
    const weights = this.metrics().map(m => m.weight);
    return weights.reduce((sum, w) => sum + w * w, 0);
  });

  hhiStr = computed(() => (this.hhi() * 10000).toFixed(0));

  effectiveBets = computed(() => {
    const h = this.hhi();
    return h > 0 ? (1 / h).toFixed(1) : '--';
  });

  top5Weight = computed(() => {
    const sorted = [...this.metrics()].sort((a, b) => b.weight - a.weight);
    const sum = sorted.slice(0, 5).reduce((s, m) => s + m.weight, 0);
    return `${(sum * 100).toFixed(1)}%`;
  });

  topNData = computed<BarData[]>(() => {
    const sorted = [...this.metrics()].sort((a, b) => b.weight - a.weight);
    let cumulative = 0;
    return sorted.slice(0, 10).map(m => {
      cumulative += m.weight;
      return { label: m.ticker, value: cumulative };
    });
  });

  riskContributionData = computed<BarData[]>(() =>
    [...this.metrics()]
      .sort((a, b) => b.riskContribution - a.riskContribution)
      .slice(0, 10)
      .map(m => ({ label: m.ticker, value: m.riskContribution })),
  );

  tableColumns: TableColumn[] = [
    { key: 'ticker', label: 'Ticker', sortable: true },
    { key: 'name', label: 'Name', sortable: true },
    { key: 'weight', label: 'Weight', type: 'percentage', sortable: true },
    { key: 'riskContribution', label: 'Risk Contribution', type: 'percentage', sortable: true },
    { key: 'componentVar', label: 'Component VaR', type: 'percentage', sortable: true },
  ];

  tableRows = computed(() =>
    this.metrics().map(m => ({
      ticker: m.ticker,
      name: m.name,
      weight: m.weight,
      riskContribution: m.riskContribution,
      componentVar: m.componentVar,
    })),
  );
}
