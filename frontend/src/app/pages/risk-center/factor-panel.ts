import { Component, input, computed, ChangeDetectionStrategy } from '@angular/core';
import { StatCardComponent } from '../../shared/stat-card/stat-card';
import { EchartsBarComponent, BarData } from '../../shared/echarts-bar/echarts-bar';
import { EchartsDonutComponent, PieSegment } from '../../shared/echarts-donut/echarts-donut';
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { ChartToolbarComponent } from '../../shared/chart-toolbar/chart-toolbar';
import { readCssVar } from '../../shared/charts/echarts-theme';
import type { FactorExposure } from '../../models/risk.model';

const CHART_COLORS = [
  '--color-chart-1', '--color-chart-2', '--color-chart-3', '--color-chart-4',
  '--color-chart-5', '--color-chart-6', '--color-chart-7', '--color-chart-8',
];

@Component({
  selector: 'app-factor-panel',
  imports: [StatCardComponent, EchartsBarComponent, EchartsDonutComponent, DataTableComponent, ChartToolbarComponent],
  templateUrl: './factor-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class FactorPanelComponent {
  exposures = input<FactorExposure[]>([]);

  barData = computed<BarData[]>(() =>
    this.exposures().map(f => ({
      label: f.factor,
      value: f.exposure,
    })),
  );

  donutData = computed<PieSegment[]>(() => {
    const positive = this.exposures().filter(f => f.contribution > 0);
    return positive.map((f, i) => ({
      label: f.factor,
      value: f.contribution,
      color: readCssVar(CHART_COLORS[i % CHART_COLORS.length]),
    }));
  });

  specificRisk = computed(() => {
    const totalContribution = this.exposures().reduce((sum, f) => sum + f.contribution, 0);
    return Math.max(0, 1 - totalContribution);
  });

  specificRiskStr = computed(() => `${(this.specificRisk() * 100).toFixed(1)}%`);

  tableColumns: TableColumn[] = [
    { key: 'factor', label: 'Factor', sortable: true },
    { key: 'exposure', label: 'Exposure', type: 'ratio', sortable: true, colorBySign: true },
    { key: 'contribution', label: 'Contribution', type: 'percentage', sortable: true, colorBySign: true },
    { key: 'marginalContribution', label: 'Marginal Contribution', type: 'ratio', sortable: true, colorBySign: true },
  ];

  tableRows = computed(() =>
    this.exposures().map(f => ({
      factor: f.factor,
      exposure: f.exposure,
      contribution: f.contribution,
      marginalContribution: f.marginalContribution,
    })),
  );
}
