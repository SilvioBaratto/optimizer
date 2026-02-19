import { Component, input, computed, ChangeDetectionStrategy } from '@angular/core';
import { PercentPipe } from '@angular/common';
import { StatCardComponent } from '../../shared/stat-card/stat-card';
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { LineChartComponent, LineChartData } from '../../shared/line-chart/line-chart';
import { EchartsDonutComponent } from '../../shared/echarts-donut/echarts-donut';
import { EchartsBarComponent } from '../../shared/echarts-bar/echarts-bar';
import { EchartsScatterComponent, ScatterPoint } from '../../shared/echarts-scatter/echarts-scatter';
import { EchartsHeatmapComponent } from '../../shared/echarts-heatmap/echarts-heatmap';
import { PieSegment } from '../../shared/pie-chart/pie-chart';
import { BarData } from '../../shared/bar-chart/bar-chart';
import { PortfolioResult } from '../../models/portfolio.model';
import {
  MOCK_EFFICIENT_FRONTIER,
  MOCK_OPTIMAL_POINT,
  MOCK_CORRELATION_MATRIX,
} from '../../mocks/mock-data';

const SECTOR_COLORS: Record<string, string> = {
  'Technology': '#18181b',
  'Healthcare': '#3f3f46',
  'Financial Services': '#52525b',
  'Industrials': '#71717a',
  'Consumer Defensive': '#a1a1aa',
  'Consumer Cyclical': '#404040',
  'Energy': '#525252',
  'Communication': '#737373',
  'Utilities': '#8b8b8b',
  'Real Estate': '#d4d4d8',
};

@Component({
  selector: 'app-results-view',
  imports: [
    PercentPipe,
    StatCardComponent,
    DataTableComponent,
    LineChartComponent,
    EchartsDonutComponent,
    EchartsBarComponent,
    EchartsScatterComponent,
    EchartsHeatmapComponent,
  ],
  templateUrl: './results-view.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ResultsViewComponent {
  result = input.required<PortfolioResult>();

  weightColumns: TableColumn[] = [
    { key: 'ticker', label: 'Ticker', sortable: true },
    { key: 'name', label: 'Name' },
    { key: 'sector', label: 'Sector' },
    { key: 'weight', label: 'Weight', sortable: true, align: 'right', format: (v) => (Number(v) * 100).toFixed(2) + '%' },
  ];

  sectorSegments = computed<PieSegment[]>(() =>
    this.result().sector_allocations.map(s => ({
      label: s.sector,
      value: s.weight,
      color: SECTOR_COLORS[s.sector] ?? '#a1a1aa',
    }))
  );

  monthlyBarData = computed<BarData[]>(() =>
    this.result().monthly_returns.map(m => ({
      label: m.month,
      value: m.return_pct,
    }))
  );

  backtestChartData = computed<LineChartData[]>(() => {
    const bt = this.result().backtest_cumulative;
    if (!bt) return [];
    return bt.map(p => ({ time: p.date, value: p.cumulative_return }));
  });

  readonly frontierPoints: ScatterPoint[] = MOCK_EFFICIENT_FRONTIER;
  readonly optimalPoint: ScatterPoint = MOCK_OPTIMAL_POINT;
  readonly correlationAssets: string[] = MOCK_CORRELATION_MATRIX.assets;
  readonly correlationMatrix: number[][] = MOCK_CORRELATION_MATRIX.matrix;

  formatPct(v: number): string {
    return (v * 100).toFixed(2) + '%';
  }
}
