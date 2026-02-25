import { Component, computed, ChangeDetectionStrategy } from '@angular/core';
import { StatCardComponent } from '../../shared/stat-card/stat-card';
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { EchartsScatterComponent, ScatterPoint } from '../../shared/echarts-scatter/echarts-scatter';
import {
  MOCK_EFFICIENT_FRONTIER_POINTS,
  MOCK_STRATEGY_COMPARISONS,
  MOCK_MOMENT_OUTPUT,
} from '../../mocks/optimization-mocks';

@Component({
  selector: 'app-results-panel',
  imports: [StatCardComponent, DataTableComponent, EchartsScatterComponent],
  templateUrl: './results-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ResultsPanelComponent {
  readonly frontierPoints = computed<ScatterPoint[]>(() =>
    MOCK_EFFICIENT_FRONTIER_POINTS.map(p => ({ x: p.risk, y: p.return, label: p.label })),
  );

  readonly optimalPoint = computed<ScatterPoint | null>(() => {
    const optimal = MOCK_EFFICIENT_FRONTIER_POINTS.find(p => p.label === 'Max Sharpe');
    return optimal ? { x: optimal.risk, y: optimal.return, label: optimal.label } : null;
  });

  readonly bestStrategy = computed(() =>
    MOCK_STRATEGY_COMPARISONS.reduce((best, s) =>
      s.sharpe > best.sharpe ? s : best,
      MOCK_STRATEGY_COMPARISONS[0],
    ),
  );

  readonly statsCards = computed(() => {
    const best = this.bestStrategy();
    return [
      { label: 'Sharpe Ratio', value: best.sharpe.toFixed(3), trend: 'up' as const },
      { label: 'Annual Return', value: `${(best.annualizedReturn * 100).toFixed(1)}%`, trend: 'up' as const },
      { label: 'Annual Vol', value: `${(best.annualizedVol * 100).toFixed(1)}%`, trend: 'down' as const },
      { label: 'Max Drawdown', value: `${(best.maxDrawdown * 100).toFixed(1)}%`, trend: 'down' as const },
    ];
  });

  readonly weightsTableColumns: TableColumn[] = [
    { key: 'asset', label: 'Asset' },
    { key: 'currentWeight', label: 'Current Weight', align: 'right' },
    { key: 'optimalWeight', label: 'Optimal Weight', align: 'right' },
    { key: 'delta', label: 'Delta', align: 'right' },
    { key: 'direction', label: 'Direction' },
  ];

  readonly weightsTableRows = computed<Record<string, unknown>[]>(() => {
    const tickers = MOCK_MOMENT_OUTPUT.expectedReturns;
    const n = tickers.length;
    const equalWeight = 1 / n;
    return tickers.map((e, i) => {
      const seed = (i * 13 + 7) % 100;
      const optimalWeight = equalWeight + (seed / 100 - 0.5) * 0.02;
      const currentWeight = equalWeight;
      const delta = optimalWeight - currentWeight;
      return {
        asset: e.ticker,
        currentWeight: `${(currentWeight * 100).toFixed(1)}%`,
        optimalWeight: `${(optimalWeight * 100).toFixed(1)}%`,
        delta: `${delta >= 0 ? '+' : ''}${(delta * 100).toFixed(1)}%`,
        direction: delta > 0.001 ? 'Increase' : delta < -0.001 ? 'Decrease' : 'Hold',
      };
    });
  });

  readonly strategyTableColumns: TableColumn[] = [
    { key: 'strategy', label: 'Strategy' },
    { key: 'annualizedReturn', label: 'Annual Return', type: 'percentage', align: 'right', colorBySign: true },
    { key: 'annualizedVol', label: 'Annual Vol', type: 'percentage', align: 'right' },
    { key: 'sharpe', label: 'Sharpe', type: 'ratio', align: 'right' },
    { key: 'maxDrawdown', label: 'Max DD', type: 'percentage', align: 'right', colorBySign: true },
    { key: 'cvar95', label: 'CVaR 95%', type: 'percentage', align: 'right', colorBySign: true },
  ];

  readonly strategyTableRows = computed<Record<string, unknown>[]>(() =>
    MOCK_STRATEGY_COMPARISONS.map(s => ({ ...s })),
  );
}
