import { Component, input, output, computed, ChangeDetectionStrategy } from '@angular/core';
import { StatCardComponent } from '../../shared/stat-card/stat-card';
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { EchartsEfficientFrontierComponent } from '../../shared/echarts-efficient-frontier/echarts-efficient-frontier';
import { EchartsDonutComponent, type PieSegment } from '../../shared/echarts-donut/echarts-donut';
import { EchartsBarComponent, type BarData } from '../../shared/echarts-bar/echarts-bar';
import { tickerColorMap } from '../../shared/charts/ticker-color-map';
import type {
  OptimizationRunResponse,
  EfficientFrontierPoint,
} from '../../core/models/optimization.model';

interface StatCard {
  label: string;
  value: string;
  trend: 'up' | 'down' | 'flat';
}

interface WeightRow extends Record<string, unknown> {
  asset: string;
  weight: string;
  riskContribution: string;
}

interface StrategyRow extends Record<string, unknown> {
  metric: string;
  value: string;
}

function formatNumber(value: unknown, digits = 3): string {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';
}

function formatPercent(value: unknown, digits = 2): string {
  return typeof value === 'number' && Number.isFinite(value)
    ? `${(value * 100).toFixed(digits)}%`
    : '—';
}

@Component({
  selector: 'app-results-panel',
  imports: [
    StatCardComponent,
    DataTableComponent,
    EchartsEfficientFrontierComponent,
    EchartsDonutComponent,
    EchartsBarComponent,
  ],
  templateUrl: './results-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ResultsPanelComponent {
  readonly result = input<OptimizationRunResponse | null>(null);
  readonly hasResult = input<boolean>(false);
  readonly errorMessage = input<string | null>(null);
  readonly canSave = input<boolean>(true);
  readonly applyWeights = output<Record<string, number>>();

  exportResults(): void {
    const run = this.result();
    if (!run) return;
    const blob = new Blob([JSON.stringify(run, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `optimization-run-${run.id}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  onApplyWeights(): void {
    const weights = this.result()?.weights;
    if (!weights) return;
    this.applyWeights.emit(weights);
  }

  readonly weightsTableColumns: TableColumn[] = [
    { key: 'asset', label: 'Asset' },
    { key: 'weight', label: 'Weight', align: 'right' },
    { key: 'riskContribution', label: 'Risk Contribution', align: 'right' },
  ];

  readonly strategyTableColumns: TableColumn[] = [
    { key: 'metric', label: 'Metric' },
    { key: 'value', label: 'Value', align: 'right' },
  ];

  readonly frontier = computed<EfficientFrontierPoint[]>(() => this.normalizedFrontier());

  readonly optimal = computed<EfficientFrontierPoint | null>(() => {
    const pts = this.frontier();
    if (pts.length === 0) return null;
    return pts.reduce((acc, p) => (p.sharpe > acc.sharpe ? p : acc), pts[0]);
  });

  readonly assetMarkers = computed<EfficientFrontierPoint[]>(() => {
    const pts = this.frontier();
    if (pts.length === 0) return [];
    return [pts.reduce((acc, p) => (p.risk < acc.risk ? p : acc), pts[0])];
  });

  readonly statsCards = computed<StatCard[]>(() => {
    const metrics = this.result()?.metrics ?? {};
    return [
      this.buildCard('Sharpe Ratio', metrics['annualized_sharpe_ratio'] ?? metrics['sharpe'], 'ratio'),
      this.buildCard('Sortino Ratio', metrics['annualized_sortino_ratio'], 'ratio'),
      this.buildCard('Annual Vol', metrics['annualized_volatility'], 'percent'),
      this.buildCard('Max Drawdown', metrics['max_drawdown'], 'percent'),
    ];
  });

  readonly weightDonutSegments = computed<PieSegment[]>(() => {
    const weights = this.result()?.weights;
    if (!weights) return [];
    const sorted = Object.keys(weights).sort((a, b) => weights[b] - weights[a]);
    const palette = tickerColorMap(sorted);
    return sorted.map((ticker) => ({
      label: ticker,
      value: weights[ticker],
      color: palette[ticker],
    }));
  });

  readonly weightBarsData = computed<BarData[]>(() => {
    const weights = this.result()?.weights;
    if (!weights) return [];
    const sorted = Object.keys(weights).sort((a, b) => weights[b] - weights[a]);
    const palette = tickerColorMap(sorted);
    return sorted.map((ticker) => ({
      label: ticker,
      value: weights[ticker],
      color: palette[ticker],
    }));
  });

  readonly weightsTableRows = computed<WeightRow[]>(() => {
    const run = this.result();
    if (!run) return [];
    const contrib = run.riskContributions ?? {};
    return Object.entries(run.weights ?? {})
      .sort((a, b) => b[1] - a[1])
      .map(([asset, w]) => ({
        asset,
        weight: formatPercent(w),
        riskContribution: formatPercent(contrib[asset]),
      }));
  });

  readonly strategyTableRows = computed<StrategyRow[]>(() => {
    const run = this.result();
    if (!run) return [];
    return [
      { metric: 'Optimizer', value: run.optimizerType },
      { metric: 'Status', value: run.status },
      { metric: 'Universe Size', value: `${run.universeTickers.length}` },
      {
        metric: 'Duration',
        value: run.durationSeconds !== null ? `${run.durationSeconds.toFixed(2)}s` : '—',
      },
      { metric: 'Run ID', value: run.id },
    ];
  });

  private normalizedFrontier(): EfficientFrontierPoint[] {
    const raw = this.result()?.efficientFrontier;
    if (!Array.isArray(raw)) return [];
    return raw
      .map((p) => this.toFrontierPoint(p))
      .filter((p): p is EfficientFrontierPoint => p !== null);
  }

  private toFrontierPoint(raw: unknown): EfficientFrontierPoint | null {
    if (!raw || typeof raw !== 'object') return null;
    const entry = raw as Record<string, unknown>;
    const risk = this.asNumber(entry['risk']);
    const ret = this.asNumber(entry['return']);
    if (risk === null || ret === null) return null;
    const sharpe = this.asNumber(entry['sharpe']) ?? (risk > 0 ? ret / risk : 0);
    const label = typeof entry['label'] === 'string' ? (entry['label'] as string) : undefined;
    return { risk, return: ret, sharpe, label };
  }

  private asNumber(value: unknown): number | null {
    return typeof value === 'number' && Number.isFinite(value) ? value : null;
  }

  private buildCard(label: string, raw: unknown, kind: 'ratio' | 'percent'): StatCard {
    const value = kind === 'ratio' ? formatNumber(raw) : formatPercent(raw, 1);
    const trend: StatCard['trend'] =
      typeof raw === 'number' && raw > 0 ? 'up' : typeof raw === 'number' && raw < 0 ? 'down' : 'flat';
    return { label, value, trend };
  }
}
