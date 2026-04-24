import {
  ChangeDetectionStrategy,
  Component,
  computed,
  input,
} from '@angular/core';

import { EchartsDrawdownComponent, type DrawdownPoint } from '../../shared/echarts-drawdown/echarts-drawdown';
import { EchartsLineComponent } from '../../shared/echarts-line/echarts-line';
import {
  EchartsRollingMetricsComponent,
  type RollingMetricSeries,
} from '../../shared/echarts-rolling-metrics/echarts-rolling-metrics';
import type { BacktestRunResponse } from '../../models/backtest.model';

interface SummaryCell {
  key: string;
  value: string;
}

// Drawdown keys are ISO dates; the max_drawdown summary leaks into the same
// dict server-side — we skip it so the chart plots only date-indexed points.
const DRAWDOWN_SUMMARY_KEYS: ReadonlySet<string> = new Set(['max_drawdown']);

/**
 * Summary + charts rendered after a backtest run completes (issue #465).
 *
 * Pure presentation: parent owns the fetch / loading / error signals and
 * passes them via inputs. Reuses the three shared ECharts components the
 * acceptance criteria point at (line / drawdown / rolling-metrics).
 */
@Component({
  selector: 'app-backtest-results-panel',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [EchartsDrawdownComponent, EchartsLineComponent, EchartsRollingMetricsComponent],
  template: `
    @if (loading()) {
      <div
        data-testid="loading"
        class="space-y-4"
      >
        <div class="h-24 skeleton rounded-lg"></div>
        <div class="h-64 skeleton rounded-lg"></div>
        <div class="h-44 skeleton rounded-lg"></div>
        <div class="h-64 skeleton rounded-lg"></div>
      </div>
    } @else if (error(); as err) {
      <div
        data-testid="error"
        class="bg-loss/10 border border-loss/30 text-loss text-sm rounded-lg p-3"
      >
        Failed to load backtest run: {{ err }}
      </div>
    } @else if (run(); as r) {
      <section
        data-testid="panel"
        class="space-y-4"
        aria-label="Backtest run results"
      >
        <!-- Summary stats -->
        <div class="bg-surface-raised border border-border rounded-lg p-4">
          <h3 class="text-sm font-semibold text-text mb-3">Summary Stats</h3>
          @if (summaryCells().length === 0) {
            <p class="text-sm text-text-secondary">No summary statistics.</p>
          } @else {
            <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
              @for (cell of summaryCells(); track cell.key) {
                <div class="rounded-md border border-border bg-surface px-3 py-2">
                  <div class="text-data-xs text-text-tertiary uppercase tracking-wide">
                    {{ cell.key }}
                  </div>
                  <div class="text-sm font-mono text-text">{{ cell.value }}</div>
                </div>
              }
            </div>
          }
        </div>

        <!-- Equity curve -->
        <div class="bg-surface-raised border border-border rounded-lg p-4">
          <h3 class="text-sm font-semibold text-text mb-3">Equity Curve</h3>
          @if (equityLabels().length === 0) {
            <p class="text-sm text-text-secondary">No equity curve data.</p>
          } @else {
            <app-echarts-line
              [labels]="equityLabels()"
              [values]="equityValues()"
              [areaFill]="true"
              [yAxisLabel]="'Cumulative return'"
            />
          }
        </div>

        <!-- Drawdowns -->
        <div class="bg-surface-raised border border-border rounded-lg p-4">
          <h3 class="text-sm font-semibold text-text mb-3">Drawdowns</h3>
          @if (drawdownPoints().length === 0) {
            <p class="text-sm text-text-secondary">No drawdown data.</p>
          } @else {
            <app-echarts-drawdown [data]="drawdownPoints()" />
          }
        </div>

        <!-- Rolling metrics -->
        <div class="bg-surface-raised border border-border rounded-lg p-4">
          <h3 class="text-sm font-semibold text-text mb-3">Rolling Metrics</h3>
          <app-echarts-rolling-metrics [series]="rollingSeries()" />
        </div>
      </section>
    }
  `,
})
export class BacktestResultsPanelComponent {
  readonly run = input<BacktestRunResponse | null>(null);
  readonly loading = input<boolean>(false);
  readonly error = input<string | null>(null);

  readonly summaryCells = computed<SummaryCell[]>(() =>
    Object.entries(this.run()?.summaryStats ?? {})
      .filter(([, value]) => value !== null && value !== undefined)
      .map(([key, value]) => ({
        key,
        value: formatNumber(value as number),
      })),
  );

  readonly equityLabels = computed(() => this.sortedDateKeys(this.run()?.equityCurve ?? {}));

  readonly equityValues = computed(() => {
    const dict = this.run()?.equityCurve ?? {};
    return this.equityLabels().map((d) => dict[d] ?? 0);
  });

  readonly drawdownPoints = computed<DrawdownPoint[]>(() => {
    const dict = this.run()?.drawdowns ?? {};
    return this.sortedDateKeys(dict, DRAWDOWN_SUMMARY_KEYS).map((date) => ({
      date,
      drawdown: dict[date] ?? 0,
    }));
  });

  readonly rollingSeries = computed<RollingMetricSeries[]>(() =>
    Object.entries(this.run()?.rollingMetrics ?? {}).map(([name, dict]) => ({
      name: formatRollingMetricName(name),
      values: this.sortedDateKeys(dict).map((date) => ({ date, value: dict[date] ?? 0 })),
      formatter: name.includes('vol') ? 'percent' : 'ratio',
    })),
  );

  private sortedDateKeys(
    dict: Record<string, number | null>,
    skip: ReadonlySet<string> = new Set(),
  ): string[] {
    return Object.keys(dict).filter((k) => !skip.has(k)).sort();
  }
}

function formatNumber(value: number): string {
  if (!Number.isFinite(value)) return '—';
  const abs = Math.abs(value);
  if (abs !== 0 && abs < 0.01) return value.toExponential(2);
  return value.toFixed(4);
}

function formatRollingMetricName(name: string): string {
  return name
    .split('_')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}
