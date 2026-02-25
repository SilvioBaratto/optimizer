import {
  Component,
  input,
  computed,
  ChangeDetectionStrategy,
} from '@angular/core';
import { EchartsStackedAreaComponent, AreaSeries } from '../../shared/echarts-stacked-area/echarts-stacked-area';
import { EchartsHeatmapComponent } from '../../shared/echarts-heatmap/echarts-heatmap';
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { ChartToolbarComponent } from '../../shared/chart-toolbar/chart-toolbar';
import { readCssVar } from '../../shared/charts/echarts-theme';
import type { RegimeDetection, HmmState } from '../../models/factor.model';

const STATE_LABELS: Record<HmmState, string> = {
  low_vol: 'Low Vol',
  medium_vol: 'Medium Vol',
  high_vol: 'High Vol',
};

const STATES: HmmState[] = ['low_vol', 'medium_vol', 'high_vol'];

@Component({
  selector: 'app-regime-panel',
  imports: [EchartsStackedAreaComponent, EchartsHeatmapComponent, DataTableComponent, ChartToolbarComponent],
  templateUrl: './regime-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class RegimePanelComponent {
  regimeHistory = input<RegimeDetection[]>([]);

  readonly stateLabels = Object.values(STATE_LABELS);

  // Sample every 5th entry to avoid overcrowding
  sampledHistory = computed(() => {
    const history = this.regimeHistory();
    return history.filter((_, i) => i % 5 === 0);
  });

  areaLabels = computed(() => this.sampledHistory().map(e => e.date));

  areaSeries = computed<AreaSeries[]>(() => {
    const sampled = this.sampledHistory();
    return STATES.map((state, i) => {
      const colors = [
        readCssVar('--color-chart-2'),
        readCssVar('--color-chart-4'),
        readCssVar('--color-chart-5'),
      ];
      return {
        name: STATE_LABELS[state],
        values: sampled.map(e => e.probabilities[state] ?? 0),
        color: colors[i],
      };
    });
  });

  transitionMatrix = computed<number[][]>(() => {
    const history = this.regimeHistory();
    const counts: number[][] = STATES.map(() => new Array(3).fill(0) as number[]);

    for (let i = 1; i < history.length; i++) {
      const from = STATES.indexOf(history[i - 1].state);
      const to = STATES.indexOf(history[i].state);
      if (from >= 0 && to >= 0) {
        counts[from][to]++;
      }
    }

    return counts.map(row => {
      const total = row.reduce((a, b) => a + b, 0);
      if (total === 0) return row.map(() => 0);
      return row.map(v => Math.round((v / total) * 100) / 100);
    });
  });

  regimeStatsColumns: TableColumn[] = [
    { key: 'state', label: 'State', type: 'text', sortable: true },
    { key: 'days', label: 'Days', type: 'number', sortable: true },
    { key: 'pctTime', label: '% Time', type: 'percentage', sortable: true },
    { key: 'avgDuration', label: 'Avg Duration (days)', type: 'number', sortable: true },
  ];

  regimeStatsRows = computed(() => {
    const history = this.regimeHistory();
    const total = history.length;

    return STATES.map(state => {
      const days = history.filter(e => e.state === state).length;
      const pctTime = total > 0 ? days / total : 0;

      // Compute average run length for this state
      let runCount = 0;
      let currentRun = 0;
      let totalRunLength = 0;

      for (const entry of history) {
        if (entry.state === state) {
          currentRun++;
        } else if (currentRun > 0) {
          runCount++;
          totalRunLength += currentRun;
          currentRun = 0;
        }
      }
      if (currentRun > 0) {
        runCount++;
        totalRunLength += currentRun;
      }

      const avgDuration = runCount > 0 ? Math.round(totalRunLength / runCount) : 0;

      return {
        state: STATE_LABELS[state],
        days,
        pctTime,
        avgDuration,
      } as Record<string, unknown>;
    });
  });
}
