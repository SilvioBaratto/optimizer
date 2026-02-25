import { Component, input, computed, inject, ChangeDetectionStrategy } from '@angular/core';
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { EchartsLineComponent } from '../../shared/echarts-line/echarts-line';
import { ChartToolbarComponent } from '../../shared/chart-toolbar/chart-toolbar';
import { FormatService } from '../../services/format.service';
import type { RebalancingHistoryEntry, RebalancingTrigger } from '../../models/rebalancing.model';

const TRIGGER_BADGE_MAP: Record<RebalancingTrigger, { value: string; colorClass: string }> = {
  calendar:  { value: 'Calendar',  colorClass: 'bg-sky-500/15 text-sky-400' },
  threshold: { value: 'Threshold', colorClass: 'bg-amber-500/15 text-amber-400' },
  hybrid:    { value: 'Hybrid',    colorClass: 'bg-purple-500/15 text-purple-400' },
};

@Component({
  selector: 'app-history-panel',
  imports: [DataTableComponent, EchartsLineComponent, ChartToolbarComponent],
  templateUrl: './history-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class HistoryPanelComponent {
  history = input<RebalancingHistoryEntry[]>([]);

  private fmt = inject(FormatService);

  readonly triggerBadgeMap = TRIGGER_BADGE_MAP;

  readonly columns: TableColumn[] = [
    { key: 'date',        label: 'Date',          type: 'date',       sortable: true,  dateFormat: 'medium' },
    { key: 'trigger',     label: 'Trigger',        type: 'badge',      sortable: true,  badgeMap: TRIGGER_BADGE_MAP },
    { key: 'tradesExecuted', label: 'Trades',      type: 'number',     sortable: true,  align: 'right' },
    { key: 'turnover',    label: 'Turnover',       type: 'percentage', sortable: true,  align: 'right' },
    { key: 'cost',        label: 'Cost',           type: 'currency',   sortable: true,  align: 'right' },
    { key: 'preDriftMax', label: 'Max Pre-Drift',  type: 'percentage', sortable: true,  align: 'right' },
  ];

  readonly tableRows = computed(() =>
    this.history().map(e => ({ ...e } as Record<string, unknown>)),
  );

  readonly chronologicalHistory = computed(() => [...this.history()].reverse());

  readonly chartLabels = computed(() =>
    this.chronologicalHistory().map(e =>
      this.fmt.formatDate(e.date, 'short'),
    ),
  );

  readonly cumulativeTurnoverValues = computed(() => {
    let sum = 0;
    return this.chronologicalHistory().map(e => {
      sum += e.turnover;
      return parseFloat((sum * 100).toFixed(4));
    });
  });

  readonly cumulativeCostValues = computed(() => {
    let sum = 0;
    return this.chronologicalHistory().map(e => {
      sum += e.cost;
      return parseFloat(sum.toFixed(2));
    });
  });
}
