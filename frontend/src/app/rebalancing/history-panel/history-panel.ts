import {
  ChangeDetectionStrategy,
  Component,
  computed,
  input,
} from '@angular/core';
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { PageErrorBannerComponent } from '../../shared/components/page-error-banner/page-error-banner';
import type { SnapshotDto } from '../../core/models/portfolio-api.model';

const TYPE_BADGE_MAP: Record<string, { value: string; colorClass: string }> = {
  optimization: { value: 'Optimization', colorClass: 'bg-chart-1/15 text-chart-1' },
  rebalance: { value: 'Rebalance', colorClass: 'bg-chart-3/15 text-chart-3' },
  manual: { value: 'Manual', colorClass: 'bg-chart-4/15 text-chart-4' },
};

@Component({
  selector: 'app-history-panel',
  imports: [DataTableComponent, PageErrorBannerComponent],
  templateUrl: './history-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class HistoryPanelComponent {
  readonly snapshots = input<SnapshotDto[]>([]);
  readonly error = input<string | null>(null);

  readonly columns: TableColumn[] = [
    { key: 'snapshot_date', label: 'Date', type: 'date', sortable: true },
    { key: 'snapshot_type', label: 'Type', type: 'badge', badgeMap: TYPE_BADGE_MAP, sortable: true },
    { key: 'holding_count', label: 'Holdings', type: 'number', sortable: true, align: 'right' },
    { key: 'turnover', label: 'Turnover', sortable: true, align: 'right' },
    { key: 'created_at', label: 'Created at', sortable: true },
  ];

  readonly tableRows = computed(() => {
    const sorted = [...this.snapshots()].sort(
      (a, b) => b.snapshot_date.localeCompare(a.snapshot_date),
    );
    return sorted.map((s) => ({
      snapshot_date: s.snapshot_date,
      snapshot_type: s.snapshot_type,
      holding_count: s.holding_count,
      turnover: s.turnover !== null ? `${(s.turnover * 100).toFixed(2)}%` : '—',
      created_at: s.created_at,
    }) as Record<string, unknown>);
  });
}
