import {
  ChangeDetectionStrategy,
  Component,
  computed,
  input,
  output,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import {
  EchartsRebalancingDiffComponent,
  type RebalancingDiffData,
} from '../../shared/echarts-rebalancing-diff/echarts-rebalancing-diff';
import { PageErrorBannerComponent } from '../../shared/components/page-error-banner/page-error-banner';
import type { DriftApiResponse, DriftEntryDto } from '../rebalancing.model';

@Component({
  selector: 'app-status-panel',
  imports: [
    FormsModule,
    DataTableComponent,
    EchartsRebalancingDiffComponent,
    PageErrorBannerComponent,
  ],
  templateUrl: './status-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class StatusPanelComponent {
  readonly drift = input<DriftApiResponse | null>(null);
  readonly threshold = input<number>(0.05);
  readonly error = input<string | null>(null);

  readonly thresholdChange = output<number>();

  readonly driftColumns: TableColumn[] = [
    { key: 'ticker', label: 'Ticker', sortable: true },
    { key: 'name', label: 'Name', sortable: true },
    { key: 'target', label: 'Target', type: 'percentage', sortable: true, align: 'right' },
    { key: 'actual', label: 'Actual', type: 'percentage', sortable: true, align: 'right' },
    { key: 'drift', label: 'Drift', type: 'percentage', sortable: true, align: 'right', colorBySign: true },
    {
      key: 'status',
      label: 'Status',
      type: 'badge',
      badgeMap: {
        true: { value: 'BREACHED', colorClass: 'bg-loss/15 text-loss' },
        false: { value: 'OK', colorClass: 'bg-gain/15 text-gain' },
      },
    },
  ];

  readonly driftEntries = computed<DriftEntryDto[]>(
    () => this.drift()?.entries ?? [],
  );

  readonly driftRows = computed(() =>
    this.driftEntries().map((e) => ({
      ticker: e.ticker,
      name: e.name ?? '',
      target: e.target,
      actual: e.actual,
      drift: e.drift,
      status: String(e.breached),
    } as Record<string, unknown>)),
  );

  readonly diffData = computed<RebalancingDiffData[]>(() =>
    this.driftEntries().map((e) => ({
      asset: e.ticker,
      oldWeight: e.actual,
      newWeight: e.target,
    })),
  );

  readonly totalDrift = computed(() => this.drift()?.totalDrift ?? 0);
  readonly breachedCount = computed(() => this.drift()?.breachedCount ?? 0);

  onThresholdInput(value: number): void {
    this.thresholdChange.emit(value);
  }
}
