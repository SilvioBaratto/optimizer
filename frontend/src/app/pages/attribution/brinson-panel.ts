import {
  ChangeDetectionStrategy,
  Component,
  computed,
  inject,
  input,
} from '@angular/core';
import { EchartsWaterfallComponent } from '../../shared/echarts-waterfall/echarts-waterfall';
import { EchartsStackedAreaComponent, AreaSeries } from '../../shared/echarts-stacked-area/echarts-stacked-area';
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { ChartToolbarComponent } from '../../shared/chart-toolbar/chart-toolbar';
import { FormatService } from '../../core/services/format.service';
import type { BrinsonApiResponse } from '../../models/attribution.model';

@Component({
  selector: 'app-brinson-panel',
  imports: [
    EchartsWaterfallComponent,
    EchartsStackedAreaComponent,
    DataTableComponent,
    ChartToolbarComponent,
  ],
  templateUrl: './brinson-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class BrinsonPanelComponent {
  readonly response = input<BrinsonApiResponse | null>(null);

  private readonly fmt = inject(FormatService);

  readonly hasData = computed(() => (this.response()?.sectors.length ?? 0) > 0);

  readonly waterfallCategories = computed(() => [
    'Allocation',
    'Selection',
    'Interaction',
    'Total',
  ]);

  readonly waterfallValues = computed(() => {
    const r = this.response();
    if (!r) return [0, 0, 0, 0];
    return [
      r.totalAllocation,
      r.totalSelection,
      r.totalInteraction,
      r.totalActiveReturn,
    ].map((v) => Math.round(v * 10000));
  });

  readonly sectorTableColumns: TableColumn[] = [
    { key: 'sector', label: 'Sector', sortable: true },
    { key: 'portfolioWeight', label: 'Port Wt', type: 'percentage', sortable: true, align: 'right' },
    { key: 'benchmarkWeight', label: 'Bench Wt', type: 'percentage', sortable: true, align: 'right' },
    { key: 'portfolioReturn', label: 'Port Ret', type: 'percentage', sortable: true, align: 'right' },
    { key: 'benchmarkReturn', label: 'Bench Ret', type: 'percentage', sortable: true, align: 'right' },
    { key: 'allocationEffect', label: 'Allocation', type: 'percentage', sortable: true, align: 'right', colorBySign: true },
    { key: 'selectionEffect', label: 'Selection', type: 'percentage', sortable: true, align: 'right', colorBySign: true },
    { key: 'interactionEffect', label: 'Interaction', type: 'percentage', sortable: true, align: 'right', colorBySign: true },
    { key: 'totalEffect', label: 'Total', type: 'percentage', sortable: true, align: 'right', colorBySign: true },
  ];

  readonly sectorTableRows = computed(() =>
    (this.response()?.sectors ?? []).map((s) => ({
      sector: s.sector,
      portfolioWeight: s.portfolioWeight,
      benchmarkWeight: s.benchmarkWeight,
      portfolioReturn: s.portfolioReturn,
      benchmarkReturn: s.benchmarkReturn,
      allocationEffect: s.allocationEffect,
      selectionEffect: s.selectionEffect,
      interactionEffect: s.interactionEffect,
      totalEffect: s.totalEffect,
    }) as Record<string, unknown>),
  );

  readonly cumulativeAreaSeries = computed<AreaSeries[]>(() => {
    const sectors = this.response()?.sectors ?? [];
    const cumSum = (values: number[]) => {
      let running = 0;
      return values.map((v) => {
        running += v;
        return Math.round(running * 1e6) / 1e6;
      });
    };
    return [
      { name: 'Allocation', values: cumSum(sectors.map((s) => s.allocationEffect)) },
      { name: 'Selection', values: cumSum(sectors.map((s) => s.selectionEffect)) },
      { name: 'Interaction', values: cumSum(sectors.map((s) => s.interactionEffect)) },
    ];
  });

  readonly cumulativeAreaLabels = computed(() =>
    (this.response()?.sectors ?? []).map((s) => s.sector),
  );

  formatPercent(v: number): string {
    return this.fmt.formatPercent(v);
  }
}
