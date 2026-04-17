import {
  ChangeDetectionStrategy,
  Component,
  computed,
  input,
} from '@angular/core';
import { EchartsWaterfallComponent } from '../../shared/echarts-waterfall/echarts-waterfall';
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { ChartToolbarComponent } from '../../shared/chart-toolbar/chart-toolbar';
import type { BrinsonApiResponse } from '../../models/attribution.model';

interface LevelRow extends Record<string, unknown> {
  level: string;
  name: string;
  weight: number;
  returnPct: number;
  contribution: number;
}

@Component({
  selector: 'app-saa-taa-panel',
  imports: [EchartsWaterfallComponent, DataTableComponent, ChartToolbarComponent],
  templateUrl: './saa-taa-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class SaaTaaPanelComponent {
  readonly brinson = input<BrinsonApiResponse | null>(null);

  readonly hasData = computed(() => (this.brinson()?.sectors.length ?? 0) > 0);

  // Derived multi-level view: sector level from the Brinson response.
  // Deeper levels (industry, ticker) are not available from the current
  // backend schema; the panel therefore renders sector-level only.
  readonly multiLevel = computed<LevelRow[]>(() => {
    const sectors = this.brinson()?.sectors ?? [];
    return sectors.map((s) => ({
      level: 'Sector',
      name: s.sector,
      weight: s.portfolioWeight,
      returnPct: s.portfolioReturn,
      contribution: s.totalEffect,
    }));
  });

  readonly waterfallCategories = computed(() =>
    this.multiLevel().map((m) => m.name),
  );

  readonly waterfallValues = computed(() =>
    this.multiLevel().map((m) => Math.round(m.contribution * 10000)),
  );

  readonly summaryTableColumns: TableColumn[] = [
    { key: 'level', label: 'Level', sortable: true },
    { key: 'name', label: 'Name', sortable: true },
    { key: 'weight', label: 'Weight', type: 'percentage', sortable: true, align: 'right' },
    { key: 'returnPct', label: 'Return', type: 'percentage', sortable: true, align: 'right' },
    { key: 'contribution', label: 'Contribution', type: 'percentage', sortable: true, align: 'right', colorBySign: true },
  ];

  readonly summaryTableRows = computed(() => this.multiLevel());
}
