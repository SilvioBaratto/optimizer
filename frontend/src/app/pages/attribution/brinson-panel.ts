import {
  Component,
  input,
  computed,
  inject,
  ChangeDetectionStrategy,
} from '@angular/core';
import { EchartsWaterfallComponent } from '../../shared/echarts-waterfall/echarts-waterfall';
import { EchartsStackedAreaComponent, AreaSeries } from '../../shared/echarts-stacked-area/echarts-stacked-area';
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { ChartToolbarComponent } from '../../shared/chart-toolbar/chart-toolbar';
import { FormatService } from '../../services/format.service';
import type { BrinsonAttribution } from '../../models/attribution.model';

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
  attribution = input.required<BrinsonAttribution>();

  private fmt = inject(FormatService);

  waterfallCategories = computed(() => ['Allocation', 'Selection', 'Interaction']);

  waterfallValues = computed(() => {
    const a = this.attribution();
    return [a.totalAllocation, a.totalSelection, a.totalInteraction];
  });

  sectorTableColumns: TableColumn[] = [
    { key: 'sector', label: 'Sector', sortable: true },
    { key: 'portfolioWeight', label: 'Portfolio Wt', type: 'percentage', sortable: true },
    { key: 'benchmarkWeight', label: 'Benchmark Wt', type: 'percentage', sortable: true },
    { key: 'portfolioReturn', label: 'Portfolio Ret', type: 'percentage', sortable: true },
    { key: 'benchmarkReturn', label: 'Benchmark Ret', type: 'percentage', sortable: true },
    { key: 'allocationEffect', label: 'Allocation', type: 'percentage', sortable: true, colorBySign: true },
    { key: 'selectionEffect', label: 'Selection', type: 'percentage', sortable: true, colorBySign: true },
    { key: 'interactionEffect', label: 'Interaction', type: 'percentage', sortable: true, colorBySign: true },
    { key: 'totalEffect', label: 'Total', type: 'percentage', sortable: true, colorBySign: true },
  ];

  sectorTableRows = computed(() =>
    this.attribution().sectors.map(s => ({
      sector: s.sector,
      portfolioWeight: s.portfolioWeight,
      benchmarkWeight: s.benchmarkWeight,
      portfolioReturn: s.portfolioReturn,
      benchmarkReturn: s.benchmarkReturn,
      allocationEffect: s.allocationEffect,
      selectionEffect: s.selectionEffect,
      interactionEffect: s.interactionEffect,
      totalEffect: s.totalEffect,
    })),
  );

  cumulativeAreaSeries = computed<AreaSeries[]>(() => {
    const sectors = this.attribution().sectors;
    const labels = sectors.map(s => s.sector);

    const allocationValues = sectors.map(s => s.allocationEffect);
    const selectionValues = sectors.map(s => s.selectionEffect);
    const interactionValues = sectors.map(s => s.interactionEffect);

    const cumSum = (arr: number[]): number[] => {
      let running = 0;
      return arr.map(v => {
        running += v;
        return Math.round(running * 1e6) / 1e6;
      });
    };

    return [
      { name: 'Allocation', values: cumSum(allocationValues) },
      { name: 'Selection', values: cumSum(selectionValues) },
      { name: 'Interaction', values: cumSum(interactionValues) },
    ];
  });

  cumulativeAreaLabels = computed(() =>
    this.attribution().sectors.map(s => s.sector),
  );

  formatPercent(v: number): string {
    return this.fmt.formatPercent(v);
  }
}
