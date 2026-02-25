import {
  Component,
  input,
  signal,
  computed,
  ChangeDetectionStrategy,
} from '@angular/core';
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { EchartsScatterComponent, ScatterPoint } from '../../shared/echarts-scatter/echarts-scatter';
import type { CMASet } from '../../models/factor.model';

@Component({
  selector: 'app-cma-builder-panel',
  imports: [DataTableComponent, EchartsScatterComponent],
  templateUrl: './cma-builder-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class CmaBuilderPanelComponent {
  cmaSets = input<CMASet[]>([]);

  selectedSet = signal(0);

  activeSet = computed(() => this.cmaSets()[this.selectedSet()]);

  readonly returnsColumns: TableColumn[] = [
    { key: 'ticker', label: 'Asset', type: 'text', sortable: true },
    { key: 'expectedReturn', label: 'Expected Return', type: 'percentage', sortable: true, colorBySign: true },
    { key: 'expectedVol', label: 'Expected Vol', type: 'percentage', sortable: true },
  ];

  returnsRows = computed(() => {
    const set = this.activeSet();
    if (!set) return [];
    return set.assets.map(a => ({
      ticker: a.ticker,
      expectedReturn: a.expectedReturn,
      expectedVol: a.expectedVol,
    }) as Record<string, unknown>);
  });

  scatterPoints = computed<ScatterPoint[]>(() => {
    const set = this.activeSet();
    if (!set) return [];
    return set.assets.map(a => ({
      x: a.expectedVol,
      y: a.expectedReturn,
      label: a.ticker,
    }));
  });
}
