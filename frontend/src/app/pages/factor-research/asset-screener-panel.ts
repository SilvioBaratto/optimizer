import {
  Component,
  input,
  signal,
  computed,
  ChangeDetectionStrategy,
} from '@angular/core';
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { EchartsScatterComponent, ScatterPoint } from '../../shared/echarts-scatter/echarts-scatter';
import type { FactorICReport } from '../../models/factor.model';

const IC_BADGE_MAP: Record<string, { value: string; colorClass: string }> = {
  true: { value: 'YES', colorClass: 'bg-emerald-500/15 text-emerald-400' },
  false: { value: 'NO', colorClass: 'bg-zinc-500/15 text-zinc-400' },
};

@Component({
  selector: 'app-asset-screener-panel',
  imports: [DataTableComponent, EchartsScatterComponent],
  templateUrl: './asset-screener-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class AssetScreenerPanelComponent {
  icReports = input<FactorICReport[]>([]);

  activeFilters = signal<string[]>([]);

  availableGroups = computed(() => {
    const groups = this.icReports().map(r => r.group);
    return [...new Set(groups)];
  });

  filteredReports = computed(() => {
    const filters = this.activeFilters();
    const reports = this.icReports();
    if (filters.length === 0) return reports;
    return reports.filter(r => filters.includes(r.group));
  });

  readonly icColumns: TableColumn[] = [
    { key: 'factor', label: 'Factor', type: 'text', sortable: true },
    { key: 'group', label: 'Group', type: 'text', sortable: true },
    { key: 'ic', label: 'IC', type: 'ratio', sortable: true, colorBySign: true },
    { key: 'icir', label: 'ICIR', type: 'ratio', sortable: true, colorBySign: true },
    { key: 'tStat', label: 't-Stat', type: 'ratio', sortable: true, colorBySign: true },
    { key: 'pValue', label: 'p-Value', type: 'ratio', sortable: true },
    { key: 'vif', label: 'VIF', type: 'ratio', sortable: true },
    {
      key: 'significant',
      label: 'Significant',
      type: 'badge',
      badgeMap: IC_BADGE_MAP,
    },
  ];

  tableRows = computed(() =>
    this.filteredReports().map(r => ({
      factor: r.factor.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
      group: r.group.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
      ic: r.ic,
      icir: r.icir,
      tStat: r.tStat,
      pValue: r.pValue,
      vif: r.vif,
      significant: String(r.significant),
    }) as Record<string, unknown>),
  );

  scatterPoints = computed<ScatterPoint[]>(() =>
    this.filteredReports().map(r => ({
      x: r.ic,
      y: r.icir,
      label: r.factor.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
    })),
  );

  isGroupActive(group: string): boolean {
    return this.activeFilters().includes(group);
  }

  toggleGroup(group: string) {
    this.activeFilters.update(filters => {
      if (filters.includes(group)) {
        return filters.filter(g => g !== group);
      }
      return [...filters, group];
    });
  }
}
