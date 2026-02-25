import { Component, input, signal, computed, inject, ChangeDetectionStrategy } from '@angular/core';
import { StatCardComponent } from '../../shared/stat-card/stat-card';
import { EchartsWaterfallComponent } from '../../shared/echarts-waterfall/echarts-waterfall';
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { ChartToolbarComponent } from '../../shared/chart-toolbar/chart-toolbar';
import { FormatService } from '../../services/format.service';
import type { StressScenario } from '../../models/risk.model';

const SECTOR_IMPACTS: Record<string, { categories: string[]; values: number[] }> = {
  s1: { categories: ['Technology', 'Healthcare', 'Financials', 'Consumer', 'Energy', 'Industrials', 'Utilities', 'Total'], values: [-0.08, -0.04, -0.06, -0.05, -0.04, -0.03, -0.01, -0.285] },
  s2: { categories: ['Technology', 'Healthcare', 'Financials', 'Consumer', 'Energy', 'Industrials', 'Utilities', 'Total'], values: [-0.06, -0.03, -0.12, -0.05, -0.04, -0.05, -0.02, -0.382] },
  s3: { categories: ['Technology', 'Healthcare', 'Financials', 'Consumer', 'Energy', 'Industrials', 'Utilities', 'Total'], values: [-0.14, -0.03, -0.04, -0.03, -0.02, -0.04, -0.02, -0.312] },
  s4: { categories: ['Technology', 'Healthcare', 'Financials', 'Consumer', 'Energy', 'Industrials', 'Utilities', 'Total'], values: [-0.03, -0.01, -0.02, -0.01, -0.01, -0.01, -0.02, -0.108] },
  s5: { categories: ['Technology', 'Healthcare', 'Financials', 'Consumer', 'Energy', 'Industrials', 'Utilities', 'Total'], values: [-0.06, -0.02, -0.02, -0.03, -0.01, -0.02, -0.01, -0.165] },
  s6: { categories: ['Technology', 'Healthcare', 'Financials', 'Consumer', 'Energy', 'Industrials', 'Utilities', 'Total'], values: [-0.02, -0.01, -0.01, -0.02, -0.02, -0.02, -0.01, -0.092] },
  s7: { categories: ['Technology', 'Healthcare', 'Financials', 'Consumer', 'Energy', 'Industrials', 'Utilities', 'Total'], values: [-0.05, -0.02, -0.04, -0.03, -0.02, -0.03, -0.01, -0.198] },
  s8: { categories: ['Technology', 'Healthcare', 'Financials', 'Consumer', 'Energy', 'Industrials', 'Utilities', 'Total'], values: [-0.03, -0.01, -0.02, -0.01, -0.01, -0.01, -0.00, -0.088] },
};

@Component({
  selector: 'app-stress-panel',
  imports: [StatCardComponent, EchartsWaterfallComponent, DataTableComponent, ChartToolbarComponent],
  templateUrl: './stress-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class StressPanelComponent {
  scenarios = input<StressScenario[]>([]);

  private fmt = inject(FormatService);

  selectedScenarioId = signal<string | null>(null);

  selectedScenario = computed(() => {
    const id = this.selectedScenarioId();
    if (!id) return null;
    return this.scenarios().find(s => s.id === id) ?? null;
  });

  waterfallCategories = computed(() => {
    const s = this.selectedScenario();
    if (!s) return [];
    return SECTOR_IMPACTS[s.id]?.categories ?? [];
  });

  waterfallValues = computed(() => {
    const s = this.selectedScenario();
    if (!s) return [];
    return SECTOR_IMPACTS[s.id]?.values ?? [];
  });

  impactStr = computed(() => {
    const s = this.selectedScenario();
    return s ? this.fmt.formatPercent(s.portfolioImpact) : '--';
  });

  benchmarkStr = computed(() => {
    const s = this.selectedScenario();
    return s ? this.fmt.formatPercent(s.benchmarkImpact) : '--';
  });

  worstAssetStr = computed(() => {
    const s = this.selectedScenario();
    return s ? `${s.worstAsset} (${this.fmt.formatPercent(s.worstAssetImpact)})` : '--';
  });

  comparisonColumns: TableColumn[] = [
    { key: 'name', label: 'Scenario', sortable: true },
    { key: 'portfolioImpact', label: 'Portfolio Impact', type: 'percentage', sortable: true, colorBySign: true },
    { key: 'benchmarkImpact', label: 'Benchmark Impact', type: 'percentage', sortable: true, colorBySign: true },
    { key: 'worstAsset', label: 'Worst Asset', sortable: true },
    { key: 'worstAssetImpact', label: 'Worst Asset Impact', type: 'percentage', sortable: true, colorBySign: true },
  ];

  comparisonRows = computed(() =>
    this.scenarios().map(s => ({
      name: s.name,
      portfolioImpact: s.portfolioImpact,
      benchmarkImpact: s.benchmarkImpact,
      worstAsset: s.worstAsset,
      worstAssetImpact: s.worstAssetImpact,
    })),
  );

  selectScenario(id: string) {
    this.selectedScenarioId.set(this.selectedScenarioId() === id ? null : id);
  }
}
