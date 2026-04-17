import {
  ChangeDetectionStrategy,
  Component,
  computed,
  input,
  signal,
} from '@angular/core';
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import type {
  BrinsonApiResponse,
  BrinsonSectorRowDto,
} from '../../models/attribution.model';

type ViewMode = 'top10' | 'bottom10' | 'all';

interface HoldingRow extends Record<string, unknown> {
  sector: string;
  portfolioWeight: number;
  benchmarkWeight: number;
  portfolioReturn: number;
  benchmarkReturn: number;
  activeWeight: number;
  contribution: number;
}

@Component({
  selector: 'app-holdings-attribution-panel',
  imports: [DataTableComponent],
  templateUrl: './holdings-attribution-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class HoldingsAttributionPanelComponent {
  readonly brinson = input<BrinsonApiResponse | null>(null);
  readonly portfolioWeights = input<Record<string, number>>({});
  readonly viewMode = signal<ViewMode>('top10');

  readonly hasData = computed(() => (this.brinson()?.sectors.length ?? 0) > 0);

  readonly viewModes: { value: ViewMode; label: string }[] = [
    { value: 'top10', label: 'Top 10' },
    { value: 'bottom10', label: 'Bottom 10' },
    { value: 'all', label: 'All' },
  ];

  readonly holdingsTableColumns: TableColumn[] = [
    { key: 'sector', label: 'Sector', sortable: true },
    { key: 'portfolioWeight', label: 'Port Wt', type: 'percentage', sortable: true, align: 'right' },
    { key: 'benchmarkWeight', label: 'Bench Wt', type: 'percentage', sortable: true, align: 'right' },
    { key: 'activeWeight', label: 'Active Wt', type: 'percentage', sortable: true, align: 'right', colorBySign: true },
    { key: 'portfolioReturn', label: 'Port Ret', type: 'percentage', sortable: true, align: 'right' },
    { key: 'benchmarkReturn', label: 'Bench Ret', type: 'percentage', sortable: true, align: 'right' },
    { key: 'contribution', label: 'Contribution', type: 'percentage', sortable: true, align: 'right', colorBySign: true },
  ];

  readonly tickerHoldings = computed(() =>
    Object.entries(this.portfolioWeights())
      .map(([ticker, weight]) => ({ ticker, weight }))
      .sort((a, b) => b.weight - a.weight),
  );

  readonly hasPerTicker = computed(() => this.tickerHoldings().length > 0);

  readonly tickerTableColumns: TableColumn[] = [
    { key: 'ticker', label: 'Ticker', sortable: true },
    { key: 'weight', label: 'Weight', type: 'percentage', sortable: true, align: 'right' },
  ];

  readonly tickerTableRows = computed(() =>
    this.tickerHoldings().map((t) => ({
      ticker: t.ticker,
      weight: t.weight,
    }) as Record<string, unknown>),
  );

  readonly filteredRows = computed<HoldingRow[]>(() => {
    const sectors = this.brinson()?.sectors ?? [];
    const rows = sectors.map((s) => this.sectorToRow(s));
    return this.applyViewMode(rows);
  });

  setViewMode(mode: ViewMode): void {
    this.viewMode.set(mode);
  }

  private sectorToRow(s: BrinsonSectorRowDto): HoldingRow {
    return {
      sector: s.sector,
      portfolioWeight: s.portfolioWeight,
      benchmarkWeight: s.benchmarkWeight,
      portfolioReturn: s.portfolioReturn,
      benchmarkReturn: s.benchmarkReturn,
      activeWeight: s.portfolioWeight - s.benchmarkWeight,
      contribution: s.totalEffect,
    };
  }

  private applyViewMode(rows: HoldingRow[]): HoldingRow[] {
    const mode = this.viewMode();
    const sorted = [...rows].sort((a, b) => b.contribution - a.contribution);
    if (mode === 'top10') return sorted.slice(0, 10);
    if (mode === 'bottom10') return sorted.slice(-10).reverse();
    return sorted;
  }
}
