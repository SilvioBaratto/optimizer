import {
  Component,
  inject,
  input,
  output,
  signal,
  computed,
  ChangeDetectionStrategy,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { LucideAngularModule } from 'lucide-angular';
import { DataTableComponent, TableColumn } from '../../shared/data-table/data-table';
import { UniverseTicker } from '../../models/portfolio-builder.model';
import { FormatService } from '../../services/format.service';

@Component({
  selector: 'app-universe-panel',
  imports: [FormsModule, DataTableComponent, LucideAngularModule],
  templateUrl: './universe-panel.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class UniversePanelComponent {
  private readonly fmt = inject(FormatService);

  tickers = input.required<UniverseTicker[]>();
  tickersChange = output<UniverseTicker[]>();

  searchTerm = signal('');
  filterSector = signal('all');

  uniqueSectors = computed(() => {
    const sectors = this.tickers().map((t) => t.sector);
    return ['all', ...Array.from(new Set(sectors)).sort()];
  });

  filteredTickers = computed(() => {
    const search = this.searchTerm().toLowerCase().trim();
    const sector = this.filterSector();
    return this.tickers().filter((t) => {
      const matchesSearch =
        !search ||
        t.ticker.toLowerCase().includes(search) ||
        t.name.toLowerCase().includes(search);
      const matchesSector = sector === 'all' || t.sector === sector;
      return matchesSearch && matchesSector;
    });
  });

  selectedCount = computed(
    () => this.tickers().filter((t) => t.selected).length,
  );

  tableColumns: TableColumn[] = [
    { key: 'ticker', label: 'Ticker', sortable: true, type: 'text' },
    { key: 'name', label: 'Name', sortable: true, type: 'text' },
    { key: 'sector', label: 'Sector', sortable: true, type: 'text' },
    {
      key: 'marketCap',
      label: 'Market Cap',
      sortable: true,
      align: 'right',
      format: (val: unknown) =>
        typeof val === 'number' ? this.fmt.formatCurrencyCompact(val) : '--',
    },
    {
      key: 'weight',
      label: 'Weight',
      sortable: true,
      align: 'right',
      type: 'percentage',
    },
    {
      key: 'selectedLabel',
      label: 'Selected',
      align: 'right',
      type: 'badge',
      badgeMap: {
        Yes: { value: 'Yes', colorClass: 'inline-flex items-center px-2 py-0.5 rounded-full text-label bg-gain-bg text-gain font-medium' },
        No: { value: 'No', colorClass: 'inline-flex items-center px-2 py-0.5 rounded-full text-label bg-surface-inset text-text-secondary font-medium' },
      },
    },
  ];

  tableRows = computed(() =>
    this.filteredTickers().map((t) => ({
      ticker: t.ticker,
      name: t.name,
      sector: t.sector,
      marketCap: t.marketCap,
      weight: t.weight,
      selectedLabel: t.selected ? 'Yes' : 'No',
    })),
  );

  applyPreset(preset: 'sp500' | 'msci' | 'sector_leaders') {
    const limits: Record<typeof preset, number> = {
      sp500: 30,
      msci: 50,
      sector_leaders: 15,
    };
    const limit = limits[preset];
    const updated = this.tickers().map((t, i) => ({
      ...t,
      selected: i < limit,
    }));
    this.tickersChange.emit(updated);
  }

  selectAll() {
    const filtered = this.filteredTickers();
    const filteredTickers = new Set(filtered.map((t) => t.ticker));
    const updated = this.tickers().map((t) => ({
      ...t,
      selected: filteredTickers.has(t.ticker) ? true : t.selected,
    }));
    this.tickersChange.emit(updated);
  }

  deselectAll() {
    const filtered = this.filteredTickers();
    const filteredTickers = new Set(filtered.map((t) => t.ticker));
    const updated = this.tickers().map((t) => ({
      ...t,
      selected: filteredTickers.has(t.ticker) ? false : t.selected,
    }));
    this.tickersChange.emit(updated);
  }
}
