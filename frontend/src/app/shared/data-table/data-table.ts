import { Component, input, output, computed, signal, inject, ChangeDetectionStrategy } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { FormatService } from '../../services/format.service';

export interface TableColumn {
  key: string;
  label: string;
  sortable?: boolean;
  align?: 'left' | 'right';
  format?: (value: unknown) => string;
  type?: 'text' | 'number' | 'percentage' | 'currency' | 'bps' | 'ratio' | 'date' | 'badge';
  colorBySign?: boolean;
  filterLabel?: string;
  badgeMap?: Record<string, { value: string; colorClass: string }>;
  currency?: string;
  dateFormat?: 'medium' | 'short' | 'iso';
  hiddenOnMobile?: boolean;
}

const NUMERIC_TYPES = new Set(['number', 'percentage', 'currency', 'bps', 'ratio']);

@Component({
  selector: 'app-data-table',
  imports: [FormsModule],
  templateUrl: './data-table.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class DataTableComponent {
  columns = input<TableColumn[]>([]);
  rows = input<Record<string, unknown>[]>([]);
  total = input(0);
  pageSize = input(15);
  loading = input(false);
  clickable = input(false);
  dense = input(false);
  stickyHeader = input(false);
  stickyFirstColumn = input(false);
  showExport = input(false);
  showCopy = input(false);
  exportFilename = input('export');

  pageChange = output<number>();
  sortChange = output<{ key: string; direction: 'asc' | 'desc' }>();
  rowClick = output<Record<string, unknown>>();

  currentPage = signal(1);
  sortKey = signal('');
  sortDir = signal<'asc' | 'desc'>('asc');
  columnFilters = signal<Record<string, string>>({});

  private fmt = inject(FormatService);

  totalPages = computed(() => Math.max(1, Math.ceil(this.total() / this.pageSize())));

  showToolbar = computed(() => this.showExport() || this.showCopy());

  hasFilters = computed(() => this.columns().some(c => c.filterLabel));

  sortedRows = computed(() => {
    const data = [...this.rows()];
    const key = this.sortKey();
    if (!key) return data;

    const dir = this.sortDir() === 'asc' ? 1 : -1;
    return data.sort((a, b) => {
      const av = a[key];
      const bv = b[key];
      if (typeof av === 'number' && typeof bv === 'number') return (av - bv) * dir;
      return String(av ?? '').localeCompare(String(bv ?? '')) * dir;
    });
  });

  filteredRows = computed(() => {
    const sorted = this.sortedRows();
    const filters = this.columnFilters();
    const activeFilters = Object.entries(filters).filter(([, v]) => v.trim().length > 0);
    if (activeFilters.length === 0) return sorted;

    return sorted.filter(row =>
      activeFilters.every(([key, term]) => {
        const val = row[key];
        return String(val ?? '').toLowerCase().includes(term.toLowerCase());
      }),
    );
  });

  onSort(col: TableColumn) {
    if (!col.sortable) return;
    if (this.sortKey() === col.key) {
      this.sortDir.update(d => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      this.sortKey.set(col.key);
      this.sortDir.set('asc');
    }
    this.sortChange.emit({ key: col.key, direction: this.sortDir() });
  }

  onPage(page: number) {
    if (page < 1 || page > this.totalPages()) return;
    this.currentPage.set(page);
    this.pageChange.emit(page);
  }

  onRowClick(row: Record<string, unknown>) {
    if (this.clickable()) this.rowClick.emit(row);
  }

  onFilterChange(key: string, value: string) {
    this.columnFilters.update(f => ({ ...f, [key]: value }));
  }

  isNumericColumn(col: TableColumn): boolean {
    return NUMERIC_TYPES.has(col.type ?? '') || col.align === 'right';
  }

  getCellValue(row: Record<string, unknown>, col: TableColumn): string {
    const val = row[col.key];
    if (col.format) return col.format(val);
    if (val == null) return '\u2014';

    switch (col.type) {
      case 'number':
        return typeof val === 'number' ? this.fmt.formatInteger(val) : String(val);
      case 'percentage':
        return typeof val === 'number' ? this.fmt.formatPercent(val) : String(val);
      case 'currency':
        return typeof val === 'number' ? this.fmt.formatCurrency(val, col.currency) : String(val);
      case 'bps':
        return typeof val === 'number' ? this.fmt.formatBps(val) : String(val);
      case 'ratio':
        return typeof val === 'number' ? this.fmt.formatRatio(val) : String(val);
      case 'date':
        return this.fmt.formatDate(val as Date | string, col.dateFormat);
      case 'badge':
        return col.badgeMap?.[String(val)]?.value ?? String(val);
      default:
        return String(val);
    }
  }

  getCellColorClass(row: Record<string, unknown>, col: TableColumn): string {
    if (!col.colorBySign) return '';
    const val = row[col.key];
    if (typeof val !== 'number') return '';
    if (val > 0) return 'text-gain';
    if (val < 0) return 'text-loss';
    return 'text-flat';
  }

  getBadgeClasses(row: Record<string, unknown>, col: TableColumn): string {
    if (col.type !== 'badge' || !col.badgeMap) return '';
    const val = String(row[col.key] ?? '');
    return col.badgeMap[val]?.colorClass ?? '';
  }

  exportCsv() {
    const cols = this.columns();
    const header = cols.map(c => c.label).join(',');
    const body = this.filteredRows()
      .map(row => cols.map(c => `"${String(row[c.key] ?? '').replace(/"/g, '""')}"`).join(','))
      .join('\n');

    const blob = new Blob([`${header}\n${body}`], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${this.exportFilename()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  copyToClipboard() {
    const cols = this.columns();
    const header = cols.map(c => c.label).join('\t');
    const body = this.filteredRows()
      .map(row => cols.map(c => String(row[c.key] ?? '')).join('\t'))
      .join('\n');

    navigator.clipboard.writeText(`${header}\n${body}`);
  }
}
