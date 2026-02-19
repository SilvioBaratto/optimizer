import { Component, input, output, computed, signal, ChangeDetectionStrategy } from '@angular/core';

export interface TableColumn {
  key: string;
  label: string;
  sortable?: boolean;
  align?: 'left' | 'right';
  format?: (value: unknown) => string;
}

@Component({
  selector: 'app-data-table',
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

  pageChange = output<number>();
  sortChange = output<{ key: string; direction: 'asc' | 'desc' }>();
  rowClick = output<Record<string, unknown>>();

  currentPage = signal(1);
  sortKey = signal('');
  sortDir = signal<'asc' | 'desc'>('asc');

  totalPages = computed(() => Math.max(1, Math.ceil(this.total() / this.pageSize())));

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

  onSort(col: TableColumn) {
    if (!col.sortable) return;
    if (this.sortKey() === col.key) {
      this.sortDir.update(d => d === 'asc' ? 'desc' : 'asc');
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

  getCellValue(row: Record<string, unknown>, col: TableColumn): string {
    const val = row[col.key];
    if (col.format) return col.format(val);
    if (val == null) return '—';
    return String(val);
  }
}
