import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed } from '../../../testing';
import { ICON_PROVIDER } from '../../icons';
import { DataTableComponent, type TableColumn } from './data-table';

const COLS: TableColumn[] = [
  { key: 'name', label: 'Name', sortable: true, filterLabel: 'Name' },
  { key: 'val', label: 'Val', sortable: true, type: 'number' },
  { key: 'st', label: 'Status', type: 'badge', badgeMap: { ok: { value: 'OK', colorClass: 'text-gain' } } },
];

const ROWS: Record<string, unknown>[] = [
  { name: 'Banana', val: 2, st: 'ok' },
  { name: 'Apple', val: 1, st: 'bad' },
];

describe('DataTableComponent', () => {
  let fixture: ComponentFixture<DataTableComponent>;
  let comp: DataTableComponent;

  beforeEach(async () => {
    await configureTestBed({ imports: [DataTableComponent], withHttp: false, providers: [ICON_PROVIDER] });
    fixture = TestBed.createComponent(DataTableComponent);
    comp = fixture.componentInstance;
    fixture.componentRef.setInput('columns', COLS);
  });

  it('when loading, a skeleton is rendered', () => {
    fixture.componentRef.setInput('loading', true);
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.animate-pulse')).not.toBeNull();
  });

  it('when rows are empty, the empty state is shown', () => {
    fixture.componentRef.setInput('rows', []);
    fixture.detectChanges();
    expect(fixture.nativeElement.textContent).toContain('No data available');
  });

  it('when showExport is set, the toolbar is shown', () => {
    fixture.componentRef.setInput('showExport', true);
    expect(comp.showToolbar()).toBe(true);
  });

  it('when a sortable column is sorted, the rows order by that key ascending then descending', () => {
    fixture.componentRef.setInput('rows', ROWS);
    let event: { key: string; direction: string } | undefined;
    comp.sortChange.subscribe((e) => (event = e));
    comp.onSort(COLS[1]); // val asc
    expect(comp.sortedRows()[0]['val']).toBe(1);
    expect(event).toEqual({ key: 'val', direction: 'asc' });
    comp.onSort(COLS[1]); // toggle desc
    expect(comp.sortedRows()[0]['val']).toBe(2);
  });

  it('when a non-sortable column is sorted, nothing changes', () => {
    fixture.componentRef.setInput('rows', ROWS);
    comp.onSort(COLS[2]); // badge, not sortable
    expect(comp.sortKey()).toBe('');
  });

  it('when a column filter is applied, only matching rows remain', () => {
    fixture.componentRef.setInput('rows', ROWS);
    comp.onFilterChange('name', 'app');
    expect(comp.filteredRows().length).toBe(1);
    expect(comp.filteredRows()[0]['name']).toBe('Apple');
  });

  it('when total exceeds the page size, totalPages reflects the count', () => {
    fixture.componentRef.setInput('total', 30);
    fixture.componentRef.setInput('pageSize', 15);
    expect(comp.totalPages()).toBe(2);
  });

  it('when a valid page is selected, pageChange emits; out-of-range is ignored', () => {
    fixture.componentRef.setInput('total', 30);
    const pages: number[] = [];
    comp.pageChange.subscribe((p) => pages.push(p));
    comp.onPage(2);
    comp.onPage(99);
    expect(pages).toEqual([2]);
  });

  it('when not clickable, a row click does not emit; when clickable, it emits the row', () => {
    let clicked: Record<string, unknown> | undefined;
    comp.rowClick.subscribe((r) => (clicked = r));
    comp.onRowClick(ROWS[0]);
    expect(clicked).toBeUndefined();
    fixture.componentRef.setInput('clickable', true);
    comp.onRowClick(ROWS[0]);
    expect(clicked).toBe(ROWS[0]);
  });

  it('when a badge cell is rendered, its mapped value and colour class are used', () => {
    expect(comp.getCellValue(ROWS[0], COLS[2])).toBe('OK');
    expect(comp.getBadgeClasses(ROWS[0], COLS[2])).toBe('text-gain');
  });

  it('when a numeric cell is rendered, the FormatService output is used', () => {
    expect(comp.getCellValue(ROWS[1], COLS[1])).toBe('1');
  });
});
