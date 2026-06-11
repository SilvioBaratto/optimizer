import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed } from '../../../testing';
import { ICON_PROVIDER } from '../../icons';
import { DataTableComponent, type TableColumn } from './data-table';

// Render-coverage spec: mounts the template to exercise every cell-type arm,
// the colour-by-sign branches, toolbar/pagination/filter DOM interactions and
// sticky/dense class bindings. Logic-only assertions live in data-table.spec.ts.

const COLS: TableColumn[] = [
  { key: 'name', label: 'Name', sortable: true, filterLabel: 'Name' },
  { key: 'num', label: 'Num', type: 'number', align: 'right' },
  { key: 'pct', label: 'Pct', type: 'percentage', colorBySign: true },
  { key: 'cur', label: 'Cur', type: 'currency', currency: 'EUR' },
  { key: 'bps', label: 'Bps', type: 'bps' },
  { key: 'ratio', label: 'Ratio', type: 'ratio' },
  { key: 'date', label: 'Date', type: 'date', dateFormat: 'iso' },
  { key: 'st', label: 'Status', type: 'badge', badgeMap: { ok: { value: 'OK', colorClass: 'text-gain' } } },
  { key: 'fmt', label: 'Fmt', format: (v) => `F:${v}` },
  { key: 'txt', label: 'Txt', type: 'text', hiddenOnMobile: true },
  { key: 'none', label: 'None' },
];

const ROWS: Record<string, unknown>[] = [
  { name: 'A', num: 1000, pct: 0.05, cur: 12.5, bps: 0.001, ratio: 1.234, date: '2024-01-02', st: 'ok', fmt: 'x', txt: 'hello', none: null },
  { name: 'B', num: -2, pct: -0.03, cur: 0, bps: 0.002, ratio: 0, date: '2024-02-02', st: 'bad', fmt: 'y', txt: 'world', none: 5 },
  { name: 'C', num: 'n/a', pct: 0, cur: null, bps: null, ratio: null, date: null, st: 'ok', fmt: 'z', txt: 1, none: 'x' },
  { name: 'D', num: 3, pct: 'flat', cur: 9, bps: 0.003, ratio: 2, date: '2024-03-02', st: 'ok', fmt: 'w', txt: 'z', none: 2 },
];

describe('DataTableComponent render-coverage', () => {
  let fixture: ComponentFixture<DataTableComponent>;
  let comp: DataTableComponent;

  beforeEach(async () => {
    await configureTestBed({ imports: [DataTableComponent], withHttp: false, providers: [ICON_PROVIDER] });
    fixture = TestBed.createComponent(DataTableComponent);
    comp = fixture.componentInstance;
    fixture.componentRef.setInput('columns', COLS);
    fixture.componentRef.setInput('rows', ROWS);
  });

  it('when populated, one tbody row renders per filtered row', () => {
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelectorAll('tbody tr').length).toBe(ROWS.length);
  });

  it('when a badge column is present, a rounded-full badge span renders', () => {
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('span.rounded-full')).not.toBeNull();
  });

  it('when sticky and dense inputs are set, the header carries the sticky/dense classes', () => {
    fixture.componentRef.setInput('stickyHeader', true);
    fixture.componentRef.setInput('stickyFirstColumn', true);
    fixture.componentRef.setInput('dense', true);
    fixture.detectChanges();
    const thead = fixture.nativeElement.querySelector('thead') as HTMLElement;
    expect(thead.className).toContain('sticky');
    const firstTh = fixture.nativeElement.querySelector('thead th') as HTMLElement;
    expect(firstTh.className).toContain('px-2');
  });

  it('renders each cell type through getCellValue without error', () => {
    fixture.detectChanges();
    expect(comp.getCellValue(ROWS[0], COLS[1])).toBe('1,000'); // number
    expect(comp.getCellValue(ROWS[0], COLS[2])).toBe('5.00%'); // percentage
    expect(comp.getCellValue(ROWS[0], COLS[3])).toContain('12.50'); // currency EUR
    expect(comp.getCellValue(ROWS[0], COLS[4])).toBe('10 bps'); // bps
    expect(comp.getCellValue(ROWS[0], COLS[5])).toBe('1.23'); // ratio
    expect(comp.getCellValue(ROWS[0], COLS[6])).toBe('2024-01-02'); // date iso
    expect(comp.getCellValue(ROWS[0], COLS[8])).toBe('F:x'); // format fn
    expect(comp.getCellValue(ROWS[0], COLS[10])).toBe('—'); // null → em dash
  });

  it('falls back to String(val) when a numeric-typed cell holds a non-number', () => {
    expect(comp.getCellValue(ROWS[2], COLS[1])).toBe('n/a');
  });

  it('colours cells by sign for a colorBySign column', () => {
    expect(comp.getCellColorClass(ROWS[0], COLS[2])).toBe('text-gain'); // +0.05
    expect(comp.getCellColorClass(ROWS[1], COLS[2])).toBe('text-loss'); // -0.03
    expect(comp.getCellColorClass(ROWS[2], COLS[2])).toBe('text-flat'); // 0
    expect(comp.getCellColorClass(ROWS[3], COLS[2])).toBe(''); // non-number
    expect(comp.getCellColorClass(ROWS[0], COLS[1])).toBe(''); // not colorBySign
  });

  it('returns an empty badge class for non-badge or unmapped values', () => {
    expect(comp.getBadgeClasses(ROWS[1], COLS[7])).toBe(''); // 'bad' unmapped
    expect(comp.getBadgeClasses(ROWS[0], COLS[1])).toBe(''); // not a badge column
  });

  it('marks numeric and right-aligned columns as numeric', () => {
    expect(comp.isNumericColumn(COLS[1])).toBe(true); // type number + align right
    expect(comp.isNumericColumn(COLS[9])).toBe(false); // text
  });

  describe('toolbar interactions', () => {
    it('when showExport is on, clicking Export CSV runs the CSV export', () => {
      fixture.componentRef.setInput('showExport', true);
      fixture.detectChanges();
      const createSpy = spyOn(URL, 'createObjectURL').and.returnValue('blob:csv');
      spyOn(URL, 'revokeObjectURL');
      const btn = [...fixture.nativeElement.querySelectorAll('button')].find(
        (b: HTMLButtonElement) => b.textContent?.includes('Export CSV'),
      ) as HTMLButtonElement;
      btn.click();
      expect(createSpy).toHaveBeenCalled();
    });

    it('when showCopy is on, clicking Copy writes to the clipboard', () => {
      fixture.componentRef.setInput('showCopy', true);
      fixture.detectChanges();
      // spyOn auto-restores in afterEach — do NOT replace navigator.clipboard
      // wholesale (a permanent stub leaks a spy into later specs).
      const writeText = spyOn(navigator.clipboard, 'writeText').and.returnValue(
        Promise.resolve(),
      );
      const btn = [...fixture.nativeElement.querySelectorAll('button')].find(
        (b: HTMLButtonElement) => b.textContent?.includes('Copy'),
      ) as HTMLButtonElement;
      btn.click();
      expect(writeText).toHaveBeenCalled();
    });
  });

  describe('header sort and filter interactions', () => {
    it('when a sortable header is clicked, sortChange emits', () => {
      fixture.detectChanges();
      let event: { key: string; direction: string } | undefined;
      comp.sortChange.subscribe((e) => (event = e));
      const headerCells = fixture.nativeElement.querySelectorAll('thead th');
      (headerCells[0] as HTMLElement).click(); // 'Name' is sortable
      expect(event).toEqual({ key: 'name', direction: 'asc' });
    });

    it('when a filter input receives text, only matching rows remain', () => {
      fixture.detectChanges();
      const input = fixture.nativeElement.querySelector('input[type="text"]') as HTMLInputElement;
      input.value = 'app';
      input.dispatchEvent(new Event('input'));
      fixture.detectChanges();
      expect(comp.filteredRows().length).toBe(0); // no 'app' match among A/B/C/D names
    });
  });

  describe('pagination interactions', () => {
    beforeEach(() => {
      fixture.componentRef.setInput('total', 30);
      fixture.componentRef.setInput('pageSize', 15);
      fixture.detectChanges();
    });

    it('renders pagination controls when more than one page exists', () => {
      expect(fixture.nativeElement.textContent).toContain('Page 1 of 2');
    });

    it('disables Prev on the first page and advances on Next', () => {
      const buttons = [...fixture.nativeElement.querySelectorAll('button')] as HTMLButtonElement[];
      const prev = buttons.find((b) => b.textContent?.trim() === 'Prev')!;
      const next = buttons.find((b) => b.textContent?.trim() === 'Next')!;
      expect(prev.disabled).toBe(true);

      const pages: number[] = [];
      comp.pageChange.subscribe((p) => pages.push(p));
      next.click();
      expect(pages).toEqual([2]);
      expect(comp.currentPage()).toBe(2);
    });
  });
});
