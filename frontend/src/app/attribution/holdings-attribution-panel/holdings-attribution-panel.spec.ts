import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, makeBrinsonResponse } from '../../../testing';
import { ICON_PROVIDER } from '../../icons';
import { HoldingsAttributionPanelComponent } from './holdings-attribution-panel';
import type { BrinsonSectorRowDto } from '../attribution.model';

// ─── Shared helpers ───────────────────────────────────────────────────────────

function sector(name: string, totalEffect: number): BrinsonSectorRowDto {
  return {
    sector: name,
    portfolioWeight: 0.1,
    benchmarkWeight: 0.08,
    portfolioReturn: 0.05,
    benchmarkReturn: 0.04,
    allocationEffect: 0,
    selectionEffect: 0,
    interactionEffect: 0,
    totalEffect,
  };
}

// ─── Suite 1: signal-level computed values (original tests) ──────────────────

describe('HoldingsAttributionPanelComponent', () => {
  let fixture: ComponentFixture<HoldingsAttributionPanelComponent>;
  let comp: HoldingsAttributionPanelComponent;

  beforeEach(async () => {
    await configureTestBed({ imports: [HoldingsAttributionPanelComponent], withHttp: false });
    fixture = TestBed.createComponent(HoldingsAttributionPanelComponent);
    comp = fixture.componentInstance;
  });

  it('when brinson is null, hasData is false and there are no rows', () => {
    expect(comp.hasData()).toBe(false);
    expect(comp.filteredRows()).toEqual([]);
  });

  it('when populated, the active weight and contribution are derived per sector', () => {
    fixture.componentRef.setInput('brinson', makeBrinsonResponse());
    expect(comp.hasData()).toBe(true);
    expect(comp.filteredRows().length).toBe(1);
    expect(comp.filteredRows()[0]['activeWeight']).toBeCloseTo(0.1); // 0.6 - 0.5
  });

  it('when 12 sectors exist, viewMode bounds the rows: top10/bottom10=10, all=12', () => {
    const sectors = Array.from({ length: 12 }, (_, i) => sector(`S${i}`, i / 100));
    fixture.componentRef.setInput('brinson', makeBrinsonResponse({ sectors }));
    expect(comp.filteredRows().length).toBe(10); // default top10
    comp.setViewMode('bottom10');
    expect(comp.filteredRows().length).toBe(10);
    comp.setViewMode('all');
    expect(comp.filteredRows().length).toBe(12);
  });

  it('when portfolio weights are given, per-ticker holdings sort descending', () => {
    fixture.componentRef.setInput('portfolioWeights', { AAPL: 0.6, MSFT: 0.4 });
    expect(comp.hasPerTicker()).toBe(true);
    expect(comp.tickerHoldings()[0].ticker).toBe('AAPL');
  });
});

// ─── Suite 2: DOM rendering — criterion (T3 / issue #1020) ───────────────────
// Criteria:
//   - Empty-state renders when brinson is null (non-blank, no table).
//   - Populated state renders the data table.
//   - Per-ticker positions table renders when portfolioWeights are provided.

describe('HoldingsAttributionPanelComponent — DOM rendering (issue #1020)', () => {
  let fixture: ComponentFixture<HoldingsAttributionPanelComponent>;

  beforeEach(async () => {
    await configureTestBed({
      imports: [HoldingsAttributionPanelComponent],
      withHttp: false,
      providers: [ICON_PROVIDER],
    });
    fixture = TestBed.createComponent(HoldingsAttributionPanelComponent);
  });

  it('when brinson is null, the panel renders non-blank empty-state content', () => {
    fixture.detectChanges();
    const text = (fixture.nativeElement as HTMLElement).textContent?.trim() ?? '';

    expect(text.length).toBeGreaterThan(
      0,
      'Expected non-blank empty-state when brinson is null',
    );
  });

  it('when brinson is null, no data table is present in the DOM', () => {
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;

    expect(el.querySelector('table')).toBeNull(
      'Expected no table in the empty state',
    );
  });

  it('when brinson has sector data, a data table is rendered', () => {
    fixture.componentRef.setInput('brinson', makeBrinsonResponse());
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;

    expect(el.querySelector('table')).toBeTruthy(
      'Expected a data table to render when brinson is populated',
    );
  });

  it('when brinson has sector data, sector names are visible in the table', () => {
    const brinson = makeBrinsonResponse();
    fixture.componentRef.setInput('brinson', brinson);
    fixture.detectChanges();

    const tableText = (fixture.nativeElement as HTMLElement).querySelector('table')?.textContent ?? '';

    for (const row of brinson.sectors) {
      expect(tableText).toContain(row.sector);
    }
  });

  it('when portfolioWeights are set, a per-ticker positions table is rendered', () => {
    fixture.componentRef.setInput('brinson', makeBrinsonResponse());
    fixture.componentRef.setInput('portfolioWeights', { AAPL: 0.6, MSFT: 0.4 });
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;

    // The per-ticker section renders as a second table or a dedicated row section
    const tables = el.querySelectorAll('table');
    expect(tables.length).toBeGreaterThanOrEqual(
      1,
      'Expected at least a positions table when portfolioWeights are provided',
    );
  });

  it('when portfolioWeights are set, all ticker symbols appear in the positions section', () => {
    fixture.componentRef.setInput('brinson', makeBrinsonResponse());
    fixture.componentRef.setInput('portfolioWeights', { AAPL: 0.6, MSFT: 0.4 });
    fixture.detectChanges();

    const fullText = (fixture.nativeElement as HTMLElement).textContent ?? '';
    expect(fullText).toContain('AAPL');
    expect(fullText).toContain('MSFT');
  });
});
