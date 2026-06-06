import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, makeBrinsonResponse } from '../../../testing';
import { HoldingsAttributionPanelComponent } from './holdings-attribution-panel';
import type { BrinsonSectorRowDto } from '../attribution.model';

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
