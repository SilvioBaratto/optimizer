import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, makeBrinsonResponse } from '../../../testing';
import { BrinsonPanelComponent } from './brinson-panel';

describe('BrinsonPanelComponent', () => {
  let fixture: ComponentFixture<BrinsonPanelComponent>;
  let comp: BrinsonPanelComponent;

  beforeEach(async () => {
    await configureTestBed({ imports: [BrinsonPanelComponent], withHttp: false });
    fixture = TestBed.createComponent(BrinsonPanelComponent);
    comp = fixture.componentInstance;
  });

  it('when response is null, hasData is false and the waterfall is zeroed', () => {
    expect(comp.hasData()).toBe(false);
    expect(comp.waterfallValues()).toEqual([0, 0, 0, 0]);
    expect(comp.sectorTableRows()).toEqual([]);
  });

  it('when populated, hasData is true and the waterfall derives from the totals (bps)', () => {
    fixture.componentRef.setInput('response', makeBrinsonResponse());
    expect(comp.hasData()).toBe(true);
    expect(comp.waterfallValues()).toEqual([20, 10, 0, 30]); // totals × 10000
  });

  it('when populated, the sector table maps one row per sector', () => {
    fixture.componentRef.setInput('response', makeBrinsonResponse());
    expect(comp.sectorTableRows().length).toBe(1);
    expect(comp.sectorTableRows()[0]['sector']).toBe('Technology');
  });
});
