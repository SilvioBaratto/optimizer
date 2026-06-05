import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, makeBrinsonResponse } from '../../../testing';
import { SaaTaaPanelComponent } from './saa-taa-panel';

describe('SaaTaaPanelComponent', () => {
  let fixture: ComponentFixture<SaaTaaPanelComponent>;
  let comp: SaaTaaPanelComponent;

  beforeEach(async () => {
    await configureTestBed({ imports: [SaaTaaPanelComponent], withHttp: false });
    fixture = TestBed.createComponent(SaaTaaPanelComponent);
    comp = fixture.componentInstance;
  });

  it('when brinson is null, multiLevel is empty and hasData is false', () => {
    expect(comp.hasData()).toBe(false);
    expect(comp.multiLevel()).toEqual([]);
  });

  it('when populated, multiLevel derives one sector-level row from the brinson sectors', () => {
    fixture.componentRef.setInput('brinson', makeBrinsonResponse());
    expect(comp.hasData()).toBe(true);
    expect(comp.multiLevel().length).toBe(1);
    expect(comp.multiLevel()[0].level).toBe('Sector');
    expect(comp.multiLevel()[0].name).toBe('Technology');
    expect(comp.waterfallValues()).toEqual([30]); // totalEffect 0.003 × 10000
  });
});
