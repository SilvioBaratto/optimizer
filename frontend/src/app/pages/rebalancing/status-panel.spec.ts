import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, makeDriftResponse } from '../../../testing';
import { StatusPanelComponent } from './status-panel';

describe('StatusPanelComponent', () => {
  let fixture: ComponentFixture<StatusPanelComponent>;
  let comp: StatusPanelComponent;

  beforeEach(async () => {
    await configureTestBed({ imports: [StatusPanelComponent], withHttp: false });
    fixture = TestBed.createComponent(StatusPanelComponent);
    comp = fixture.componentInstance;
  });

  it('when drift is null, totals are zero and there are no rows', () => {
    expect(comp.totalDrift()).toBe(0);
    expect(comp.breachedCount()).toBe(0);
    expect(comp.driftRows()).toEqual([]);
  });

  it('when populated, drift rows and totals derive from the response', () => {
    fixture.componentRef.setInput('drift', makeDriftResponse({ totalDrift: 0.07, breachedCount: 1 }));
    expect(comp.totalDrift()).toBe(0.07);
    expect(comp.breachedCount()).toBe(1);
    expect(comp.driftRows().length).toBe(1);
    expect(comp.driftRows()[0]['status']).toBe('false'); // breached false → 'false'
  });

  it('when the threshold input changes, thresholdChange emits the value', () => {
    let value: number | undefined;
    comp.thresholdChange.subscribe((v) => (value = v));
    comp.onThresholdInput(0.1);
    expect(value).toBe(0.1);
  });
});
