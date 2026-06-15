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

  // ── Issue #1019: error input ─────────────────────────────────────────────────

  it('when error is set, a role="alert" banner is rendered with the error text', () => {
    fixture.componentRef.setInput('error', 'Drift data unavailable');
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    const alert = el.querySelector('[role="alert"]');
    expect(alert).not.toBeNull();
    expect(alert?.textContent).toContain('Drift data unavailable');
  });

  it('when error is null, no role="alert" banner is rendered', () => {
    fixture.componentRef.setInput('error', null);
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('[role="alert"]')).toBeNull();
  });
});
