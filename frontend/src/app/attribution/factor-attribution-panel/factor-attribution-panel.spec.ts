import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureTestBed, makeFactorAttributionResponse } from '../../../testing';
import { FactorAttributionPanelComponent } from './factor-attribution-panel';

describe('FactorAttributionPanelComponent', () => {
  let fixture: ComponentFixture<FactorAttributionPanelComponent>;
  let comp: FactorAttributionPanelComponent;

  beforeEach(async () => {
    await configureTestBed({ imports: [FactorAttributionPanelComponent], withHttp: false });
    fixture = TestBed.createComponent(FactorAttributionPanelComponent);
    comp = fixture.componentInstance;
  });

  it('when response is null, KPIs show the dash placeholder and hasData is false', () => {
    expect(comp.hasData()).toBe(false);
    expect(comp.kpiPortfolioReturn()).toBe('—');
    expect(comp.kpiExplained()).toBe('—');
    expect(comp.kpiResidual()).toBe('—');
    expect(comp.factorTableRows()).toEqual([]);
  });

  it('when populated, KPIs and the factor table derive from the response', () => {
    fixture.componentRef.setInput('response', makeFactorAttributionResponse());
    expect(comp.hasData()).toBe(true);
    expect(comp.kpiResidual()).toBe('2.00%');
    expect(comp.factorTableRows().length).toBe(1);
    expect(comp.factorTableRows()[0]['factorName']).toBe('momentum');
  });

  // ── issue #997, 13c: 404 empty-state vs generic "no run yet" ───────────────

  it('when notAvailable is true and there is no data, the "Factor scores not available" message is shown', () => {
    fixture.componentRef.setInput('notAvailable', true);
    fixture.detectChanges();
    const text = (fixture.nativeElement as HTMLElement).textContent ?? '';
    expect(text).toContain('Factor scores not available');
  });

  it('when notAvailable is false and there is no data, the generic "no run yet" message is shown', () => {
    fixture.componentRef.setInput('notAvailable', false);
    fixture.detectChanges();
    const text = (fixture.nativeElement as HTMLElement).textContent ?? '';
    expect(text).toContain('No factor attribution result yet');
    expect(text).not.toContain('Factor scores not available');
  });

  it('when data is present, the not-available empty-state is not shown even if notAvailable is true', () => {
    fixture.componentRef.setInput('response', makeFactorAttributionResponse());
    fixture.componentRef.setInput('notAvailable', true);
    fixture.detectChanges();
    const text = (fixture.nativeElement as HTMLElement).textContent ?? '';
    expect(text).not.toContain('Factor scores not available');
  });
});
